#!/usr/bin/env python3
"""
统一日志配置模块
为机器人端和服务端提供一致的日志记录方案
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime, timedelta
import threading
import json
import queue
import requests


# 全局请求ID存储（用于追踪分布式请求）
_request_id_storage = threading.local()


class RequestIDFilter(logging.Filter):
    """为日志记录添加请求ID（用于追踪完整请求链路）"""
    
    def filter(self, record):
        record.request_id = getattr(_request_id_storage, 'request_id', '-')
        return True


class RemoteLogHandler(logging.Handler):
    """远程日志推送Handler"""
    
    def __init__(self, log_server_url: str, device_name: str, max_queue_size: int = 1000):
        """
        初始化远程日志Handler
        
        Args:
            log_server_url: 日志服务器URL（如 http://127.0.0.1:8888）
            device_name: 设备名称（server/jetson/nuc）
            max_queue_size: 队列最大长度，超过后丢弃旧日志
        """
        super().__init__()
        self.log_server_url = log_server_url.rstrip('/')
        self.device_name = device_name
        self.max_queue_size = max_queue_size
        
        # 使用队列存储日志，后台线程处理
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        
        # 启动后台推送线程
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="RemoteLogWorker")
        self._worker_thread.start()
    
    def emit(self, record):
        """发送日志记录（非阻塞，放入队列）"""
        try:
            # 格式化时间戳
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            
            # 构建日志数据
            log_data = {
                'timestamp': timestamp,
                'level': record.levelname,
                'module': record.name,
                'request_id': getattr(record, 'request_id', None),
                'message': record.getMessage(),
                'file': getattr(record, 'pathname', None),
                'line': getattr(record, 'lineno', None),
                'device': self.device_name
            }
            
            # 非阻塞放入队列，如果队列满了就丢弃（避免阻塞主流程）
            try:
                self.log_queue.put_nowait(log_data)
            except queue.Full:
                # 队列满了，丢弃这条日志（不影响主流程）
                pass
        
        except Exception:
            # 任何异常都静默处理，不影响主流程
            pass
    
    def _worker_loop(self):
        """后台工作线程：从队列取日志并发送到服务器"""
        while not self._stop_event.is_set():
            try:
                # 从队列获取日志（带超时，以便定期检查停止事件）
                try:
                    log_data = self.log_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 发送日志到服务器
                self._send_single_log(log_data)
                
                # 标记任务完成
                self.log_queue.task_done()
                
            except Exception:
                # 任何异常都静默处理，继续处理下一条日志
                pass
    
    def _send_single_log(self, log_data):
        """发送单条日志到服务器（带超时，快速失败）"""
        try:
            response = requests.post(
                f"{self.log_server_url}/api/logs",
                json=log_data,
                timeout=0.5,  # 0.5秒超时，快速失败
                headers={'Connection': 'close'}  # 确保连接关闭
            )
            # 只检查状态码，不关心具体内容
            if response.status_code != 200:
                # 非200状态码，静默失败
                pass
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException):
            # 网络错误，静默失败（不影响主流程）
            pass
        except Exception:
            # 其他异常，静默失败
            pass
    
    def close(self):
        """关闭Handler，等待队列处理完成"""
        # 设置停止事件
        self._stop_event.set()
        
        # 等待工作线程结束（最多等待2秒）
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        
        super().close()


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    is_robot: bool = False,
    remote_log_url: str = None,
    device_name: str = None
) -> logging.Logger:
    """
    创建统一配置的logger
    
    Args:
        name: logger名称（通常是模块名）
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_dir: 日志文件目录（None则使用./logs）
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志文件数量
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        json_format: 是否使用JSON格式（便于日志分析）
        is_robot: 是否为机器人端（影响日志格式的详细程度）
    
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False  # 避免重复输出
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 添加请求ID过滤器
    request_filter = RequestIDFilter()
    logger.addFilter(request_filter)
    
    # === 控制台输出 ===
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        if is_robot:
            # 机器人端：简洁格式
            console_format = '%(asctime)s [%(levelname)s] %(message)s'
        else:
            # 服务端：详细格式（包含请求ID）
            console_format = '%(asctime)s [%(levelname)s] [%(name)s] [ReqID:%(request_id)s] %(message)s'
        
        console_formatter = logging.Formatter(
            console_format,
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # === 文件输出 ===
    if file_output:
        # 确定日志目录
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        log_file = log_dir / f"{name}.log"
        
        # 使用RotatingFileHandler（按大小轮转）
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if json_format:
            # JSON格式（便于日志分析工具解析）
            file_format = '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","request_id":"%(request_id)s","message":"%(message)s"}'
        elif is_robot:
            # 机器人端：简洁格式
            file_format = '%(asctime)s [%(levelname)s] %(message)s'
        else:
            # 服务端：详细格式
            file_format = '%(asctime)s [%(levelname)s] [%(name)s] [ReqID:%(request_id)s] [%(filename)s:%(lineno)d] %(message)s'
        
        file_formatter = logging.Formatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # === 远程日志推送 ===
    if remote_log_url:
        # 自动检测设备名称
        if device_name is None:
            # 从环境变量获取，如果没有则根据模块类型推断
            device_name = os.getenv('LOG_DEVICE_NAME')
            if not device_name:
                device_name = 'jetson' if (is_robot or name in ['pipeline', 'mqtt_manager']) else 'server'
        
        try:
            remote_handler = RemoteLogHandler(
                log_server_url=remote_log_url,
                device_name=device_name,
                max_queue_size=1000  # 队列最大1000条，超过后丢弃
            )
            remote_handler.setLevel(logging.DEBUG)
            logger.addHandler(remote_handler)
        except Exception:
            # 远程日志推送失败不影响主流程
            pass
    
    return logger


def set_request_id(request_id: str):
    """设置当前线程的请求ID（用于追踪分布式请求）"""
    _request_id_storage.request_id = request_id


def get_request_id() -> str:
    """获取当前线程的请求ID"""
    return getattr(_request_id_storage, 'request_id', '-')


def clear_request_id():
    """清除当前线程的请求ID"""
    if hasattr(_request_id_storage, 'request_id'):
        delattr(_request_id_storage, 'request_id')


# === 预定义的logger配置 ===

def create_robot_logger(name: str, level: str = "INFO", remote_log_url: str = None) -> logging.Logger:
    """
    创建机器人端logger
    - 简洁格式
    - 控制台输出为主
    - 文件日志可选
    - 可选远程日志推送
    """
    # 从环境变量获取远程日志URL
    if remote_log_url is None:
        remote_log_url = os.getenv('LOG_SERVER_URL')
    
    return setup_logger(
        name=name,
        level=level,
        console_output=True,
        file_output=True,
        json_format=False,
        is_robot=True,
        remote_log_url=remote_log_url,
        device_name=os.getenv('LOG_DEVICE_NAME', 'jetson')
    )


def create_server_logger(name: str, level: str = "INFO", json_format: bool = False, remote_log_url: str = None) -> logging.Logger:
    """
    创建服务端logger
    - 详细格式
    - 支持请求ID追踪
    - 控制台+文件双输出
    - 可选JSON格式
    - 可选远程日志推送
    """
    # 从环境变量获取远程日志URL
    if remote_log_url is None:
        # remote_log_url = os.getenv('LOG_SERVER_URL')
        remote_log_url = "http://127.0.0.1:8888"
    
    return setup_logger(
        name=name,
        level=level,
        console_output=True,
        file_output=True,
        json_format=json_format,
        is_robot=False,
        remote_log_url=remote_log_url,
        device_name=os.getenv('LOG_DEVICE_NAME', 'server')
    )


# === 便捷函数：统一的日志消息格式 ===

def log_request_start(logger: logging.Logger, endpoint: str, method: str = "POST"):
    """记录请求开始"""
    logger.info(f"请求开始 [{method}] {endpoint}")


def log_request_end(logger: logging.Logger, status_code: int = None, endpoint: str = None, duration_ms: float = None, status: str = "success"):
    """记录请求结束
    
    Args:
        logger: logger实例
        status_code: HTTP状态码（如200, 400, 500）
        endpoint: 端点路径（可选）
        duration_ms: 耗时（毫秒，可选）
        status: 状态描述（可选）
    """
    if status_code is not None:
        # 简化调用：只传状态码
        logger.info(f"请求完成 状态码: {status_code}")
    elif endpoint:
        # 完整调用：包含端点和耗时
        if duration_ms is not None:
            logger.info(f"请求结束 [{endpoint}] 耗时: {duration_ms:.2f}ms 状态: {status}")
        else:
            logger.info(f"请求结束 [{endpoint}] 状态: {status}")
    else:
        logger.info(f"请求结束 状态: {status}")


def log_tool_call(logger: logging.Logger, tool_name: str, params: dict = None):
    """记录工具调用"""
    if params:
        logger.info(f"工具调用: {tool_name} 参数: {params}")
    else:
        logger.info(f"工具调用: {tool_name}")


def log_mqtt_publish(logger: logging.Logger, topic: str, payload: str):
    """记录MQTT消息发布"""
    logger.debug(f"MQTT发布 [{topic}] {payload[:100]}")


def log_mqtt_receive(logger: logging.Logger, topic: str, payload: str):
    """记录MQTT消息接收"""
    logger.debug(f"MQTT接收 [{topic}] {payload[:100]}")


def log_task_add(logger: logging.Logger, task_type: str, queue_len: int):
    """记录任务添加"""
    logger.info(f"任务入队: {task_type} 队列长度: {queue_len}")


def log_task_start(logger: logging.Logger, task_type: str):
    """记录任务开始"""
    logger.info(f"任务开始: {task_type}")


def log_task_complete(logger: logging.Logger, task_type: str, duration_s: float):
    """记录任务完成"""
    logger.info(f"任务完成: {task_type} 耗时: {duration_s:.2f}s")


def log_vad_event(logger: logging.Logger, event: str):
    """记录VAD事件"""
    logger.debug(f"VAD事件: {event}")


def log_asr_result(logger: logging.Logger, text: str):
    """记录ASR识别结果"""
    logger.info(f"ASR识别: {text}")


def log_tts_request(logger: logging.Logger, text: str):
    """记录TTS合成请求"""
    preview = text[:30] + "..." if len(text) > 30 else text
    logger.info(f"TTS合成: {preview}")


if __name__ == "__main__":
    # 测试示例
    print("=== 机器人端日志示例 ===")
    robot_logger = create_robot_logger("test_robot", level="DEBUG")
    robot_logger.debug("这是调试信息")
    robot_logger.info("这是普通信息")
    robot_logger.warning("这是警告信息")
    robot_logger.error("这是错误信息")
    
    print("\n=== 服务端日志示例 ===")
    server_logger = create_server_logger("test_server", level="DEBUG")
    set_request_id("req-12345")
    server_logger.debug("这是调试信息")
    server_logger.info("这是普通信息")
    log_request_start(server_logger, "/api/test")
    log_tool_call(server_logger, "go_to_office", {"x": 1.0, "y": 2.0})
    log_request_end(server_logger, "/api/test", 125.5)
    
    print("\n=== 日志文件已写入 ./logs/ 目录 ===")

