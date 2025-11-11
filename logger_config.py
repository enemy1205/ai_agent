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
from datetime import datetime
import threading

# 全局请求ID存储（用于追踪分布式请求）
_request_id_storage = threading.local()


class RequestIDFilter(logging.Filter):
    """为日志记录添加请求ID（用于追踪完整请求链路）"""
    
    def filter(self, record):
        record.request_id = getattr(_request_id_storage, 'request_id', '-')
        return True


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器（终端输出）"""
    
    # ANSI颜色码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    is_robot: bool = False
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
        
        console_formatter = ColoredFormatter(
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

def create_robot_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    创建机器人端logger（轻量级）
    - 简洁格式
    - 控制台输出为主
    - 文件日志可选
    """
    return setup_logger(
        name=name,
        level=level,
        console_output=True,
        file_output=True,
        json_format=False,
        is_robot=True
    )


def create_server_logger(name: str, level: str = "INFO", json_format: bool = False) -> logging.Logger:
    """
    创建服务端logger（详细）
    - 详细格式
    - 支持请求ID追踪
    - 控制台+文件双输出
    - 可选JSON格式
    """
    return setup_logger(
        name=name,
        level=level,
        console_output=True,
        file_output=True,
        json_format=json_format,
        is_robot=False
    )


# === 便捷函数：统一的日志消息格式 ===

def log_request_start(logger: logging.Logger, endpoint: str, method: str = "POST"):
    """记录请求开始"""
    logger.info(f"请求开始 [{method}] {endpoint}")


def log_request_end(logger: logging.Logger, endpoint: str, duration_ms: float, status: str = "success"):
    """记录请求结束"""
    logger.info(f"请求结束 [{endpoint}] 耗时: {duration_ms:.2f}ms 状态: {status}")


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

