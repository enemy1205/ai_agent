#!/usr/bin/python3
# coding=UTF-8
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from navigation_msgs.msg import NavigationStatus
import math
from imrobot_msg.msg import ArmDrive, ArmStatus,ArmPositionDrive
from std_msgs.msg import Int32, Bool
import paho.mqtt.client as mqtt
import ast
import time
import socket
import threading  
from collections import deque
from enum import Enum
import os

# === 导入统一日志配置 ===
from logger_config import (
    create_robot_logger,
    log_mqtt_receive,
    log_mqtt_publish,
    log_task_add,
    log_task_start,
    log_task_complete
)

# 创建logger实例（机器人端，简化格式）
logger = create_robot_logger("mqtt_manager", level=os.getenv("LOG_LEVEL", "INFO"))

class TaskType(Enum):
    """任务类型枚举"""
    NAVIGATION = "navigation"
    ARM_COMMAND = "arm_command"
    ARM_COORDINATE = "arm_coordinate"
    GRIPPER = "gripper"

class Task:
    """任务对象"""
    def __init__(self, task_type, payload):
        self.task_type = task_type
        self.payload = payload
        self.start_time = None
        
    def __repr__(self):
        return f"Task(type={self.task_type.value}, payload={self.payload})"

class NavigationManager:
    def __init__(self):
        # 状态记录
        self.nav_status = 0
        self.last_nav_status = None
        self.arm_running_status = False
        self.last_arm_running_status = None

        # 任务队列
        self.task_queue = deque()
        self.current_task = None
        self.task_paused = False  # 任务暂停标志
        self.queue_lock = threading.RLock()
        
        # 任务完成后的额外等待时间(秒) - 可配置参数
        self.task_completion_delay = 0.5  # 任务完成后等待2秒再执行下一个
        self.task_completion_time = None  # 记录任务完成的时间

        # 初始化ROS节点和发布者/订阅者
        self.init_ros()

        self.lock = threading.RLock()
        
        # 启动任务执行线程
        self.executor_thread = threading.Thread(target=self._task_executor, name='TaskExecutor')
        self.executor_thread.daemon = True
        self.executor_thread.start()
        logger.info("任务队列执行器已启动")
        logger.info(f"任务完成后等待时间: {self.task_completion_delay}秒") 

    def init_ros(self):
        # 初始化发布者
        self.pub_nav_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.pub_arm_drive = rospy.Publisher("/Arm_Position_Drive", ArmPositionDrive, queue_size=10)
        self.pub_cmd_end_effector = rospy.Publisher("/cmdeffector", Int32, queue_size=10)
        # 订阅状态（用于日志记录）
        rospy.Subscriber("/navigation_status", NavigationStatus, self.update_nav_status, queue_size=10)
        rospy.Subscriber("/arm_status", ArmStatus, self.update_arm_status, queue_size=10)

    def update_arm_status(self, data):
        """更新机械臂状态"""
        with self.lock:
            current_running_status = data.running_status
            if current_running_status != self.last_arm_running_status:
                self.arm_running_status = current_running_status
                logger.info(f"机械臂运行状态更新: running_status={self.arm_running_status}")
                self.last_arm_running_status = current_running_status

    def update_nav_status(self, data):
        """更新导航状态"""
        with self.lock: 
            current_status = data.state
            if current_status != self.last_nav_status:
                self.nav_status = current_status
                self.last_nav_status = current_status                         
                status_desc = {1: "导航异常", 2: "导航完成/正常", 3: "正在导航", 4: "手柄控制模式"}
                logger.info(f"当前导航状态: {self.nav_status} ({status_desc.get(current_status, '未知')})")

    def publish_navigation(self, x, y, z, orientation=None):
        """直接发布导航目标"""
        logger.info(f"发布导航目标: ({x}, {y}, {z})")
        if orientation:
            logger.info(f"目标朝向: {orientation}")

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z

        # 处理朝向：如果提供了orientation则使用，否则使用默认值(0,0,0,1)
        if orientation and isinstance(orientation, dict):
            # 使用提供的四元数
            q = Quaternion()
            q.x = orientation.get('x', 0.0)
            q.y = orientation.get('y', 0.0)
            q.z = orientation.get('z', 0.0)
            q.w = orientation.get('w', 1.0)
            goal.pose.orientation = q
            logger.info(f"使用自定义朝向: ({q.x}, {q.y}, {q.z}, {q.w})")
        else:
            # 使用默认朝向(0,0,0,1)
            q = rpy2elements(0, 0, 0)
            goal.pose.orientation = q
            logger.info("使用默认朝向: (0, 0, 0, 1)")

        # 发布目标
        self.pub_nav_goal.publish(goal)
        logger.info(f"已发布导航目标到 /move_base_simple/goal")

    def publish_arm_command(self, command):
        """直接发布机械臂命令"""
        logger.info(f"发布机械臂命令: {command}")
        
        # 根据命令类型设置机械臂参数
        arm_cmd = ArmPositionDrive()
        if command == 0:
            # 回到原位
            arm_cmd.x = -183.396
            arm_cmd.y = 32.867
            arm_cmd.z = -100.611
            arm_cmd.rx = -7.378
            arm_cmd.ry = 89.048
            arm_cmd.rz = 0
            logger.info("执行回到原位指令")
        elif command == 1:
            # 夹取指令
            arm_cmd.x = -92.346
            arm_cmd.y = -50.122
            arm_cmd.z = -53.531
            arm_cmd.rx = 17.066
            arm_cmd.ry = 89.044
            arm_cmd.rz = 0
            logger.info("执行夹取指令")
        elif command == 2:
            # 释放指令
            arm_cmd.x = -92.346
            arm_cmd.y = -50.122
            arm_cmd.z = -53.531
            arm_cmd.rx = 17.066
            arm_cmd.ry = 89.044
            arm_cmd.rz = 0
            logger.info("执行释放指令")
        elif command == 3:
            # 搬运指令
            arm_cmd.x = -88.396
            arm_cmd.y = 39.867
            arm_cmd.z = -100.611
            arm_cmd.rx = -7.378
            arm_cmd.ry = 89.048
            arm_cmd.rz = 0
            logger.info("执行搬运指令")
        else:
            logger.warning(f"未知命令: {command}，不执行任何操作")
            return

        # 发布机械臂控制指令
        self.pub_arm_drive.publish(arm_cmd)
        logger.info(f"已发布机械臂指令到 /arm_drive: {arm_cmd}")

    def publish_arm_coordinate(self, x, y, z, rx, ry, rz):
        """直接发布带坐标的机械臂命令"""
        logger.info(f"发布机械臂坐标指令: 位置({x}, {y}, {z}), 姿态({rx}, {ry}, {rz})")
        
        # 创建机械臂控制指令
        arm_cmd = ArmPositionDrive()
        arm_cmd.x = x
        arm_cmd.y = y
        arm_cmd.z = z
        arm_cmd.rx = rx
        arm_cmd.ry = ry
        arm_cmd.rz = rz
        
        # 发布机械臂控制指令
        self.pub_arm_drive.publish(arm_cmd)
        logger.info(f"已发布机械臂坐标指令到 /arm_drive: {arm_cmd}")

    def publish_gripper_command(self, command):
        """直接发布夹爪命令"""
        logger.info(f"发布夹爪命令: {command}")
        
        # 命令描述
        command_desc = {
            2: "张开",
            1: "闭合"
        }.get(command, "未知")
        
        logger.info(f"夹爪命令: {command_desc}（指令值: {command}）")
        
        # 直接发布
        self.pub_cmd_end_effector.publish(Int32(command))
        logger.info(f"已将夹爪命令 {command} 发送到 /cmdeffector 话题")

    def add_task(self, task_type, payload):
        """添加任务到队列"""
        with self.queue_lock:
            if self.task_paused:
                logger.warning(f"任务队列已暂停,拒绝添加新任务: {task_type.value}")
                return False
            
            task = Task(task_type, payload)
            self.task_queue.append(task)
            log_task_add(logger, task_type.value, payload)
            logger.info(f"当前队列长度: {len(self.task_queue)}")
            return True
    
    def _task_executor(self):
        """任务执行器主循环"""
        logger.info("任务执行器线程已启动")
        rate = rospy.Rate(10)  # 10Hz检查频率
        
        while not rospy.is_shutdown():
            try:
                with self.queue_lock:
                    # 如果任务暂停,跳过执行
                    if self.task_paused:
                        rate.sleep()
                        continue
                    
                    # 如果当前有任务在执行,检查是否完成
                    if self.current_task is not None:
                        if self._is_task_completed(self.current_task):
                            # 计算任务持续时间
                            duration_s = time.time() - self.current_task.start_time if self.current_task.start_time else 0.0
                            log_task_complete(logger, self.current_task.task_type.value, duration_s)
                            # 记录任务完成时间,开始等待
                            self.task_completion_time = time.time()
                            self.current_task = None
                            logger.info(f"等待{self.task_completion_delay}秒后执行下一个任务...")
                        else:
                            # 任务未完成,继续等待
                            rate.sleep()
                            continue
                    
                    # 如果有任务刚完成,需要等待一段时间
                    if self.task_completion_time is not None:
                        elapsed = time.time() - self.task_completion_time
                        if elapsed < self.task_completion_delay:
                            # 还在等待期,继续等待
                            rate.sleep()
                            continue
                        else:
                            # 等待期结束
                            logger.info(f"等待完成,准备执行下一个任务")
                            self.task_completion_time = None
                    
                    # 如果没有当前任务且队列不为空,取出下一个任务
                    if self.current_task is None and len(self.task_queue) > 0:
                        self.current_task = self.task_queue.popleft()
                        log_task_start(logger, self.current_task.task_type.value)
                        logger.info(f"剩余任务: {len(self.task_queue)}")
                        self._execute_task(self.current_task)
                
                rate.sleep()
                
            except Exception as e:
                logger.error(f"任务执行器异常: {str(e)}", exc_info=True)
                rate.sleep()
    
    def _execute_task(self, task):
        """执行单个任务"""
        task.start_time = time.time()
        
        if task.task_type == TaskType.NAVIGATION:
            self._execute_navigation(task.payload)
        elif task.task_type == TaskType.ARM_COMMAND:
            self._execute_arm_command(task.payload)
        elif task.task_type == TaskType.ARM_COORDINATE:
            self._execute_arm_coordinate(task.payload)
        elif task.task_type == TaskType.GRIPPER:
            self._execute_gripper(task.payload)
    
    def _execute_navigation(self, payload):
        """执行导航任务"""
        if isinstance(payload, dict):
            x = payload.get('x', 0)
            y = payload.get('y', 0)
            z = payload.get('z', 0)
            orientation = payload.get('orientation', None)
            self.publish_navigation(x, y, z, orientation)
    
    def _execute_arm_command(self, payload):
        """执行机械臂命令任务"""
        if isinstance(payload, dict):
            command = payload.get('command', 0)
            self.publish_arm_command(command)
    
    def _execute_arm_coordinate(self, payload):
        """执行机械臂坐标任务"""
        if isinstance(payload, dict):
            x = payload.get('x', 0)
            y = payload.get('y', 0)
            z = payload.get('z', 0)
            rx = payload.get('rx', 0)
            ry = payload.get('ry', 0)
            rz = payload.get('rz', 0)
            self.publish_arm_coordinate(x, y, z, rx, ry, rz)
    
    def _execute_gripper(self, payload):
        """执行机械爪任务"""
        if isinstance(payload, dict):
            command = payload.get('command', 1)
            self.publish_gripper_command(command)
    
    def _is_task_completed(self, task):
        """检查任务是否完成"""
        if task is None:
            return True
        
        # 计算任务执行时长
        elapsed_time = time.time() - task.start_time
        
        # 导航任务: 需要等待2秒后再检查状态
        if task.task_type == TaskType.NAVIGATION:
            if elapsed_time < 2.0:
                return False  # 还没到2秒,继续等待
            
            with self.lock:
                status = self.nav_status
            
            if status == 1:
                logger.error("导航异常! 终止当前任务并暂停任务队列")
                self._pause_task_queue()
                return True
            elif status == 4:
                logger.warning("检测到手柄控制模式! 终止当前任务并暂停任务队列")
                self._pause_task_queue()
                return True
            elif status == 2:
                logger.info("导航任务已完成")
                return True
            elif status == 3:
                # 正在导航中,继续等待
                return False
            else:
                logger.warning(f"未知导航状态: {status}, 继续等待")
                return False
        
        # 机械臂任务: 需要等待2秒后再检查状态
        elif task.task_type == TaskType.ARM_COMMAND or task.task_type == TaskType.ARM_COORDINATE:
            if elapsed_time < 2.0:
                return False  # 还没到2秒,继续等待
            
            with self.lock:
                running = self.arm_running_status
            
            if running:
                # 机械臂正在运行,继续等待
                return False
            else:
                logger.info("机械臂任务已完成")
                return True
        
        # 机械爪任务: 等待一定时间确保动作完成
        elif task.task_type == TaskType.GRIPPER:
            # 机械爪需要时间完成动作,等待1.5秒
            if elapsed_time < 1.5:
                return False  # 继续等待
            logger.info("机械爪任务已完成")
            return True
        
        return True
    
    def _pause_task_queue(self):
        """暂停任务队列"""
        with self.queue_lock:
            self.task_paused = True
            # 清空当前任务
            self.current_task = None
            logger.warning(f"任务队列已暂停! 队列中剩余 {len(self.task_queue)} 个任务被取消")
            # 清空队列
            self.task_queue.clear()
    
    def resume_task_queue(self):
        """恢复任务队列"""
        with self.queue_lock:
            self.task_paused = False
            logger.info("任务队列已恢复")
    
    def set_task_completion_delay(self, delay):
        """设置任务完成后的等待时间
        
        Args:
            delay: 等待时间(秒),建议范围1.0-5.0
        """
        if delay < 0:
            logger.warning(f"等待时间不能为负数,保持当前值: {self.task_completion_delay}秒")
            return
        
        old_delay = self.task_completion_delay
        self.task_completion_delay = delay
        logger.info(f"任务完成等待时间已更新: {old_delay}秒 → {delay}秒")


MQTT_BROKER = "127.0.0.1"  # 改为回环地址连接本地Broker
MQTT_PORT = 1883
MQTT_TOPICS = [
    ("robot/navigation", 1),
    ("robot/arm/control", 1),
    ("robot/arm/coordinate", 1),
    ("robot/gripper/control",1),
    ("robot/navigation_status", 1)
]


def get_local_ip():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "未知"


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info(f"成功连接到MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        logger.info(f"本地IP地址: {get_local_ip()}")
        client.subscribe(MQTT_TOPICS)
        logger.info(f"已订阅主题: {[t[0] for t in MQTT_TOPICS]}")
    else:
        logger.error(f"连接失败，错误码: {rc} - {mqtt.connack_string(rc)}")


def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    logger.debug(f"订阅确认: MID={mid}, QOS={granted_qos}")


def on_disconnect(client, userdata, rc, properties=None):
    if rc != 0:
        logger.warning(f"意外断开连接，错误码: {rc}")
        logger.info("尝试重新连接...")
        try:
            client.reconnect()
        except Exception as e:
            logger.error(f"重连失败: {str(e)}")


def on_message(client, userdata, msg):
    log_mqtt_receive(logger, msg.topic, str(msg.payload)[:100])
    logger.debug(f"原始消息: {msg.payload}")
    
    try:
        payload_str = msg.payload.decode('utf-8')
        logger.debug(f"解码内容: {payload_str}")
        
        # 尝试解析为Python字典
        try:
            payload = ast.literal_eval(payload_str)
            logger.info(f"解析后的指令: {payload}")
        except:
            # 如果不是Python字典格式，尝试作为纯文本处理
            payload = payload_str
            logger.warning("消息不是字典格式，作为文本处理")
        
        # 处理不同主题的指令 - 直接执行
        if msg.topic == "robot/navigation":
            handle_navigation(payload)
        elif "arm/control" in msg.topic:
            handle_arm_command(payload)
        elif "arm/coordinate" in msg.topic:
            handle_arm_coordinate_command(payload)
        elif "gripper/control" in msg.topic:
            handle_gripper_command(payload)
            
    except Exception as e:
        logger.error(f"处理消息时出错: {str(e)}", exc_info=True)


def rpy2elements(roll, pitch, yaw):
    """欧拉角转四元素"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    q.w = cy * cp * cr + sy * sp * sr
    return q


def handle_navigation(payload):
    """处理导航指令 - 添加到任务队列"""
    logger.info(f"收到导航指令")
    
    # 提取导航参数（导航消息包含x/y/z和可选的orientation）
    if isinstance(payload, dict):
        x = payload.get('x', 0)
        y = payload.get('y', 0)
        z = payload.get('z', 0)
        orientation = payload.get('orientation', None)
        
        logger.info(f"目标坐标: ({x}, {y}, {z})")
        if orientation:
            logger.info(f"目标朝向: {orientation}")

        # 添加到任务队列
        if nav_manager:
            nav_manager.add_task(TaskType.NAVIGATION, payload)
    else:
        logger.warning(f"导航指令格式不正确: {payload}")


def handle_arm_command(payload):
    """处理机械臂控制指令 - 添加到任务队列"""
    logger.info("收到机械臂控制指令")
    
    if isinstance(payload, dict):
        command = payload.get('command', 0)
        # 验证命令有效性
        if command not in [0, 1, 2, 3]:
            logger.warning(f"无效的机械臂命令: {command}（必须为0-3）")
            return
        
        # 命令描述
        command_desc = {
            0: "回到原位",
            1: "夹取",
            2: "释放",
            3: "搬运"
        }[command]
        logger.info(f"机械臂命令: {command_desc}（指令值: {command}）")
        
        # 添加到任务队列
        if nav_manager:
            nav_manager.add_task(TaskType.ARM_COMMAND, payload)
    else:
        logger.warning("机械臂指令格式不正确，需为字典类型")


def handle_arm_coordinate_command(payload):
    """处理带坐标的机械臂控制指令 - 添加到任务队列"""
    logger.info("收到带坐标的机械臂控制指令")
    
    if isinstance(payload, dict):
        x = payload.get('x', 0)
        y = payload.get('y', 0)
        z = payload.get('z', 0)
        rx = payload.get('rx', 0)
        ry = payload.get('ry', 0)
        rz = payload.get('rz', 0)
        
        logger.info(f"机械臂坐标指令: 位置({x}, {y}, {z}), 姿态({rx}, {ry}, {rz})")
        
        # 添加到任务队列
        if nav_manager:
            nav_manager.add_task(TaskType.ARM_COORDINATE, payload)
    else:
        logger.warning("带坐标机械臂指令格式不正确，需为字典类型")

def handle_gripper_command(payload):
    """处理机械爪控制指令 - 添加到任务队列"""
    logger.info("收到机械爪控制指令")
    
    if isinstance(payload, dict):
        command = payload.get('command', 1)
        # 验证命令有效性（机械爪通常有开合两种状态）
        if command not in [1, 2]:
            logger.warning(f"无效的机械爪命令: {command}（必须为1或2）")
            return
        
        # 添加到任务队列
        if nav_manager:
            nav_manager.add_task(TaskType.GRIPPER, payload)
    else:
        logger.warning("机械爪指令格式不正确，需为字典类型")


def start_mqtt_client():
    # client = mqtt.Client(
    #     callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    #     client_id=f"nav_receiver_{get_local_ip()}"
    # )
    client = mqtt.Client(
    client_id=f"nav_receiver_{get_local_ip()}"  # 移除 callback_api_version 参数
      )
    # 设置回调函数
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe
    client.on_disconnect = on_disconnect
    
    # 设置连接参数
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        logger.info("启动MQTT监听循环...")
        client.loop_forever()  # 在独立线程中运行，不阻塞ROS
    except Exception as e:
        logger.error(f"连接异常: {str(e)}", exc_info=True)


# 全局导航管理器实例
nav_manager = None

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("启动MQTT导航指令接收器")
    logger.info(f"目标Broker: {MQTT_BROKER}:{MQTT_PORT}")
    logger.info(f"本地IP地址: {get_local_ip()}")
    logger.info("=" * 60)

    try:
        # 初始化ROS节点
        rospy.init_node('mqtt_navigation_receiver', anonymous=True)
        # 创建导航管理器
        nav_manager = NavigationManager()
        logger.info("导航管理器初始化完成")

        # 启动MQTT客户端线程  
        mqtt_thread = threading.Thread(target=start_mqtt_client, name='MQTTThread') 
        mqtt_thread.daemon = True  
        mqtt_thread.start()  
        logger.info("MQTT客户端线程已启动")  

        rospy.spin()  

    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常终止: {str(e)}", exc_info=True)
        
