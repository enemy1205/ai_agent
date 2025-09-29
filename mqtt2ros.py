#!/usr/bin/python3
# coding=UTF-8
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from navigation_msgs.msg import NavigationStatus
import math
from imrobot_msg.msg import ArmDrive, ArmStatus
from std_msgs.msg import Int32, Bool
import paho.mqtt.client as mqtt
import logging
import ast
import time
import socket
import threading  

class NavigationManager:
    def __init__(self):
        # 存储导航目标和命令
        self.nav_target = None  # 格式: {'x': x, 'y': y, 'yaw': yaw}
        self.command = None
        self.nav_status = 0
        self.last_nav_status = None
        self.last_arm_running_status = True
        self.arm_running_status = False
        self.send_cmd = 99
        self.waiting_for_end_effector = False  # 标记是否需要等待end_effector话题的消息

        # 任务队列系统（FIFO）
        self.task_queue = []  # 任务队列（先进先出）
        self.current_task = None  # 当前执行的任务

        # 初始化ROS节点和发布者/订阅者
        self.init_ros()

        self.lock = threading.RLock() 

    def add_task(self, task_type, task_data):
        """添加任务到队列（FIFO）"""
        with self.lock:
            task = {
                'type': task_type,
                'data': task_data,
                'timestamp': time.time()
            }
            # FIFO：直接追加到队尾
            self.task_queue.append(task)
            rospy.loginfo(f"添加任务到队列: {task_type} (FIFO)")
            rospy.loginfo(f"当前队列长度: {len(self.task_queue)}")
            # 如果没有当前任务，开始执行下一个任务
            if self.current_task is None:
                self.execute_next_task()

    def execute_next_task(self):
        """执行队列中的下一个任务"""
        with self.lock:
            if not self.task_queue:
                rospy.loginfo("任务队列为空")
                return
            
            if self.current_task is not None:
                rospy.loginfo("当前有任务正在执行，等待完成")
                return
            
            # 获取优先级最高的任务
            self.current_task = self.task_queue.pop(0)
            task_type = self.current_task['type']
            task_data = self.current_task['data']
            
            rospy.loginfo(f"开始执行任务: {task_type}")
            
            if task_type == 'navigation':
                self.execute_navigation_task(task_data)
            elif task_type == 'arm':
                self.execute_arm_task(task_data)
            elif task_type == 'gripper':
                self.execute_gripper_task(task_data)

    def complete_current_task(self):
        """标记当前任务完成，开始下一个任务"""
        with self.lock:
            if self.current_task:
                rospy.loginfo(f"任务完成: {self.current_task['type']}")
                self.current_task = None
            
            # 执行下一个任务
            self.execute_next_task()

    def execute_navigation_task(self, task_data):
        """执行导航任务"""
        # 兼容 z/yaw 字段别名：如果上层误传 z 或角度为 deg 可在服务端处理，这里仅兜底
        x = task_data.get('x', task_data.get('lon', 0))
        y = task_data.get('y', task_data.get('lat', 0))
        yaw = task_data.get('yaw', task_data.get('theta', 0))
        
        rospy.loginfo(f"执行导航任务: 目标坐标({x}, {y}), 朝向: {yaw}")
        self.nav_target = {'x': x, 'y': y, 'yaw': yaw}
        self.publish_goal()

    def execute_arm_task(self, task_data):
        """执行机械臂任务"""
        command = task_data.get('command', 0)
        rospy.loginfo(f"执行机械臂任务: 命令 {command}")
        self.set_arm_command(command)

    def execute_gripper_task(self, task_data):
        """执行机械爪任务"""
        # 兼容 action/command 两种字段
        command = task_data.get('command', task_data.get('action', 0))
        rospy.loginfo(f"执行机械爪任务: 命令 {command}")
        self.pub_cmd_end_effector.publish(Int32(command))
        rospy.loginfo(f"已将机械爪命令 {command} 发送到 /cmdeffector 话题")
        # 机械爪任务立即完成
        self.complete_current_task()

    def init_ros(self):
        # 初始化发布者
        self.pub_nav_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.pub_arm_drive = rospy.Publisher("/arm_drive", ArmDrive, queue_size=10)
        self.pub_cmd_end_effector = rospy.Publisher("/cmdeffector", Int32, queue_size=10)
        # 订阅导航状态
        rospy.Subscriber("/navigation_status", NavigationStatus, self.update_nav_status, queue_size=10)
        rospy.Subscriber("/arm_status", ArmStatus, self.update_arm_status, queue_size=10)
        rospy.Subscriber("end_effector_jiazhua", Bool, self.handle_end_effector, queue_size=10)

    def update_arm_status(self, data):
        """更新机械臂状态，重点关注running_status"""
        with self.lock:
            current_running_status = data.running_status
            if current_running_status != self.last_arm_running_status:
                self.arm_running_status = current_running_status
                rospy.loginfo(f"机械臂运行状态更新: running_status={self.arm_running_status}")
                # 当状态从True（运行中）变为False（已完成）时，发送话题
                if self.last_arm_running_status is True and current_running_status is False:
                    rospy.loginfo("机械臂已完成动作，开始末端操作")
                    self.pub_cmd_end_effector.publish(Int32(self.send_cmd))
                    rospy.loginfo(f"已将命令 {self.send_cmd} 发送到 /cmdeffector 话题")
                    self.send_cmd = None
                    
                    # 如果当前任务是机械臂任务，标记任务完成
                    if self.current_task and self.current_task['type'] == 'arm':
                        rospy.loginfo("机械臂任务完成")
                        self.complete_current_task()
                        
                # 最后更新last_arm_running_status，确保下次判断正确
                self.last_arm_running_status = current_running_status

    def update_nav_status(self, data):
        """更新导航状态"""
        with self.lock: 
            current_status = data.state
            if current_status != self.last_nav_status:
                self.nav_status = current_status
                self.last_nav_status = current_status                         
                rospy.loginfo(f"当前导航状态: {self.nav_status}")
                
                # 如果导航完成且当前任务是导航任务，标记任务完成
                if current_status == 2 and self.current_task and self.current_task['type'] == 'navigation':
                    rospy.loginfo("导航任务完成")
                    self.complete_current_task()
                else:
                    self.check_and_execute()

    def handle_end_effector(self, data):
        """处理end_effector话题消息"""
        with self.lock:
            # 仅在等待状态且消息为True时处理
            if self.waiting_for_end_effector and data.data and self.command is None:
                rospy.loginfo("收到end_effector消息为True，将命令设置为3")
                self.set_arm_command(3)  # 触发命令3的执行
                rospy.loginfo("夹爪完成任务")
                self.waiting_for_end_effector = False  # 重置等待状态

    def check_and_execute(self):
        """根据导航状态执行相应操作（支持独立机械臂命令）"""
        # 这个方法现在主要用于向后兼容，实际任务执行由队列系统管理
        if self.nav_status == 2:
            if self.nav_target is not None:
                self.publish_goal()  # 有导航目标则发布
            elif self.nav_target is None and self.command is not None:
                self.execute_command()  # 无导航目标但有命令则执行

    def publish_goal(self):
        """发布导航目标"""
        if not self.nav_target:
            rospy.logwarn("没有可发布的导航目标")
            return

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = self.nav_target['x']
        goal.pose.position.y = self.nav_target['y']
        goal.pose.position.z = 0.0

        # 将yaw转换为四元数（roll和pitch为0）
        q = rpy2elements(0, 0, self.nav_target['yaw'])
        goal.pose.orientation = q

        # 发布目标
        self.pub_nav_goal.publish(goal)
        rospy.loginfo(f"已发布导航目标: x={self.nav_target['x']}, y={self.nav_target['y']}, yaw={self.nav_target['yaw']}")

        # 发布后清空目标
        self.nav_target = None

    def execute_command(self):
        """执行命令（实现机械臂控制）"""
        rospy.loginfo(f"收到命令: {self.command}，准备执行机械臂操作...")
        with self.lock:
            # 检查机械臂是否处于可执行状态（running_status为False）
            if self.arm_running_status:
                rospy.logwarn("机械臂正在运行中（running_status=True），无法发布新指令")
                return
            
            if self.send_cmd in [1, 2]:
                self.waiting_for_end_effector = True
                rospy.loginfo(f"命令{self.send_cmd}执行完成，等待end_effector消息...")

            # 根据命令类型设置机械臂参数（与robot_arm_navigation.py保持一致）
            arm_cmd = ArmDrive()
            if self.command == 0:
                # 回到原位
                arm_cmd.x = -183.396
                arm_cmd.y = 32.867
                arm_cmd.z = -100.611
                arm_cmd.rx = -7.378
                arm_cmd.ry = 89.048
                arm_cmd.rz = 0
                rospy.loginfo("执行回到原位指令")
            elif self.command == 1:
                # 夹取指令
                arm_cmd.x = -92.346
                arm_cmd.y = -50.122
                arm_cmd.z = -53.531
                arm_cmd.rx = 17.066
                arm_cmd.ry = 89.044
                arm_cmd.rz = 0
                rospy.loginfo("执行夹取指令")
            elif self.command == 2:
                # 释放指令
                arm_cmd.x = -92.346
                arm_cmd.y = -50.122
                arm_cmd.z = -53.531
                arm_cmd.rx = 17.066
                arm_cmd.ry = 89.044
                arm_cmd.rz = 0
                rospy.loginfo("执行释放指令")
            elif self.command == 3:
                # 搬运指令
                arm_cmd.x = -88.396
                arm_cmd.y = 39.867
                arm_cmd.z = -100.611
                arm_cmd.rx = -7.378
                arm_cmd.ry = 89.048
                arm_cmd.rz = 0
                rospy.loginfo("执行搬运指令")
            else:
                rospy.logwarn(f"未知命令: {self.command}，不执行任何操作")
                self.command = None
                return

            # 发布机械臂控制指令
            self.pub_arm_drive.publish(arm_cmd)
            rospy.loginfo(f"已发布机械臂指令: {arm_cmd}")

            # 执行后清空命令
            self.command = None

    def update_nav_target(self, x, y, yaw):
        """更新导航目标和命令（向后兼容方法）"""
        with self.lock:
            self.nav_target = {'x': x, 'y': y, 'yaw': yaw}
            rospy.loginfo(f"更新导航目标: x={x}, y={y}, yaw={yaw}")
            # 立即检查是否可以执行
            self.check_and_execute()

    def set_arm_command(self, command):
        """单独设置机械臂命令（不涉及导航目标）"""
        with self.lock:
            self.command = command
            self.send_cmd = command
            #末端执行器控制
            #self.pub_cmd_end_effector.publish(Int32(self.command))
            #rospy.loginfo(f"已将命令 {self.command} 发送到 /cmdeffector 话题")

            self.nav_target = None  # 机械臂命令无需导航目标
            rospy.loginfo(f"更新机械臂命令: {command}")
            self.check_and_execute()  # 立即检查并执行


MQTT_BROKER = "127.0.0.1"  # 改为回环地址连接本地Broker
MQTT_PORT = 1883
MQTT_TOPICS = [
    ("robot/navigation/gooffice", 1),
    ("robot/navigation/gorestroom", 1),
    ("robot/navigation/gocorridor", 1), 
    ("robot/arm/control", 1),
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
        rospy.loginfo(f"成功连接到MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        rospy.loginfo(f"本地IP地址: {get_local_ip()}")
        client.subscribe(MQTT_TOPICS)
        rospy.loginfo(f"已订阅主题: {[t[0] for t in MQTT_TOPICS]}")
    else:
        rospy.logerr(f"连接失败，错误码: {rc} - {mqtt.connack_string(rc)}")


def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    rospy.logdebug(f"订阅确认: MID={mid}, QOS={granted_qos}")


def on_disconnect(client, userdata, rc, properties=None):
    if rc != 0:
        rospy.logwarn(f"意外断开连接，错误码: {rc}")
        rospy.loginfo("尝试重新连接...")
        try:
            client.reconnect()
        except Exception as e:
            rospy.logerr(f"重连失败: {str(e)}")


def on_message(client, userdata, msg):
    rospy.loginfo(f"收到消息 [主题: {msg.topic}]")
    rospy.logdebug(f"原始消息: {msg.payload}")
    
    try:
        payload_str = msg.payload.decode('utf-8')
        rospy.logdebug(f"解码内容: {payload_str}")
        
        # 尝试解析为Python字典
        try:
            payload = ast.literal_eval(payload_str)
            rospy.loginfo(f"解析后的指令: {payload}")
        except:
            # 如果不是Python字典格式，尝试作为纯文本处理
            payload = payload_str
            rospy.logwarn("消息不是字典格式，作为文本处理")
        
        # 处理不同主题的指令
        if "gooffice" in msg.topic:
            handle_navigation("办公室", payload)
        elif "gorestroom" in msg.topic:
            handle_navigation("休息室", payload)
        elif "gocorridor" in msg.topic:  
            handle_navigation("走廊", payload)
        elif "arm/control" in msg.topic:
            handle_arm_command(payload)
        elif "gripper/control" in msg.topic:
            handle_gripper_command(payload)
            
    except Exception as e:
        rospy.logerr(f"处理消息时出错: {str(e)}")


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


def handle_navigation(destination, payload):
    """处理导航指令"""
    rospy.loginfo(f"开始处理{destination}导航指令")
    
    # 提取导航参数（导航消息不含command，仅x/y/yaw）
    if isinstance(payload, dict):
        x = payload.get('x', 0)
        y = payload.get('y', 0)
        yaw = payload.get('yaw', 0)
        
        rospy.loginfo(f"目标坐标: ({x}, {y}), 朝向: {yaw}弧度")

        # 将导航任务添加到队列
        if nav_manager:
            nav_manager.add_task('navigation', {'x': x, 'y': y, 'yaw': yaw})
    else:
        rospy.loginfo(f"接收到的指令内容: {payload}")
    
    rospy.loginfo(f"{destination}导航指令处理完成\n")


def handle_arm_command(payload):
    """处理机械臂控制指令"""
    rospy.loginfo("开始处理机械臂控制指令")
    
    if isinstance(payload, dict):
        command = payload.get('command', 0)
        # 验证命令有效性（与robot_arm_navigation.py保持一致）
        if command not in [0, 1, 2, 3]:
            rospy.logwarn(f"无效的机械臂命令: {command}（必须为0-3）")
            return
        
        # 命令描述
        command_desc = {
            0: "回到原位",
            1: "夹取",
            2: "释放",
            3: "搬运"
        }[command]
        rospy.loginfo(f"机械臂命令: {command_desc}（指令值: {command}）")
        
        # 将机械臂任务添加到队列
        if nav_manager:
            nav_manager.add_task('arm', {'command': command})
    else:
        rospy.logwarn("机械臂指令格式不正确，需为字典类型")


def handle_gripper_command(payload):
    """处理机械爪控制指令"""
    rospy.loginfo("开始处理机械爪控制指令")
    
    if isinstance(payload, dict):
        command = payload.get('command', 0)
        # 验证命令有效性（机械爪通常有开合两种状态）
        if command not in [0, 1]:
            rospy.logwarn(f"无效的机械爪命令: {command}（必须为0或1）")
            return
        
        # 命令描述
        command_desc = {
            0: "张开",
            1: "闭合"
        }[command]
        rospy.loginfo(f"机械爪命令: {command_desc}（指令值: {command}）")
        
        # 将机械爪任务添加到队列
        if nav_manager:
            nav_manager.add_task('gripper', {'command': command})
    else:
        rospy.logwarn("机械爪指令格式不正确，需为字典类型")


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
        rospy.loginfo("启动MQTT监听循环...")
        client.loop_forever()  # 在独立线程中运行，不阻塞ROS
    except Exception as e:
        rospy.logerr(f"连接异常: {str(e)}")


# 全局导航管理器实例
nav_manager = None

if __name__ == "__main__":
    rospy.loginfo("启动MQTT导航指令接收器")
    rospy.loginfo(f"目标Broker: {MQTT_BROKER}:{MQTT_PORT}")
    rospy.loginfo(f"本地IP地址: {get_local_ip()}")

    try:
        # 初始化ROS节点
        rospy.init_node('mqtt_navigation_receiver', anonymous=True)
        # 创建导航管理器
        nav_manager = NavigationManager()
        rospy.loginfo("导航管理器初始化完成")

        # 启动MQTT客户端线程  
        mqtt_thread = threading.Thread(target=start_mqtt_client, name='MQTTThread') 
        mqtt_thread.daemon = True  
        mqtt_thread.start()  
        rospy.loginfo("MQTT客户端线程已启动")  

        rospy.spin()  

    except KeyboardInterrupt:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"程序异常终止: {str(e)}")
        
