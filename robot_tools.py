#!/usr/bin/env python3
"""
机器人控制工具模块
"""

import sys
import json
from pathlib import Path
import logging
import paho.mqtt.client as mqtt
import time
from typing import Any
from langchain.tools import StructuredTool

# 日志配置
logger = logging.getLogger('RobotController')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# 避免向根日志传播导致重复输出
logger.propagate = False

if sys.platform == 'win32':
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

# MQTT 配置
MQTT_BROKER = "10.194.142.142"
MQTT_PORT = 1883
MQTT_TOPIC_GOOFFICE = "robot/navigation/gooffice"
MQTT_TOPIC_GORESTROOM = "robot/navigation/gorestroom"
MQTT_TOPIC_GOCORRIDOR = "robot/navigation/gocorridor"
MQTT_TOPIC_ARM_CONTROL = "robot/arm/control"
MQTT_TOPIC_GRIPPER_CONTROL = "robot/gripper/control"

# ========== MQTT通信函数 ==========

def connect_mqtt():
    client = mqtt.Client()

    def _on_connect(client, userdata, flags, rc):
        try:
            if rc == 0:
                logger.debug("MQTT连接成功")
            else:
                logger.error(f"MQTT连接失败: {rc}")
        except Exception as e:
            logger.error(f"on_connect 回调异常: {e}")

    def _on_publish(client, userdata, mid):
        try:
            logger.debug(f"消息已发送 (ID: {mid})")
        except Exception as e:
            logger.error(f"on_publish 回调异常: {e}")

    def _on_disconnect(client, userdata, rc):
        try:
            logger.warning(f"MQTT断开连接: {rc}")
        except Exception as e:
            logger.error(f"on_disconnect 回调异常: {e}")

    client.on_connect = _on_connect
    client.on_publish = _on_publish
    client.on_disconnect = _on_disconnect

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        return client
    except Exception as e:
        logger.error(f"MQTT连接失败: {e}")
        return None

def _send_navigation(client, topic, x, y, z):
    payload = json.dumps({"x": x, "y": y, "z": z})
    logger.debug(f"发送导航指令: {topic} → {payload}")
    try:
        result = client.publish(topic, payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            # 使用更短的超时时间
            result.wait_for_publish(timeout=2)
            logger.debug("发布成功")
            return True
        else:
            logger.error(f"发布失败，错误码: {result.rc}")
            return False
    except TimeoutError:
        logger.warning("发布超时")
        return False
    except RuntimeError as e:
        logger.error(f"发布失败: {e}")
        return False
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return False

def _send_arm_command(client, topic, command):
    payload = json.dumps({"command": command})
    logger.debug(f"发送机械臂指令: {command} → {payload}")
    try:
        result = client.publish(topic, payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            # 使用更短的超时时间
            result.wait_for_publish(timeout=2)
            logger.debug("发布成功")
            return True
        else:
            logger.error(f"发布失败，错误码: {result.rc}")
            return False
    except TimeoutError:
        logger.warning("发布超时")
        return False
    except RuntimeError as e:
        logger.error(f"发布失败: {e}")
        return False
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return False

def _send_gripper_command(client, topic, action):
    payload = json.dumps({"action": action})
    logger.debug(f"发送夹爪指令: {action} → {payload}")
    try:
        result = client.publish(topic, payload, qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            # 使用更短的超时时间
            result.wait_for_publish(timeout=2)
            logger.debug("发布成功")
            return True
        else:
            logger.error(f"发布失败，错误码: {result.rc}")
            return False
    except TimeoutError:
        logger.warning("发布超时")
        return False
    except RuntimeError as e:
        logger.error(f"发布失败: {e}")
        return False
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return False

# ========== 机器人控制工具函数 ==========

def arm_control(command: int) -> dict:
    """
    控制机械臂整体动作和姿态（不移动机器人底盘）
    适用场景：用户说"机械臂归位"、"机械臂准备"、"机械臂搬运模式"等，需要机械臂整体动作时调用。
    注意：此工具不控制夹爪开合，夹爪控制请使用gripper_control工具。
    参数 command:
        0 → 机械臂回到原位（归位到初始姿态）
        1 → 机械臂准备抓取姿态（移动到抓取位置）
        2 → 机械臂准备递送姿态（移动到递送位置）
        3 → 机械臂搬运模式（保持当前姿态用于移动）
    返回:
        {"sent": True, "message": str} 或 {"sent": False, "error": str}
    """
    # 确保参数是整数类型
    try:
        command = int(command)
    except (ValueError, TypeError):
        return {"sent": False, "error": f"command 参数类型错误: {type(command)}, 值: {command}"}
    
    if command not in [0, 1, 2, 3]:
        return {"sent": False, "error": "command 必须是 0, 1, 2 或 3"}

    client = connect_mqtt()
    if client is None:
        return {"sent": False, "error": "MQTT连接失败"}
    
    client.loop_start()
    time.sleep(0.3)
    if not client.is_connected():
        client.loop_stop()
        return {"sent": False, "error": "MQTT连接失败"}

    success = _send_arm_command(client, MQTT_TOPIC_ARM_CONTROL, command)
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()

    desc = {0: "归位", 1: "准备抓取", 2: "准备递送", 3: "搬运模式"}[command]
    if success:
        return {"sent": True, "message": f"✅ 已发送机械臂「{desc}」指令 (command={command})"}
    else:
        return {"sent": False, "error": "MQTT消息发送失败"}

def gripper_control(action: int) -> dict:
    """
    控制夹爪开合动作（仅控制夹爪的夹紧和松开）
    适用场景：用户说"夹爪夹紧"、"夹爪松开"、"夹取物品"、"放开物品"等，需要控制夹爪开合时调用。
    注意：此工具只控制夹爪开合，机械臂整体动作请使用arm_control工具。
    参数 action:
        1 → 夹爪夹紧（夹取物品）
        2 → 夹爪松开（释放物品）
    返回:
        {"sent": True, "message": str} 或 {"sent": False, "error": str}
    """
    # 确保参数是整数类型
    try:
        action = int(action)
    except (ValueError, TypeError):
        return {"sent": False, "error": f"action 参数类型错误: {type(action)}, 值: {action}"}
    
    if action not in [1, 2]:
        return {"sent": False, "error": "action 必须是 1 或 2"}

    client = connect_mqtt()
    if client is None:
        return {"sent": False, "error": "MQTT连接失败"}
    
    client.loop_start()
    time.sleep(0.3)
    if not client.is_connected():
        client.loop_stop()
        return {"sent": False, "error": "MQTT连接失败"}

    success = _send_gripper_command(client, MQTT_TOPIC_GRIPPER_CONTROL, action)
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()

    desc = {1: "夹紧", 2: "松开"}[action]
    if success:
        return {"sent": True, "message": f"✅ 已发送夹爪「{desc}」指令 (action={action})"}
    else:
        return {"sent": False, "error": "MQTT消息发送失败"}

def _load_locations_config() -> dict:
    """加载坐标配置文件 config/locations.json"""
    try:
        config_path = Path(__file__).parent / "config" / "locations.json"
        # 兼容从项目根路径运行
        if not config_path.exists():
            config_path = Path.cwd() / "config" / "locations.json"
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载坐标配置失败: {e}")
        return {}

def go_to_office() -> dict:
    """
    让机器人导航到办公室（不操作机械臂）
    适用场景：用户说"去办公室"、"到办公室去"等，不需要拿/放物品时调用。
    返回:
        {"sent": True, "message": str} 或 {"sent": False, "error": str}
    """
    locations = _load_locations_config()
    pos = locations.get("office", {})
    x, y, z = pos.get("x", 74.814), pos.get("y", 77.791), pos.get("z", 0.0)
    client = connect_mqtt()
    client.loop_start()
    time.sleep(0.3)
    if not client.is_connected():
        client.loop_stop()
        return {"sent": False, "error": "MQTT连接失败"}

    success = _send_navigation(client, MQTT_TOPIC_GOOFFICE, x, y, z)
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()

    if success:
        return {"sent": True, "message": "✅ 已发送前往「办公室」的导航指令"}
    else:
        return {"sent": False, "error": "MQTT消息发送失败"}

def go_to_restroom() -> dict:
    """
    让机器人导航到休息室（不操作机械臂）
    适用场景：用户说"去休息室"、"到休息室"等，不需要拿/放物品时调用。
    返回:
        {"sent": True, "message": str} 或 {"sent": False, "error": str}
    """
    locations = _load_locations_config()
    pos = locations.get("restroom", {})
    x, y, z = pos.get("x", 86.846), pos.get("y", 92.542), pos.get("z", 0.0)
    client = connect_mqtt()
    client.loop_start()
    time.sleep(0.3)
    if not client.is_connected():
        client.loop_stop()
        return {"sent": False, "error": "MQTT连接失败"}

    success = _send_navigation(client, MQTT_TOPIC_GORESTROOM, x, y, z)
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()

    if success:
        return {"sent": True, "message": "✅ 已发送前往「休息室」的导航指令"}
    else:
        return {"sent": False, "error": "MQTT消息发送失败"}

def go_to_corridor() -> dict:
    """
    让机器人导航到走廊（不操作机械臂）
    适用场景：用户说"去走廊"、"到走廊中间"等，不需要拿/放物品时调用。
    返回:
        {"sent": True, "message": str} 或 {"sent": False, "error": str}
    """
    locations = _load_locations_config()
    pos = locations.get("corridor", {})
    x, y, z = pos.get("x", 97.678375), pos.get("y", 90.0347824), pos.get("z", 0.0)
    client = connect_mqtt()
    client.loop_start()
    time.sleep(0.3)
    if not client.is_connected():
        client.loop_stop()
        return {"sent": False, "error": "MQTT连接失败"}

    success = _send_navigation(client, MQTT_TOPIC_GOCORRIDOR, x, y, z)
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()

    if success:
        return {"sent": True, "message": "✅ 已发送前往「走廊」的导航指令"}
    else:
        return {"sent": False, "error": "MQTT消息发送失败"}

def complex_task(location: str, arm_command: int) -> dict:
    """
    执行组合任务：先导航到指定位置，再执行机械臂动作
    适用场景：
      - "去办公室拿瓶水" → location="office", arm_command=1
      - "把水送到休息室" → location="restroom", arm_command=3
      - "去走廊然后放下东西" → location="corridor", arm_command=2
    参数:
      location: "office" | "restroom" | "corridor"
      arm_command: 0=归位, 1=夹取, 2=释放, 3=搬运
    返回:
      {"sent": True, "message": str} 或 {"sent": False, "error": str, "step": str}
    """
    if location not in ["office", "restroom", "corridor"]:
        return {"sent": False, "error": "location 必须是 office, restroom 或 corridor"}
    if arm_command not in [0, 1, 2, 3]:
        return {"sent": False, "error": "arm_command 必须是 0, 1, 2 或 3"}

    # 导航
    nav_functions = {
        "office": go_to_office,
        "restroom": go_to_restroom,
        "corridor": go_to_corridor
    }
    nav_result = nav_functions[location]()
    if not nav_result.get("sent"):
        return {"sent": False, "error": f"导航失败: {nav_result.get('error', '未知错误')}", "step": "navigation"}

    # 机械臂
    arm_result = arm_control(arm_command)
    if not arm_result.get("sent"):
        return {"sent": False, "error": f"机械臂指令失败: {arm_result.get('error', '未知错误')}", "step": "arm_control"}

    location_names = {"office": "办公室", "restroom": "休息室", "corridor": "走廊"}
    arm_names = ["归位", "夹取", "释放", "搬运"]
    return {"sent": True, "message": f"✅ 已发送组合任务：前往「{location_names[location]}」 + 机械臂「{arm_names[arm_command]}」"}

# ========== 创建LangChain工具 ==========

# 使用 StructuredTool.from_function 创建工具（显式提供描述，避免 docstring 中大括号被 PromptTemplate 误解析）
ArmControlTool = StructuredTool.from_function(
    arm_control,
    name="arm_control",
    description=(
        "控制机械臂整体动作和姿态（不控制夹爪开合）。"
        "参数: command (0=归位, 1=准备抓取, 2=准备递送, 3=搬运模式)。"
        "返回字段: sent(布尔), message/错误信息。"
    ),
)

GoToOfficeTool = StructuredTool.from_function(
    go_to_office,
    name="go_to_office",
    description=(
        "导航到办公室。返回字段: sent(布尔), message/错误信息。"
    ),
)

GoToRestroomTool = StructuredTool.from_function(
    go_to_restroom,
    name="go_to_restroom",
    description=(
        "导航到休息室。返回字段: sent(布尔), message/错误信息。"
    ),
)

GoToCorridorTool = StructuredTool.from_function(
    go_to_corridor,
    name="go_to_corridor",
    description=(
        "导航到走廊。返回字段: sent(布尔), message/错误信息。"
    ),
)

ComplexTaskTool = StructuredTool.from_function(
    complex_task,
    name="complex_task",
    description=(
        "执行组合任务：先导航到地点，再执行机械臂动作。"
        "参数: location(office/restroom/corridor), arm_command(0-3)。"
        "返回字段: sent(布尔), message/错误信息。"
    ),
)

GripperControlTool = StructuredTool.from_function(
    gripper_control,
    name="gripper_control",
    description=(
        "控制夹爪开合动作（仅控制夹爪夹紧和松开）。"
        "参数: action (1=夹紧, 2=松开)。"
        "返回字段: sent(布尔), message/错误信息。"
    ),
)

# ========== 工具列表 ==========
ALL_TOOLS = [
    ArmControlTool,
    GoToOfficeTool,
    GoToRestroomTool,
    GoToCorridorTool,
    ComplexTaskTool,
    GripperControlTool
]

def get_all_tools():
    """获取所有工具列表"""
    return ALL_TOOLS

def get_tool_names():
    """获取所有工具名称列表"""
    return [tool.name for tool in ALL_TOOLS]

def get_tool_by_name(name: str):
    """根据名称获取工具"""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None

def get_tools_info():
    """获取工具信息字典"""
    return [
        {
            "name": tool.name,
            "description": tool.description
        }
        for tool in ALL_TOOLS
    ]


if __name__ == "__main__":
    # 显示工具信息
    print("\n🤖 机器人控制工具:")
    for tool_info in get_tools_info():
        print(f"  - {tool_info['name']}: {tool_info['description'][:80]}...")
    
    print("\n🤖 机器人工具模块测试完成！")
