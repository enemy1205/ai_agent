import sys
import logging
import paho.mqtt.client as mqtt
import time
import json
from pathlib import Path

# 日志配置
logger = logging.getLogger('RobotController')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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

# ========== MQTT通信函数 ==========

def connect_mqtt():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = lambda c, u, f, rc, p=None: logger.info("✅ MQTT连接成功") if rc == 0 else logger.error(f"❌ 连接失败: {rc}")
    client.on_publish = lambda c, u, mid, rc, p: logger.info(f"消息已发送 (ID: {mid})")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    return client

def _send_navigation(client, topic, x, y, z):
    payload = str({"x": x, "y": y, "z": z})
    logger.info(f"发送导航指令: {topic} → {payload}")
    result = client.publish(topic, payload, qos=1)
    try:
        return result.wait_for_publish(timeout=5)
    except:
        return False

def _send_arm_command(client, topic, command):
    payload = str({"command": command})
    logger.info(f"发送机械臂指令: {command} → {payload}")
    result = client.publish(topic, payload, qos=1)
    try:
        return result.wait_for_publish(timeout=5)
    except:
        return False

# ========== Agent可调用工具函数 ==========

def arm_control(command: int) -> dict:
    """
    控制机械臂执行动作（不移动机器人）
    适用场景：用户说“拿起水”、“放下杯子”、“机械臂归位”等，不需要机器人移动位置时调用。
    参数 command:
        0 → 机械臂回到原位（归位）
        1 → 夹取物品（如拿水）
        2 → 释放物品（如递给用户）
        3 → 搬运模式（移动中保持夹持）
    返回:
        {"sent": True, "message": "✅ 机械臂指令已发送"} → 成功
        {"sent": False, "error": "原因"} → 失败
    """
    if command not in [0, 1, 2, 3]:
        return {"sent": False, "error": "command 必须是 0, 1, 2 或 3"}

    client = connect_mqtt()
    client.loop_start()
    time.sleep(0.3)
    if not client.is_connected():
        client.loop_stop()
        return {"sent": False, "error": "MQTT连接失败"}

    success = _send_arm_command(client, MQTT_TOPIC_ARM_CONTROL, command)
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()

    desc = {0: "归位", 1: "夹取", 2: "释放", 3: "搬运"}[command]
    if success:
        return {"sent": True, "message": f"✅ 已发送机械臂「{desc}」指令 (command={command})"}
    else:
        return {"sent": False, "error": "MQTT消息发送失败"}


def _load_locations_config() -> dict:
    """加载坐标配置文件 config/locations.json"""
    try:
        config_path = Path(__file__).parent / "config" / "locations.json"
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
    适用场景：用户说“去办公室”、“到办公室去”等，不需要拿/放物品时调用。
    返回:
        {"sent": True, "message": "✅ 前往办公室指令已发送"} → 成功
        {"sent": False, "error": "原因"} → 失败
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
    适用场景：用户说“去休息室”、“到休息室”等，不需要拿/放物品时调用。
    返回:
        {"sent": True, "message": "✅ 前往休息室指令已发送"} → 成功
        {"sent": False, "error": "原因"} → 失败
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
    适用场景：用户说“去走廊”、“到走廊中间”等，不需要拿/放物品时调用。
    返回:
        {"sent": True, "message": "✅ 前往走廊指令已发送"} → 成功
        {"sent": False, "error": "原因"} → 失败
    """
    locations = _load_locations_config()
    pos = locations.get("corridor", {})
    x, y, z = pos.get("x", 97.407), pos.get("y", 55.386), pos.get("z", 0.0)
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
      - “去办公室拿瓶水” → location="office", arm_command=1
      - “把水送到休息室” → location="restroom", arm_command=3
      - “去走廊然后放下东西” → location="corridor", arm_command=2
    参数:
      location: "office" | "restroom" | "corridor"
      arm_command: 0=归位, 1=夹取, 2=释放, 3=搬运
    返回:
      {"sent": True, "message": "..."} → 两个指令均已发送
      {"sent": False, "error": "...", "step": "navigation|arm_control"} → 哪一步失败
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
    if not nav_result["sent"]:
        return {
            "sent": False,
            "error": f"导航失败: {nav_result.get('error', '未知错误')}",
            "step": "navigation"
        }

    # 机械臂
    arm_result = arm_control(arm_command)
    if not arm_result["sent"]:
        return {
            "sent": False,
            "error": f"机械臂指令失败: {arm_result.get('error', '未知错误')}",
            "step": "arm_control"
        }

    return {
        "sent": True,
        "message": f"✅ 已发送组合任务：前往「{location}」 + 机械臂「{['归位','夹取','释放','搬运'][arm_command]}」"
    }

# ========== 测试用（可选）==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n🧪 测试：机械臂夹取")
    print(arm_control(1))
    print("\n🧪 测试：去办公室")
    print(go_to_office())
    print("\n🧪 测试：去休息室并夹取")
    print(complex_task("restroom", 1))