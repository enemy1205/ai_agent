# [<title="语音识别服务节点 (Flask)">]
# -*- coding: utf-8 -*-
"""
语音识别服务节点 (Flask)
接收 Base64 编码的 WAV 音频数据，调用腾讯云 ASR，并返回识别结果。
"""

import os
import sys
import base64
import io
import json
import time
from flask import Flask, request, jsonify

# --- 导入你的腾讯云识别逻辑 ---
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models

# --- 配置 ---
SECRET_ID = os.getenv("TENCENTCLOUD_SECRET_ID")
SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY")
ENGINE_MODEL_TYPE = "16k_zh"
VOICE_FORMAT = "wav"

if not SECRET_ID or not SECRET_KEY:
    print("错误: 请设置环境变量 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY")
    sys.exit(1)

def recognize_audio_with_tencent(audio_data):
    """使用腾讯云 SentenceRecognition 接口识别音频数据 (内部函数)"""
    try:
        cred = credential.Credential(SECRET_ID, SECRET_KEY)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "asr.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = asr_client.AsrClient(cred, "ap-guangzhou", clientProfile)

        req = models.SentenceRecognitionRequest()
        params = {
            "ProjectId": 0,
            "SubServiceType": 2,
            "EngSerViceType": ENGINE_MODEL_TYPE,
            "SourceType": 1,
            "VoiceFormat": VOICE_FORMAT,
            "UsrAudioKey": f"audio_{int(time.time())}",
            "Data": base64.b64encode(audio_data).decode('utf-8'),
            "DataLen": len(audio_data)
        }
        req.from_json_string(json.dumps(params))

        print("🔄 正在调用腾讯云识别...")
        resp = client.SentenceRecognition(req)
        print("✅ 腾讯云识别完成.")
        return {
            "success": True,
            "result": resp.Result if hasattr(resp, 'Result') else "",
            "request_id": resp.RequestId,
            "duration": resp.AudioDuration if hasattr(resp, 'AudioDuration') else None
        }
    except TencentCloudSDKException as err:
        print(f"❌ 腾讯云 SDK 错误: {err}")
        return {"success": False, "error": f"Tencent SDK Error: {err}"}
    except Exception as e:
        print(f"❌ 其他识别错误: {e}")
        return {"success": False, "error": f"General Error: {e}"}

# --- Flask 应用 ---
app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize():
    """处理 /recognize 路由的 POST 请求"""
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    audio_base64 = data.get('audio_base64')
    sample_rate = data.get('sample_rate', 16000) # 可选参数

    if not audio_base64:
        return jsonify({"error": "缺少 'audio_base64' 字段"}), 400

    try:
        # 1. 解码 Base64
        print("-> 接收到 Base64 音频数据，正在解码...")
        audio_data = base64.b64decode(audio_base64)
        print(f"-> 解码完成，音频数据大小: {len(audio_data)} 字节")

        # 2. 调用识别函数
        result = recognize_audio_with_tencent(audio_data)

        # 3. 返回 JSON 响应
        return jsonify(result)

    except Exception as e:
        print(f"! 处理请求时出错: {e}")
        return jsonify({"success": False, "error": f"Server Error: {e}"}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "语音识别服务节点已启动", "status": "OK"})

if __name__ == '__main__':
    print("🚀 启动语音识别服务节点...")
    print("请确保已设置环境变量 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY")
    app.run(host='0.0.0.0', port=4999, debug=False) # 在所有接口监听，端口 4999