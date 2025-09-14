# [<title="腾讯云 TTS 文本转语音服务节点 (Flask)">]
# -*- coding: utf-8 -*-
"""
腾讯云 TTS 文本转语音服务节点 (Flask)
接收文本，调用腾讯云 TTS API 合成语音，并返回 Base64 编码的音频数据。
依赖: pip install flask tencentcloud-sdk-python
环境变量: TENCENTCLOUD_SECRET_ID, TENCENTCLOUD_SECRET_KEY
"""

import os
import sys
import base64
from flask import Flask, request, jsonify

# --- 导入腾讯云 TTS SDK ---
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tts.v20190823 import tts_client, models

# ================== 配置区 ==================
# 请通过环境变量设置密钥
SECRET_ID = os.getenv("TENCENTCLOUD_SECRET_ID")
SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY")

if not SECRET_ID or not SECRET_KEY:
    print("错误: 请设置环境变量 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY", file=sys.stderr)


# 默认 TTS 配置参数 (可以在请求中覆盖)
DEFAULT_VOICE_TYPE = 101001 # 默认音色 ID
DEFAULT_PRIMARY_LANGUAGE = 1 # 默认主语言 (1=中文)
DEFAULT_SAMPLE_RATE = 16000 # 默认采样率
DEFAULT_SPEED = 0 # 默认语速
DEFAULT_CODEC = "wav" # 默认返回格式

# ============================================

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    """根路径，用于健康检查"""
    return jsonify({"message": "腾讯云 TTS 服务节点已启动", "status": "OK"})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """处理 /synthesize 路由的 POST 请求"""
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "请求中缺少 'text' 字段或文本为空"}), 400

    # 从请求中获取可选参数，或使用默认值
    voice_type = data.get('voice_type', DEFAULT_VOICE_TYPE)
    primary_language = data.get('primary_language', DEFAULT_PRIMARY_LANGUAGE)
    sample_rate = data.get('sample_rate', DEFAULT_SAMPLE_RATE)
    speed = data.get('speed', DEFAULT_SPEED)
    codec = data.get('codec', DEFAULT_CODEC).lower()

    # 验证 codec 参数
    if codec not in ["wav", "mp3", "pcm"]:
        return jsonify({"error": "不支持的 'codec' 格式，支持: wav, mp3, pcm"}), 400

    try:
        # --- 调用腾讯云 TTS API ---
        cred = credential.Credential(SECRET_ID, SECRET_KEY)

        httpProfile = HttpProfile()
        httpProfile.endpoint = "tts.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile

        # 注意：区域 "ap-guangzhou" 可根据需要更改
        client = tts_client.TtsClient(cred, "ap-guangzhou", clientProfile)

        req = models.TextToVoiceRequest()
        req.Text = text
        req.VoiceType = voice_type
        req.PrimaryLanguage = primary_language
        req.SampleRate = sample_rate
        # 使用当前时间戳生成唯一的 SessionId
        import time
        req.SessionId = f"tts_service_{int(time.time() * 1000)}" 
        req.Speed = speed
        req.Codec = codec

        print(f"-> 接收到合成请求: '{text[:30]}...' (参数: VT={voice_type}, SR={sample_rate}, C={codec})")
        resp = client.TextToVoice(req)
        print("<- 腾讯云 TTS 合成完成.")

        if resp.Audio and resp.SessionId:
            # resp.Audio 已经是 Base64 编码的字符串
            audio_base64 = resp.Audio
            print(f"-> 成功生成音频，Base64 长度: {len(audio_base64)} 字符")

            # 构造并返回成功的 JSON 响应
            return jsonify({
                "success": True,
                "message": "语音合成成功",
                "audio_base64": audio_base64, # 返回 Base64 音频数据
                "session_id": resp.SessionId,
                "sample_rate": sample_rate,
                "codec": codec
            })

        else:
            error_msg = "腾讯云 TTS API 返回响应中没有音频数据"
            print(f"! 错误: {error_msg}")
            return jsonify({
                "success": False,
                "error": error_msg
            }), 500

    except TencentCloudSDKException as err:
        error_msg = f"Tencent Cloud SDK Error: {err}"
        print(f"! 腾讯云 SDK 错误: {err}")
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500
    except Exception as e:
        error_msg = f"General Server Error: {e}"
        print(f"! 其他错误: {e}")
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

if __name__ == '__main__':
    print("🚀 启动腾讯云 TTS 服务节点...")
    print("请确保已设置环境变量 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY")
    # 启动 Flask 应用
    # host='0.0.0.0' 允许外部访问
    # debug=True 有助于开发时看到错误，生产环境应设为 False
    app.run(host='0.0.0.0', port=5001, debug=True)