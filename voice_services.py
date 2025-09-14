# [<title="优化版统一语音服务节点 (Flask)">]
# -*- coding: utf-8 -*-
"""
优化版统一语音服务节点 (Flask)
整合 ASR (语音识别) 和 TTS (文本转语音) 功能。
- ASR: POST /asr/recognize
- TTS: POST /tts/synthesize
"""

import os
import sys
import base64
import json
import time
import logging
from flask import Flask, request, jsonify

# --- 腾讯云 SDK 导入 ---
# ASR
from tencentcloud.common import credential as asr_credential
from tencentcloud.common.profile.client_profile import ClientProfile as asr_client_profile
from tencentcloud.common.profile.http_profile import HttpProfile as asr_http_profile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException as AsrException
from tencentcloud.asr.v20190614 import asr_client, models as asr_models

# TTS
from tencentcloud.common import credential as tts_credential
from tencentcloud.common.profile.client_profile import ClientProfile as tts_client_profile
from tencentcloud.common.profile.http_profile import HttpProfile as tts_http_profile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException as TtsException
from tencentcloud.tts.v20190823 import tts_client, models as tts_models

# ================== 配置区 ==================
# 从环境变量获取腾讯云密钥
SECRET_ID = os.getenv("TENCENTCLOUD_SECRET_ID")
SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY")

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    # level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # handlers=[
    #     logging.StreamHandler(sys.stdout),
    #     logging.FileHandler('voice_service.log', encoding='utf-8')
    # ]
)
logger = logging.getLogger(__name__)

# ASR 配置
ASR_ENGINE_MODEL_TYPE = "16k_zh" # 适用于中文普通话
ASR_VOICE_FORMAT = "wav"

# TTS 配置
TTS_DEFAULT_VOICE_TYPE = 101001 # 默认音色
TTS_DEFAULT_PRIMARY_LANGUAGE = 1 # 1=中文
TTS_DEFAULT_SAMPLE_RATE = 16000  # 采样率
TTS_DEFAULT_SPEED = 0            # 语速 (-2 到 6)
TTS_DEFAULT_CODEC = "wav"        # 输出格式
# ============================================

app = Flask(__name__)

# --- ASR 核心逻辑 ---
def recognize_audio_with_tencent(audio_data: bytes) -> dict:
    """调用腾讯云 ASR 服务识别音频数据"""
    if not SECRET_ID or not SECRET_KEY:
        return {"success": False, "error": "腾讯云凭证未配置"}

    try:
        cred = asr_credential.Credential(SECRET_ID, SECRET_KEY)
        http_profile = asr_http_profile()
        http_profile.endpoint = "asr.tencentcloudapi.com"
        client_profile = asr_client_profile()
        client_profile.httpProfile = http_profile
        client = asr_client.AsrClient(cred, "ap-guangzhou", client_profile)

        req = asr_models.SentenceRecognitionRequest()
        params = {
            "ProjectId": 0,
            "SubServiceType": 2,
            "EngSerViceType": ASR_ENGINE_MODEL_TYPE,
            "SourceType": 1, # 1: 语音 URL 或语音数据 (Base64)
            "VoiceFormat": ASR_VOICE_FORMAT,
            "UsrAudioKey": f"audio_{int(time.time())}",
            "Data": base64.b64encode(audio_data).decode('utf-8'),
            "DataLen": len(audio_data)
        }
        req.from_json_string(json.dumps(params))

        logger.info("🔄 正在调用 ASR...")
        resp = client.SentenceRecognition(req)
        logger.info("✅ ASR 识别完成.")
        return {
            "success": True,
            "result": getattr(resp, 'Result', ""),
            "request_id": resp.RequestId,
            "duration": getattr(resp, 'AudioDuration', None)
        }
    except AsrException as err:
        logger.error(f"❌ ASR SDK 错误: {err}")
        return {"success": False, "error": f"Tencent ASR SDK Error: {err}"}
    except Exception as e:
        logger.error(f"❌ ASR 其他识别错误: {e}")
        return {"success": False, "error": f"ASR General Error: {e}"}

# --- TTS 核心逻辑 ---
def synthesize_text_with_tencent(text: str, voice_type: int, primary_language: int,
                                 sample_rate: int, speed: int, codec: str) -> dict:
    """调用腾讯云 TTS 服务合成语音"""
    if not SECRET_ID or not SECRET_KEY:
        return {"success": False, "error": "腾讯云凭证未配置"}

    try:
        cred = tts_credential.Credential(SECRET_ID, SECRET_KEY)
        http_profile = tts_http_profile()
        http_profile.endpoint = "tts.tencentcloudapi.com"
        client_profile = tts_client_profile()
        client_profile.httpProfile = http_profile
        client = tts_client.TtsClient(cred, "ap-guangzhou", client_profile)

        req = tts_models.TextToVoiceRequest()
        req.Text = text
        req.VoiceType = voice_type
        req.PrimaryLanguage = primary_language
        req.SampleRate = sample_rate
        req.SessionId = f"tts_service_{int(time.time() * 1000)}"
        req.Speed = speed
        req.Codec = codec

        logger.info(f"-> 调用 TTS: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        resp = client.TextToVoice(req)
        logger.info("<- TTS 合成完成.")

        if resp.Audio and resp.SessionId:
            return {
                "success": True,
                "message": "语音合成成功",
                "audio_base64": resp.Audio,
                "session_id": resp.SessionId,
                "sample_rate": sample_rate,
                "codec": codec
            }
        else:
            error_msg = "TTS API 返回响应中没有音频数据"
            logger.error(f"! 错误: {error_msg}")
            return {"success": False, "error": error_msg}

    except TtsException as err:
        error_msg = f"Tencent Cloud TTS SDK Error: {err}"
        logger.error(f"!TTS SDK 错误: {err}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"TTS General Server Error: {e}"
        logger.error(f"!TTS 其他错误: {e}")
        return {"success": False, "error": error_msg}

# --- Flask 路由 ---
@app.route('/', methods=['GET'])
def home():
    """根路径，服务健康检查"""
    return jsonify({
        "message": "统一语音服务节点已启动",
        "status": "OK",
        "endpoints": {
            "asr": "/asr/recognize",
            "tts": "/tts/synthesize"
        }
    })

@app.route('/asr/recognize', methods=['POST'])
def asr_recognize():
    """ASR 识别接口"""
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    audio_base64 = data.get('audio_base64')

    if not audio_base64:
        return jsonify({"error": "缺少 'audio_base64' 字段"}), 400

    try:
        logger.info("-> ASR 接收到 Base64 音频数据，正在解码...")
        # 腾讯云 SDK 内部期望的是 bytes，base64.b64decode 直接返回 bytes
        audio_data = base64.b64decode(audio_base64)
        logger.info(f"-> ASR 解码完成，音频数据大小: {len(audio_data)} 字节")

        result = recognize_audio_with_tencent(audio_data)
        # 错误已在函数内处理
        return jsonify(result)

    except Exception as e:
        logger.error(f"! ASR 处理请求时出错: {e}")
        return jsonify({"success": False, "error": f"ASR Server Error: {e}"}), 500

@app.route('/tts/synthesize', methods=['POST'])
def tts_synthesize():
    """TTS 合成接口"""
    if not request.is_json:
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "请求中缺少 'text' 字段或文本为空"}), 400

    # 获取并验证参数，使用默认值
    voice_type = data.get('voice_type', TTS_DEFAULT_VOICE_TYPE)
    primary_language = data.get('primary_language', TTS_DEFAULT_PRIMARY_LANGUAGE)
    sample_rate = data.get('sample_rate', TTS_DEFAULT_SAMPLE_RATE)
    speed = data.get('speed', TTS_DEFAULT_SPEED)
    codec = data.get('codec', TTS_DEFAULT_CODEC).lower()

    if codec not in ["wav", "mp3", "pcm"]:
        return jsonify({"error": "不支持的 'codec' 格式，支持: wav, mp3, pcm"}), 400

    result = synthesize_text_with_tencent(text, voice_type, primary_language, sample_rate, speed, codec)
    # 错误已在函数内处理
    return jsonify(result)

# --- 可选：文件上传识别接口 (方便测试) ---
@app.route('/asr/recognize_file', methods=['POST'])
def asr_recognize_file():
    """通过上传 WAV 文件进行 ASR 识别"""
    if 'file' not in request.files:
        return jsonify({"error": "请求中缺少 'file' 字段"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    try:
        file_content = file.read()
        logger.info(f"-> ASR 文件上传识别，文件大小: {len(file_content)} 字节")
        result = recognize_audio_with_tencent(file_content)
        return jsonify(result)

    except Exception as e:
        logger.error(f"! ASR 文件处理时出错: {e}")
        return jsonify({"success": False, "error": f"ASR File Error: {e}"}), 500


if __name__ == '__main__':
    if not SECRET_ID or not SECRET_KEY:
        logger.warning("警告: 未设置环境变量 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY。服务功能将受限。")

    # 从环境变量获取host和port，如果没有设置则使用默认值
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 4999))
    
    logger.info(f"🚀 启动语音服务节点 (host={host}, port={port})...")
    app.run(host=host, port=port, debug=False)
