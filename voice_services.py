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
import uuid
from typing import Optional
from speaker_local import LocalSpeaker
from flask import Flask, request, jsonify

# === 导入统一日志配置 ===
from logger_config import (
    create_server_logger,
    set_request_id,
    log_request_start,
    log_request_end,
    log_asr_result,
    log_tts_request
)

# 创建logger实例（服务器端）
logger = create_server_logger("voice_services", level=os.getenv("LOG_LEVEL", "INFO"))

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

# 添加响应后处理，确保连接正确关闭
@app.after_request
def after_request(response):
    """确保响应后连接正确关闭"""
    # 设置连接关闭头，避免 keepalive 导致连接积累
    response.headers['Connection'] = 'close'
    return response

# --- 说话人识别配置 ---
SPEAKER_MODEL_DIR = os.getenv("SPEAKER_MODEL_DIR", "/home/lxc/.wespeaker/chinese")
SPEAKER_DB_DIR = os.getenv("SPEAKER_DB_DIR", "./speaker_db")
SPEAKER_THRESHOLD = float(os.getenv("SPEAKER_THRESHOLD", "0.62"))
SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "cuda:0") 
UNREGISTERED_ID = os.getenv("SPEAKER_UNREGISTERED_ID", "UNREGISTERED")

# 懒加载全局 LocalSpeaker 实例
_local_speaker_instance: Optional[LocalSpeaker] = None

def get_local_speaker() -> LocalSpeaker:
    global _local_speaker_instance
    if _local_speaker_instance is None:
        logger.info("初始化 LocalSpeaker 模型与数据库...")
        spk = LocalSpeaker(model_name_or_dir=SPEAKER_MODEL_DIR, db_dir=SPEAKER_DB_DIR)
        # 统一设置设备（优先环境变量），speaker_local 默认 cuda:0，可能在无 GPU 环境报错
        if SPEAKER_DEVICE:
            try:
                spk.set_device(SPEAKER_DEVICE)
            except Exception as e:
                logger.warning(f"设置设备为 {SPEAKER_DEVICE} 失败，回退默认设备: {e}")
        _local_speaker_instance = spk
        logger.info("LocalSpeaker 初始化完成")
    return _local_speaker_instance

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

        logger.debug("正在调用腾讯云ASR...")
        resp = client.SentenceRecognition(req)
        result_text = getattr(resp, 'Result', "")
        log_asr_result(logger, result_text or "(空)")
        return {
            "success": True,
            "result": result_text,
            "request_id": resp.RequestId,
            "duration": getattr(resp, 'AudioDuration', None)
        }
    except AsrException as err:
        logger.error(f"ASR SDK 错误: {err}")
        return {"success": False, "error": f"Tencent ASR SDK Error: {err}"}
    except Exception as e:
        logger.error(f"ASR 其他识别错误: {e}", exc_info=True)
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

        log_tts_request(logger, text)
        logger.debug("正在调用腾讯云TTS...")
        resp = client.TextToVoice(req)
        logger.info("TTS 合成完成")

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
            logger.error(f"错误: {error_msg}")
            return {"success": False, "error": error_msg}

    except TtsException as err:
        error_msg = f"Tencent Cloud TTS SDK Error: {err}"
        logger.error(f"TTS SDK 错误: {err}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = f"TTS General Server Error: {e}"
        logger.error(f"TTS 其他错误: {e}", exc_info=True)
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
            "tts": "/tts/synthesize",
            "speaker_register": "/speaker/register",
            "speaker_verify": "/speaker/verify"
        }
    })

@app.route('/asr/recognize', methods=['POST'])
def asr_recognize():
    """ASR 识别接口"""
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    log_request_start(logger, "/asr/recognize", "POST")
    
    if not request.is_json:
        logger.warning("请求不是JSON格式")
        log_request_end(logger, 400)
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    audio_base64 = data.get('audio_base64')

    if not audio_base64:
        logger.warning("缺少 audio_base64 字段")
        log_request_end(logger, 400)
        return jsonify({"error": "缺少 'audio_base64' 字段"}), 400

    try:
        logger.debug("ASR 接收到 Base64 音频数据，正在解码...")
        # 腾讯云 SDK 内部期望的是 bytes，base64.b64decode 直接返回 bytes
        audio_data = base64.b64decode(audio_base64)
        logger.debug(f"ASR 解码完成，音频数据大小: {len(audio_data)} 字节")

        result = recognize_audio_with_tencent(audio_data)
        log_request_end(logger, 200)
        # 错误已在函数内处理
        return jsonify(result)

    except Exception as e:
        logger.error(f"ASR 处理请求时出错: {e}", exc_info=True)
        log_request_end(logger, 500)
        return jsonify({"success": False, "error": f"ASR Server Error: {e}"}), 500

@app.route('/tts/synthesize', methods=['POST'])
def tts_synthesize():
    """TTS 合成接口"""
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    log_request_start(logger, "/tts/synthesize", "POST")
    
    if not request.is_json:
        logger.warning("请求不是JSON格式")
        log_request_end(logger, 400)
        return jsonify({"error": "请求必须是 JSON 格式"}), 400

    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        logger.warning("缺少 text 字段或文本为空")
        log_request_end(logger, 400)
        return jsonify({"error": "请求中缺少 'text' 字段或文本为空"}), 400

    # 获取并验证参数，使用默认值
    voice_type = data.get('voice_type', TTS_DEFAULT_VOICE_TYPE)
    primary_language = data.get('primary_language', TTS_DEFAULT_PRIMARY_LANGUAGE)
    sample_rate = data.get('sample_rate', TTS_DEFAULT_SAMPLE_RATE)
    speed = data.get('speed', TTS_DEFAULT_SPEED)
    codec = data.get('codec', TTS_DEFAULT_CODEC).lower()

    if codec not in ["wav", "mp3", "pcm"]:
        logger.warning(f"不支持的codec格式: {codec}")
        log_request_end(logger, 400)
        return jsonify({"error": "不支持的 'codec' 格式，支持: wav, mp3, pcm"}), 400

    result = synthesize_text_with_tencent(text, voice_type, primary_language, sample_rate, speed, codec)
    log_request_end(logger, 200)
    # 错误已在函数内处理
    return jsonify(result)

# --- 可选：文件上传识别接口 (方便测试) ---
@app.route('/asr/recognize_file', methods=['POST'])
def asr_recognize_file():
    """通过上传 WAV 文件进行 ASR 识别"""
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    log_request_start(logger, "/asr/recognize_file", "POST")
    
    if 'file' not in request.files:
        logger.warning("请求中缺少 file 字段")
        log_request_end(logger, 400)
        return jsonify({"error": "请求中缺少 'file' 字段"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("未选择文件")
        log_request_end(logger, 400)
        return jsonify({"error": "未选择文件"}), 400

    try:
        file_content = file.read()
        logger.info(f"ASR 文件上传识别，文件大小: {len(file_content)} 字节")
        result = recognize_audio_with_tencent(file_content)
        log_request_end(logger, 200)
        return jsonify(result)

    except Exception as e:
        logger.error(f"ASR 文件处理时出错: {e}", exc_info=True)
        log_request_end(logger, 500)
        return jsonify({"success": False, "error": f"ASR File Error: {e}"}), 500


# --- 说话人注册与认证 ---
@app.route('/speaker/register', methods=['POST'])
def speaker_register():
    """语音身份注册：接收 { id, audio_base64 }，返回是否成功"""
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    log_request_start(logger, "/speaker/register", "POST")
    
    if not request.is_json:
        logger.warning("请求不是JSON格式")
        log_request_end(logger, 400)
        return jsonify({"success": False, "error": "请求必须是 JSON"}), 400

    data = request.get_json()
    register_id = (data.get('id') or data.get('register_id') or '').strip()
    audio_base64 = data.get('audio_base64')

    if not register_id:
        logger.warning("缺少 id 字段")
        log_request_end(logger, 400)
        return jsonify({"success": False, "error": "缺少 'id'"}), 400
    if not audio_base64:
        logger.warning("缺少 audio_base64 字段")
        log_request_end(logger, 400)
        return jsonify({"success": False, "error": "缺少 'audio_base64'"}), 400

    try:
        logger.info(f"注册声纹: {register_id}")
        spk = get_local_speaker()
        result = spk.register(register_id, audio_base64)
        logger.info(f"声纹注册成功: {register_id}")
        log_request_end(logger, 200)
        return jsonify({"success": True, "id": result.get("name"), "path": result.get("path")})
    except Exception as e:
        logger.error(f"说话人注册失败: {e}", exc_info=True)
        log_request_end(logger, 500)
        return jsonify({"success": False, "error": f"register failed: {e}"}), 500


@app.route('/speaker/verify', methods=['POST'])
def speaker_verify():
    """语音身份认证：接收 { audio_base64 }，返回匹配的 id 或 UNREGISTERED"""
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    log_request_start(logger, "/speaker/verify", "POST")
    
    if not request.is_json:
        logger.warning("请求不是JSON格式")
        log_request_end(logger, 400)
        return jsonify({"success": False, "error": "请求必须是 JSON"}), 400

    data = request.get_json()
    audio_base64 = data.get('audio_base64')
    threshold = float(data.get('threshold', SPEAKER_THRESHOLD))

    if not audio_base64:
        logger.warning("缺少 audio_base64 字段")
        log_request_end(logger, 400)
        return jsonify({"success": False, "error": "缺少 'audio_base64'"}), 400

    try:
        logger.debug("声纹认证中...")
        spk = get_local_speaker()
        res = spk.recognize(audio_base64)
        name = res.get('name')
        confidence = float(res.get('confidence') or 0.0)
        is_registered = bool(name) and confidence >= threshold
        final_id = name if is_registered else UNREGISTERED_ID
        
        if is_registered:
            logger.info(f"声纹认证通过: {final_id} (置信度: {confidence:.2f})")
        else:
            logger.info(f"声纹未识别 (置信度: {confidence:.2f})")
        
        log_request_end(logger, 200)
        return jsonify({
            "success": True,
            "id": final_id,
            "confidence": confidence,
            "threshold": threshold,
            "registered": is_registered
        })
    except Exception as e:
        logger.error(f"说话人认证失败: {e}", exc_info=True)
        log_request_end(logger, 500)
        return jsonify({"success": False, "error": f"verify failed: {e}"}), 500

if __name__ == '__main__':
    if not SECRET_ID or not SECRET_KEY:
        logger.warning("警告: 未设置环境变量 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY。服务功能将受限。")

    # 从环境变量获取host和port，如果没有设置则使用默认值
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 4999))
    
    logger.info("=" * 60)
    logger.info("启动语音服务节点")
    logger.info(f"服务地址: http://{host}:{port}")
    logger.info("可用端点:")
    logger.info("  - POST /asr/recognize")
    logger.info("  - POST /tts/synthesize")
    logger.info("  - POST /speaker/register")
    logger.info("  - POST /speaker/verify")
    logger.info("=" * 60)
    
    # 使用 threaded=True 支持并发连接，避免连接阻塞
    # 添加连接超时和 keepalive 配置
    app.run(host=host, port=port, debug=False, threaded=True)
