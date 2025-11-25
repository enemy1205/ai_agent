import os
import pyaudio
import numpy as np
import queue
import threading
import time
import base64
import requests
import json
import uuid
from io import BytesIO
from scipy.io import wavfile

# === 导入统一日志配置 ===
from logger_config import (
    create_robot_logger,
    set_request_id,
    log_asr_result,
    log_tts_request,
    log_vad_event
)

# 创建logger实例
logger = create_robot_logger("pipeline", level=os.getenv("LOG_LEVEL", "INFO"))

# --- Silero VAD 相关 ---
try:
    from silero_vad import load_silero_vad, VADIterator
except ImportError:
    logger.critical("未找到 silero-vad 库，请运行 'pip install silero-vad' 安装")
    exit(1)

# --- 配置 ---
# 麦克风录音参数
FORMAT = pyaudio.paFloat32  # Silero VAD 期望 float32
CHANNELS = 1
SAMPLE_RATE = 16000         # Silero VAD 推荐 16kHz
CHUNK = 512                 # 每次读取的样本数

# VAD 参数
VAD_THRESHOLD = 0.5         # 语音置信度阈值
MIN_SILENCE_DURATION_MS = 300 # 语音结束判断所需的最小静音时长 (毫秒)
SPEECH_PAD_MS = 100         # 在语音开始前/结束后填充的静音时长 (毫秒)

# 服务器地址配置
VOICE_SERVER_IP = "202.38.214.151"
VOICE_SERVER_PORT = 4999
VOICE_SERVER_BASE_URL = f"http://{VOICE_SERVER_IP}:{VOICE_SERVER_PORT}"
ASR_ENDPOINT = f"{VOICE_SERVER_BASE_URL}/asr/recognize"
TTS_ENDPOINT = f"{VOICE_SERVER_BASE_URL}/tts/synthesize"
SPEAKER_VERIFY_ENDPOINT = f"{VOICE_SERVER_BASE_URL}/speaker/verify"

# LLM服务配置（支持本地和云端）
LLM_SERVER_IP = "202.38.214.151" # <-- 修改为你的大模型服务IP
LLM_SERVER_PORT = 5000           # <-- 修改为你的大模型服务端口
LLM_API_BASE = f"http://{LLM_SERVER_IP}:{LLM_SERVER_PORT}/v1"
LLM_ENDPOINT = f"{LLM_API_BASE}/chat/completions"  # 使用 chat/completions 端点（支持记忆功能）

# LLM参数配置 - 现在由服务器端统一管理

# 系统提示词 (可选)
# SYSTEM_PROMPT = "你是搭载在迎宾服务机器人上的AI智能体，你的名字叫Siri。请用中文回答相应我的需求。你只需对我的要求做出语言回应，涉及到真实执行的并不需要你实际去做，比如抱或者拿某个东西等等，你只需要回答我。"
# 如果需要系统提示词，可以在构建 full_prompt 时加入

# --- 全局变量 ---
audio_queue = queue.Queue()
speech_buffer = []  # 存储当前检测到的语音数据
is_speaking = False
vad_iterator = None
pyaudio_instance = None
stream = None
playback_stream = None # 用于播放 TTS 音频的 PyAudio 流

# --- 防止音频反馈循环的标志 ---
is_playing_tts = False
playback_lock = threading.Lock()  # 播放锁，确保同一时间只有一个音频在播放

# --- 对话历史 ---
conversation_history = []
# 会话ID（用于维持对话记忆）
llm_session_id = None

# --- 声纹认证控制 ---
ENABLE_SPEAKER_AUTH = os.getenv("ENABLE_SPEAKER_AUTH", "false").lower() == "true"  # 声纹认证开关
is_register_mode = False  # 注册模式标志
pending_register_id = None  # 待注册的用户ID

def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio 回调函数，将录音数据放入队列"""
    if status:
        logger.warning(f"音频回调状态异常: {status}")
    
    try:
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        audio_queue.put(audio_chunk)
    except Exception as e:
        logger.error(f"音频回调错误: {e}", exc_info=True)
    
    return (in_data, pyaudio.paContinue)

def play_audio_from_base64(audio_base64_str, sample_rate=16000, codec="wav"):
    """播放 Base64 编码的音频数据"""
    global pyaudio_instance, playback_stream, is_playing_tts, playback_lock
    
    # 使用锁确保同一时间只有一个线程在播放音频
    with playback_lock:
        try:
            if not audio_base64_str:
                logger.warning("TTS返回音频数据为空")
                return

            is_playing_tts = True

            # 1. 解码 Base64
            audio_bytes = base64.b64decode(audio_base64_str)

            # 2. 确定播放参数 (简化处理)
            if codec.lower() in ["wav", "pcm"]:
                audio_format = pyaudio.paInt16
                width = 2
            elif codec.lower() == "mp3":
                logger.warning("MP3格式需要额外解码库，假设为PCM")
                audio_format = pyaudio.paInt16
                width = 2
            else:
                logger.error(f"不支持的音频格式: {codec}")
                is_playing_tts = False
                return

            # 3. 关闭旧的播放流并重新打开（避免 underrun）
            if playback_stream and not playback_stream.is_stopped():
                try:
                    playback_stream.stop_stream()
                    playback_stream.close()
                except:
                    pass
            
            if not pyaudio_instance:
                logger.error("PyAudio实例未初始化")
                is_playing_tts = False
                return
                
            playback_stream = pyaudio_instance.open(
                format=audio_format,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=1024
            )

            # 4. 播放音频数据
            logger.info("播放TTS音频")
            playback_stream.write(audio_bytes)
            
            # 5. 播放完毕后关闭流
            playback_stream.stop_stream()
            playback_stream.close()
            playback_stream = None

        except Exception as e:
            logger.error(f"播放音频出错: {e}", exc_info=True)
            # 出错时尝试清理播放流
            if playback_stream:
                try:
                    playback_stream.stop_stream()
                    playback_stream.close()
                    playback_stream = None
                except:
                    pass
        finally:
            is_playing_tts = False

def send_audio_to_asr_server(audio_data_float32, sample_rate):
    """
    将 float32 音频数据转换为 Base64 并发送到 ASR 服务器
    """
    try:
        # 1. 转换为 int16 (标准 WAV 格式)
        audio_data_float32 = np.clip(audio_data_float32, -1.0, 1.0)
        audio_data_int16 = (audio_data_float32 * 32767).astype(np.int16)

        # 2. 写入内存缓冲区 (BytesIO)
        buffer = BytesIO()
        wavfile.write(buffer, sample_rate, audio_data_int16)
        buffer.seek(0)

        # 3. 编码为 Base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        # 4. 准备并发送 POST 请求到 ASR 服务
        payload = {
            "audio_base64": audio_base64,
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(ASR_ENDPOINT, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                recognized_text = data.get("result", "").strip()
                if recognized_text:
                    log_asr_result(logger, recognized_text)
                    # 识别成功，将文本发送给 LLM
                    threading.Thread(target=process_with_llm, args=(recognized_text,), daemon=True).start()
                else:
                    logger.info("ASR识别结果为空")
            else:
                logger.error(f"ASR识别失败: {data.get('error')}")
        else:
            logger.error(f"ASR服务响应错误 ({response.status_code}): {response.text}")

    except requests.exceptions.RequestException as e:
        logger.error(f"ASR请求异常: {e}")
    except Exception as e:
        logger.error(f"ASR处理异常: {e}", exc_info=True)

def send_audio_to_asr_server_and_get_text(audio_data_float32, sample_rate):
    """
    将 float32 音频数据转换为 Base64 并发送到 ASR 服务器，返回识别结果
    """
    try:
        # 1. 转换为 int16 (标准 WAV 格式)
        audio_data_float32 = np.clip(audio_data_float32, -1.0, 1.0)
        audio_data_int16 = (audio_data_float32 * 32767).astype(np.int16)

        # 2. 写入内存缓冲区 (BytesIO)
        buffer = BytesIO()
        wavfile.write(buffer, sample_rate, audio_data_int16)
        buffer.seek(0)

        # 3. 编码为 Base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        # 4. 准备并发送 POST 请求到 ASR 服务
        payload = {
            "audio_base64": audio_base64,
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(ASR_ENDPOINT, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                recognized_text = data.get("result", "").strip()
                return recognized_text
            else:
                logger.error(f"ASR识别失败: {data.get('error')}")
                return None
        else:
            logger.error(f"ASR服务响应错误 ({response.status_code}): {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"ASR请求异常: {e}")
        return None
    except Exception as e:
        logger.error(f"ASR处理异常: {e}", exc_info=True)
        return None

def encode_float32_audio_to_base64_wav(audio_data_float32, sample_rate):
    """
    将 float32 PCM 音频编码为 WAV(Base64)。
    返回 (audio_base64, ok)。
    """
    try:
        audio_data_float32 = np.clip(audio_data_float32, -1.0, 1.0)
        audio_data_int16 = (audio_data_float32 * 32767).astype(np.int16)
        buffer = BytesIO()
        wavfile.write(buffer, sample_rate, audio_data_int16)
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64, True
    except Exception as e:
        logger.error(f"编码WAV失败: {e}")
        return "", False

def verify_speaker_before_asr(audio_data_float32, sample_rate, threshold=None):
    """
    先调用说话人认证服务，通过则返回 (True, name, confidence)，否则 (False, None, confidence)。
    """
    try:
        audio_base64, ok = encode_float32_audio_to_base64_wav(audio_data_float32, sample_rate)
        if not ok:
            return False, None, 0.0
        payload = {"audio_base64": audio_base64}
        if threshold is not None:
            payload["threshold"] = float(threshold)
        headers = {'Content-Type': 'application/json'}
        logger.info("进行声纹认证...")
        resp = requests.post(SPEAKER_VERIFY_ENDPOINT, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            logger.debug(f"声纹认证响应: {data}")
            if data.get("success"):
                is_registered = bool(data.get("registered"))
                name = data.get("id")
                confidence = float(data.get("confidence", 0.0))
                if is_registered:
                    logger.info(f"认证通过: {name} (置信度: {confidence:.2f})")
                else:
                    logger.warning(f"未注册用户 (置信度: {confidence:.2f})")
                return is_registered, name, confidence
            else:
                logger.error(f"声纹认证失败: {data.get('error')}")
                return False, None, 0.0
        else:
            logger.error(f"声纹认证服务响应错误 ({resp.status_code}): {resp.text}")
            return False, None, 0.0
    except requests.exceptions.RequestException as e:
        logger.error(f"声纹认证请求异常: {e}")
        return False, None, 0.0
    except Exception as e:
        logger.error(f"声纹认证异常: {e}", exc_info=True)
        return False, None, 0.0

def register_speaker(audio_data_float32, sample_rate, user_id):
    """注册说话人声纹"""
    try:
        audio_base64, ok = encode_float32_audio_to_base64_wav(audio_data_float32, sample_rate)
        if not ok:
            return False, "音频编码失败"
        
        payload = {
            "id": user_id,
            "audio_base64": audio_base64
        }
        headers = {'Content-Type': 'application/json'}
        
        logger.info(f"注册用户声纹: {user_id}")
        response = requests.post(f"{VOICE_SERVER_BASE_URL}/speaker/register", 
                               json=payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                logger.info(f"声纹注册成功: {user_id}")
                return True, f"用户 {user_id} 注册成功"
            else:
                error_msg = data.get("error", "注册失败")
                logger.error(f"声纹注册失败: {error_msg}")
                return False, error_msg
        else:
            error_msg = f"注册服务响应错误 ({response.status_code}): {response.text}"
            logger.error(error_msg)
            return False, error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"注册请求出错: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"处理注册时出错: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def handle_captured_speech(audio_data_float32, sample_rate):
    """新的语音处理流程：先ASR，根据结果决定后续处理"""
    global is_register_mode, pending_register_id
    
    # 1. 先进行ASR识别
    logger.info("进行语音识别...")
    recognized_text = send_audio_to_asr_server_and_get_text(audio_data_float32, sample_rate)
    
    if not recognized_text:
        logger.warning("ASR识别失败或结果为空")
        return
    
    log_asr_result(logger, recognized_text)
    
    # 2. 检查是否包含"注册新用户"指令
    if "注册新用户" in recognized_text:
        logger.info("检测到注册指令，进入注册模式")
        is_register_mode = True
        pending_register_id = f"user_{int(time.time())}"
        send_text_to_tts(f"请说一段话用于注册，您的用户ID是 {pending_register_id}")
        return
    
    # 3. 如果当前在注册模式，进行声纹注册
    if is_register_mode and pending_register_id:
        logger.info(f"注册用户声纹: {pending_register_id}")
        success, message = register_speaker(audio_data_float32, sample_rate, pending_register_id)
        send_text_to_tts(message)
        is_register_mode = False
        pending_register_id = None
        return
    
    # 4. 正常模式：检查声纹认证开关
    if not ENABLE_SPEAKER_AUTH:
        logger.info("声纹认证已关闭，直接LLM处理")
        process_with_llm(recognized_text)
        return
    
    # 5. 进行声纹认证
    is_ok, name, conf = verify_speaker_before_asr(audio_data_float32, sample_rate)
    if not is_ok:
        logger.warning("未注册用户，拒绝处理")
        send_text_to_tts("用户尚未注册")
        return
    
    logger.info(f"认证通过: {name} (置信度: {conf:.2f})")
    process_with_llm(recognized_text)

def call_local_llm(messages, session_id=None):
    """
    调用LLM服务（支持云端和本地，使用 chat/completions 端点）
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
        session_id: 可选的会话ID，用于维持对话记忆
    
    Returns:
        (reply_text, success, new_session_id): 回复文本、是否成功、新的会话ID
    """
    global llm_session_id
    try:
        # 构建请求数据 - 使用 chat/completions 格式
        data = {
            "messages": messages
        }
        
        # 如果有会话ID，添加到请求中
        if session_id:
            data["session_id"] = session_id
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            LLM_ENDPOINT,
            headers=headers,
            json=data,
            timeout=120  # 120秒超时
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                # 提取回复内容
                choice = result["choices"][0]
                if "message" in choice:
                    reply_text = choice["message"].get("content", "").strip()
                else:
                    # 兼容旧格式
                    reply_text = choice.get("text", "").strip()
                
                # 提取会话ID（如果返回了）
                metadata = result.get("metadata", {})
                if metadata.get("session_id"):
                    llm_session_id = metadata["session_id"]
                
                return reply_text, True, llm_session_id
            else:
                logger.error("LLM服务返回格式异常")
                return "", False, session_id
        else:
            logger.error(f"LLM服务调用失败，状态码: {response.status_code}")
            logger.debug(f"错误信息: {response.text}")
            return "", False, session_id
            
    except requests.exceptions.ConnectionError:
        logger.error(f"无法连接到LLM服务 ({LLM_ENDPOINT})")
        return "", False, session_id
    except requests.exceptions.Timeout:
        logger.error("LLM服务调用超时")
        return "", False, session_id
    except Exception as e:
        logger.error(f"LLM服务调用出错: {e}", exc_info=True)
        return "", False, session_id

def chat_with_local_llm(user_input, conversation_history):
    """
    与LLM进行对话，使用 chat/completions 格式
    兼容 http_agent_server_v2.py（本地LLM，有记忆）和 http_agent_server_v3.py（云端LLM，有记忆）
    
    两个版本都支持服务器端会话记忆，因此只需要传递当前用户消息即可。
    服务器会自动管理对话历史，无需客户端传递历史消息。
    """
    global llm_session_id
    
    # 构建消息列表（只包含当前用户消息，服务器端会管理历史）
    messages = [{
        "role": "user",
        "content": user_input
    }]
    
    # 调用LLM（使用会话ID以维持记忆）
    reply, success, new_session_id = call_local_llm(messages, llm_session_id)
    
    if success and reply:
        # 更新会话ID（服务器返回新的或现有的session_id）
        if new_session_id:
            llm_session_id = new_session_id
        
        # 更新本地对话历史（用于日志记录和备用）
        updated_history = conversation_history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply}
        ]
        return reply, updated_history
    else:
        return "", conversation_history

def process_with_llm(user_input):
    """
    处理ASR识别后的文本：调用LLM并播放回复
    在独立线程中运行，避免阻塞ASR响应处理。
    """
    global conversation_history
    if not user_input.strip():
        logger.debug("输入为空，跳过LLM处理")
        return

    if user_input.lower() in ['quit', 'exit', '退出', '再见']:
        send_text_to_tts("好的，再见！")
        return

    logger.info(f"用户输入: {user_input}")
    reply, conversation_history = chat_with_local_llm(user_input, conversation_history)
    
    if not reply:
        send_text_to_tts("抱歉，我没有听清楚，请再说一遍。")
        return

    logger.info(f"AI回复: {reply}")
    send_text_to_tts(reply)

def send_text_to_tts(text):
    """将文本发送到 TTS 服务并播放返回的音频"""
    if not text.strip():
        return

    try:
        log_tts_request(logger, text)
        payload = {"text": text}
        headers = {'Content-Type': 'application/json'}

        response = requests.post(TTS_ENDPOINT, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                audio_b64 = data.get("audio_base64")
                sample_rate = data.get("sample_rate", SAMPLE_RATE)
                codec = data.get("codec", "wav")
                play_audio_from_base64(audio_b64, sample_rate, codec)
            else:
                logger.error(f"TTS服务错误: {data.get('error')}")
        else:
            logger.error(f"TTS服务响应错误 ({response.status_code})")

    except requests.exceptions.RequestException as e:
        logger.error(f"TTS请求异常: {e}")
    except Exception as e:
        logger.error(f"TTS处理异常: {e}", exc_info=True)

def main_loop():
    """主循环，处理音频队列和 VAD 事件"""
    global is_speaking, speech_buffer

    logger.info("开始监听麦克风 (按 Ctrl+C 停止)")
    
    try:
        while stream.is_active():
            if not audio_queue.empty():
                chunk = audio_queue.get()

                # 防止音频反馈循环
                if is_playing_tts:
                    continue

                # VAD 检测
                speech_dict = vad_iterator(chunk, return_seconds=False)

                if speech_dict:
                    if 'start' in speech_dict:
                        log_vad_event(logger, "语音开始")
                        is_speaking = True
                        speech_buffer = []

                    if 'end' in speech_dict:
                        log_vad_event(logger, "语音结束")
                        is_speaking = False
                        if len(speech_buffer) > 0:
                            full_speech = np.concatenate(speech_buffer)
                            duration = len(full_speech) / SAMPLE_RATE
                            if duration > 0.5: # 至少 0.5 秒
                                logger.debug(f"捕获语音段 (时长: {duration:.2f}s)")
                                # 在新线程中先认证，后根据结果决定是否继续 ASR
                                threading.Thread(target=handle_captured_speech, args=(full_speech, SAMPLE_RATE), daemon=True).start()
                            else:
                                logger.debug("语音段过短，已丢弃")
                        speech_buffer = []

                # 如果正在说话，将当前块添加到缓冲区
                if is_speaking:
                    speech_buffer.append(chunk)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("停止监听...")
    finally:
        # 清理资源
        if stream:
            try:
                if not stream.is_stopped():
                    stream.stop_stream()
                stream.close()
            except:
                pass
        if playback_stream:
            try:
                if not playback_stream.is_stopped():
                    playback_stream.stop_stream()
                playback_stream.close()
            except:
                pass
        if pyaudio_instance:
            try:
                pyaudio_instance.terminate()
            except:
                pass
        logger.info("资源已释放")

def test_server_connections():
    """测试与ASR, TTS, LLM服务器的连接"""
    logger.info("测试服务器连接...")
    
    # 测试 ASR (发送一个空的Base64，期望得到错误响应)
    try:
        response = requests.post(ASR_ENDPOINT, json={"audio_base64": ""}, headers={'Content-Type': 'application/json'}, timeout=5)
        if response.status_code in [200, 400]: # 400是预期的参数错误
             logger.info(f"ASR服务正常 ({ASR_ENDPOINT})")
        else:
             logger.warning(f"ASR服务异常 ({ASR_ENDPOINT}), 状态码: {response.status_code}")
    except:
        logger.error(f"无法连接ASR服务 ({ASR_ENDPOINT})")

    # 测试 TTS (发送一个空文本，期望得到错误响应)
    try:
        response = requests.post(TTS_ENDPOINT, json={"text": ""}, headers={'Content-Type': 'application/json'}, timeout=5)
        if response.status_code in [200, 400]:
             logger.info(f"TTS服务正常 ({TTS_ENDPOINT})")
        else:
             logger.warning(f"TTS服务异常 ({TTS_ENDPOINT}), 状态码: {response.status_code}")
    except:
        logger.error(f"无法连接TTS服务 ({TTS_ENDPOINT})")

    # 测试 LLM (发送一个简单请求，使用 chat/completions 格式)
    try:
        data = {
            "messages": [
                {"role": "user", "content": "Hello, just reply 'OK' please."}
            ]
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(LLM_ENDPOINT, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            logger.info(f"LLM服务正常 ({LLM_ENDPOINT})")
        else:
            logger.warning(f"LLM服务异常 ({LLM_ENDPOINT}), 状态码: {response.status_code}")
    except:
        logger.error(f"无法连接LLM服务 ({LLM_ENDPOINT})")


def list_audio_devices():
    """列出所有可用的音频设备"""
    p = pyaudio.PyAudio()
    logger.info("可用音频设备列表:")
    default_input = p.get_default_input_device_info()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # 只显示输入设备
            is_default = " [默认]" if i == default_input['index'] else ""
            logger.info(f"  [{i}] {info['name']}{is_default}")
            logger.debug(f"      采样率: {int(info['defaultSampleRate'])} Hz, "
                  f"输入通道: {info['maxInputChannels']}")
    logger.info(f"使用默认设备: [{default_input['index']}] {default_input['name']}")
    p.terminate()
    return default_input['index']

if __name__ == "__main__":
    # 0. 显示配置信息
    logger.info("=" * 50)
    logger.info("语音监听客户端启动")
    logger.info(f"声纹认证: {'开启' if ENABLE_SPEAKER_AUTH else '关闭'}")
    logger.info(f"注册指令: '注册新用户'")
    logger.info("=" * 50)
    
    # 1. 测试服务器连接
    test_server_connections()

    # 2. 列出音频设备
    default_device_index = list_audio_devices()
    
    # 3. 加载 Silero VAD 模型
    logger.info("正在加载 Silero VAD 模型...")
    model = load_silero_vad(onnx=True)
    logger.info(f"VAD模型加载完成")
    
    # 4. 创建 VAD Iterator
    vad_iterator = VADIterator(
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS
    )

    # 5. 初始化 PyAudio 和音频流
    pyaudio_instance = pyaudio.PyAudio()
    
    # 获取设备详细信息以便确认
    device_info = pyaudio_instance.get_device_info_by_index(default_device_index)
    logger.info("正在打开麦克风设备:")
    logger.info(f"   设备索引: {default_device_index}")
    logger.info(f"   设备名称: {device_info['name']}")
    logger.debug(f"   格式: Float32, 采样率: {SAMPLE_RATE} Hz, 通道: {CHANNELS}, 块大小: {CHUNK}")
    
    try:
        stream = pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=default_device_index,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback
        )
        
        # 确保流已启动
        if not stream.is_active():
            logger.warning("音频流未激活，正在启动...")
            stream.start_stream()
        
        logger.info("音频流已成功启动")
        
    except Exception as e:
        logger.critical(f"打开音频流失败: {e}", exc_info=True)
        pyaudio_instance.terminate()
        exit(1)

    # 6. 启动主循环
    main_loop()




