import os
import pyaudio
import numpy as np
import queue
import threading
import time
import base64
import requests
import json
from io import BytesIO
from scipy.io import wavfile

# --- Silero VAD 相关 ---
try:
    from silero_vad import load_silero_vad, VADIterator
except ImportError:
    print("错误: 未找到 silero-vad 库。请运行 'pip install silero-vad' 安装。")
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
VOICE_SERVER_IP = "202.38.209.227"
VOICE_SERVER_PORT = 4999
VOICE_SERVER_BASE_URL = f"http://{VOICE_SERVER_IP}:{VOICE_SERVER_PORT}"
ASR_ENDPOINT = f"{VOICE_SERVER_BASE_URL}/asr/recognize"
TTS_ENDPOINT = f"{VOICE_SERVER_BASE_URL}/tts/synthesize"
SPEAKER_VERIFY_ENDPOINT = f"{VOICE_SERVER_BASE_URL}/speaker/verify"

# 本地LLM服务配置
LLM_SERVER_IP = "202.38.209.227" # <-- 修改为你的大模型服务IP
LLM_SERVER_PORT = 5000           # <-- 修改为你的大模型服务端口
LLM_API_BASE = f"http://{LLM_SERVER_IP}:{LLM_SERVER_PORT}/v1"
LLM_ENDPOINT = f"{LLM_API_BASE}/completions"

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

# --- 对话历史 ---
conversation_history = []

# --- 声纹认证控制 ---
ENABLE_SPEAKER_AUTH = os.getenv("ENABLE_SPEAKER_AUTH", "true").lower() == "true"  # 声纹认证开关
is_register_mode = False  # 注册模式标志
pending_register_id = None  # 待注册的用户ID

def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio 回调函数，将录音数据放入队列"""
    audio_chunk = np.frombuffer(in_data, dtype=np.float32)
    audio_queue.put(audio_chunk)
    return (in_data, pyaudio.paContinue)

def play_audio_from_base64(audio_base64_str, sample_rate=16000, codec="wav"):
    """播放 Base64 编码的音频数据"""
    global pyaudio_instance, playback_stream, is_playing_tts
    try:
        if not audio_base64_str:
            print("! TTS 返回的音频数据为空")
            return

        print("-> 开始播放 TTS 音频，暂停麦克风监听...")
        is_playing_tts = True

        # 1. 解码 Base64
        print("-> 正在解码 TTS 返回的 Base64 音频数据...")
        audio_bytes = base64.b64decode(audio_base64_str)
        print(f"-> 解码完成，音频数据大小: {len(audio_bytes)} 字节")

        # 2. 确定播放参数 (简化处理)
        if codec.lower() in ["wav", "pcm"]:
            audio_format = pyaudio.paInt16
            width = 2
        elif codec.lower() == "mp3":
            print("⚠️  注意: 客户端直接播放 MP3 需要额外解码库 (如 pydub)。这里假设数据是 PCM。")
            audio_format = pyaudio.paInt16
            width = 2
        else:
            print(f"! 不支持的音频格式用于播放: {codec}")
            is_playing_tts = False
            return

        # 3. 打开播放流 (如果尚未打开或已关闭)
        if not playback_stream or playback_stream.is_stopped():
            if not pyaudio_instance:
                 print("! PyAudio 实例未初始化，无法播放音频。")
                 is_playing_tts = False
                 return
            playback_stream = pyaudio_instance.open(
                format=audio_format,
                channels=1,
                rate=sample_rate,
                output=True
            )
            print("-> 已打开音频播放流。")

        # 4. 播放音频数据
        print("🔊 开始播放 TTS 音频...")
        playback_stream.write(audio_bytes)
        print("✅ TTS 音频播放完毕。")

    except Exception as e:
        print(f"! 播放音频时出错: {e}")
    finally:
        print("-> TTS 播放结束，恢复麦克风监听。")
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

        print(f"-> 正在发送 {len(audio_data_float32) / sample_rate:.2f} 秒的语音数据到 ASR (Base64 长度: {len(audio_base64)})...")
        response = requests.post(ASR_ENDPOINT, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            print(f"<- ASR 服务器响应: {data}")
            if data.get("success"):
                recognized_text = data.get("result", "").strip()
                if recognized_text:
                    print(f"🗣️ 识别结果: {recognized_text}")
                    # 识别成功，将文本发送给 LLM
                    threading.Thread(target=process_with_llm, args=(recognized_text,), daemon=True).start()
                else:
                    print("🗣️ 识别结果为空。")
            else:
                print(f"! ASR 识别失败: {data.get('error')}")
        else:
            print(f"! ASR 服务器响应错误 ({response.status_code}): {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"! 发送 ASR 请求时出错: {e}")
    except Exception as e:
        print(f"! 处理或编码 ASR 音频时出错: {e}")

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

        print(f"-> 正在发送 {len(audio_data_float32) / sample_rate:.2f} 秒的语音数据到 ASR...")
        response = requests.post(ASR_ENDPOINT, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                recognized_text = data.get("result", "").strip()
                return recognized_text
            else:
                print(f"! ASR 识别失败: {data.get('error')}")
                return None
        else:
            print(f"! ASR 服务器响应错误 ({response.status_code}): {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"! 发送 ASR 请求时出错: {e}")
        return None
    except Exception as e:
        print(f"! 处理或编码 ASR 音频时出错: {e}")
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
        print(f"! 编码 WAV(Base64) 失败: {e}")
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
        print("-> 正在进行说话人认证...")
        resp = requests.post(SPEAKER_VERIFY_ENDPOINT, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            print(f"<- 说话人认证响应: {data}")
            if data.get("success"):
                is_registered = bool(data.get("registered"))
                name = data.get("id")
                confidence = float(data.get("confidence", 0.0))
                return is_registered, name, confidence
            else:
                print(f"! 说话人认证失败: {data.get('error')}")
                return False, None, 0.0
        else:
            print(f"! 说话人认证服务响应错误 ({resp.status_code}): {resp.text}")
            return False, None, 0.0
    except requests.exceptions.RequestException as e:
        print(f"! 说话人认证请求出错: {e}")
        return False, None, 0.0
    except Exception as e:
        print(f"! 处理说话人认证时出错: {e}")
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
        
        print(f"-> 正在注册用户 {user_id} 的声纹...")
        response = requests.post(f"{VOICE_SERVER_BASE_URL}/speaker/register", 
                               json=payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ 用户 {user_id} 声纹注册成功")
                return True, f"用户 {user_id} 注册成功"
            else:
                error_msg = data.get("error", "注册失败")
                print(f"❌ 声纹注册失败: {error_msg}")
                return False, error_msg
        else:
            error_msg = f"注册服务响应错误 ({response.status_code}): {response.text}"
            print(f"❌ {error_msg}")
            return False, error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"注册请求出错: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"处理注册时出错: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg

def handle_captured_speech(audio_data_float32, sample_rate):
    """新的语音处理流程：先ASR，根据结果决定后续处理"""
    global is_register_mode, pending_register_id
    
    # 1. 先进行ASR识别
    print("🎤 正在进行语音识别...")
    recognized_text = send_audio_to_asr_server_and_get_text(audio_data_float32, sample_rate)
    
    if not recognized_text:
        print("❌ ASR识别失败或结果为空")
        return
    
    print(f"🗣️ 识别结果: {recognized_text}")
    
    # 2. 检查是否包含"注册新用户"指令
    if "注册新用户" in recognized_text:
        print("📝 检测到注册新用户指令，进入注册模式")
        is_register_mode = True
        pending_register_id = f"user_{int(time.time())}"  # 生成临时用户ID
        send_text_to_tts(f"请说一段话用于注册，您的用户ID是 {pending_register_id}")
        return
    
    # 3. 如果当前在注册模式，进行声纹注册
    if is_register_mode and pending_register_id:
        print(f"🔐 正在注册用户 {pending_register_id} 的声纹...")
        success, message = register_speaker(audio_data_float32, sample_rate, pending_register_id)
        send_text_to_tts(message)
        is_register_mode = False
        pending_register_id = None
        return
    
    # 4. 正常模式：检查声纹认证开关
    if not ENABLE_SPEAKER_AUTH:
        print("🔓 声纹认证已关闭，直接进行LLM处理")
        process_with_llm(recognized_text)
        return
    
    # 5. 进行声纹认证
    is_ok, name, conf = verify_speaker_before_asr(audio_data_float32, sample_rate)
    if not is_ok:
        print("🔒 未注册用户，拒绝后续处理")
        send_text_to_tts("用户尚未注册")
        return
    
    print(f"✅ 认证通过: id={name}, confidence={conf:.2f}，开始LLM处理...")
    process_with_llm(recognized_text)

def call_local_llm(prompt):
    """
    调用本地部署的大模型服务
    """
    try:
        # 构建请求数据 - 不传参数，让服务器端使用自己的配置
        data = {
            "prompt": prompt,
            "stop": ["\n\n", "Human:", "Assistant:"]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print("🧠 正在调用本地LLM服务...")
        response = requests.post(
            LLM_ENDPOINT,
            headers=headers,
            json=data,
            timeout=120  # 120秒超时
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                reply_text = result["choices"][0]["text"].strip()
                print("✅ LLM服务调用成功")
                return reply_text, True
            else:
                print("❌ LLM服务返回格式异常")
                return "", False
        else:
            print(f"❌ LLM服务调用失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return "", False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到LLM服务 ({LLM_ENDPOINT})")
        print("请确保LLM服务正在运行")
        return "", False
    except requests.exceptions.Timeout:
        print("❌ LLM服务调用超时")
        return "", False
    except Exception as e:
        print(f"❌ LLM服务调用出错: {e}")
        return "", False

def chat_with_local_llm(user_input, conversation_history):
    """
    与本地LLM进行对话，支持上下文历史
    """
    # 构建完整的提示词 (可加入 SYSTEM_PROMPT)
    # full_prompt = f"{SYSTEM_PROMPT}\n\n"
    full_prompt = ""
    
    # 添加对话历史 (只保留最近几轮)
    for msg in conversation_history[-6:]: # 例如只保留最近3轮对话
        if msg["role"] == "user":
            full_prompt += f"Human: {msg['content']}\n"
        elif msg["role"] == "assistant":
            full_prompt += f"Assistant: {msg['content']}\n"
    
    # 添加当前用户输入
    full_prompt += f"Human: {user_input}\nAssistant:"
    
    # 调用LLM
    reply, success = call_local_llm(full_prompt)
    
    if success and reply:
        # 更新对话历史
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
        print("-> ASR 结果为空，跳过LLM处理。")
        return

    if user_input.lower() in ['quit', 'exit', '退出', '再见']:
        print("👋 用户请求退出!")
        # 可以在这里添加退出逻辑，例如设置一个退出标志
        # 为了简化，我们只打印信息
        # 你可以设置一个全局标志 `should_exit = True` 并在 main_loop 中检查
        goodbye_text = "好的，再见！"
        # 直接调用TTS播放告别语
        send_text_to_tts(goodbye_text)
        return

    print("🧠 正在思考...")
    reply, conversation_history = chat_with_local_llm(user_input, conversation_history)
    
    if not reply:
        print("❌ 模型未能生成回复。")
        error_reply = "抱歉，我没有听清楚，请再说一遍。"
        send_text_to_tts(error_reply)
        return

    print(f"🤖 模型回复: {reply}")
    # 将LLM的回复发送给TTS服务
    send_text_to_tts(reply)

def send_text_to_tts(text):
    """将文本发送到 TTS 服务并播放返回的音频"""
    if not text.strip():
        print("-> TTS 文本为空，跳过合成。")
        return

    try:
        payload = {
            "text": text,
            # 可以添加其他 TTS 参数
        }
        headers = {'Content-Type': 'application/json'}

        print(f"-> 正在向 TTS 服务发送文本: '{text}'")
        response = requests.post(TTS_ENDPOINT, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                audio_b64 = data.get("audio_base64")
                sample_rate = data.get("sample_rate", SAMPLE_RATE)
                codec = data.get("codec", "wav")
                print(f"<- TTS 服务响应成功。")
                # 播放音频
                play_audio_from_base64(audio_b64, sample_rate, codec)
            else:
                print(f"<- TTS 服务返回错误: {data.get('error')}")
        else:
            print(f"<- TTS 服务响应错误 ({response.status_code}): {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"! 发送 TTS 请求时出错: {e}")
    except Exception as e:
        print(f"! 处理 TTS 响应时出错: {e}")

def main_loop():
    """主循环，处理音频队列和 VAD 事件"""
    global is_speaking, speech_buffer

    print("开始监听麦克风... (按 Ctrl+C 停止)")
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
                        print(">>> 检测到说话开始")
                        is_speaking = True
                        speech_buffer = []

                    if 'end' in speech_dict:
                        print("<<< 检测到说话结束")
                        is_speaking = False
                        if len(speech_buffer) > 0:
                            full_speech = np.concatenate(speech_buffer)
                            duration = len(full_speech) / SAMPLE_RATE
                            if duration > 0.5: # 至少 0.5 秒
                                print(f"  -> 捕获到一段语音，时长: {duration:.2f} 秒")
                                # 在新线程中先认证，后根据结果决定是否继续 ASR
                                threading.Thread(target=handle_captured_speech, args=(full_speech, SAMPLE_RATE), daemon=True).start()
                            else:
                                print("  -> 语音段太短，已丢弃")
                        speech_buffer = []

                # 如果正在说话，将当前块添加到缓冲区
                if is_speaking:
                    speech_buffer.append(chunk)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n停止监听...")
    finally:
        # 清理资源
        if stream:
            stream.stop_stream()
            stream.close()
        if playback_stream:
            playback_stream.stop_stream()
            playback_stream.close()
        if pyaudio_instance:
            pyaudio_instance.terminate()
        print("资源已释放。")

def test_server_connections():
    """测试与ASR, TTS, LLM服务器的连接"""
    print("🔍 正在测试服务器连接...")
    
    # 测试 ASR (发送一个空的Base64，期望得到错误响应)
    try:
        response = requests.post(ASR_ENDPOINT, json={"audio_base64": ""}, headers={'Content-Type': 'application/json'}, timeout=5)
        if response.status_code == 200 or response.status_code == 400: # 400是预期的参数错误
             print(f"✅ ASR 服务连接正常 ({ASR_ENDPOINT})")
        else:
             print(f"❌ ASR 服务连接异常 ({ASR_ENDPOINT}), 状态码: {response.status_code}")
    except:
        print(f"❌ 无法连接到 ASR 服务 ({ASR_ENDPOINT})")

    # 测试 TTS (发送一个空文本，期望得到错误响应)
    try:
        response = requests.post(TTS_ENDPOINT, json={"text": ""}, headers={'Content-Type': 'application/json'}, timeout=5)
        if response.status_code == 200 or response.status_code == 400:
             print(f"✅ TTS 服务连接正常 ({TTS_ENDPOINT})")
        else:
             print(f"❌ TTS 服务连接异常 ({TTS_ENDPOINT}), 状态码: {response.status_code}")
    except:
        print(f"❌ 无法连接到 TTS 服务 ({TTS_ENDPOINT})")

    # 测试 LLM (发送一个简单请求)
    try:
        data = {
            "prompt": "Hello, just reply 'OK' please.",
            "max_tokens": 10,
            "temperature": 0.7,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(LLM_ENDPOINT, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            print(f"✅ LLM 服务连接正常 ({LLM_ENDPOINT})")
        else:
            print(f"❌ LLM 服务连接异常 ({LLM_ENDPOINT}), 状态码: {response.status_code}")
    except:
        print(f"❌ 无法连接到 LLM 服务 ({LLM_ENDPOINT})")


if __name__ == "__main__":
    # 0. 显示配置信息
    print("=" * 50)
    print("🎤 语音监听客户端启动")
    print(f"🔐 声纹认证: {'开启' if ENABLE_SPEAKER_AUTH else '关闭'}")
    print(f"🎯 注册指令: '注册新用户'")
    print("=" * 50)
    
    # 1. 测试服务器连接
    test_server_connections()

    # 2. 加载 Silero VAD 模型
    print("正在加载 Silero VAD 模型...")
    model = load_silero_vad(onnx=True)
    print(f"模型加载完成: {type(model)}")
    
    # 3. 创建 VAD Iterator
    vad_iterator = VADIterator(
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS
    )

    # 4. 初始化 PyAudio 和音频流
    pyaudio_instance = pyaudio.PyAudio()
    stream = pyaudio_instance.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    # 5. 启动主循环
    main_loop()




