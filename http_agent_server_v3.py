#!/usr/bin/env python3
"""
HTTP Agent Server V3 - 集成腾讯混元大模型
将交互式AI agent改造为HTTP服务端，支持客户端通过HTTP请求调用本地自建工具
使用腾讯混元大模型作为LLM服务
"""

import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
import json
import uuid
import threading

# 导入机器人控制工具
from robot_tools import (
    get_all_tools, get_tool_names, get_tools_info
)

# === 导入统一日志配置 ===
from logger_config import (
    create_server_logger,
    set_request_id,
    log_request_start,
    log_request_end,
    log_tool_call
)

# 创建logger实例（服务器端，包含request_id）
logger = create_server_logger("http_agent_server_v3", level=os.getenv("LOG_LEVEL", "INFO"))

# Flask应用配置
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 添加响应后处理，确保连接正确关闭
@app.after_request
def after_request(response):
    """确保响应后连接正确关闭，避免 CLOSE_WAIT 连接积累"""
    # 对于长时间运行的连接，设置关闭头
    # 注意：对于需要 keepalive 的场景，可以条件性设置
    if request.path.startswith('/v1/chat/completions'):
        # LLM 请求可能较长，响应后关闭连接
        response.headers['Connection'] = 'close'
    return response

# ================== 腾讯混元配置 ==================
# 从环境变量获取腾讯混元 API Key
HUNYUAN_API_KEY = os.getenv("HUNYUAN_API_KEY")
HUNYUAN_BASE_URL = os.getenv("HUNYUAN_BASE_URL", "https://api.hunyuan.cloud.tencent.com/v1")
HUNYUAN_MODEL = os.getenv("HUNYUAN_MODEL", "hunyuan-turbos-latest")
# ============================================

# 会话管理相关全局变量
sessions_lock = threading.Lock()
sessions: Dict[str, Dict] = {}  # {session_id: {memory, agent_executor, last_active}}

# 会话配置
SESSION_TIMEOUT = timedelta(hours=2)  # 会话超时时间
MAX_SESSIONS = 100  # 最大会话数
MEMORY_WINDOW_SIZE = 10  # 保留最近10轮对话

class ToolResultCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器，用于捕获工具执行结果"""
    
    def __init__(self):
        super().__init__()
        self.tool_outputs = []  # 存储所有工具的返回值
        self.tool_calls = []    # 存储工具调用信息
    
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """工具开始执行时调用"""
        tool_name = serialized.get('name', 'unknown')
        try:
            safe_input = (
                input_str if isinstance(input_str, str)
                else json.dumps(input_str, ensure_ascii=False)
            )
        except Exception:
            safe_input = str(input_str)
        # 精简日志：只显示工具名称和输入参数
        logger.info(f"调用工具: {tool_name}({safe_input[:100]}{'...' if len(safe_input) > 100 else ''})")
        self.tool_calls.append({
            'name': tool_name,
            'input': safe_input,
            'status': 'started'
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具执行完成时调用"""
        # 规范化输出为字符串
        if isinstance(output, dict):
            text = output.get('message') or output.get('error') or output.get('text', '')
            if not isinstance(text, str):
                try:
                    text = json.dumps(output, ensure_ascii=False)
                except Exception:
                    text = str(output)
        else:
            text = str(output)
        
        if self.tool_calls:
            tool_name = self.tool_calls[-1]['name']
            # 提取关键信息（成功/失败）
            if isinstance(output, dict):
                result_msg = output.get('text', '')[:80] if output.get('text') else '完成'
                logger.info(f"工具执行完成: {tool_name} - {result_msg}")
            else:
                logger.info(f"工具执行完成: {tool_name} - {text[:80]}{'...' if len(text) > 80 else ''}")
        
        self.tool_outputs.append(text)
        
        # 更新最后一个工具调用的状态
        if self.tool_calls:
            self.tool_calls[-1]['status'] = 'completed'
            self.tool_calls[-1]['output'] = output
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """工具执行出错时调用"""
        logger.error(f"工具执行出错: {error}", exc_info=True)
        if self.tool_calls:
            self.tool_calls[-1]['status'] = 'error'
            self.tool_calls[-1]['error'] = str(error)
    
    def get_tool_outputs(self):
        """获取所有工具的输出"""
        return self.tool_outputs
    
    def get_tool_calls(self):
        """获取所有工具调用信息"""
        return self.tool_calls
    
    def clear(self):
        """清空存储的结果"""
        self.tool_outputs.clear()
        self.tool_calls.clear()


def create_enhanced_prompt() -> str:
    """创建增强的提示词，包含记忆使用说明"""
    return """你是搭载在迎宾服务机器人上的AI智能体，你的名字叫Siri。任何情况都请用中文回答用户的需求。你可以通过调用相应的工具函数来控制机器人的导航和机械臂/夹爪操作。

【核心能力】
1. 理解用户意图并制定执行计划
2. 调用工具控制机器人的导航和机械臂/夹爪操作
3. 根据执行结果动态调整策略
4. **记住对话历史，提供连贯的交互体验**

【重要！对话记忆使用规则】
在回答每个问题前，必须：
1. 仔细查看对话历史（chat_history），了解之前的所有对话内容
2. 识别用户在对话中提到的个人信息（姓名、偏好、身份等）
3. 如果用户提到"刚才"、"之前"、"上次"等词，必须引用对话历史
4. 保持对话的连贯性，基于历史上下文回答

【工作流程】
对于每个用户请求，请按以下步骤处理：

Step 0 - 【查看对话历史】！
- 检查对话历史中是否有用户的个人信息
- 查看之前的对话主题和上下文
- 识别对话中的指代关系

Step 1 - 【理解意图】
- 结合对话历史分析用户的真实需求
- 识别关键词和动作指令
- 判断是否需要调用工具

Step 2 - 【制定计划】（复杂任务时）
- 如果任务需要多个步骤，先列出执行计划
- 确定工具调用顺序
- 预测可能的问题

Step 3 - 【执行操作】
- 按计划调用相应工具
- 观察工具返回结果
- 如果失败，分析原因

Step 4 - 【反馈调整】
- 根据执行结果判断是否成功
- 如果需要，调整计划并重试
- 向用户反馈执行状态

【可用工具及调用条件】

导航工具 - 用户明确表达"去"、"到"、"导航"、"前往"等移动意图时使用
- go_to_office: 去办公室（关键词：办公室、office）
- go_to_restroom: 去休息室（关键词：休息室、restroom）  
- go_to_corridor: 去走廊（关键词：走廊、corridor）

机械臂工具(arm_control) - 用户明确表达"拿起"、"放下"、"机械臂"等操作意图时使用
- 参数: command (0=归位, 1=夹取, 2=释放, 3=搬运)

夹爪工具(gripper_control) - 用户明确表达"夹爪"、"夹"、"抓"等动作时使用
- 参数: command (1=夹紧, 2=松开)

复合任务工具 - 用户同时提出导航+操作需求时使用
- complex_task: 先导航再执行机械臂动作
- get_water_bottle: 拿水瓶的完整自动化流程

视觉抓取工具 - 用户明确提出“拿起/夹取/抓取某个具体物体”且需要视觉定位时使用
- vision_detect_and_grasp: 仅发送视觉识别+抓取姿态估计请求，参数 object_name（需用英文描述，如 water bottle、banana），不负责移动机械臂/夹爪以外的动作
- 禁止：用户只要求机械臂/夹爪动作或已给出明确坐标时不要调用；不要与导航同一步混用

【重要原则】
1. ✅ 明确识别：只在用户明确表达意图时调用工具
2. ✅ 先思考后行动：复杂任务先说明计划，再执行
3. ✅ 结果验证：观察工具返回结果，必要时重试
4. ✅ 记忆对话：利用对话历史提供连贯服务
5. ❌ 不要过度解读：用户说"你好"只需问候，不调用工具
6. ❌ 不要臆测：用户没提到地点，不要假设导航目标；未提及需要视觉定位时不要擅自调用视觉抓取工具

【正确示例】

示例1 - 记忆用户信息：
用户："你好，我是小明"
思考：用户在自我介绍，需要记住这个信息
回复：你好，小明！我是Siri，迎宾服务机器人。有什么可以帮您的吗？

用户："我叫什么名字？"
思考：查看对话历史 → 发现用户之前说"我是小明"
回复：您叫小明。

示例2 - 工具调用：
用户："去办公室拿水瓶"
思考：这是一个复杂任务，需要导航+机械臂操作
计划：
  1. 导航到办公室
  2. 机械臂移动到水瓶位置
  3. 夹爪夹取
  4. 机械臂抬升
执行：调用 get_water_bottle()
反馈：根据返回结果告知用户

示例3 - 简单问候：
用户："你好"
思考：这是问候，不需要调用工具
回复：你好！我是Siri，迎宾服务机器人。有什么可以帮您的吗？

示例4 - 上下文理解：
用户："去办公室"
回复：好的，正在为您导航到办公室...

用户："到了吗？"
思考：查看对话历史 → 用户之前要求去办公室，现在询问是否到达
回复：根据上次导航任务的状态回答

严格遵循上述流程，特别是要查看对话历史！"""


def create_agent_with_memory(memory: ConversationBufferWindowMemory) -> AgentExecutor:
    """创建带记忆的 Agent Executor"""
    tools = get_all_tools()

    # 检查API Key配置
    if not HUNYUAN_API_KEY:
        logger.warning("警告: 未设置环境变量 HUNYUAN_API_KEY，将使用空字符串（可能导致认证失败）")
    
    # 初始化腾讯混元LLM客户端（兼容OpenAI接口）
    # 使用 ChatOpenAI 而不是 OpenAI，因为腾讯混元主要支持 /v1/chat/completions 端点
    llm = ChatOpenAI(
        openai_api_key=HUNYUAN_API_KEY or "EMPTY",
        openai_api_base=HUNYUAN_BASE_URL,
        model_name=HUNYUAN_MODEL,
        max_tokens=2000,
        temperature=0.2,
        top_p=0.95,
        default_headers={"Content-Type": "application/json"},
        request_timeout=120,
    )

    # 创建增强的提示词，明确包含记忆占位符说明
    enhanced_prompt = create_enhanced_prompt()
    
    # 添加记忆相关的提示
    memory_suffix = """

当前对话历史：
{chat_history}

当前用户输入：
{input}

请基于上述对话历史和当前输入，给出你的回答。
{agent_scratchpad}"""
    
    # 创建 agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,  # 添加记忆
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=30,
        early_stopping_method="generate",
        agent_kwargs={
            "prefix": enhanced_prompt,
            "suffix": memory_suffix,
            "input_variables": ["input", "chat_history", "agent_scratchpad"]
        }
    )
    
    return agent


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Dict]:
    """获取或创建会话"""
    with sessions_lock:
        # 清理过期会话
        _cleanup_expired_sessions()
        
        # 如果没有提供 session_id，创建新会话
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # 如果会话已存在，更新最后活跃时间
        if session_id in sessions:
            sessions[session_id]['last_active'] = datetime.now()
            return session_id, sessions[session_id]
        
        # 创建新会话
        if len(sessions) >= MAX_SESSIONS:
            # 删除最旧的会话
            oldest_id = min(sessions.keys(), key=lambda k: sessions[k]['last_active'])
            del sessions[oldest_id]
        
        # 初始化会话记忆
        memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # 创建 Agent Executor
        agent_executor = create_agent_with_memory(memory)
        
        # 存储会话
        sessions[session_id] = {
            'memory': memory,
            'agent_executor': agent_executor,
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'request_count': 0
        }
        
        return session_id, sessions[session_id]


def _cleanup_expired_sessions():
    """清理过期会话（内部使用，需要持有锁）"""
    now = datetime.now()
    expired = [
        sid for sid, session in sessions.items()
        if now - session['last_active'] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del sessions[sid]



def _clean_agent_output(output) -> str:
    """清理Agent输出，处理字典格式和提取最终答案"""
    if not output:
        return "抱歉，未能生成回复。"
    
    # 如果是字典格式，提取 action_input
    if isinstance(output, dict):
        if 'action_input' in output:
            return str(output['action_input'])
        # 尝试其他可能的键
        for key in ['text', 'content', 'message', 'output']:
            if key in output and isinstance(output[key], str):
                return output[key]
        # 如果都不存在，转换为JSON字符串
        return json.dumps(output, ensure_ascii=False)
    
    # 如果是字符串，处理 "Final Answer:" 格式
    output_str = str(output) if not isinstance(output, str) else output
    
    # 如果包含 Final Answer，提取最后一部分
    if "Final Answer:" in output_str:
        parts = output_str.split("Final Answer:")
        final_answer = parts[-1].strip()
        if final_answer:
            # 取第一个非空段落
            paragraphs = [p.strip() for p in final_answer.split('\n\n') if p.strip()]
            if paragraphs:
                return paragraphs[0]
            return final_answer
    
    # 移除 Thought/Action/Observation 等调试信息
    lines = output_str.split('\n')
    clean_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        line = line.strip()
        # 跳过调试标记
        if line.startswith(('Thought:', 'Action:', 'Observation:', 'Action Input:')):
            skip_next = True
            continue
        if skip_next and not line:
            skip_next = False
            continue
        if line and not line.startswith('【'):
            clean_lines.append(line)
    
    if clean_lines:
        result = '\n'.join(clean_lines)
        # 如果结果太长，只取第一段
        if len(result) > 500:
            paragraphs = result.split('\n\n')
            if paragraphs:
                return paragraphs[0]
        return result
    
    return output_str.strip() or "抱歉，未能生成回复。"


def _post_process_response(original_prompt, agent_output, tool_outputs):
    """清理并组合LLM输出和工具结果的text部分"""
    # 先清理agent输出（处理字典格式等）
    cleaned_output = _clean_agent_output(agent_output)
    
    # 提取工具结果的text字段
    tool_texts = []
    for tool_output in tool_outputs:
        if isinstance(tool_output, dict) and 'text' in tool_output:
            text = tool_output['text']
            if text and isinstance(text, str):
                tool_texts.append(text)
    
    # 组合LLM输出和工具结果
    if tool_texts:
        final_text = f"{cleaned_output}\n\n" + "\n".join(tool_texts)
    else:
        final_text = cleaned_output
    
    return final_text

# --- HTTP API 路由 ---

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "message": "HTTP Agent Server V3 (腾讯混元) 正在运行",
        "tools_available": get_tool_names(),
        "llm_provider": "腾讯混元",
        "llm_model": HUNYUAN_MODEL
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    聊天completions端点，支持对话格式和会话记忆
    """
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    
    try:
        data = request.get_json()
        if not data:
            logger.warning("未提供JSON数据")
            return jsonify({"error": "未提供JSON数据"}), 400
        
        messages = data.get('messages', [])
        if not messages:
            logger.warning("未提供消息")
            return jsonify({"error": "未提供消息"}), 400
        
        # 获取会话ID（如果提供）
        session_id = data.get('session_id')
        
        # 提取最后一条用户消息
        user_message = None
        for message in reversed(messages):
            if message.get('role') == 'user':
                user_message = message.get('content', '')
                break
        
        if not user_message:
            logger.warning("未找到用户消息")
            return jsonify({"error": "未找到用户消息"}), 400
        
        try:
            # 获取或创建会话（带记忆）
            session_id, session = get_or_create_session(session_id)
            agent_executor = session['agent_executor']
            session['request_count'] += 1
            
            # 精简日志：只显示用户消息内容
            logger.info(f"用户消息: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
            
            # 创建回调处理器
            callback_handler = ToolResultCallbackHandler()
            
            # 使用带记忆的 agent executor 调用
            response = agent_executor.invoke(
                {"input": user_message},
                config={"callbacks": [callback_handler]}
            )
            
            # 获取输出，如果是字典格式则提取文本内容
            raw_output = response.get('output', '未收到输出')
            # 如果输出是字典格式（如 {"action": "Final Answer", "action_input": "..."}），直接提取 action_input
            if isinstance(raw_output, dict):
                output_text = raw_output.get('action_input', raw_output.get('text', str(raw_output)))
            else:
                output_text = raw_output
            
            # 从回调处理器获取工具执行结果
            tool_outputs = callback_handler.get_tool_outputs()
            
            # 统一进行后处理，无论是否有工具调用
            final_text = _post_process_response(user_message, output_text, tool_outputs)
            
            # 获取记忆统计
            memory_messages_count = len(session['memory'].chat_memory.messages)
            tool_calls = callback_handler.get_tool_calls()
            tool_calls_count = len([c for c in tool_calls if c.get('status') != 'error'])
            
            result = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": final_text
                        },
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(final_text.split()),
                    "total_tokens": len(user_message.split()) + len(final_text.split())
                },
                "model": HUNYUAN_MODEL,
                "object": "chat.completion",
                "metadata": {
                    "session_id": session_id,
                    "memory_messages_count": memory_messages_count,
                    "tool_calls_count": tool_calls_count,
                    "request_count": session['request_count']
                }
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"❌ Agent处理出错: {e}")
            return jsonify({
                "error": f"Agent处理错误: {str(e)}",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"抱歉，处理您的消息时出现错误：{str(e)}"
                        },
                        "index": 0,
                        "finish_reason": "error"
                    }
                ]
            }), 500
            
    except Exception as e:
        logger.error(f"❌ 请求处理出错: {e}")
        return jsonify({"error": f"请求处理错误: {str(e)}"}), 500

@app.route('/tools', methods=['GET'])
def list_tools():
    """列出可用的工具"""
    return jsonify({
        "tools": get_tools_info(),
        "count": len(get_tools_info())
    })

@app.route('/status', methods=['GET'])
def status():
    """服务状态信息"""
    with sessions_lock:
        active_sessions = len(sessions)
    
    return jsonify({
        "status": "running",
        "base_directory": os.getcwd(),
        "available_tools": get_tool_names(),
        "llm_provider": "腾讯混元",
        "llm_model": HUNYUAN_MODEL,
        "llm_base_url": HUNYUAN_BASE_URL,
        "api_key_configured": bool(HUNYUAN_API_KEY),
        "features": [
            "会话记忆管理",
            "多轮对话支持",
            "工具调用追踪"
        ],
        "session_stats": {
            "active_sessions": active_sessions,
            "max_sessions": MAX_SESSIONS,
            "memory_window_size": MEMORY_WINDOW_SIZE,
            "session_timeout_hours": SESSION_TIMEOUT.total_seconds() / 3600
        }
    })

# --- 错误处理 ---

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "端点未找到"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "内部服务器错误"}), 500

# --- 主程序 ---

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HTTP Agent Server V3 - 集成腾讯混元大模型的AI Agent HTTP服务')
    parser.add_argument(
        '--base-dir', 
        type=str, 
        default=None,
        help='指定工作目录路径（默认为当前目录）'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务监听主机（默认: 0.0.0.0）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='服务监听端口（默认: 5000）'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    return parser.parse_args()


def main():
    """主程序入口"""
    
    # 解析命令行参数
    args = parse_arguments()

    # 检查必要的环境变量
    if not HUNYUAN_API_KEY:
        logger.warning("=" * 60)
        logger.warning("警告: 未设置环境变量 HUNYUAN_API_KEY")
        logger.warning("请设置环境变量: export HUNYUAN_API_KEY='your_api_key'")
        logger.warning("或在控制台创建 API KEY: https://console.cloud.tencent.com/hunyuan/apiKey")
        logger.warning("=" * 60)
    
    logger.info("=" * 60)
    logger.info("启动HTTP Agent Server V3")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info(f"LLM模型: {HUNYUAN_MODEL}")
    logger.info("=" * 60)
    
    # 启动Flask应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == "__main__":
    main()

