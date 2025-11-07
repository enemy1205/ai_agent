#!/usr/bin/env python3
"""
HTTP Agent Server V2 - 改进版
新增功能：
1. 会话记忆管理 (Memory)
2. 增强规划能力 (Planning)
3. 工具结果反馈循环
4. 重构API减少代码重复
"""

import os
import argparse
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain_openai import OpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
import logging
import json
import threading

# 导入机器人控制工具
from robot_tools import (
    get_all_tools, get_tool_names, get_tools_info
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask应用配置
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
llm_endpoint = "http://localhost:8000/v1"
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
        logger.info(f"🛠️ 工具 {tool_name} 开始执行，输入: {safe_input}")
        self.tool_calls.append({
            'name': tool_name,
            'input': safe_input,
            'status': 'started',
            'timestamp': datetime.now().isoformat()
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具执行完成时调用"""
        # 规范化输出为字符串
        if isinstance(output, dict):
            text = output.get('text') or output.get('message') or output.get('error')
            if not isinstance(text, str):
                try:
                    text = json.dumps(output, ensure_ascii=False)
                except Exception:
                    text = str(output)
        else:
            text = str(output)
        logger.info(f"✅ 工具执行完成，返回值: {text}")
        self.tool_outputs.append(output)
        
        # 更新最后一个工具调用的状态
        if self.tool_calls:
            self.tool_calls[-1]['status'] = 'completed'
            self.tool_calls[-1]['output'] = output
            self.tool_calls[-1]['completed_at'] = datetime.now().isoformat()
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """工具执行出错时调用"""
        logger.error(f"❌ 工具执行出错: {error}")
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
    return """你是搭载在迎宾服务机器人上的AI智能体，你的名字叫Siri。任何情况都请用中文回答用户的需求。

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

【重要原则】
1. ✅ 明确识别：只在用户明确表达意图时调用工具
2. ✅ 先思考后行动：复杂任务先说明计划，再执行
3. ✅ 结果验证：观察工具返回结果，必要时重试
4. ✅ 记忆对话：利用对话历史提供连贯服务
5. ❌ 不要过度解读：用户说"你好"只需问候，不调用工具
6. ❌ 不要臆测：用户没提到地点，不要假设导航目标

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


def create_agent_with_memory(memory: ConversationBufferWindowMemory, llm_endpoint: str) -> AgentExecutor:
    """创建带记忆的 Agent Executor"""
    tools = get_all_tools()
    
    # 初始化LLM客户端
    llm = OpenAI(
        openai_api_key="EMPTY",
        openai_api_base=llm_endpoint,
        model="",
        max_tokens=2000,
        temperature=0.7,  # ← 提高温度，增加创造性和上下文理解能力
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
        verbose=False,  # 避免中间步骤污染输出
        handle_parsing_errors=True,
        max_iterations=5,  # ← 恢复到5次，记忆相关推理可能需要更多步骤
        max_execution_time=30,  # 30秒超时限制
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
            logger.info(f"📝 创建新会话: {session_id}")
        
        # 如果会话已存在，更新最后活跃时间
        if session_id in sessions:
            sessions[session_id]['last_active'] = datetime.now()
            logger.info(f"♻️ 复用现有会话: {session_id}")
            return session_id, sessions[session_id]
        
        # 创建新会话
        if len(sessions) >= MAX_SESSIONS:
            # 删除最旧的会话
            oldest_id = min(sessions.keys(), key=lambda k: sessions[k]['last_active'])
            del sessions[oldest_id]
            logger.warning(f"⚠️ 会话数达到上限，删除最旧会话: {oldest_id}")
        
        # 初始化会话记忆
        memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # 创建 Agent Executor
        agent_executor = create_agent_with_memory(memory, llm_endpoint)
        
        # 存储会话
        sessions[session_id] = {
            'memory': memory,
            'agent_executor': agent_executor,
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'request_count': 0
        }
        
        logger.info(f"✅ 新会话已创建: {session_id}")
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
        logger.info(f"🗑️ 清理过期会话: {sid}")


def _clean_agent_output(output: str) -> str:
    """清理Agent输出，移除重复的思考过程，只保留最终答案"""
    if not output:
        return "抱歉，未能生成回复。"
    
    # 如果包含 Final Answer，只保留最后一个
    if "Final Answer:" in output:
        parts = output.split("Final Answer:")
        final_answer = parts[-1].strip()
        if final_answer:
            # 取第一个非空段落
            paragraphs = [p.strip() for p in final_answer.split('\n\n') if p.strip()]
            if paragraphs:
                return paragraphs[0]
            return final_answer
    
    # 移除 Thought/Action/Observation 等调试信息
    lines = output.split('\n')
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
    
    return output.strip() or "抱歉，未能生成回复。"


def _process_agent_request(
    user_input: str,
    session_id: Optional[str] = None,
    include_planning: bool = True
) -> Dict:
    """统一的 Agent 请求处理逻辑
    
    Args:
        user_input: 用户输入内容
        session_id: 会话ID（可选）
        include_planning: 是否在响应中包含规划信息
    
    Returns:
        包含响应内容和元数据的字典
    """
    # 获取或创建会话
    session_id, session = get_or_create_session(session_id)
    agent_executor = session['agent_executor']
    session['request_count'] += 1
    
    logger.info(f"📨 处理请求 [会话: {session_id[:8]}...] [第{session['request_count']}次请求]")
    logger.info(f"💬 用户输入: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
    
    # 创建回调处理器
    callback_handler = ToolResultCallbackHandler()
    
    try:
        # 调用 Agent Executor（支持多轮迭代）
        response = agent_executor.invoke(
            {"input": user_input},
            config={"callbacks": [callback_handler]}
        )
        
        output_text = response.get('output', '未收到输出')
        intermediate_steps = response.get('intermediate_steps', [])
        
        # 清理输出，移除重复内容和调试信息
        output_text = _clean_agent_output(output_text)
        
        # 获取工具调用信息，过滤掉内部错误工具
        all_tool_calls = callback_handler.get_tool_calls()
        tool_calls = [
            call for call in all_tool_calls
            if call.get('name') not in ['_Exception', 'invalid_tool']
        ]
        
        # 构建响应元数据
        metadata = {
            'session_id': session_id,
            'request_count': session['request_count'],
            'tool_calls_count': len(tool_calls),
            'tool_calls': tool_calls if include_planning else [],
            'has_memory': True,
            'memory_messages_count': len(session['memory'].chat_memory.messages),
            'intermediate_steps_count': len(intermediate_steps)
        }
        
        logger.info(f"✅ 请求处理完成 [工具调用: {len(tool_calls)}次]")
        
        return {
            'output': output_text,
            'metadata': metadata,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Agent 执行出错: {e}", exc_info=True)
        return {
            'output': f"抱歉，处理您的请求时出现错误：{str(e)}",
            'metadata': {
                'session_id': session_id,
                'error': str(e),
                'success': False
            },
            'success': False
        }


# --- HTTP API 路由 ---

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    with sessions_lock:
        active_sessions = len(sessions)
    
    return jsonify({
        "status": "healthy",
        "message": "HTTP Agent Server V2 正在运行",
        "version": "2.0",
        "features": [
            "会话记忆管理",
            "增强规划能力",
            "工具结果反馈循环",
            "多轮迭代支持"
        ],
        "tools_available": get_tool_names(),
        "active_sessions": active_sessions,
        "max_sessions": MAX_SESSIONS
    })


@app.route('/v1/completions', methods=['POST'])
def completions():
    """
    文本补全端点（兼容 OpenAI API 格式）
    支持会话管理
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供JSON数据"}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "未提供prompt"}), 400
        
        # 获取可选的 session_id
        session_id = data.get('session_id')
        
        # 处理请求
        result = _process_agent_request(prompt, session_id)
        
        if not result['success']:
            return jsonify({
                "error": result['output'],
                "metadata": result['metadata']
            }), 500
        
        # 构建 OpenAI 兼容响应
        response = {
            "choices": [
                {
                    "text": result['output'],
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(result['output'].split()),
                "total_tokens": len(prompt.split()) + len(result['output'].split())
            },
            "model": "local-agent-v2",
            "object": "text_completion",
            "metadata": result['metadata']  # 额外的元数据
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"请求处理出错: {e}")
        return jsonify({"error": f"请求处理错误: {str(e)}"}), 500


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    聊天补全端点（兼容 OpenAI Chat API 格式）
    推荐使用此端点，支持完整的会话管理
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供JSON数据"}), 400
        
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "未提供消息"}), 400
        
        # 获取会话ID
        session_id = data.get('session_id')
        
        # 提取最新的用户消息
        # 注意：历史消息已经存储在 memory 中，这里只需要最新消息
        user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        if not user_message:
            return jsonify({"error": "未找到用户消息"}), 400
        
        # 处理请求
        result = _process_agent_request(user_message, session_id, include_planning=True)
        
        if not result['success']:
            return jsonify({
                "error": result['output'],
                "metadata": result['metadata']
            }), 500
        
        # 构建 OpenAI Chat 兼容响应
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": result['output']
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(result['output'].split()),
                "total_tokens": len(user_message.split()) + len(result['output'].split())
            },
            "model": "local-agent-v2",
            "object": "chat.completion",
            "metadata": result['metadata']  # 包含会话ID和工具调用信息
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"聊天请求处理出错: {e}")
        return jsonify({"error": f"请求处理错误: {str(e)}"}), 500


@app.route('/sessions/<session_id>', methods=['GET'])
def get_session_info(session_id):
    """获取会话信息"""
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "会话不存在"}), 404
        
        session = sessions[session_id]
        return jsonify({
            "session_id": session_id,
            "created_at": session['created_at'].isoformat(),
            "last_active": session['last_active'].isoformat(),
            "request_count": session['request_count'],
            "memory_messages_count": len(session['memory'].chat_memory.messages),
            "active": True
        })


@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """删除会话"""
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "会话不存在"}), 404
        
        del sessions[session_id]
        logger.info(f"🗑️ 手动删除会话: {session_id}")
        return jsonify({"message": "会话已删除", "session_id": session_id})


@app.route('/sessions', methods=['GET'])
def list_sessions():
    """列出所有活跃会话"""
    with sessions_lock:
        session_list = [
            {
                "session_id": sid,
                "created_at": session['created_at'].isoformat(),
                "last_active": session['last_active'].isoformat(),
                "request_count": session['request_count']
            }
            for sid, session in sessions.items()
        ]
        return jsonify({
            "sessions": session_list,
            "total": len(session_list),
            "max_sessions": MAX_SESSIONS
        })


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
        "version": "2.0",
        "features": {
            "memory": True,
            "planning": True,
            "feedback_loop": True,
            "multi_iteration": True
        },
        "base_directory": os.getcwd(),
        "available_tools": get_tool_names(),
        "active_sessions": active_sessions,
        "max_sessions": MAX_SESSIONS,
        "session_timeout_hours": SESSION_TIMEOUT.total_seconds() / 3600
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
    parser = argparse.ArgumentParser(
        description='HTTP Agent Server V2 - 带记忆和规划能力的AI Agent服务'
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
        '--llm-endpoint',
        type=str,
        default='http://localhost:8000/v1',
        help='LLM服务端点（默认: http://localhost:8000/v1）'
    )
    parser.add_argument(
        '--max-sessions',
        type=int,
        default=100,
        help='最大会话数（默认: 100）'
    )
    parser.add_argument(
        '--session-timeout',
        type=int,
        default=2,
        help='会话超时时间（小时）（默认: 2）'
    )
    parser.add_argument(
        '--memory-window',
        type=int,
        default=10,
        help='记忆窗口大小（保留最近N轮对话）（默认: 10）'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    return parser.parse_args()


def main():
    """主程序入口"""
    global llm_endpoint, MAX_SESSIONS, SESSION_TIMEOUT, MEMORY_WINDOW_SIZE
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置全局配置
    llm_endpoint = args.llm_endpoint
    MAX_SESSIONS = args.max_sessions
    SESSION_TIMEOUT = timedelta(hours=args.session_timeout)
    MEMORY_WINDOW_SIZE = args.memory_window
    
    print("=" * 70)
    print("🚀 启动 HTTP Agent Server V2")
    print("=" * 70)
    print(f"🧠 LLM端点: {llm_endpoint}")
    print(f"🔧 可用工具: {', '.join(get_tool_names())}")
    print(f"🌐 服务地址: http://{args.host}:{args.port}")
    print()
    print("📋 新功能:")
    print("  ✅ 会话记忆管理 (每个会话独立的对话历史)")
    print("  ✅ 增强规划能力 (思考-计划-执行-反馈流程)")
    print("  ✅ 工具结果反馈循环 (支持最多5轮迭代)")
    print("  ✅ 多轮对话支持 (记住最近10轮对话)")
    print()
    print("📋 可用端点:")
    print("  - GET  /health - 健康检查")
    print("  - POST /v1/completions - 文本补全（支持会话）")
    print("  - POST /v1/chat/completions - 聊天补全（推荐）")
    print("  - GET  /sessions - 列出所有会话")
    print("  - GET  /sessions/<id> - 获取会话信息")
    print("  - DELETE /sessions/<id> - 删除会话")
    print("  - GET  /tools - 列出可用工具")
    print("  - GET  /status - 服务状态")
    print()
    print(f"⚙️  配置:")
    print(f"  - 最大会话数: {MAX_SESSIONS}")
    print(f"  - 会话超时: {args.session_timeout} 小时")
    print(f"  - 记忆窗口: {MEMORY_WINDOW_SIZE} 轮对话")
    print("=" * 70)
    print("\n按 Ctrl+C 停止服务\n")
    
    # 启动Flask应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    main()

