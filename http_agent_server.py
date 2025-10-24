#!/usr/bin/env python3
"""
HTTP Agent Server
将交互式AI agent改造为HTTP服务端，支持客户端通过HTTP请求调用本地自建工具
"""

import os
import argparse
from pathlib import Path
from typing import Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI
from langchain_core.callbacks import BaseCallbackHandler
import logging
import json

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

# 全局变量存储agent实例和配置
agent = None
llm_endpoint = "http://localhost:8000/v1"

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
            'status': 'started'
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具执行完成时调用 - 这是关键方法！"""
        # 规范化输出为字符串
        if isinstance(output, dict):
            text = output.get('message') or output.get('error')
            if not isinstance(text, str):
                try:
                    text = json.dumps(output, ensure_ascii=False)
                except Exception:
                    text = str(output)
        else:
            text = str(output)
        logger.info(f"✅ 工具执行完成，返回值: {text}")
        self.tool_outputs.append(text)
        
        # 更新最后一个工具调用的状态
        if self.tool_calls:
            self.tool_calls[-1]['status'] = 'completed'
            self.tool_calls[-1]['output'] = output
    
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


def create_agent(llm_endpoint="http://localhost:8000/v1") -> Any:
    """创建并初始化LangChain agent，配置工具和LLM"""
    tools = get_all_tools()
    
    logger.info(f"已创建工具: {get_tool_names()}")

    # 初始化LLM客户端
    llm = OpenAI(
        openai_api_key="EMPTY",
        openai_api_base=llm_endpoint,
        model="",
        max_tokens=2000,
        temperature=0.2,
        top_p=0.95,
        default_headers={"Content-Type": "application/json"},
        request_timeout=120,
    )
    logger.info(f"LLM已初始化，端点: {llm_endpoint}")

    # 创建agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        prompt="""你是搭载在迎宾服务机器人上的AI智能体，你的名字叫Siri。请用中文回答用户的需求。你可以通过调用相应的工具函数来控制机器人的导航和机械臂/夹爪操作。

可用工具:
- arm_control: 控制机械臂执行动作
  - 参数: command (0=归位, 1=夹取, 2=释放, 3=搬运)
  - 适用场景: "拿起水"、"放下杯子"、"机械臂归位"等
- gripper_control: 控制夹爪开合
  - 参数: command (1=夹紧, 2=松开)
  - 适用场景: "夹爪夹紧"、"夹爪松开" 等
- go_to_office: 导航到办公室
  - 适用场景: "去办公室"、"到办公室去"等
- go_to_restroom: 导航到休息室
  - 适用场景: "去休息室"、"到休息室"等
- go_to_corridor: 导航到走廊
  - 适用场景: "去走廊"、"到走廊中间"等
- complex_task: 执行组合任务（先导航再操作机械臂）
  - 参数: location ("office"/"restroom"/"corridor"), arm_command (0-3)
  - 适用场景: "去办公室拿瓶水"、"把水送到休息室"等

顺序策略（非常重要）：
1) 如果需求涉及“去某地并做某事”，请先调用导航工具，再调用机械臂，然后根据需要调用夹爪。
2) 仅当必须要连续执行多个工具时，按以下顺序依次调用：导航 → 机械臂 → 夹爪。
3) 如果用户只提出单一动作（如只夹紧夹爪），则直接调用该工具，不要添加无关步骤。
4) 工具之间不要并行调用，等待上一步完成再进行下一步。

使用示例:
- "去办公室" → 使用 go_to_office()
- "拿起水" → 使用 arm_control(1)
- "去办公室拿瓶水" → 使用 complex_task("office", 1)
- "把水送到休息室" → 使用 complex_task("restroom", 3)
- "去走廊然后放下东西" → 使用 complex_task("corridor", 2)

根据用户的具体需求选择合适的工具。若用户要求“去某地做某事”，请显式先导航再执行机械臂/夹爪；若已有更细分的步骤，则按导航→机械臂→夹爪的顺序分步调用工具。""",
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True  # 启用返回中间步骤
    )
    
    logger.info("Agent已初始化，启用中间步骤返回")
    return agent

def initialize_agent_globally():
    """全局初始化agent"""
    global agent
    if agent is None:
        logger.info("正在初始化AI Agent...")
        agent = create_agent(llm_endpoint)
        logger.info("AI Agent初始化完成")

# --- HTTP API 路由 ---

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "message": "HTTP Agent Server正在运行",
        "tools_available": get_tool_names()
    })

@app.route('/v1/completions', methods=['POST'])
def completions():
    """
    主要的completions端点，兼容OpenAI API格式
    支持客户端发送prompt并获取AI回复
    """
    try:
        # 确保agent已初始化
        initialize_agent_globally()
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供JSON数据"}), 400
        
        # 提取prompt参数
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "未提供prompt"}), 400
        
        logger.info(f"收到请求 - Prompt: {prompt[:100]}...")
        
        # 调用agent处理请求
        try:
            # 创建回调处理器
            callback_handler = ToolResultCallbackHandler()
            
            # 使用回调处理器调用agent
            response = agent.invoke(
                {"input": prompt},
                config={"callbacks": [callback_handler]}
            )
            output_text = response.get('output', '未收到输出')
            
            # 从回调处理器获取工具执行结果
            tool_outputs = callback_handler.get_tool_outputs()
            
            # 决定返回给客户端的内容
            if tool_outputs:
                # 如果有工具返回值，只取每个工具结果的第一段话（第一个\n之前）
                first_lines = []
                for tool_output in tool_outputs:
                    try:
                        text = tool_output if isinstance(tool_output, str) else json.dumps(tool_output, ensure_ascii=False)
                    except Exception:
                        text = str(tool_output)
                    first_line = text.split('\n')[0] if '\n' in text else text
                    first_lines.append(first_line)
                
                tool_results_text = "\n".join(first_lines)
                final_text = f"{tool_results_text}\n\n{output_text}"
                logger.info(f"返回 {len(tool_outputs)} 个工具执行结果+LLM输出给客户端")
            else:
                # 如果工具没有返回值，直接返回LLM的输出
                final_text = output_text
                logger.info("返回LLM输出结果给客户端")
            
            # 构建响应格式，兼容OpenAI API
            result = {
                "choices": [
                    {
                        "text": final_text,
                        "index": 0,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(final_text.split()),
                    "total_tokens": len(prompt.split()) + len(final_text.split())
                },
                "model": "local-agent",
                "object": "text_completion"
            }
            
            logger.info("请求处理成功")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Agent处理出错: {e}")
            return jsonify({
                "error": f"Agent处理错误: {str(e)}",
                "choices": [
                    {
                        "text": f"抱歉，处理您的请求时出现错误：{str(e)}",
                        "index": 0,
                        "finish_reason": "error"
                    }
                ]
            }), 500
            
    except Exception as e:
        logger.error(f"请求处理出错: {e}")
        return jsonify({"error": f"请求处理错误: {str(e)}"}), 500

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    聊天completions端点，支持对话格式
    """
    try:
        initialize_agent_globally()
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供JSON数据"}), 400
        
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "未提供消息"}), 400
        
        # 将消息转换为prompt
        prompt = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'user':
                prompt += f"Human: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        
        logger.info(f"收到聊天请求 - Messages: {len(messages)}条")
        
        try:
            # 创建回调处理器
            callback_handler = ToolResultCallbackHandler()
            
            # 使用回调处理器调用agent
            response = agent.invoke(
                {"input": prompt},
                config={"callbacks": [callback_handler]}
            )
            output_text = response.get('output', '未收到输出')
            
            # 从回调处理器获取工具执行结果
            tool_outputs = callback_handler.get_tool_outputs()
            
            # 决定返回给客户端的内容
            if tool_outputs:
                # 如果有工具返回值，只取每个工具结果的第一段话（第一个\n之前）
                first_lines = []
                for tool_output in tool_outputs:
                    first_line = tool_output.split('\n')[0] if '\n' in tool_output else tool_output
                    first_lines.append(first_line)
                
                tool_results_text = "\n".join(first_lines)
                final_text = f"{tool_results_text}\n\n{output_text}"
                logger.info(f"返回 {len(tool_outputs)} 个工具执行结果+LLM输出给客户端")
            else:
                # 如果工具没有返回值，直接返回LLM的输出
                final_text = output_text
                logger.info("返回LLM输出结果给客户端")
            
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
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(final_text.split()),
                    "total_tokens": len(prompt.split()) + len(final_text.split())
                },
                "model": "local-agent",
                "object": "chat.completion"
            }
            
            logger.info("聊天请求处理成功")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Agent处理聊天请求出错: {e}")
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
        logger.error(f"聊天请求处理出错: {e}")
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
    return jsonify({
        "status": "running",
        "agent_initialized": agent is not None,
        "base_directory": os.getcwd(),
        "available_tools": get_tool_names()
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
    parser = argparse.ArgumentParser(description='HTTP Agent Server - 支持本地工具的AI Agent HTTP服务')
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
        '--llm-endpoint',
        type=str,
        default='http://localhost:8000/v1',
        help='LLM服务端点（默认: http://localhost:8000/v1）'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    return parser.parse_args()


def main():
    """主程序入口"""
    global llm_endpoint
    
    # 解析命令行参数
    args = parse_arguments()

    
    # 设置LLM端点
    llm_endpoint = args.llm_endpoint
    
    print("🚀 启动HTTP Agent Server...")
    print("🧠 LLM端点:", llm_endpoint)
    print("🔧 可用工具:", get_tool_names())
    print(f"🌐 服务将在 http://{args.host}:{args.port} 启动")
    print("📋 可用端点:")
    print("  - GET  /health - 健康检查")
    print("  - POST /v1/completions - 文本补全（兼容OpenAI API）")
    print("  - POST /v1/chat/completions - 聊天补全")
    print("  - GET  /tools - 列出可用工具")
    print("  - GET  /status - 服务状态")
    print("\n按 Ctrl+C 停止服务")
    
    # 启动Flask应用
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == "__main__":
    main()