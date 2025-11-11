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
import json
import uuid

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
logger = create_server_logger("http_agent_server", level=os.getenv("LOG_LEVEL", "INFO"))

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
        log_tool_call(logger, tool_name, {"input": safe_input})
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
        logger.info(f"工具执行完成，返回值: {text[:100]}")
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
        prompt="""你是搭载在迎宾服务机器人上的AI智能体，你的名字叫Siri。任何情况都请用中文回答用户的需求。你可以通过调用相应的工具函数来控制机器人的导航和机械臂/夹爪操作。

【核心原则 - 必须严格遵守】
1. **明确识别原则**：只根据用户明确表达的意图调用工具，不要推测或过度解读
2. **关键词匹配原则**：必须确认用户话中包含特定关键词才调用相应工具
3. **单一任务原则**：用户明确只要求一个动作时，只调用一个工具，不要自动添加额外步骤

【工具列表及调用条件】

导航工具 - 调用条件：用户明确表达了"去"、"到"、"导航"、"前往"等移动意图
- go_to_office: 去办公室（关键词：办公室、office）
- go_to_restroom: 去休息室（关键词：休息室、restroom）  
- go_to_corridor: 去走廊（关键词：走廊、corridor）
- 示例："去办公室"、"到休息室去"、"导航到走廊"

机械臂工具(arm_control) - 调用条件：用户明确表达了"拿起"、"放下"、"起"、"下"、"机械臂"、"搬"等操作意图
- 参数: command (0=归位, 1=夹取, 2=释放, 3=搬运)
- 示例："拿起水"、"放下杯子"、"机械臂归位"、"把它搬起来"

夹爪工具(gripper_control) - 调用条件：用户明确表达了"夹爪"、"夹"、"夹取"、"抓"、"握"等动作
- 参数: command (1=夹紧, 2=松开)
- 示例："夹爪夹紧"、"夹爪松开"、"夹取物体"

复合任务工具 - 只在用户同时提出导航+机械臂需求时使用
- complex_task: 先导航再执行机械臂动作
- get_water_bottle: 拿水瓶的完整自动化流程（适合"拿水瓶"、"拿水杯"等明确需求）

【禁用行为】
❌ 用户说"你好"只回复问候，不要调用任何工具
❌ 用户问"状态"时只回答状态信息，不要主动导航
❌ 用户说"可以吗"、"准备好了吗"时只确认，不调用工具
❌ 用户没有提到具体地点时，不要使用导航工具
❌ 用户没有提到"拿"、"放"、"机械臂"时，不要调用机械臂
❌ 用户没有提到"夹"、"爪"时，不要调用夹爪

【正确使用示例】
- "你好" → 只回复问候，不调用工具
- "去办公室" → go_to_office()（明确的导航意图）
- "拿起水" → arm_control(1)（明确的机械臂操作）
- "夹爪夹紧" → gripper_control(1)（明确的夹爪操作）
- "请帮我去拿水瓶" → get_water_bottle()（明确的完整任务）
- "去办公室拿瓶水" → complex_task("office", 1)（同时包含导航和拿取）

【错误使用示例】
- 用户："你好" → ❌ 不要执行"去办公室"
- 用户："准备好了吗" → ❌ 不要调用任何工具
- 用户："水在哪里" → ❌ 不调用导航或机械臂，只回答问题

严格遵循上述原则，确保只在用户明确表达意图时才调用相应工具。""",
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

def _post_process_response(original_prompt, agent_output, tool_outputs):
    """直接组合LLM输出和工具结果的text部分"""
    # 提取工具结果的text字段
    tool_texts = []
    for tool_output in tool_outputs:
        if isinstance(tool_output, dict) and 'text' in tool_output:
            text = tool_output['text']
            if text and isinstance(text, str):
                tool_texts.append(text)
    
    # 组合LLM输出和工具结果
    if tool_texts:
        final_text = f"{agent_output}\n\n" + "\n".join(tool_texts)
    else:
        final_text = agent_output
    
    return final_text

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
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    log_request_start(logger, "/v1/completions", "POST")
    
    try:
        # 确保agent已初始化
        initialize_agent_globally()
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            logger.warning("未提供JSON数据")
            return jsonify({"error": "未提供JSON数据"}), 400
        
        # 提取prompt参数
        prompt = data.get('prompt', '')
        if not prompt:
            logger.warning("未提供prompt")
            return jsonify({"error": "未提供prompt"}), 400
        
        logger.info(f"Prompt: {prompt[:100]}...")
        
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
            
            # 统一进行后处理，无论是否有工具调用
            final_text = _post_process_response(prompt, output_text, tool_outputs)
            logger.debug("完成后处理")
            
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
            
            log_request_end(logger, 200)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Agent处理出错: {e}", exc_info=True)
            log_request_end(logger, 500)
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
        logger.error(f"请求处理出错: {e}", exc_info=True)
        log_request_end(logger, 500)
        return jsonify({"error": f"请求处理错误: {str(e)}"}), 500

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    聊天completions端点，支持对话格式
    """
    # 为每个请求生成唯一的request_id
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    log_request_start(logger, "/v1/chat/completions", "POST")
    
    try:
        initialize_agent_globally()
        
        data = request.get_json()
        if not data:
            logger.warning("未提供JSON数据")
            return jsonify({"error": "未提供JSON数据"}), 400
        
        messages = data.get('messages', [])
        if not messages:
            logger.warning("未提供消息")
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
        
        logger.info(f"收到 {len(messages)} 条消息")
        
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
            
            # 统一进行后处理，无论是否有工具调用
            final_text = _post_process_response(prompt, output_text, tool_outputs)
            logger.debug("完成后处理")
            
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
            
            log_request_end(logger, 200)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Agent处理聊天请求出错: {e}", exc_info=True)
            log_request_end(logger, 500)
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
        logger.error(f"聊天请求处理出错: {e}", exc_info=True)
        log_request_end(logger, 500)
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
    
    logger.info("=" * 60)
    logger.info("启动HTTP Agent Server")
    logger.info(f"LLM端点: {llm_endpoint}")
    logger.info(f"可用工具: {get_tool_names()}")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info("可用端点:")
    logger.info("  - GET  /health")
    logger.info("  - POST /v1/completions")
    logger.info("  - POST /v1/chat/completions")
    logger.info("  - GET  /tools")
    logger.info("  - GET  /status")
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