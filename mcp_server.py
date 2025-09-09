#!/usr/bin/env python3
"""
MCP Server - 交互式AI Agent
支持本地工具的命令行交互式AI助手
"""

import os
from pathlib import Path
from typing import Any
from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI

# 导入共享工具
from tools import (
    get_all_tools, get_tool_names, 
    set_base_directory, get_base_directory
)

# --- 配置 ---
# 基础目录，所有文件操作都限制在此目录内
BASE_DIR = Path("./").resolve()

def create_agent() -> Any:
    """Creates and initializes the LangChain agent with tools and LLM."""
    tools = get_all_tools()
    print("Tools created:", get_tool_names())

    llm = OpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:8000/v1",
        model="", # 让 vLLM 使用默认加载的模型
        max_tokens=500,
        temperature=0.8,
        top_p=0.95,
        default_headers={"Content-Type": "application/json"},
        request_timeout=120,
    )
    print("LLM initialized.")

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        prompt="You are an expert in the fields of computer science and deep learning",
        verbose=True, # 可根据需要设置为 True 查看内部过程
        handle_parsing_errors=True
    )
    print("Agent initialized.")
    return agent

def run_interactive_loop(agent: Any) -> None:
    """Runs the interactive command-line loop for the agent."""
    print("\n--- Interactive Mode ---")
    print("Enter your queries. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue # 忽略空输入
            if user_input.lower() in ['quit', 'exit']:
                print("Agent: Goodbye!")
                break
            
            print("\n--- Agent is processing... ---")
            response: dict = agent.invoke({"input": user_input})
            # agent.invoke 返回一个字典，主要输出在 'output' 键下
            output_text: str = response.get('output', 'No output received.')
            print(f"Agent: {output_text}\n")
            
        except KeyboardInterrupt:
            print("\nAgent: Received interrupt signal. Goodbye!")
            break
        except Exception as e:
            print(f"Agent: An unexpected error occurred: {e}\n")

def main() -> None:
    """Main entry point of the application."""
    # 设置基础目录
    set_base_directory(str(BASE_DIR))
    
    print("🚀 启动MCP Server...")
    print("📁 基础目录:", get_base_directory())
    print("🔧 可用工具:", get_tool_names())
    
    agent = create_agent()
    run_interactive_loop(agent)

if __name__ == "__main__":
    main()