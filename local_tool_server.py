"""
MCP Server 提供本地工具服务。

此脚本使用 mcp 库创建一个 MCP 服务器，
将本地文件操作和计算器功能通过 MCP 协议暴露出去。

其他支持 MCP 的客户端（如 Claude Desktop, Cursor）可以连接此服务器并调用这些工具。
"""

import asyncio
import os
import math
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging


from mcp import  types
from mcp.server import Server
from mcp.server.stdio import stdio_server


# --- 配置 ---
# 定义工具操作的基础目录，限制文件访问范围，增强安全性
BASE_DIR = Path("./").resolve()  # 使用相对路径下的 test 文件夹
os.makedirs(BASE_DIR, exist_ok=True)  # 确保目录存在

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server")

# 创建服务器实例
server = Server("local-tools-server")

# --- 工具函数实现 ---
async def list_files() -> List[str]:
    """
    异步列出基础目录下的所有文件。
    返回相对路径列表。
    """
    logger.info("Executing list_files")
    file_list = []
    try:
        for item in BASE_DIR.rglob("*"):
            if item.is_file():
                try:
                    relative_path = str(item.relative_to(BASE_DIR))
                    file_list.append(relative_path)
                except ValueError:
                    logger.warning(f"Skipping file outside base dir: {item}")
                    continue
    except Exception as e:
        logger.error(f"Error listing files: {e}")
    return file_list

async def read_file(name: str) -> Union[str, types.TextContent]:
    """
    异步读取指定文件的内容。
    """
    logger.info(f"Executing read_file with name: {name}")
    try:
        # 构建完整路径
        file_path = BASE_DIR / name
        
        # 安全检查：确保解析后的绝对路径在 BASE_DIR 之内
        resolved_path = file_path.resolve()
        if not str(resolved_path).startswith(str(BASE_DIR)):
            error_msg = f"Access denied. File '{name}' is outside the allowed directory."
            logger.warning(error_msg)
            return types.TextContent(type="text", text=error_msg)
        
        # 检查文件是否存在
        if not file_path.exists():
            error_msg = f"File '{name}' not found."
            logger.info(error_msg)
            return types.TextContent(type="text", text=error_msg)
        
        # 读取文件内容
        content = file_path.read_text(encoding='utf-8')
        return types.TextContent(type="text", text=content)
    
    except PermissionError:
        error_msg = f"Permission denied reading file '{name}'."
        logger.warning(error_msg)
        return types.TextContent(type="text", text=error_msg)
    except Exception as e:
        error_msg = f"An error occurred while reading '{name}': {e}"
        logger.error(error_msg, exc_info=True)
        return types.TextContent(type="text", text=error_msg)

# 允许使用的模块和函数，用于安全计算
ALLOWED_MATH_MODULES = {"math": math, "random": random}
ALLOWED_MATH_NAMES = {
    "abs", "round", "min", "max", "sum", "pow", "sqrt",
    "sin", "cos", "tan", "log", "log10", "exp", "pi", "e",
    "randint", "random", "choice", "uniform"
}

async def calculator(python_expression: str) -> Union[str, types.TextContent]:
    """
    安全地计算数学表达式。
    """
    logger.info(f"Executing calculator with expression: {python_expression}")
    try:
        # 创建一个受限的全局命名空间
        safe_globals = {"__builtins__": {}}
        
        # 添加允许的模块和函数
        for mod_name, mod in ALLOWED_MATH_MODULES.items():
            safe_globals[mod_name] = mod
        
        # 添加允许的名称
        for name in ALLOWED_MATH_NAMES:
            for mod in ALLOWED_MATH_MODULES.values():
                if hasattr(mod, name):
                    safe_globals[name] = getattr(mod, name)
                    break
        
        # 执行计算
        result = eval(python_expression, safe_globals)
        result_text = f"Calculation result: `{python_expression}` = `{result}`"
        return types.TextContent(type="text", text=result_text)
    
    except ZeroDivisionError:
        error_msg = "Error: Division by zero."
        logger.warning(error_msg)
        return types.TextContent(type="text", text=error_msg)
    except SyntaxError:
        error_msg = f"Error: Invalid syntax in expression '{python_expression}'."
        logger.warning(error_msg)
        return types.TextContent(type="text", text=error_msg)
    except NameError as e:
        error_msg = f"Error: Invalid name or function used: {e}."
        logger.warning(error_msg)
        return types.TextContent(type="text", text=error_msg)
    except Exception as e:
        error_msg = f"An error occurred during calculation: {e}"
        logger.error(error_msg, exc_info=True)
        return types.TextContent(type="text", text=error_msg)

# --- 注册 MCP 工具 ---
@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """返回服务器提供的工具列表"""
    return [
        types.Tool(
            name="list_files",
            description="List all files in the server's base directory.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="read_file",
            description="Read the contents of a file in the server's base directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the file to read."
                    }
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="calculator",
            description="Evaluate a safe mathematical expression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "python_expression": {
                        "type": "string",
                        "description": "A safe mathematical expression to evaluate."
                    }
                },
                "required": ["python_expression"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """处理工具调用请求"""
    logger.info(f"Calling tool: {name} with arguments: {arguments}")
    
    try:
        if name == "list_files":
            files = await list_files()
            content = "\n".join(files) if files else "No files found."
            return [types.TextContent(type="text", text=content)]
        
        elif name == "read_file":
            filename = arguments.get("name")
            if not filename:
                return [types.TextContent(type="text", text="Error: Missing required argument 'name'.")]
            result = await read_file(filename)
            return [result] if isinstance(result, types.TextContent) else [types.TextContent(type="text", text=result)]
        
        elif name == "calculator":
            expression = arguments.get("python_expression")
            if not expression:
                return [types.TextContent(type="text", text="Error: Missing required argument 'python_expression'.")]
            result = await calculator(expression)
            return [result] if isinstance(result, types.TextContent) else [types.TextContent(type="text", text=result)]
        
        else:
            error_msg = f"Unknown tool: {name}"
            logger.warning(error_msg)
            return [types.TextContent(type="text", text=error_msg)]
    
    except Exception as e:
        error_msg = f"Error executing tool {name}: {e}"
        logger.error(error_msg, exc_info=True)
        return [types.TextContent(type="text", text=error_msg)]

async def handle_initialize(
    params: types.InitializeParams
) -> types.InitializeResult:
    """处理初始化请求"""
    logger.info(f"Initializing server for client: {params.client_info}")
    return types.InitializeResult(
        server_info=types.ServerInfo(
            name="local-tools-server",
            version="0.1.0"
        ),
        capabilities=types.ServerCapabilities(
            tools=types.ToolsOptions(
                dynamic_registration=False
            )
        )
    )

async def main():
    """主函数：运行 MCP 服务器"""
    logger.info("Starting MCP server...")
    
    # 使用 stdio 传输层运行服务器
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")