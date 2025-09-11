# mcp_tool_wrapper.py （放在服务端）

from langchain_core.tools import BaseTool
from pydantic import Field
import requests
import json

class MCToolWrapper(BaseTool):
    name: str = "mcp_tool"
    description: str = "通过 MCP 协议调用远程客户端工具"
    mcp_server_url: str = "http://CLIENT_IP:8080/mcp"  # 客户端 MCP 服务地址
    tool_name: str = Field(..., description="要调用的远程工具名称")

    def _run(self, **kwargs) -> str:
        try:
            # 构造 MCP callTool 请求
            payload = {
                "method": "callTool",
                "params": {
                    "name": self.tool_name,
                    "arguments": kwargs
                }
            }

            response = requests.post(self.mcp_server_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                return f"工具执行失败: {result['error']}"

            return result.get("content", "无返回内容")

        except Exception as e:
            return f"调用远程工具失败: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        # 异步版本
        return self._run(**kwargs)