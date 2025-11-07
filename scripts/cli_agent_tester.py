#!/usr/bin/env python3
"""
交互式 CLI 测试脚本

用法示例：
  python3 scripts/cli_agent_tester.py --host 127.0.0.1 --port 5000 --path /v1/chat/completions

说明：
- 针对 HTTP Agent Server 的 `POST /v1/completions` 接口，发送用户在终端输入的中文文本。
- 服务器会将“工具输出的首行”与“LLM 的最终回复”组合返回，本脚本尝试将两部分分开展示，便于观察工具调用与反馈。
"""

import argparse
import json
import sys
import time
from typing import Dict, Any

import requests


def build_url(host: str, port: int, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{host}:{port}{path}"


def post_chat_completions(url: str, user_text: str, session_id: str = None, timeout: float = 30.0) -> Dict[str, Any]:
    """
    适配 http_agent_server.py 的 /v1/chat/completions：
    请求: {"messages": [{"role": "user", "content": "..."}], "session_id": "..."}
    返回: choices[0].message.content + metadata.session_id
    """
    headers = {"Content-Type": "application/json"}
    payload = {"messages": [{"role": "user", "content": user_text}]}
    
    # 如果有 session_id，添加到请求中以维持会话
    if session_id:
        payload["session_id"] = session_id
    
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    # 尝试返回 JSON；非 2xx 也读取文本便于调试
    try:
        data = resp.json()
    except Exception:
        data = {"error": f"Non-JSON response (status {resp.status_code})", "text": resp.text}
    data["http_status"] = resp.status_code
    return data


def extract_texts_from_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    服务器返回格式（/v1/chat/completions）：
    {
      "choices": [{"message": {"role": "assistant", "content": "<工具若干行首行合并>\n\n<LLM输出>"}, ...}],
      ...
    }
    这里尝试用第一个空行将“工具反馈（可能多行）”与“模型回复”分离，
    若无法分离则整体作为模型回复展示。
    """
    result = {
        "tool_feedback": None,
        "assistant_text": None,
    }
    try:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            # 优先解析 chat.completions 的 message.content
            message = choices[0].get("message") or {}
            raw_text = None
            if isinstance(message, dict):
                raw_text = message.get("content")
            # 兜底兼容 /v1/completions 的 text
            if raw_text is None:
                raw_text = choices[0].get("text")
            if isinstance(raw_text, str):
                # 按双空行拆分（服务器合成时用了一段空行）
                parts = raw_text.split("\n\n", 1)
                if len(parts) == 2:
                    result["tool_feedback"], result["assistant_text"] = parts[0], parts[1]
                else:
                    result["assistant_text"] = raw_text
            else:
                result["assistant_text"] = str(raw_text)
    except Exception:
        # 保底
        result["assistant_text"] = json.dumps(data, ensure_ascii=False)
    return result


def interactive_loop(url: str, show_raw: bool, timeout: float):
    print(f"目标接口: {url}")
    print("输入中文指令回车发送。输入 'exit' 或 Ctrl+C 退出。")
    print("🧠 启用会话记忆功能，对话将被记住。\n")
    
    # 维护会话ID以保持记忆
    session_id = None
    request_count = 0
    
    while True:
        try:
            user_text = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return

        if user_text.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            return
        if not user_text:
            continue

        t0 = time.time()
        try:
            data = post_chat_completions(url, user_text, session_id=session_id, timeout=timeout)
        except requests.exceptions.RequestException as e:
            print(f"[请求错误] {e}")
            continue
        dt = (time.time() - t0) * 1000
        request_count += 1

        status = data.get("http_status")
        if status and status >= 400:
            print(f"[HTTP {status}] {data}")
            continue

        # 提取并保存 session_id
        metadata = data.get("metadata", {})
        if metadata.get("session_id"):
            new_session_id = metadata["session_id"]
            if session_id is None:
                session_id = new_session_id
                print(f"🔑 会话已创建: {session_id[:8]}...\n")
            elif session_id != new_session_id:
                # 理论上不应该发生，除非服务器重启
                session_id = new_session_id
                print(f"⚠️  会话ID已更新: {session_id[:8]}...\n")

        if show_raw:
            print("\n=== 原始响应 JSON ===")
            print(json.dumps(data, ensure_ascii=False, indent=2))

        split_texts = extract_texts_from_response(data)
        tool_feedback = split_texts.get("tool_feedback")
        assistant_text = split_texts.get("assistant_text")

        # 显示会话信息
        memory_count = metadata.get("memory_messages_count", 0)
        tool_calls_count = metadata.get("tool_calls_count", 0)
        
        print(f"\n(耗时: {dt:.0f} ms | 第{request_count}次请求 | 记忆:{memory_count}条消息 | 工具调用:{tool_calls_count}次)")
        
        if tool_feedback:
            print("—— 工具反馈 ——")
            print(tool_feedback)
        print("—— 模型回复 ——")
        if assistant_text:
            print(assistant_text)
        else:
            print("<无文本>")
        print()


def main():
    parser = argparse.ArgumentParser(description="AI Agent CLI 测试器")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Agent 服务主机")
    parser.add_argument("--port", type=int, default=5000, help="Agent 服务端口")
    parser.add_argument("--path", type=str, default="/v1/chat/completions", help="测试接口路径")
    parser.add_argument("--timeout", type=float, default=30.0, help="请求超时秒数")
    parser.add_argument("--show-raw", action="store_true", help="打印原始 JSON 响应")
    args = parser.parse_args()

    url = build_url(args.host, args.port, args.path)
    interactive_loop(url=url, show_raw=args.show_raw, timeout=args.timeout)


if __name__ == "__main__":
    main()


