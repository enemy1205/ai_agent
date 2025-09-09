#!/bin/bash

# 一键停止 LLM / Agent / ASR 服务

# 默认端口（需与 start_server.sh 保持一致）
AGENT_PORT="5000"
LLM_PORT="8000"
ASR_PORT="4999"

show_help() {
  echo "一键停止服务脚本"
  echo ""
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --agent-port PORT   Agent 端口（默认: 5000）"
  echo "  --llm-port PORT     LLM 端口（默认: 8000）"
  echo "  --asr-port PORT     ASR 端口（默认: 4999）"
  echo "  -h, --help          显示此帮助信息"
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --agent-port)
      AGENT_PORT="$2"; shift 2;;
    --llm-port)
      LLM_PORT="$2"; shift 2;;
    --asr-port)
      ASR_PORT="$2"; shift 2;;
    -h|--help)
      show_help; exit 0;;
    *)
      echo "未知参数: $1"; show_help; exit 1;;
  esac
done

terminate_pids() {
  local pids="$1"
  local name="$2"
  local timeout=10

  if [ -z "$pids" ]; then
    echo "ℹ️  未发现 $name 相关进程"
    return 0
  fi

  echo "⛔ 尝试停止 $name: $pids"
  kill $pids 2>/dev/null

  local waited=0
  while kill -0 $pids 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [ $waited -ge $timeout ]; then
      echo "⚠️  停止超时，对 $name 使用强制终止"
      kill -9 $pids 2>/dev/null
      break
    fi
  done

  if ! kill -0 $pids 2>/dev/null; then
    echo "✅ 已停止 $name"
  else
    echo "❌ 停止 $name 失败 (PID: $pids)"
  fi
}

pids_by_port() {
  local port="$1"
  # 使用 lsof 查找占用端口的进程ID
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -i :"$port" 2>/dev/null | tr '\n' ' '
  else
    # 备选方式：ss
    if command -v ss >/dev/null 2>&1; then
      ss -lpn 2>/dev/null | awk -v p=":$port" '$0 ~ p {print $NF}' | sed -E 's/.*pid=([0-9]+).*/\1/' | tr '\n' ' '
    fi
  fi
}

pids_by_pattern() {
  local pattern="$1"
  pgrep -f "$pattern" 2>/dev/null | tr '\n' ' '
}

echo "🚦 开始停止服务..."

# LLM (vllm serve)
LLM_PIDS="$(pids_by_port "$LLM_PORT")"
if [ -z "$LLM_PIDS" ]; then
  # 兜底：匹配 vllm serve 进程
  LLM_PIDS="$(pids_by_pattern "vllm serve")"
fi
terminate_pids "$LLM_PIDS" "LLM(vLLM:$LLM_PORT)"

# Agent (http_agent_server.py)
AGENT_PIDS="$(pids_by_port "$AGENT_PORT")"
if [ -z "$AGENT_PIDS" ]; then
  AGENT_PIDS="$(pids_by_pattern "http_agent_server.py")"
fi
terminate_pids "$AGENT_PIDS" "Agent($AGENT_PORT)"

# ASR (asr_server.py)
ASR_PIDS="$(pids_by_port "$ASR_PORT")"
if [ -z "$ASR_PIDS" ]; then
  ASR_PIDS="$(pids_by_pattern "asr_server.py")"
fi
terminate_pids "$ASR_PIDS" "ASR($ASR_PORT)"

echo "🏁 处理完成。"


