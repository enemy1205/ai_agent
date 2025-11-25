#!/bin/bash

# 一键停止 HTTP Agent Server V3 相关服务（使用腾讯混元云端LLM）

# 默认端口（需与 start_server_v3.sh 保持一致）
AGENT_PORT="5000"
VOICE_PORT="4999"

show_help() {
  echo "一键停止 HTTP Agent Server V3 相关服务脚本"
  echo ""
  echo "用法: $0 [选项]"
  echo ""
  echo "选项:"
  echo "  --agent-port PORT   Agent 端口（默认: 5000）"
  echo "  --voice-port PORT    语音服务端口（默认: 4999）"
  echo "  -h, --help            显示此帮助信息"
  echo ""
  echo "说明:"
  echo "  此脚本用于停止使用腾讯混元云端LLM的服务"
  echo "  - Agent Server V3 (http_agent_server_v3.py)"
  echo "  - 语音服务 (voice_services.py)"
  echo "  注意：不会停止本地 vLLM 服务（因为使用的是云端LLM）"
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --agent-port)
      AGENT_PORT="$2"; shift 2;;
    --voice-port)
      VOICE_PORT="$2"; shift 2;;
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

echo "🚦 开始停止 HTTP Agent Server V3 相关服务..."
echo "☁️  使用腾讯混元云端LLM（不会停止本地vLLM服务）"
echo ""

# Agent V3 (http_agent_server_v3.py)
AGENT_PIDS="$(pids_by_port "$AGENT_PORT")"
if [ -z "$AGENT_PIDS" ]; then
  AGENT_PIDS="$(pids_by_pattern "http_agent_server_v3.py")"
fi
terminate_pids "$AGENT_PIDS" "Agent V3($AGENT_PORT)"

# 语音服务 (voice_services.py)
VOICE_PIDS="$(pids_by_port "$VOICE_PORT")"
if [ -z "$VOICE_PIDS" ]; then
  VOICE_PIDS="$(pids_by_pattern "voice_services.py")"
fi
terminate_pids "$VOICE_PIDS" "语音服务($VOICE_PORT)"

echo ""
echo "🏁 处理完成。"
echo ""
echo "💡 提示:"
echo "  - 如果使用本地vLLM服务，请使用 kill_server.sh 停止"
echo "  - 查看进程: ps aux | grep -E 'http_agent_server_v3|voice_services'"

