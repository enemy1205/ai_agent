#!/bin/bash

# HTTP Agent Server V3 启动脚本（使用腾讯混元云端LLM）

# 默认参数
BASE_DIR=""
HOST="0.0.0.0"
PORT="5000"
VOICE_HOST="0.0.0.0"
VOICE_PORT="4999"
DEBUG=""
LOG_LEVEL="INFO"  # 默认日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --asr-host)
            VOICE_HOST="$2"
            shift 2
            ;;
        --asr-port)
            VOICE_PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "HTTP Agent Server V3 启动脚本（使用腾讯混元云端LLM）"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --base-dir DIR        指定工作目录路径（默认为当前目录）"
            echo "  --host HOST           Agent监听主机（默认: 0.0.0.0）"
            echo "  --port PORT           Agent监听端口（默认: 5000）"
            echo "  --asr-host HOST       ASR监听主机（默认: 0.0.0.0）"
            echo "  --asr-port PORT       ASR监听端口（默认: 4999）"
            echo "  --log-level LEVEL     日志级别（默认: INFO, 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL）"
            echo "  --debug               启用调试模式"
            echo "  -h, --help            显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  HUNYUAN_API_KEY       腾讯混元 API Key（必需）"
            echo "  HUNYUAN_BASE_URL       腾讯混元 API 端点（可选，默认: https://api.hunyuan.cloud.tencent.com/v1）"
            echo "  HUNYUAN_MODEL          腾讯混元模型名称（可选，默认: hunyuan-turbos-latest）"
            echo "  AGENT_BASE_DIR        工作目录路径（优先级低于命令行参数）"
            echo ""
            echo "示例:"
            echo "  export HUNYUAN_API_KEY='your_api_key'"
            echo "  $0 --port 5000"
            echo "  $0 --port 5000 --log-level DEBUG"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "🚀 启动HTTP Agent Server V3（腾讯混元云端LLM）..."
echo "📝 日志级别: $LOG_LEVEL"

# 检查必要的环境变量
if [ -z "$HUNYUAN_API_KEY" ]; then
    echo "⚠️  警告: 未设置环境变量 HUNYUAN_API_KEY"
    echo "   请设置环境变量: export HUNYUAN_API_KEY='your_api_key'"
    echo "   或在控制台创建 API KEY: https://console.cloud.tencent.com/hunyuan/apiKey"
    echo ""
    echo "   继续启动，但服务可能无法正常工作..."
    echo ""
fi

# 准备日志目录
LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

# 导出日志级别环境变量（供所有Python服务使用）
export LOG_LEVEL="$LOG_LEVEL"
# export LOG_SERVER_URL="http://127.0.0.1:8888"
# 构建启动命令
CMD="python3 http_agent_server_v3.py --host $HOST --port $PORT"

if [ -n "$BASE_DIR" ]; then
    CMD="$CMD --base-dir $BASE_DIR"
fi

if [ -n "$DEBUG" ]; then
    CMD="$CMD $DEBUG"
fi

# 简要启动信息
echo "🌐 Agent: http://$HOST:$PORT"
echo "☁️  LLM:  腾讯混元云端大模型"

# 后台启动HTTP Agent Server V3
echo "🌟 启动HTTP Agent Server V3 到后台... (日志: $LOG_DIR/agent_v3.log)"
LOG_LEVEL="$LOG_LEVEL" nohup bash -c "$CMD" > "$LOG_DIR/agent_v3.log" 2>&1 &
AGENT_PID=$!

# 启动语音VOICE服务
if [ -f "voice_services.py" ]; then
	echo "🎤 启动统一语音服务到后台 ($VOICE_HOST:$VOICE_PORT)... (日志: $LOG_DIR/voice.log)"
	LOG_LEVEL="$LOG_LEVEL" FLASK_HOST="$VOICE_HOST" FLASK_PORT="$VOICE_PORT" nohup python3 voice_services.py > "$LOG_DIR/voice.log" 2>&1 &
	VOICE_PID=$!
else
	echo "ℹ️ 未发现 voice_services.py，跳过启动语音服务"
fi

echo ""
echo "📋 服务状态:"
echo "  ☁️  LLM:   腾讯混元云端大模型"
echo "  🌐 Agent:  http://$HOST:$PORT (日志: $LOG_DIR/agent_v3.log, 日志级别: $LOG_LEVEL)"
if [ -n "$VOICE_PID" ]; then
	echo "  🎤 VOICE:        运行中 (日志: $LOG_DIR/voice.log, 日志级别: $LOG_LEVEL)"
else
	echo "  🎤 VOICE:        未启动"
fi
echo ""
echo "✅ 所有服务已尝试启动到后台。按 Ctrl+C 退出此提示，不影响后台服务运行。"
echo ""
echo "💡 提示:"
echo "  - 可以通过 --log-level 参数调整日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
echo "  - 确保已设置 HUNYUAN_API_KEY 环境变量"
echo "  - 查看日志: tail -f $LOG_DIR/agent_v3.log"

