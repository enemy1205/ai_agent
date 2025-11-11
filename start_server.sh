#!/bin/bash

# HTTP Agent Server 启动脚本

# 默认参数
BASE_DIR=""
HOST="0.0.0.0"
PORT="5000"
LLM_HOST="0.0.0.0"
LLM_PORT="8000"
LLM_ENDPOINT=""
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
        --llm-endpoint)
            LLM_ENDPOINT="$2"
            shift 2
            ;;
        --llm-host)
            LLM_HOST="$2"
            shift 2
            ;;
        --llm-port)
            LLM_PORT="$2"
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
            echo "HTTP Agent Server 启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --base-dir DIR        指定工作目录路径（默认为当前目录）"
            echo "  --host HOST           Agent监听主机（默认: 0.0.0.0）"
            echo "  --port PORT           Agent监听端口（默认: 5000）"
            echo "  --llm-host HOST       LLM监听主机（默认: 0.0.0.0）"
            echo "  --llm-port PORT       LLM监听端口（默认: 8000）"
            echo "  --llm-endpoint URL    LLM服务端点"
            echo "  --asr-host HOST       ASR监听主机（默认: 0.0.0.0）"
            echo "  --asr-port PORT       ASR监听端口（默认: 4999）"
            echo "  --log-level LEVEL     日志级别（默认: INFO, 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL）"
            echo "  --debug               启用调试模式"
            echo "  -h, --help            显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  AGENT_BASE_DIR        工作目录路径（优先级低于命令行参数）"
            echo ""
            echo "示例:"
            echo "  $0 --port 8080 --llm-port 8001 --asr-port 5101"
            echo "  AGENT_BASE_DIR=/path/to/work $0 --port 8080"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "🚀 启动一键服务启动器..."
echo "📝 日志级别: $LOG_LEVEL"

# 推导 LLM 端点
if [ -z "$LLM_ENDPOINT" ]; then
    LLM_ENDPOINT="http://$LLM_HOST:$LLM_PORT"
fi

# 准备日志目录
LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"

# 导出日志级别环境变量（供所有Python服务使用）
export LOG_LEVEL="$LOG_LEVEL"

# 等待HTTP服务可用
wait_for_http() {
	URL="$1"
	TIMEOUT_SECONDS="$2"
	INTERVAL=10
	ELAPSED=0
	while (( ELAPSED < TIMEOUT_SECONDS )); do
		if curl -sSf "$URL" > /dev/null; then
			return 0
		fi
		sleep $INTERVAL
		ELAPSED=$((ELAPSED + INTERVAL))
	done
	return 1
}

# 启动LLM服务（若未运行）
echo "🔍 检查LLM服务..."
if ! curl -s "$LLM_ENDPOINT/health" > /dev/null; then
	echo "🧠 启动vLLM到后台 ($LLM_HOST:$LLM_PORT)... (日志: $LOG_DIR/llm.log)"
	nohup bash -c "CUDA_VISIBLE_DEVICES=1,2 vllm serve ~/llm_model/qwen3_4B_Instruct_2507/ \
		--tensor-parallel-size 2 \
		--max-model-len 8192 \
		--max-num-seqs 64 \
		--max-num-batched-tokens 2048 \
		--gpu-memory-utilization 0.8 \
		--dtype half \
		--host $LLM_HOST \
		--port $LLM_PORT" > "$LOG_DIR/llm.log" 2>&1 &
	LLM_PID=$!
	if wait_for_http "$LLM_ENDPOINT/health" 120; then
		echo "✅ LLM已就绪: $LLM_ENDPOINT"
	else
		echo "⚠️  LLM启动后仍未在超时内就绪，请检查 $LOG_DIR/llm.log"
	fi
else
	echo "✅ 检测到LLM已在运行: $LLM_ENDPOINT"
fi

# 构建启动命令
CMD="python3 http_agent_server.py --host $HOST --port $PORT --llm-endpoint $LLM_ENDPOINT/v1"

if [ -n "$BASE_DIR" ]; then
    CMD="$CMD --base-dir $BASE_DIR"
fi

if [ -n "$DEBUG" ]; then
    CMD="$CMD $DEBUG"
fi

# 简要启动信息
echo "🌐 Agent: http://$HOST:$PORT"
echo "🧠 LLM:   $LLM_ENDPOINT"

# 后台启动HTTP Agent Server
echo "🌟 启动HTTP Agent Server 到后台... (日志: $LOG_DIR/agent.log)"
LOG_LEVEL="$LOG_LEVEL" nohup bash -c "$CMD" > "$LOG_DIR/agent.log" 2>&1 &
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
echo "  🧠 LLM:   $LLM_ENDPOINT  (日志: $LOG_DIR/llm.log)"
echo "  🌐 Agent: http://$HOST:$PORT (日志: $LOG_DIR/agent.log, 日志级别: $LOG_LEVEL)"
if [ -n "$VOICE_PID" ]; then
	echo "  🎤 VOICE:        运行中 (日志: $LOG_DIR/voice.log, 日志级别: $LOG_LEVEL)"
else
	echo "  🎤 VOICE:        未启动"
fi
echo ""
echo "✅ 所有服务已尝试启动到后台。按 Ctrl+C 退出此提示，不影响后台服务运行。"
echo ""
echo "💡 提示: 可以通过 --log-level 参数调整日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
