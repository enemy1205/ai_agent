#!/bin/bash
# 启动集中式日志服务器

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
LOG_SERVER_HOST="${LOG_SERVER_HOST:-0.0.0.0}"
LOG_SERVER_PORT="${LOG_SERVER_PORT:-8888}"

echo "=========================================="
echo "启动集中式日志服务器"
echo "=========================================="
echo "监听地址: $LOG_SERVER_HOST:$LOG_SERVER_PORT"
echo "Web访问: http://$LOG_SERVER_HOST:$LOG_SERVER_PORT"
echo "=========================================="
echo ""

# 检查Python依赖
if ! python3 -c "import flask" 2>/dev/null; then
    echo "错误: 缺少Flask，请运行: pip install flask flask-cors flask-socketio"
    exit 1
fi

# 启动服务
export LOG_SERVER_HOST="$LOG_SERVER_HOST"
export LOG_SERVER_PORT="$LOG_SERVER_PORT"
python3 app.py

