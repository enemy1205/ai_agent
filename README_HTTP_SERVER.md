# HTTP Agent Server 使用说明

## 概述

HTTP Agent Server 是一个基于 Flask 的 HTTP 服务端，将原有的交互式 AI agent 改造为可通过 HTTP 请求调用的服务。该服务集成了本地自建工具（文件操作、计算器等），支持客户端通过 HTTP API 调用。

## 功能特性

- 🔧 **本地工具集成**: 支持文件操作、数学计算等工具
- 🌐 **HTTP API**: 兼容 OpenAI API 格式
- 💬 **多种接口**: 支持文本补全和聊天补全
- 🔒 **安全限制**: 文件操作限制在指定目录内
- 📊 **状态监控**: 提供健康检查和状态查询接口

## 可用工具

1. **文件操作工具**:
   - `read_file`: 读取文件内容
   - `list_files`: 列出目录下所有文件
   - `rename_file`: 重命名文件

2. **计算器工具**:
   - `calculator`: 安全的数学表达式计算
   - 支持 math 和 random 模块的常用函数

## 安装和启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 vLLM 服务

确保 vLLM 服务正在运行：

```bash
CUDA_VISIBLE_DEVICES=1,2 vllm serve ./qwen3_4B_Instruct_2507/ \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.8 \
  --dtype half \
  --host 0.0.0.0 \
  --port 8000
```

### 3. 启动 HTTP Agent Server

使用启动脚本：

```bash
./start_server.sh
```

或直接运行：

```bash
python3 http_agent_server.py
```

服务将在 `http://localhost:5000` 启动。

## API 接口

### 1. 健康检查

```bash
GET /health
```

**响应示例**:
```json
{
  "status": "healthy",
  "message": "HTTP Agent Server is running",
  "tools_available": ["read_file", "list_files", "rename_file", "calculator"]
}
```

### 2. 文本补全 (兼容 OpenAI API)

```bash
POST /v1/completions
```

**请求示例**:
```json
{
  "prompt": "请计算 math.sqrt(16) + 5 的结果",
  "max_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.95
}
```

**响应示例**:
```json
{
  "choices": [
    {
      "text": "根据计算结果，math.sqrt(16) + 5 = 4 + 5 = 9",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 20,
    "total_tokens": 35
  },
  "model": "local-agent",
  "object": "text_completion"
}
```

### 3. 聊天补全

```bash
POST /v1/chat/completions
```

**请求示例**:
```json
{
  "messages": [
    {"role": "user", "content": "你好，请帮我列出当前目录的文件"}
  ],
  "max_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.95
}
```

**响应示例**:
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "我来帮您列出当前目录的文件..."
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 25,
    "total_tokens": 45
  },
  "model": "local-agent",
  "object": "chat.completion"
}
```

### 4. 工具列表

```bash
GET /tools
```

**响应示例**:
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file within the base directory."
    },
    {
      "name": "list_files",
      "description": "List all files in the base directory and its subdirectories."
    }
  ],
  "count": 4
}
```

### 5. 服务状态

```bash
GET /status
```

**响应示例**:
```json
{
  "status": "running",
  "agent_initialized": true,
  "base_directory": "/home/sp/projects/mcp-calculator",
  "available_tools": ["read_file", "list_files", "rename_file", "calculator"]
}
```

## 客户端使用示例

### Python 客户端

```python
import requests

def call_agent(prompt, max_tokens=100, temperature=0.8, top_p=0.95):
    """调用HTTP Agent Server"""
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    response = requests.post(
        "http://localhost:5000/v1/completions",
        headers={"Content-Type": "application/json"},
        json=data,
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["text"]
    else:
        return f"错误: {response.status_code} - {response.text}"

# 使用示例
reply = call_agent("请计算 sin(π/2) 的值")
print(reply)
```

### cURL 示例

```bash
# 基本文本补全
curl -X POST http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请帮我列出当前目录的文件",
    "max_tokens": 100
  }'

# 聊天补全
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "max_tokens": 100
  }'
```

## 测试

运行测试客户端：

```bash
python3 test_client.py
```

测试将验证：
- 健康检查
- 工具列表获取
- 基本文本补全
- 聊天补全
- 工具使用功能

## 配置说明

### 环境变量

- `BASE_DIR`: 文件操作的基础目录（默认：当前目录）
- `FLASK_HOST`: Flask 服务主机（默认：0.0.0.0）
- `FLASK_PORT`: Flask 服务端口（默认：5000）

### vLLM 配置

确保 vLLM 服务运行在 `http://localhost:8000/v1`，或修改 `http_agent_server.py` 中的 `openai_api_base` 配置。

## 安全注意事项

1. **文件操作限制**: 所有文件操作都限制在 `BASE_DIR` 目录内
2. **计算器安全**: 使用受限的 `eval` 环境，只允许数学和随机函数
3. **路径检查**: 防止路径遍历攻击
4. **超时设置**: 请求超时设置为 120 秒

## 故障排除

### 常见问题

1. **连接 vLLM 失败**
   - 检查 vLLM 服务是否运行
   - 确认端口 8000 可访问

2. **工具调用失败**
   - 检查文件权限
   - 确认基础目录存在

3. **响应超时**
   - 增加 `request_timeout` 参数
   - 检查 vLLM 服务性能

### 日志查看

服务运行时会输出详细日志，包括：
- 工具调用记录
- 请求处理状态
- 错误信息

## 扩展功能

### 添加新工具

1. 在 `http_agent_server.py` 中定义工具函数
2. 使用 `StructuredTool.from_function()` 创建工具
3. 将工具添加到 `tools` 列表
4. 重启服务

### 自定义提示词

修改 `create_agent()` 函数中的 `prompt` 参数来自定义 AI 助手的行为。

## 许可证

本项目基于 MIT 许可证开源。
