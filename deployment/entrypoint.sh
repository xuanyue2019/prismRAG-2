#!/bin/bash

# Agent Lightning 智能体客户端入口点脚本

set -e

# 设置默认值
SERVER_URL=${SERVER_URL:-"http://localhost:8000"}
AGENT_NAME=${AGENT_NAME:-"default-agent"}
FRAMEWORK=${FRAMEWORK:-"autogen"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# 等待服务器就绪
echo "等待服务器在 $SERVER_URL 就绪..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -s -f "$SERVER_URL/health" > /dev/null; then
        echo "服务器已就绪!"
        break
    fi
    
    echo "尝试 $attempt/$max_attempts: 服务器尚未就绪，等待 5 秒..."
    sleep 5
    attempt=$((attempt + 1))
done

if [ $attempt -gt $max_attempts ]; then
    echo "错误: 服务器在 $SERVER_URL 未就绪，超时退出"
    exit 1
fi

# 根据框架选择启动脚本
case $FRAMEWORK in
    "autogen")
        echo "启动 AutoGen 智能体..."
        exec python examples/autogen/agent.py \
            --server-url "$SERVER_URL" \
            --name "$AGENT_NAME" \
            --log-level "$LOG_LEVEL"
        ;;
    "langchain")
        echo "启动 LangChain 智能体..."
        exec python examples/langchain/agent.py \
            --server-url "$SERVER_URL" \
            --name "$AGENT_NAME" \
            --log-level "$LOG_LEVEL"
        ;;
    "openai")
        echo "启动 OpenAI 智能体..."
        exec python examples/openai/agent.py \
            --server-url "$SERVER_URL" \
            --name "$AGENT_NAME" \
            --log-level "$LOG_LEVEL"
        ;;
    "custom")
        echo "启动自定义智能体..."
        if [ -z "$CUSTOM_SCRIPT" ]; then
            echo "错误: CUSTOM_SCRIPT 环境变量未设置"
            exit 1
        fi
        exec python "$CUSTOM_SCRIPT" \
            --server-url "$SERVER_URL" \
            --name "$AGENT_NAME" \
            --log-level "$LOG_LEVEL"
        ;;
    *)
        echo "错误: 不支持的框架 '$FRAMEWORK'"
        echo "支持的框架: autogen, langchain, openai, custom"
        exit 1
        ;;
esac