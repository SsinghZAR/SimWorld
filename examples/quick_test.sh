#!/bin/bash
# 快速测试脚本

echo "=========================================="
echo "SimWorld 快速测试"
echo "=========================================="
echo ""
echo "步骤1: 请确保Unreal Engine已启动"
echo "步骤2: 启动API服务器..."
echo ""

# 启动API服务器（后台运行）
python examples/api_server_example.py &
SERVER_PID=$!

echo "API服务器已启动 (PID: $SERVER_PID)"
echo "等待服务器就绪..."
sleep 3

# 运行测试脚本
echo ""
echo "步骤3: 生成agent..."
python examples/test_spawn_agent.py

# 清理：可以选择是否关闭服务器
echo ""
read -p "是否关闭API服务器? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kill $SERVER_PID
    echo "API服务器已关闭"
fi
