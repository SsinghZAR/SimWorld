#!/bin/bash
# Quick test script

echo "=========================================="
echo "SimWorld Quick Test"
echo "=========================================="
echo ""
echo "Step 1: Please make sure Unreal Engine is running"
echo "Step 2: Starting the API server..."
echo ""

# Start API server (run in background)
python examples/api_server_example.py &
SERVER_PID=$!

echo "API server started (PID: $SERVER_PID)"
echo "Waiting for the server to be ready..."
sleep 3

# Run test script
echo ""
echo "Step 3: Spawning an agent..."
python examples/test_spawn_agent.py

# Cleanup: optionally stop the server
echo ""
read -p "Do you want to stop the API server? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kill $SERVER_PID
    echo "API server stopped"
fi