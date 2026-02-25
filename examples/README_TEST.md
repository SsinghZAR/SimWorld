# Quick Test Guide

## File Location Notes

All files are located under the `SimWorld/` subdirectory, so you need to use the full path.

## Test Steps

### 1. Start Unreal Engine

Make sure Unreal Engine is running and UnrealCV is enabled.

### 2. Start the API Server

From the project root (`/home/lingjun/SimWorld`), run:

```bash
python SimWorld/examples/api_server_example.py
```

Or:

```bash
cd SimWorld
python examples/api_server_example.py
```

### 3. Run the Test Script

In another terminal, also from the project root, run:

```bash
python SimWorld/examples/test_spawn_agent.py
```

Or:

```bash
cd SimWorld
python examples/test_spawn_agent.py
```

## Or Test Directly with curl

```bash
curl -X POST "http://localhost:8000/api/agents/spawn" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
  }'
```

## Check Server Status

```bash
curl http://localhost:8000/health
```