# 快速测试指南

## 文件位置说明

所有文件都在 `SimWorld/` 子目录下，所以需要使用完整路径。

## 测试步骤

### 1. 启动 Unreal Engine
确保 UE 已启动并启用了 UnrealCV。

### 2. 启动 API 服务器

在项目根目录 (`/home/lingjun/SimWorld`) 运行：

```bash
python SimWorld/examples/api_server_example.py
```

或者：

```bash
cd SimWorld
python examples/api_server_example.py
```

### 3. 运行测试脚本

在另一个终端，同样在项目根目录运行：

```bash
python SimWorld/examples/test_spawn_agent.py
```

或者：

```bash
cd SimWorld
python examples/test_spawn_agent.py
```

## 或者使用 curl 直接测试

```bash
curl -X POST "http://localhost:8000/api/agents/spawn" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
  }'
```

## 检查服务器状态

```bash
curl http://localhost:8000/health
```
