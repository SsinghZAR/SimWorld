# SimWorld API 模块

这个模块提供了 REST API 接口，允许外部系统（如 clawbot）通过 HTTP 请求控制 SimWorld 中的 agent。

## 快速开始

### 1. 启动 API 服务器

```bash
cd SimWorld
python3 examples/api_server_example.py
```

服务器会在 `http://localhost:8000` 启动。

### 2. 从外部系统（如 clawbot）连接

clawbot 只需要发送 HTTP 请求即可。例如：

**生成 agent:**
```bash
curl -X POST "http://localhost:8000/api/agents/spawn" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
  }'
```

**控制 agent 移动:**
```bash
curl -X POST "http://localhost:8000/api/humanoid/move" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": 0,
    "action": "step_forward",
    "duration": 2.0
  }'
```

**旋转 agent:**
```bash
curl -X POST "http://localhost:8000/api/humanoid/rotate" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": 0,
    "angle": 90,
    "direction": "left"
  }'
```

## 主要 API 端点

- **生成 agent**: `POST /api/agents/spawn`
- **列出 agents**: `GET /api/agents`
- **Humanoid 移动**: `POST /api/humanoid/move`
- **Humanoid 旋转**: `POST /api/humanoid/rotate`
- **Humanoid 交互**: `POST /api/humanoid/interact`
- **Dog 移动**: `POST /api/dog/move`
- **Dog 旋转**: `POST /api/dog/rotate`
- **获取相机图像**: `GET /api/camera/{camera_id}/image`

## 完整 API 文档

- **Swagger UI**: http://localhost:8000/docs (服务器启动后访问)
- **详细文档**: 参考 `docs/API_USAGE.md`

## 注意事项

1. 确保 Unreal Engine 已启动并启用了 UnrealCV
2. API 服务器默认端口是 8000
3. 所有 API 都支持跨域访问（CORS 已启用）
