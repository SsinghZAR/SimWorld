# SimWorld API 使用指南

SimWorld 提供了 REST API 接口，允许外部系统（如 clawbot）通过 HTTP 请求控制 SimWorld 中的 agent。

## 快速开始

### 1. 启动 API 服务器

```bash
# 方式1: 使用示例脚本
python examples/api_server_example.py

# 方式2: 使用 uvicorn 直接启动
uvicorn simworld.api.server:app --host 0.0.0.0 --port 8000
```

服务器启动后，你可以访问：
- API 文档: http://localhost:8000/docs (Swagger UI)
- 健康检查: http://localhost:8000/health

### 2. 测试 API

运行客户端示例：

```bash
python examples/api_client_example.py
```

## API 端点

### 系统端点

#### GET `/`
根端点，返回服务器信息

#### GET `/health`
健康检查，返回服务器和 Unreal Engine 连接状态

### Agent 管理

#### POST `/api/agents/spawn`
生成一个新的 agent

**请求体:**
```json
{
  "agent_type": "humanoid",  // 或 "dog"
  "position": [0, 0],        // [x, y] 或 [x, y, z]
  "direction": [1, 0],       // [x, y]
  "model_path": "...",       // 可选，自定义模型路径
  "name": "..."              // 可选，自定义名称
}
```

**响应:**
```json
{
  "success": true,
  "message": "Agent spawned successfully",
  "data": {
    "agent_id": 0,
    "agent_type": "humanoid",
    "name": "GEN_BP_Humanoid_0",
    "position": [0, 0],
    "direction": [1, 0],
    "camera_id": 1
  }
}
```

#### GET `/api/agents`
列出所有已注册的 agent

#### GET `/api/agents/{agent_id}`
获取指定 agent 的信息

#### DELETE `/api/agents/{agent_id}`
销毁指定的 agent

### Humanoid 动作

#### POST `/api/humanoid/move`
控制 humanoid 移动

**请求体:**
```json
{
  "agent_id": 0,
  "action": "move_forward",  // 或 "step_forward", "stop"
  "duration": 2.0,          // step_forward 的持续时间（秒）
  "direction": 0            // step_forward 的方向
}
```

#### POST `/api/humanoid/rotate`
旋转 humanoid

**请求体:**
```json
{
  "agent_id": 0,
  "angle": 90,              // 旋转角度（度）
  "direction": "left"       // 或 "right"
}
```

#### POST `/api/humanoid/interact`
Humanoid 交互动作

**请求体:**
```json
{
  "agent_id": 0,
  "action": "pick_up",      // 或 "drop", "sit_down", "stand_up", 
                            // "argue", "discuss", "listen", 
                            // "wave_to_dog", "directing_path", "stop_action"
  "object_name": "..."      // pick_up 时需要指定对象名称
}
```

### Dog 动作

#### POST `/api/dog/move`
控制 dog 移动

**请求体:**
```json
{
  "agent_id": 0,
  "speed": 200,            // 移动速度
  "duration": 1.0,         // 持续时间（秒）
  "direction": 0            // 0=前进, 1=后退, 2=左, 3=右
}
```

#### POST `/api/dog/rotate`
旋转 dog

**请求体:**
```json
{
  "agent_id": 0,
  "angle": 90,              // 旋转角度（度）
  "duration": 0.7,          // 持续时间（秒）
  "clockwise": 1            // 1=顺时针, -1=逆时针
}
```

#### POST `/api/dog/look?agent_id={id}&direction={up|down}`
Dog 抬头/低头

### 相机操作

#### GET `/api/camera/{camera_id}/image?viewmode={lit|depth|object_mask}`
获取相机图像（返回 base64 编码的图片）

#### GET `/api/camera/{camera_id}/info`
获取相机信息（位置、旋转、FOV、分辨率等）

## 使用示例

### Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 生成 humanoid agent
response = requests.post(f"{BASE_URL}/api/agents/spawn", json={
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
})
agent_id = response.json()["data"]["agent_id"]

# 2. 让 agent 向前移动
requests.post(f"{BASE_URL}/api/humanoid/move", json={
    "agent_id": agent_id,
    "action": "step_forward",
    "duration": 2.0
})

# 3. 旋转 agent
requests.post(f"{BASE_URL}/api/humanoid/rotate", json={
    "agent_id": agent_id,
    "angle": 90,
    "direction": "left"
})

# 4. 获取相机图像
response = requests.get(f"{BASE_URL}/api/camera/1/image?viewmode=lit")
image_data = response.json()["image"]  # base64 编码的图片
```

### curl 示例

```bash
# 生成 agent
curl -X POST "http://localhost:8000/api/agents/spawn" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
  }'

# 移动 agent
curl -X POST "http://localhost:8000/api/humanoid/move" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": 0,
    "action": "step_forward",
    "duration": 2.0
  }'
```

## 集成到外部系统

### 从 clawbot 调用

clawbot 可以通过 HTTP 请求调用这些 API：

```python
# 在 clawbot 中
import requests

class SimWorldController:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def spawn_agent(self, agent_type, position, direction):
        response = requests.post(
            f"{self.api_url}/api/agents/spawn",
            json={
                "agent_type": agent_type,
                "position": position,
                "direction": direction
            }
        )
        return response.json()
    
    def move_agent(self, agent_id, action, **kwargs):
        response = requests.post(
            f"{self.api_url}/api/humanoid/move",
            json={
                "agent_id": agent_id,
                "action": action,
                **kwargs
            }
        )
        return response.json()
```

## 注意事项

1. **确保 Unreal Engine 运行**: API 服务器启动时会尝试连接 Unreal Engine，请确保 UE 已启动并启用了 UnrealCV
2. **端口配置**: 默认 API 端口是 8000，UnrealCV 端口是 9000，确保端口未被占用
3. **CORS**: API 服务器已启用 CORS，允许跨域访问
4. **异步操作**: 某些动作（如移动、旋转）可能需要时间完成，API 会等待动作完成后再返回

## 故障排除

- **连接失败**: 检查 Unreal Engine 是否运行，UnrealCV 是否启用
- **Agent 未找到**: 确保 agent_id 正确，agent 已成功生成
- **动作执行失败**: 检查 agent 类型是否匹配（humanoid vs dog）
