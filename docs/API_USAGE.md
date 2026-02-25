# SimWorld API User Guide

SimWorld provides a REST API interface that allows external systems (such as clawbot) to control agents in SimWorld via HTTP requests.

## Quick Start

### 1. Start the API Server

```bash
# Option 1: Use the example script
python examples/api_server_example.py

# Option 2: Start directly with uvicorn
uvicorn simworld.api.server:app --host 0.0.0.0 --port 8000
```

After the server starts, you can access:

* API Docs: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
* Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Test the API

Run the client example:

```bash
python examples/api_client_example.py
```

## API Endpoints

### System Endpoints

#### GET `/`

Root endpoint, returns server information

#### GET `/health`

Health check, returns server status and Unreal Engine connection status

### Agent Management

#### POST `/api/agents/spawn`

Spawn a new agent

**Request Body:**

```json
{
  "agent_type": "humanoid",  // or "dog"
  "position": [0, 0],        // [x, y] or [x, y, z]
  "direction": [1, 0],       // [x, y]
  "model_path": "...",       // optional, custom model path
  "name": "..."              // optional, custom name
}
```

**Response:**

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

List all registered agents

#### GET `/api/agents/{agent_id}`

Get information for a specific agent

#### DELETE `/api/agents/{agent_id}`

Destroy a specific agent

### Humanoid Actions

#### POST `/api/humanoid/move`

Control humanoid movement

**Request Body:**

```json
{
  "agent_id": 0,
  "action": "move_forward",  // or "step_forward", "stop"
  "duration": 2.0,           // duration in seconds for step_forward
  "direction": 0             // direction for step_forward
}
```

#### POST `/api/humanoid/rotate`

Rotate humanoid

**Request Body:**

```json
{
  "agent_id": 0,
  "angle": 90,               // rotation angle (degrees)
  "direction": "left"        // or "right"
}
```

#### POST `/api/humanoid/interact`

Humanoid interaction actions

**Request Body:**

```json
{
  "agent_id": 0,
  "action": "pick_up",       // or "drop", "sit_down", "stand_up",
                             // "argue", "discuss", "listen",
                             // "wave_to_dog", "directing_path", "stop_action"
  "object_name": "..."       // required for pick_up
}
```

### Dog Actions

#### POST `/api/dog/move`

Control dog movement

**Request Body:**

```json
{
  "agent_id": 0,
  "speed": 200,              // movement speed
  "duration": 1.0,           // duration in seconds
  "direction": 0             // 0=forward, 1=backward, 2=left, 3=right
}
```

#### POST `/api/dog/rotate`

Rotate dog

**Request Body:**

```json
{
  "agent_id": 0,
  "angle": 90,               // rotation angle (degrees)
  "duration": 0.7,           // duration in seconds
  "clockwise": 1             // 1=clockwise, -1=counterclockwise
}
```

#### POST `/api/dog/look?agent_id={id}&direction={up|down}`

Make the dog look up or down

### Camera Operations

#### GET `/api/camera/{camera_id}/image?viewmode={lit|depth|object_mask}`

Get camera image (returns a base64-encoded image)

#### GET `/api/camera/{camera_id}/info`

Get camera information (position, rotation, FOV, resolution, etc.)

## Usage Examples

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Spawn a humanoid agent
response = requests.post(f"{BASE_URL}/api/agents/spawn", json={
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
})
agent_id = response.json()["data"]["agent_id"]

# 2. Move the agent forward
requests.post(f"{BASE_URL}/api/humanoid/move", json={
    "agent_id": agent_id,
    "action": "step_forward",
    "duration": 2.0
})

# 3. Rotate the agent
requests.post(f"{BASE_URL}/api/humanoid/rotate", json={
    "agent_id": agent_id,
    "angle": 90,
    "direction": "left"
})

# 4. Get a camera image
response = requests.get(f"{BASE_URL}/api/camera/1/image?viewmode=lit")
image_data = response.json()["image"]  # base64-encoded image
```

### curl Example

```bash
# Spawn an agent
curl -X POST "http://localhost:8000/api/agents/spawn" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
  }'

# Move the agent
curl -X POST "http://localhost:8000/api/humanoid/move" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": 0,
    "action": "step_forward",
    "duration": 2.0
  }'
```

## Integration with External Systems

### Calling from clawbot

clawbot can call these APIs via HTTP requests:

```python
# In clawbot
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

## Notes

1. **Make sure Unreal Engine is running**: The API server will try to connect to Unreal Engine when starting up. Please ensure UE is running and UnrealCV is enabled.
2. **Port configuration**: The default API port is 8000, and the UnrealCV port is 9000. Make sure these ports are not occupied.
3. **CORS**: The API server has CORS enabled, allowing cross-origin access.
4. **Asynchronous operations**: Some actions (such as movement and rotation) may take time to complete. The API will wait until the action is completed before returning.

## Troubleshooting

* **Connection failed**: Check whether Unreal Engine is running and whether UnrealCV is enabled.
* **Agent not found**: Make sure the `agent_id` is correct and the agent was successfully spawned.
* **Action execution failed**: Check whether the agent type matches the endpoint (`humanoid` vs `dog`).