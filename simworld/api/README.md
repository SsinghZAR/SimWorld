# SimWorld API Module

This module provides a REST API interface that allows external systems (such as clawbot) to control agents in SimWorld via HTTP requests.

## Quick Start

### 1. Start the API Server

```bash id="x1fl74"
cd SimWorld
python3 examples/api_server_example.py
```

The server will start at `http://localhost:8000`.

### 2. Connect from an External System (e.g., clawbot)

clawbot only needs to send HTTP requests. For example:

**Spawn an agent:**

```bash id="1q122n"
curl -X POST "http://localhost:8000/api/agents/spawn" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
  }'
```

**Move an agent:**

```bash id="tzmfgu"
curl -X POST "http://localhost:8000/api/humanoid/move" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": 0,
    "action": "step_forward",
    "duration": 2.0
  }'
```

**Rotate an agent:**

```bash id="omgqbo"
curl -X POST "http://localhost:8000/api/humanoid/rotate" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": 0,
    "angle": 90,
    "direction": "left"
  }'
```

## Main API Endpoints

* **Spawn agent**: `POST /api/agents/spawn`
* **List agents**: `GET /api/agents`
* **Humanoid movement**: `POST /api/humanoid/move`
* **Humanoid rotation**: `POST /api/humanoid/rotate`
* **Humanoid interaction**: `POST /api/humanoid/interact`
* **Dog movement**: `POST /api/dog/move`
* **Dog rotation**: `POST /api/dog/rotate`
* **Get camera image**: `GET /api/camera/{camera_id}/image`

## Full API Documentation

* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs) (available after the server starts)
* **Detailed documentation**: see `docs/API_USAGE.md`

## Notes

1. Make sure Unreal Engine is running and UnrealCV is enabled.
2. The default port for the API server is 8000.
3. All APIs support cross-origin access (CORS is enabled).