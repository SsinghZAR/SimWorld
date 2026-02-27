# Clawbot × SimWorld Integration

This guide walks you through **connecting Clawbot to SimWorld via the SimWorld REST API** and running an end-to-end demo:
1) launch SimWorld (Unreal + UnrealCV)  
2) launch the FastAPI REST server  
3) verify with Swagger/cURL  
4) install/register a Clawbot skill (`skill.md`) that calls the REST endpoints  
5) run Clawbot to control agents in SimWorld (spawn → move/rotate → camera)

## Architecture overview (how the pieces talk)

Clawbot ──(HTTP REST)──> FastAPI server ──(UnrealCV TCP)──> SimWorld (Unreal)

- FastAPI listens on `http://localhost:8000` (default)
- UnrealCV typically listens on `127.0.0.1:9000` (common default, but must match your config)

## Step 1 — Start SimWorld (Unreal) with UnrealCV enabled

Example (server/headless/offscreen style; adapt to your environment):

```bash
# Example path from team notes (adapt to your actual environment)
cd $UE_PATH/Linux-new
bash ./gym_citynav.sh DefaultMap -RenderOffscreen
```
You should see logs indicating UnrealCV is loaded/enabled.
If you see connection errors later, this is the first thing to verify:
- Unreal is running
- UnrealCV plugin is enabled
- UnrealCV is listening on the configured port

## Step 2 — Start the FastAPI server (SimWorld REST API)

From the SimWorld repo:

```bash
cd SimWorld
python3 examples/api_server_example.py
```

Expected:

- Server starts at: http://localhost:8000
- Swagger UI is available at: http://localhost:8000/docs

If you run this on a remote server and want to access it from your laptop, you’ll likely need port forwarding (see Remote / server mode).

## Step 3 — Verify the API is alive (Swagger + curl)
3.1 Swagger smoke test
Open:

- http://localhost:8000/docs

Try:

- `POST /api/agents/spawn`

- `POST /api/humanoid/move`

- `GET /api/agents`

- `GET /api/camera/{camera_id}/image`

3.2 CURL smoke test (spawn)
```bash
curl -X POST "http://localhost:8000/api/agents/spawn" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
  }'
  ```
Expected response example:

```json
{
  "success": true,
  "message": "Agent spawned successfully",
  "data": {
    "agent_id": 0,
    "agent_type": "humanoid",
    "name": "GEN_BP_Humanoid_0",
    "position": [0.0, 0.0],
    "direction": [1.0, 0.0],
    "camera_id": 1
  }
}
```
3.3 CURL move/rotate
```bash
curl -X POST "http://localhost:8000/api/humanoid/move" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": 0, "action": "step_forward", "duration": 2.0}'

curl -X POST "http://localhost:8000/api/humanoid/rotate" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": 0, "angle": 90, "direction": "left"}'
  ```
If these work, your SimWorld ↔ API ↔ UnrealCV chain is healthy.

## Step 4 — Add a Clawbot skill that calls the SimWorld REST API

Clawbot needs a skill markdown file that:

- defines the base URL

- lists endpoints

- provides request/response schema examples

- lists allowable actions (move/rotate/interact)

This helps Clawbot generate correct HTTP calls reliably.

4.1 Create the skill directory

Windows
```
.openclaw\skills\SimWorld\skill.md
```

Linux / macOS
```
~/.openclaw/skills/SimWorld/skill.md
```

`SimWorld` is the skill name; you can rename it, but keep it consistent.

4.2 `skill.md` template (recommended)

Copy-paste and adjust `Base URL` as needed:

```
# SimWorld Skill (REST API)

This skill lets Clawbot control SimWorld agents via HTTP requests.

## Base URL
Default: http://localhost:8000  
If the API server runs remotely, use http://<SERVER_HOST>:8000

## API Endpoints

### Spawn an agent
POST /api/agents/spawn  
Request JSON:
{
  "agent_type": "humanoid",
  "position": [0, 0],
  "direction": [1, 0]
}

Response JSON:
{
  "success": true,
  "data": {
    "agent_id": 0,
    "agent_type": "humanoid",
    "name": "...",
    "camera_id": 1
  }
}

Use data.agent_id for later actions.

### List agents
GET /api/agents

### Humanoid move
POST /api/humanoid/move  
Request JSON:
{
  "agent_id": 0,
  "action": "step_forward",
  "duration": 2.0
}

Allowed actions (typical):
- step_forward
- step_backward
- strafe_left
- strafe_right

(If your server supports more, list them here.)

### Humanoid rotate
POST /api/humanoid/rotate  
Request JSON:
{
  "agent_id": 0,
  "angle": 90,
  "direction": "left"
}

direction in {left, right}

### Humanoid interact
POST /api/humanoid/interact  
Request JSON:
{
  "agent_id": 0,
  "action": "interact"
}

(If target/object fields exist in your implementation, document them here.)

### Dog move / rotate
POST /api/dog/move  
POST /api/dog/rotate  

(Same schema style as humanoid.)

### Get camera image
GET /api/camera/{camera_id}/image
```

## Step 5 — Run Clawbot to control SimWorld

At this point:

- SimWorld UE is running + UnrealCV OK

- FastAPI is running on :8000

- Clawbot has a SimWorld skill markdown

Run Clawbot with an instruction like:

**Example task prompt**

“Spawn a humanoid at (0,0) facing (1,0). Move forward for 2s, rotate left 90 degrees, move forward 1s. Then fetch the camera image.”

Clawbot should:

- call `POST /api/agents/spawn`

- call `POST /api/humanoid/move`

- call `POST /api/humanoid/rotate`

- call `POST /api/humanoid/move`

- call `GET /api/camera/{camera_id}/image`