## Clawbot × SimWorld — Quick Start

```bash
# 1) Start SimWorld (Unreal + UnrealCV) so the city is running (UnrealCV port must match, e.g., 9000)
cd $UE_PATH/Linux-new && bash ./gym_citynav.sh DefaultMap -RenderOffscreen

# 2) Start the REST API server (FastAPI) that Clawbot will call
cd SimWorld && python3 examples/api_server_example.py  # Swagger: http://localhost:8000/docs

# 3) Spawn a humanoid agent via REST (creates agent_id / camera_id)
curl -X POST http://localhost:8000/api/agents/spawn -H "Content-Type: application/json" -d '{"agent_type":"humanoid","position":[0,0],"direction":[1,0]}'

# 4) Move the agent forward for 2 seconds (control via REST)
curl -X POST http://localhost:8000/api/humanoid/move -H "Content-Type: application/json" -d '{"agent_id":0,"action":"step_forward","duration":2.0}'

# 5) Register the Clawbot skill so Clawbot knows these endpoints (then prompt Clawbot to do spawn→move→rotate→camera)
mkdir -p ~/.openclaw/skills/SimWorld && printf "Base URL: http://localhost:8000\nPOST /api/agents/spawn\nPOST /api/humanoid/move\nPOST /api/humanoid/rotate\nGET /api/camera/{camera_id}/image\n" > ~/.openclaw/skills/SimWorld/skill.md

```