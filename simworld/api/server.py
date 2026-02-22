"""FastAPI server for SimWorld external control."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import base64
import io

from simworld.communicator.unrealcv import UnrealCV
from simworld.communicator.communicator import Communicator
from simworld.agent.humanoid import Humanoid
from simworld.agent.scooter import Scooter
from simworld.utils.vector import Vector
from simworld.api.models import (
    AgentSpawnRequest, AgentInfo, Response,
    HumanoidMoveRequest, HumanoidRotateRequest, HumanoidInteractionRequest,
    DogMoveRequest, DogRotateRequest, CameraRequest
)
from simworld.api.agent_manager import AgentManager

app = FastAPI(
    title="SimWorld API",
    description="REST API for controlling SimWorld agents",
    version="1.0.0"
)

# Enable CORS for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
communicator: Optional[Communicator] = None
agent_manager: Optional[AgentManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize SimWorld connection on startup."""
    global communicator, agent_manager
    try:
        print("正在连接 UnrealCV...")
        ucv = UnrealCV()
        print("UnrealCV 连接成功")
        print("正在初始化 Communicator...")
        communicator = Communicator(ucv)
        print("Communicator 初始化成功")
        print("正在初始化 AgentManager...")
        agent_manager = AgentManager(communicator)
        print("AgentManager 初始化成功")
        print("✓ SimWorld API 服务器已完全启动")
    except Exception as e:
        print(f"✗ 错误: 无法连接到 Unreal Engine: {e}")
        print("请确保:")
        print("  1. Unreal Engine 正在运行")
        print("  2. UnrealCV 插件已启用")
        print("  3. UnrealCV 在 9000 端口监听")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常，让服务器知道启动失败


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global communicator
    if communicator:
        communicator.disconnect()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SimWorld API Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    global communicator
    if communicator and communicator.unrealcv:
        connected = communicator.unrealcv.client.isconnected()
        return {
            "status": "healthy" if connected else "disconnected",
            "unreal_engine_connected": connected
        }
    return {"status": "uninitialized"}


# Agent Management Endpoints

@app.post("/api/agents/spawn", response_model=Response)
async def spawn_agent(request: AgentSpawnRequest):
    """Spawn a new agent in the simulation.
    
    Args:
        request: Agent spawn request
        
    Returns:
        Agent information
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    try:
        # Create agent object
        position = Vector(request.position[0], request.position[1])
        direction = Vector(request.direction[0], request.direction[1])
        
        if request.agent_type == "humanoid":
            agent = Humanoid(position, direction)
            model_path = request.model_path or '/Game/Human_Avatar/DefaultCharacter/Blueprint/BP_Default_Character.BP_Default_Character_C'
            agent_name = request.name or f'GEN_BP_Humanoid_{agent.id}'
        elif request.agent_type == "dog":
            # For dog, we need to spawn it differently
            agent_name = request.name or "Demo_Robot"
            model_path = request.model_path or "/Game/Robot_Dog/Blueprint/BP_SpotRobot.BP_SpotRobot_C"
            
            # Spawn the dog directly
            communicator.unrealcv.spawn_bp_asset(model_path, agent_name)
            z = request.position[2] if len(request.position) > 2 else 20
            communicator.unrealcv.set_location((request.position[0], request.position[1], z), agent_name)
            communicator.unrealcv.enable_controller(agent_name, True)
            
            # Create a simple agent object for tracking
            agent = type('DogAgent', (), {
                'id': agent_manager._next_agent_id,
                'position': position,
                'direction': direction,
                'camera_id': None
            })()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}")
        
        # Spawn agent in Unreal Engine
        if request.agent_type == "humanoid":
            communicator.spawn_agent(agent=agent, name=agent_name, model_path=model_path)
        
        # Register agent
        agent_id = agent_manager.register_agent(agent, request.agent_type, agent_name)
        
        agent_info = agent_manager.get_agent_info(agent_id)
        
        return Response(
            success=True,
            message=f"Agent spawned successfully",
            data=agent_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents", response_model=Response)
async def list_agents():
    """List all registered agents.
    
    Returns:
        List of agent information
    """
    global agent_manager
    
    if not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agents = agent_manager.list_agents()
    return Response(
        success=True,
        message=f"Found {len(agents)} agents",
        data=agents
    )


@app.get("/api/agents/{agent_id}", response_model=Response)
async def get_agent(agent_id: int):
    """Get agent information by ID.
    
    Args:
        agent_id: Agent ID
        
    Returns:
        Agent information
    """
    global agent_manager
    
    if not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_info = agent_manager.get_agent_info(agent_id)
    if not agent_info:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return Response(
        success=True,
        message="Agent found",
        data=agent_info
    )


@app.delete("/api/agents/{agent_id}", response_model=Response)
async def destroy_agent(agent_id: int):
    """Destroy an agent.
    
    Args:
        agent_id: Agent ID
        
    Returns:
        Success response
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_data = agent_manager.get_agent(agent_id)
    if not agent_data:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    try:
        agent_name = agent_data['name']
        communicator.unrealcv.destroy(agent_name)
        del agent_manager.agents[agent_id]
        
        return Response(
            success=True,
            message=f"Agent {agent_id} destroyed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Humanoid Action Endpoints

@app.post("/api/humanoid/move", response_model=Response)
async def humanoid_move(request: HumanoidMoveRequest):
    """Control humanoid movement.
    
    Args:
        request: Movement request
        
    Returns:
        Success response
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_data = agent_manager.get_agent(request.agent_id)
    if not agent_data or agent_data['agent_type'] != 'humanoid':
        raise HTTPException(status_code=404, detail=f"Humanoid agent {request.agent_id} not found")
    
    try:
        if request.action == "move_forward":
            communicator.humanoid_move_forward(request.agent_id)
        elif request.action == "step_forward":
            duration = request.duration or 1.0
            communicator.humanoid_step_forward(request.agent_id, duration, request.direction)
        elif request.action == "stop":
            communicator.humanoid_stop(request.agent_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
        
        return Response(
            success=True,
            message=f"Action '{request.action}' executed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/humanoid/rotate", response_model=Response)
async def humanoid_rotate(request: HumanoidRotateRequest):
    """Rotate humanoid.
    
    Args:
        request: Rotation request
        
    Returns:
        Success response
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_data = agent_manager.get_agent(request.agent_id)
    if not agent_data or agent_data['agent_type'] != 'humanoid':
        raise HTTPException(status_code=404, detail=f"Humanoid agent {request.agent_id} not found")
    
    try:
        communicator.humanoid_rotate(request.agent_id, request.angle, request.direction)
        return Response(
            success=True,
            message=f"Rotated {request.angle} degrees {request.direction}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/humanoid/interact", response_model=Response)
async def humanoid_interact(request: HumanoidInteractionRequest):
    """Humanoid interaction actions.
    
    Args:
        request: Interaction request
        
    Returns:
        Success response
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_data = agent_manager.get_agent(request.agent_id)
    if not agent_data or agent_data['agent_type'] != 'humanoid':
        raise HTTPException(status_code=404, detail=f"Humanoid agent {request.agent_id} not found")
    
    try:
        action = request.action
        
        if action == "pick_up":
            if not request.object_name:
                raise HTTPException(status_code=400, detail="object_name required for pick_up")
            communicator.humanoid_pick_up_object(request.agent_id, request.object_name)
        elif action == "drop":
            communicator.humanoid_drop_object(request.agent_id)
        elif action == "sit_down":
            communicator.humanoid_sit_down(request.agent_id)
        elif action == "stand_up":
            communicator.humanoid_stand_up(request.agent_id)
        elif action == "argue":
            communicator.unrealcv.humanoid_argue(agent_data['name'], 0)
        elif action == "discuss":
            communicator.unrealcv.humanoid_discuss(agent_data['name'], 0)
        elif action == "listen":
            communicator.unrealcv.humanoid_listen(agent_data['name'])
        elif action == "wave_to_dog":
            communicator.unrealcv.humanoid_wave_to_dog(agent_data['name'])
        elif action == "directing_path":
            communicator.unrealcv.humanoid_directing_path(agent_data['name'])
        elif action == "stop_action":
            communicator.unrealcv.humanoid_stop_current_action(agent_data['name'])
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
        
        return Response(
            success=True,
            message=f"Action '{action}' executed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dog Action Endpoints

@app.post("/api/dog/move", response_model=Response)
async def dog_move(request: DogMoveRequest):
    """Control dog movement.
    
    Args:
        request: Movement request
        
    Returns:
        Success response
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_data = agent_manager.get_agent(request.agent_id)
    if not agent_data or agent_data['agent_type'] != 'dog':
        raise HTTPException(status_code=404, detail=f"Dog agent {request.agent_id} not found")
    
    try:
        move_parameter = [request.speed, request.duration, request.direction]
        communicator.unrealcv.dog_move(agent_data['name'], move_parameter)
        
        return Response(
            success=True,
            message="Dog move action executed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dog/rotate", response_model=Response)
async def dog_rotate(request: DogRotateRequest):
    """Rotate dog.
    
    Args:
        request: Rotation request
        
    Returns:
        Success response
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_data = agent_manager.get_agent(request.agent_id)
    if not agent_data or agent_data['agent_type'] != 'dog':
        raise HTTPException(status_code=404, detail=f"Dog agent {request.agent_id} not found")
    
    try:
        rotate_parameter = [request.duration, request.angle, request.clockwise]
        communicator.unrealcv.dog_rotate(agent_data['name'], rotate_parameter)
        
        return Response(
            success=True,
            message="Dog rotate action executed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dog/look", response_model=Response)
async def dog_look(agent_id: int, direction: str):
    """Dog look up/down.
    
    Args:
        agent_id: Agent ID
        direction: 'up' or 'down'
        
    Returns:
        Success response
    """
    global communicator, agent_manager
    
    if not communicator or not agent_manager:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    agent_data = agent_manager.get_agent(agent_id)
    if not agent_data or agent_data['agent_type'] != 'dog':
        raise HTTPException(status_code=404, detail=f"Dog agent {agent_id} not found")
    
    try:
        if direction == "up":
            communicator.unrealcv.dog_look_up(agent_data['name'])
        elif direction == "down":
            communicator.unrealcv.dog_look_down(agent_data['name'])
        else:
            raise HTTPException(status_code=400, detail="direction must be 'up' or 'down'")
        
        return Response(
            success=True,
            message=f"Dog look {direction} executed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Camera Endpoints

@app.get("/api/camera/{camera_id}/image")
async def get_camera_image(camera_id: int, viewmode: str = "lit"):
    """Get camera image.
    
    Args:
        camera_id: Camera ID
        viewmode: View mode ('lit', 'depth', 'object_mask')
        
    Returns:
        Base64 encoded image
    """
    global communicator
    
    if not communicator:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    try:
        image = communicator.get_camera_observation(camera_id, viewmode)
        
        # Convert image to base64
        import cv2
        import numpy as np
        from PIL import Image
        
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return {
                "success": True,
                "image": f"data:image/png;base64,{img_base64}",
                "camera_id": camera_id,
                "viewmode": viewmode
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to get image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/camera/{camera_id}/info")
async def get_camera_info(camera_id: int):
    """Get camera information.
    
    Args:
        camera_id: Camera ID
        
    Returns:
        Camera information
    """
    global communicator
    
    if not communicator:
        raise HTTPException(status_code=503, detail="SimWorld not initialized")
    
    try:
        location = communicator.unrealcv.get_camera_location(camera_id)
        rotation = communicator.unrealcv.get_camera_rotation(camera_id)
        fov = communicator.unrealcv.get_camera_fov(camera_id)
        resolution = communicator.unrealcv.get_camera_resolution(camera_id)
        
        return {
            "success": True,
            "camera_id": camera_id,
            "location": location.tolist() if hasattr(location, 'tolist') else list(location),
            "rotation": rotation.tolist() if hasattr(rotation, 'tolist') else list(rotation),
            "fov": fov,
            "resolution": resolution.tolist() if hasattr(resolution, 'tolist') else list(resolution)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)