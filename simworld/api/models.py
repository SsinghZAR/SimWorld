"""Pydantic models for API requests and responses."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentSpawnRequest(BaseModel):
    """Request model for spawning an agent."""
    agent_type: str = Field(..., description="Agent type: 'humanoid' or 'dog'")
    position: List[float] = Field(..., description="Initial position [x, y] or [x, y, z]")
    direction: List[float] = Field(..., description="Initial direction [x, y]")
    model_path: Optional[str] = Field(None, description="Custom model path (optional)")
    name: Optional[str] = Field(None, description="Custom agent name (optional)")


class AgentInfo(BaseModel):
    """Agent information response."""
    agent_id: int
    agent_type: str
    name: str
    position: List[float]
    direction: List[float]
    camera_id: Optional[int] = None


class ActionRequest(BaseModel):
    """Base action request model."""
    agent_id: int = Field(..., description="Agent ID")
    action_type: str = Field(..., description="Action type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class HumanoidMoveRequest(BaseModel):
    """Request for humanoid move actions."""
    agent_id: int
    action: str = Field(..., description="Action: 'move_forward', 'step_forward', 'stop'")
    duration: Optional[float] = Field(None, description="Duration for step_forward")
    direction: Optional[int] = Field(0, description="Direction for step_forward")


class HumanoidRotateRequest(BaseModel):
    """Request for humanoid rotation."""
    agent_id: int
    angle: float = Field(..., description="Rotation angle in degrees")
    direction: str = Field("left", description="Rotation direction: 'left' or 'right'")


class HumanoidInteractionRequest(BaseModel):
    """Request for humanoid interactions."""
    agent_id: int
    action: str = Field(..., description="Action: 'pick_up', 'drop', 'sit_down', 'stand_up', etc.")
    object_name: Optional[str] = Field(None, description="Object name for pick_up")


class DogMoveRequest(BaseModel):
    """Request for dog movement."""
    agent_id: int
    speed: float = Field(..., description="Movement speed")
    duration: float = Field(..., description="Movement duration")
    direction: int = Field(..., description="Direction: 0=forward, 1=backward, 2=left, 3=right")


class DogRotateRequest(BaseModel):
    """Request for dog rotation."""
    agent_id: int
    angle: float = Field(..., description="Rotation angle in degrees")
    duration: float = Field(0.7, description="Rotation duration")
    clockwise: int = Field(1, description="1 for clockwise, -1 for counter-clockwise")


class CameraRequest(BaseModel):
    """Request for camera operations."""
    camera_id: int
    action: str = Field(..., description="Action: 'get_image', 'set_location', 'set_rotation', etc.")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    """Standard API response."""
    success: bool
    message: str
    data: Optional[Any] = None
