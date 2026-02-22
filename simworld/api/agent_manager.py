"""Agent manager for tracking and managing agents."""
from typing import Dict, Optional
from simworld.communicator.communicator import Communicator


class AgentManager:
    """Manages agents and their metadata."""
    
    def __init__(self, communicator: Communicator):
        """Initialize the agent manager.
        
        Args:
            communicator: Communicator instance for SimWorld.
        """
        self.communicator = communicator
        self.agents: Dict[int, Dict] = {}  # agent_id -> agent info
        self._next_agent_id = 0
    
    def register_agent(self, agent, agent_type: str, name: Optional[str] = None) -> int:
        """Register an agent and return its ID.
        
        Args:
            agent: Agent object (Humanoid, etc.)
            agent_type: Type of agent ('humanoid', 'dog', etc.)
            name: Optional custom name
            
        Returns:
            Agent ID
        """
        agent_id = self._next_agent_id
        self._next_agent_id += 1
        
        agent_name = name or f"{agent_type}_{agent_id}"
        
        self.agents[agent_id] = {
            'agent': agent,
            'agent_type': agent_type,
            'name': agent_name,
            'agent_id': agent_id,
            'camera_id': getattr(agent, 'camera_id', None)
        }
        
        return agent_id
    
    def get_agent(self, agent_id: int) -> Optional[Dict]:
        """Get agent information by ID.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent info dict or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agent_info(self, agent_id: int) -> Optional[Dict]:
        """Get agent information for API response.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent info dict with position and direction
        """
        agent_data = self.agents.get(agent_id)
        if not agent_data:
            return None
        
        agent = agent_data['agent']
        position = [agent.position.x, agent.position.y]
        direction = [agent.direction.x, agent.direction.y]
        
        return {
            'agent_id': agent_id,
            'agent_type': agent_data['agent_type'],
            'name': agent_data['name'],
            'position': position,
            'direction': direction,
            'camera_id': agent_data.get('camera_id')
        }
    
    def list_agents(self) -> list:
        """List all registered agents.
        
        Returns:
            List of agent info dicts
        """
        return [self.get_agent_info(agent_id) for agent_id in self.agents.keys()]
