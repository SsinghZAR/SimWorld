"""Example client code showing how to use the SimWorld API from external systems."""
import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the SimWorld API."""
    
    # 1. Check health
    print("1. Checking health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health: {response.json()}")
    
    # 2. Spawn a humanoid agent
    print("\n2. Spawning humanoid agent...")
    spawn_request = {
        "agent_type": "humanoid",
        "position": [0, 0],
        "direction": [1, 0],
        "model_path": "/Game/Human_Avatar/DefaultCharacter/Blueprint/BP_Default_Character.BP_Default_Character_C"
    }
    response = requests.post(f"{BASE_URL}/api/agents/spawn", json=spawn_request)
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if result.get("success"):
        agent_id = result["data"]["agent_id"]
        print(f"Agent spawned with ID: {agent_id}")
        
        # 3. List all agents
        print("\n3. Listing all agents...")
        response = requests.get(f"{BASE_URL}/api/agents")
        print(f"Agents: {json.dumps(response.json(), indent=2)}")
        
        # 4. Move the humanoid forward
        print("\n4. Moving humanoid forward...")
        move_request = {
            "agent_id": agent_id,
            "action": "step_forward",
            "duration": 2.0,
            "direction": 0
        }
        response = requests.post(f"{BASE_URL}/api/humanoid/move", json=move_request)
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 5. Rotate the humanoid
        print("\n5. Rotating humanoid...")
        rotate_request = {
            "agent_id": agent_id,
            "angle": 90,
            "direction": "left"
        }
        response = requests.post(f"{BASE_URL}/api/humanoid/rotate", json=rotate_request)
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 6. Get agent info
        print("\n6. Getting agent info...")
        response = requests.get(f"{BASE_URL}/api/agents/{agent_id}")
        print(f"Agent info: {json.dumps(response.json(), indent=2)}")
        
        # 7. Test dog spawn and control
        print("\n7. Spawning dog agent...")
        dog_spawn_request = {
            "agent_type": "dog",
            "position": [100, 100, 20],
            "direction": [0, 1],
            "name": "Demo_Robot"
        }
        response = requests.post(f"{BASE_URL}/api/agents/spawn", json=dog_spawn_request)
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("success"):
            dog_id = result["data"]["agent_id"]
            
            # Move dog
            print("\n8. Moving dog...")
            dog_move_request = {
                "agent_id": dog_id,
                "speed": 200,
                "duration": 1.0,
                "direction": 0  # forward
            }
            response = requests.post(f"{BASE_URL}/api/dog/move", json=dog_move_request)
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            # Dog look up
            print("\n9. Dog looking up...")
            response = requests.post(f"{BASE_URL}/api/dog/look?agent_id={dog_id}&direction=up")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 10. Clean up - destroy agents
        print("\n10. Destroying agents...")
        response = requests.delete(f"{BASE_URL}/api/agents/{agent_id}")
        print(f"Destroy agent {agent_id}: {json.dumps(response.json(), indent=2)}")
        
        if result.get("success"):
            response = requests.delete(f"{BASE_URL}/api/agents/{dog_id}")
            print(f"Destroy agent {dog_id}: {json.dumps(response.json(), indent=2)}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Please make sure the API server is running:")
        print("  python -m simworld.api.server")
        print("  or")
        print("  python examples/api_server_example.py")
    except Exception as e:
        print(f"Error: {e}")
