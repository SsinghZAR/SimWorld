"""最简单的测试：直接发送HTTP请求生成agent"""
import requests
import json

API_URL = "http://localhost:8000"

print("=" * 50)
print("SimWorld 简单测试 - 生成agent在(0,0)")
print("=" * 50)

# 1. 检查服务器
print("\n1. 检查服务器状态...")
try:
    health = requests.get(f"{API_URL}/health", timeout=2)
    print(f"   状态码: {health.status_code}")
    if health.status_code == 200:
        health_data = health.json()
        print(f"   状态: {health_data.get('status')}")
        if health_data.get('unreal_engine_connected'):
            print("   ✓ Unreal Engine 已连接")
        else:
            print("   ⚠ Unreal Engine 未连接（但可以继续测试）")
except requests.exceptions.ConnectionError:
    print("   ✗ 无法连接到服务器")
    print("   请确保API服务器正在运行:")
    print("     cd /home/lingjun/SimWorld/SimWorld")
    print("     python3 examples/api_server_example.py")
    exit(1)
except Exception as e:
    print(f"   ✗ 错误: {e}")
    exit(1)

# 2. 生成agent
print("\n2. 发送请求生成agent...")
spawn_request = {
    "agent_type": "humanoid",
    "position": [0, 0],
    "direction": [1, 0]
}

try:
    print(f"   请求: POST {API_URL}/api/agents/spawn")
    print(f"   数据: {json.dumps(spawn_request, indent=6)}")
    
    response = requests.post(
        f"{API_URL}/api/agents/spawn",
        json=spawn_request,
        timeout=10
    )
    
    print(f"   响应状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   响应: {json.dumps(result, indent=6, ensure_ascii=False)}")
        
        if result.get("success"):
            agent_info = result["data"]
            print("\n" + "=" * 50)
            print("✓ 成功！Agent已生成")
            print("=" * 50)
            print(f"Agent ID: {agent_info['agent_id']}")
            print(f"位置: {agent_info['position']}")
            print(f"方向: {agent_info['direction']}")
            print(f"名称: {agent_info['name']}")
            if agent_info.get('camera_id'):
                print(f"相机ID: {agent_info['camera_id']}")
        else:
            print("\n✗ 生成失败")
            print(f"错误信息: {result.get('message')}")
    else:
        print(f"\n✗ HTTP错误: {response.status_code}")
        print(f"响应内容: {response.text}")
        
except requests.exceptions.Timeout:
    print("   ✗ 请求超时")
except Exception as e:
    print(f"   ✗ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
