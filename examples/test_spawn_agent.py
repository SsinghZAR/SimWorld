"""简单的测试脚本：生成一个在(0,0)位置的agent"""
import requests
import time
import sys
from pathlib import Path

# Add the parent directory to Python path if needed
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

API_URL = "http://localhost:8000"

def check_server():
    """检查API服务器是否运行"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def wait_for_server(max_wait=30):
    """等待服务器启动"""
    print("等待API服务器启动...")
    for i in range(max_wait):
        if check_server():
            print("✓ API服务器已就绪")
            return True
        time.sleep(1)
        print(f"  等待中... ({i+1}/{max_wait})")
    return False

def spawn_agent():
    """生成一个在(0,0)位置的humanoid agent"""
    print("\n正在生成agent...")
    
    spawn_request = {
        "agent_type": "humanoid",
        "position": [0, 0],
        "direction": [1, 0]  # 朝向x轴正方向
    }
    
    try:
        response = requests.post(
            f"{API_URL}/api/agents/spawn",
            json=spawn_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                agent_info = result["data"]
                print(f"✓ Agent生成成功！")
                print(f"  Agent ID: {agent_info['agent_id']}")
                print(f"  位置: {agent_info['position']}")
                print(f"  方向: {agent_info['direction']}")
                print(f"  名称: {agent_info['name']}")
                if agent_info.get('camera_id'):
                    print(f"  相机ID: {agent_info['camera_id']}")
                return True
            else:
                print(f"✗ 生成失败: {result.get('message')}")
                return False
        else:
            print(f"✗ HTTP错误: {response.status_code}")
            print(f"  响应: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到API服务器")
        print("  请确保API服务器正在运行:")
        print("    python examples/api_server_example.py")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("SimWorld Agent 生成测试")
    print("=" * 50)
    
    # 检查服务器
    if not check_server():
        print("\n⚠ API服务器未运行")
        print("\n请先启动API服务器:")
        print("  python examples/api_server_example.py")
        print("\n或者使用uvicorn:")
        print("  uvicorn simworld.api.server:app --host 0.0.0.0 --port 8000")
        print("\n然后再次运行此脚本")
        sys.exit(1)
    
    # 检查健康状态
    try:
        health = requests.get(f"{API_URL}/health").json()
        print(f"\n服务器状态: {health.get('status')}")
        if health.get('unreal_engine_connected'):
            print("✓ Unreal Engine 已连接")
        else:
            print("⚠ Unreal Engine 未连接")
            print("  请确保Unreal Engine正在运行并启用了UnrealCV")
    except:
        pass
    
    # 生成agent
    success = spawn_agent()
    
    if success:
        print("\n" + "=" * 50)
        print("测试完成！Agent已成功生成在(0,0)位置")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("测试失败")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()
