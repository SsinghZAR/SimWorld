import random

def find_random_cycle(graph, start_node, min_len, max_len, max_attempts=1000):
    """
    从 start_node 出发，随机采样一个满足长度要求的回路。
    """
    def dfs(current, path, visited):
        if len(path) > max_len:
            return None
        neighbors = graph[current][:]
        random.shuffle(neighbors)  # 打乱邻居顺序增加随机性
        for neighbor in neighbors:
            if neighbor == start_node and len(path) >= min_len:
                return path + [start_node]
            if neighbor not in visited:
                result = dfs(neighbor, path + [neighbor], visited | {neighbor})
                if result:
                    return result
        return None

    for _ in range(max_attempts):
        result = dfs(start_node, [start_node], {start_node})
        if result:
            return result
    return None



# 示例用法：
if __name__ == "__main__":
    # 构建一个简单无向图
    graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D', 'E'],
    'C': ['A', 'B', 'F'],
    'D': ['B', 'E'],
    'E': ['B', 'D', 'F', 'G'],
    'F': ['C', 'E', 'G'],
    'G': ['E', 'F', 'H'],
    'H': ['G']
}


    start_node = 'B'
    min_len = 4
    max_len = 6

    cycle = find_random_cycle(graph, start_node, min_len, max_len)
    if cycle:
        print("找到回路:", cycle)
    else:
        print("未找到满足条件的回路。")
