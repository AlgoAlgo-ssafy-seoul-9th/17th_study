## 백준_ 1922 _ 네트워크 연결

import sys
input = sys.stdin.readline

# 루트노드 찾기
def findset(node):
    while parent[node] != node:
        node = parent[node]
    return node

# 루트노드 갱신
def union(x, y):
    parent[findset(y)] = findset(x)     # y의 루트노드를 x의 루트노드로 갱신

# Kruskal
N = int(input())
M = int(input())
graph = []

# 각 정점에 대한 부모 초기화
parent = [i for i in range(N+1)]

for _ in range(M):
    a, b, w = map(int, input().split())
    graph.append([a, b, w])

# 그래프의 간선을 가중치에 따라 오름차순으로 정렬
graph.sort(key=lambda x:x[2])

cnt = 0
cost = 0

for edge in graph:
    u, v, weight = edge
    # 사이클이 형성되지 않는 경우에만 간선 선택
    # 루트 노드가 같지 않으면 이어져 있지 않은 상태
    if findset(u) != findset(v):
        cnt += 1        # 추가된 간선 수
        cost += weight  # cost 더해주고
        union(u, v)     # 이어졌으니까 루트노드 갱신해주고
        if cnt == N:
            break
print(cost)



