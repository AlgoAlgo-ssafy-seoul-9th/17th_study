# 1922_네트워크연결_network-connecting
import sys
input = sys.stdin.readline
# 4 2 6 3 8

def findset(node):
    while parent[node] != node:
        node = parent[node]
    return node


def union(x, y):
    parent[findset(y)] = findset(x)


N = int(input().strip())
M = int(input().strip())

graph = []
parent = [i for i in range(N+1)]
for _ in range(M):
    s, g, w = map(int, input().split())
    graph.append([s, g, w])

graph.sort(key=lambda x: x[2])

cost = 0
cnt = 0
for g in graph:
    u, v, w = g
    if findset(u) != findset(v):
        cnt += 1
        cost += w
        union(u, v)
        if cnt == N:
            break

# print(parent)
print(cost)