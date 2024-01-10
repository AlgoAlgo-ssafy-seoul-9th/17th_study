
# 1 > 2 > 3
# 3 부모 2 2부모 1

def findset(node):
    while parent[node] != node:
        node = parent[node]
    return node

# 3 부모 1 2부모 1 3 ,2  합쳐보자
def union(x,y):
    x = findset(x)
    y = findset(y)
    parent[y] = x

def kruskal(graph):
    graph.sort(key=lambda x: x[2])

    mst = []
    total_cost = 0
    for u,v,weight in graph:
        # 사이클이 아닌경우 간선 선택
        if findset(u) != findset(v):
            union(u,v)
            mst.append((u,v,weight))
            total_cost += weight
    return mst, total_cost

n = int(input())
m = int(input())
graph = []
for _ in range(m):
    a,b,c = map(int,input().split())
    graph.append([a,b,c])

parent = [i for i in range(n+1)]

a, b = kruskal(graph)
print(b)