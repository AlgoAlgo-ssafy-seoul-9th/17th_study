import heapq


def prim(graph,start,V):
    min_heap = [(0,start)] # 가중치와 정점
    total_cost = 0

    while min_heap:
        weight,node = heapq.heappop(min_heap)
        if not V[node]:
            V[node] = 1
            total_cost += weight
            for v,w in graph[node]:
                if not V[v]:
                    heapq.heappush(min_heap,(w,v))
    return total_cost

n = int(input())
m = int(input())
graph = [[] for _ in range(n+1)]
v = [0] * (n+1)
cost_list = [10000] * (n+1)
for _ in range(m):
    a,b,c = map(int,input().split())
    graph[a].append([b,c])
    graph[b].append([a,c])

cost = prim(graph, 1, v)
print(cost)


