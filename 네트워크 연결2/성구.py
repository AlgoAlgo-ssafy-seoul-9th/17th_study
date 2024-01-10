# 1922 네트워크 연결 - Prim 
import sys, heapq
input = sys.stdin.readline


def prim(start:int) -> int:
    que = [(0, start)]
    heapq.heapify(que)
    visited = [0] * (N+1)
    total_w = cnt = 0
    while que:
        w, spot = heapq.heappop(que)
        if visited[spot]:
            continue
        visited[spot] = 1
        total_w += w
        for next, nw in graph[spot]:
            if not visited[next]:
                heapq.heappush(que, (nw, next))
    return total_w


if __name__ == "__main__":
    N = int(input())
    M = int(input())
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b, c = map(int, input().split())
        graph[a].append((b,c))
        graph[b].append((a,c))
    ans = prim(1)
    print(ans)