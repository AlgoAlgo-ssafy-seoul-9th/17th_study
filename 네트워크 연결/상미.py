## 백준 1922 _ 네트워크 연결
## 답 안 나옴

import sys
input = sys.stdin.readline

N = int(input())
M = int(input())
arr = [[0] * (N+1) for _ in range(N+1)]
for _ in range(M):
    a, b, c = map(int, input().split())
    arr[a][b] = c
    arr[b][a] = c
visited = [0] * (N+1)
minC = 100000000

def sol(arr, tmp, cost):
    visited[tmp] = 1    # 방문 처리
    if visited == [1] * (N+1):  # 모두 다 방문했으면
        if minC > cost:
            minC = cost
        return
    for i in range(N+1):    # 연결되어 있고 방문 안 한 곳이라면
        if arr[tmp][i] and not visited[i]:
            sol(arr, i, cost + arr[tmp][i]) # 방문
        else:
            return
        
print(sol(arr, 1, 0))
print(minC)