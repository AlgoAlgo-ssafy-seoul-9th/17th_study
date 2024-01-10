## 백준 1074_ Z

import sys
input = sys.stdin.readline

def solve(n, r, c):
    if n == 0:
        return 0

    half = 2 ** (n-1)
    quad = 0

    if r >= half:
        quad += 2   # 행은 다음 행으로 갈 때 사분할 2번 지나감
        r -= half   

    if c >= half:
        quad += 1
        c -= half   # 열은 다음 열로 갈 때 사분할 1번 지나감

    return quad * half * half + solve(n-1, r, c)


N, r, c = map(int, input().split())
result = solve(N, r, c)
print(result)

