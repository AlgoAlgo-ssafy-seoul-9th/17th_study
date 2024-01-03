import sys
from itertools import combinations
input = sys.stdin.readline

n, m = map(int, input().split().strip())
field = [list(map(int, input().split())) for _ in range(n)]
tmp_field = [[0 for _ in range(n)] for _ in range(n)]

hos = 0
hos_cord = []
peo_cord = []
ans = float('inf')

for i in range(n):
    for j in range(n):
        if field[i][j] == 2:
            hos += 1
            hos_cord.append([i, j])
        elif field[i][j] == 1:
            tmp_field[i][j] = 1
            peo_cord.append([i, j])

num_lst = [i for i in range(hos)]
comb = combinations(num_lst, m)
for c in comb:
    tmp_hos = []
    for v in c:
        tmp_hos.append(hos_cord[v])
    tmp_ans = 0
    for p in peo_cord:
        dis = float('inf')
        for th in tmp_hos:
            dis = min(dis, abs(th[0] - p[0]) + abs(th[1] - p[1]))
        tmp_ans += dis
        if tmp_ans > ans:
            break
    else:
        ans = tmp_ans
print(ans)
