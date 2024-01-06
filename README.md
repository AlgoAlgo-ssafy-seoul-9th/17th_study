# 17th_study

### 17주차 알고리즘스터디

# 지난주 문제

<details>
<summary>접기/펼치기</summary>
<div markdown="1">

## [병원거리 최소화하기](https://www.codetree.ai/training-field/frequent-problems/problems/min-of-hospital-distance/submissions?page=3&pageSize=20)

### [민웅](<./병원거리 최소화하기/민웅.py>)

```py
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

```

### [상미](<./병원거리 최소화하기/상미.py>)

```py

```

### [병국](<./병원거리 최소화하기/병국.py>)

```py

```

### [성구](./병원거리%20최소화하기/성구.py)

```py
import sys
from itertools import combinations
input = sys.stdin.readline

def distance(x1:int, y1:int, x2:int, y2:int) -> int:
    return abs(x2-x1) + abs(y2-y1)


def solution(n:int, m:int, field:list):
    hospital = []
    person = []

    for i in range(n):
        for j in range(n):
            if field[i][j] == 2:
                hospital.append((i,j))
            elif field[i][j] == 1:
                person.append((i,j))
    # 후보군 선정
    cons = list(combinations(range(len(hospital)), m))
    h_dist = [0] * len(cons)
    for i in range(len(cons)):
        # 사람 별 가까운 병원 거리 저장
        p_dist = [1000000] * len(person)
        for idx in cons[i]:
            for p in range(len(person)):
                p_dist[p] = min(p_dist[p], distance(hospital[idx][1], hospital[idx][0], person[p][1], person[p][0]))
        # m 개의 병원에서 가장 가까운 사람 수를 모두 더해 저장
        h_dist[i] = sum(p_dist)
    return min(h_dist)


if __name__ == "__main__":
    n, m = map(int, input().split())
    field = [list(map(int, input().split())) for _ in range(n)]
    ans = solution(n, m, field)
    print(ans)
```

<br/><br/>

## [Z](https://www.acmicpc.net/problem/1074)

### [민웅](./Z/민웅.py)

```py

```

### [상미](./Z/상미.py)

```py

```

### [병국](./Z/병국.py)

```py

```

### [성구](./Z/성구.py)

```py
# 1074 Z
import sys
sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def z(n:int, r:int, c:int) -> int:
    if n == 1:
        return 0
    half = n // 2
    seq = 0
    if r >= half:
        seq += 2
        r -= half

    if c >= half:
        seq += 1
        c -= half
    

    return seq * half * half + z(half, r, c)

if __name__ == "__main__":
    N, r, c = map(int, input().split())
    cnt = z(2**N, r, c)
    print(cnt)
```

</div>
</details>

</br></br></br>


# 이번주 문제

<details open>
<summary>접기/펼치기</summary>
<div markdown="1">

## [네트워크 연결](https://www.acmicpc.net/problem/1922)

### [민웅](<./네트워크 연결/민웅.py>)

```py


```

### [상미](<./네트워크 연결/상미.py>)

```py

```

### [병국](<./네트워크 연결/병국.py>)

```py

```

### [성구](<./네트워크 연결/성구.py>)

```py
# 1922 네트워크 연결
import sys
input = sys.stdin.readline

# MST (Ksruskal Algorithm)
# root 찾는 알고리즘(parent 배열 이용)
def find(num:int) -> int:
    # 부모 노드가 나, 나 자신이 root 이면 멈춤
    # 반복문
    tmp = num
    # parent 배열을 통해 상위 노드를 찾아서 저장
    while parent[tmp] != tmp:
        tmp = parent[tmp]
        parent[num] = tmp
        
    return parent[num]
    # 재귀
    # if parent[num] == num:
    #     return num
    # parent[num] = find(parent[num])
    # return parent[num]

# 두 spot을 이어줌
def union(num1:int, num2:int) -> None:
    # 두 root를 찾아서
    x = find(num1)
    y = find(num2)
    # 순환되지 않는다면 하나의 고리로 이어줌
    if x != y:
        parent[y] = x
    return


def kruskal():
    # Kruskal 알고리즘 == 최소 신장트리
    node = []      # 지나온 노드 저장용
    cost = 0       # 비용 저장
    for i in range(M):        
        c, a, b = network[i]        # 비용이 가장 적은 노드부터 가져옴
        if find(a) == find(b):      # 순환되면 패스
            continue
        node.append(i)              # 지나온 노드는 저장
        union(a, b)                 # 지나갈 수 있다 == 이어져있다
        cost += c                   # 비용 저장
        if len(node) == N-1:        # 모든 노드 다 탐색했으면 비용 return -> 최소 비용임
            return cost            
    return cost


if __name__ == "__main__":
    N = int(input())
    M = int(input())
    network = []
    for _ in range(M):
        a, b, c = map(int, input().split())
        if a != b:      # 같을 때는 최소가 되지 않기에(c>=1) 빼줌
            network.append((c, a, b))
    # root를 저장할 parent 배열
    parent = [i for i in range(N+1)]
    # 비용순으로 정렬
    network.sort(key=lambda x:x[0])
    ans = kruskal()
    print(ans)
```


</div>
</details>
