# 1922 네트워크 연결 - Kruskal
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