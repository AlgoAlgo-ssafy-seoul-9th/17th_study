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