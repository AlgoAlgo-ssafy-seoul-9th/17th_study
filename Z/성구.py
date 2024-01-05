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


'''
def divide(n:int, r:int, c:int, sx:int,ex:int, sy:int,ey:int) -> None:
    global cnt
    if ex-sx <= 2 or ey-sx <= 2:
        cnt += conquer(r, c, sx, sy)
        return
    xnum = (sx+ex)//2
    ynum = (sy+ey)//2
    num = n // 2
    if r < ynum:
        if c < xnum:        # 2사분면
            divide(num, r, c, sx, (ex+sx)//2, sy, (ey+sy)//2)
        else:               # 1사분면
            cnt += num*num
            divide(num, r, c, (ex+sx)//2, ex, sy, (ey+sy)//2)
            
    else:                   
        if c < xnum:        # 3사분면
            cnt += num*num * 2
            divide(num, r, c, sx, (ex+sx)//2, (ey+sy)//2, ey)
        else:               # 4사분면
            cnt += num*num * 3
            divide(num, r, c, (ex+sx)//2, ex, (ey+sy)//2, ey)
    return


def conquer(r:int ,c:int, x:int, y:int) -> int:
    search = [(0,0), (0,1), (1,0), (1,1)]
    count = 0
    for di, dj in search:
        ni, nj = y+di, x+dj
        if ni == r and nj == c:
            break
        else:
            count += 1
    return count
'''
if __name__ == "__main__":
    N, r, c = map(int, input().split())
    # cnt = 0
    # divide(2**N, r, c, 0, 2**N, 0, 2**N)
    cnt = z(2**N, r, c)
    print(cnt)