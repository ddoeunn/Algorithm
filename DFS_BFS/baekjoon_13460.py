"""
https://www.acmicpc.net/problem/13460
N * M 보드
    . 빈칸
    # 벽 (보드의 모든 가장자리는 벽)
    O 구멍
    R 빨간 구슬
    B 파란 구슬

빨간 구슬을 구멍을 통해 빼내는 게임.
파란 구슬이 구멍에 들어가면 실패
빨간 구슬과 파란 구슬이 동시에 같은 칸에 있을 수 없음.
최소 몇번 만에 빨간 구슬을 구멍을 통해 빼낼 수 있는가?
(10번 초과하면 -1을 출력한다.)

3 <= N, M <= 10
"""

from collections import deque

N, M = map(int, input().split())

arr = []
for i in range(N):
    arr.append(list(input()))

q = deque()
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]
# 방문 여부를 체크할 4차원 배열
visited = [[[[False] * M for _ in range(N)] for _ in range(M)] for _ in range(N)]


def init():
    rx, ry, bx, by = 0, 0, 0, 0
    for i in range(N):
        for j in range(M):
            if arr[i][j] == 'R':
                rx, ry = i, j       # 처음 빨간 구슬의 위치
            elif arr[i][j] == 'B':
                bx, by = i, j       # 처음 파란 구슬의 위치
    visited[rx][ry][bx][by] = True  # 방문 체크
    q.append((rx, ry, bx, by, 1))   # 큐에 삽입 count =1


def move(x, y, dx, dy):
    step = 0
    while arr[x + dx][y + dy] != '#' and arr[x][y] != 'O':
        # 다음 칸이 벽이거나 지금 칸이 구멍이면 stop
        x += dx
        y += dy
        step += 1

    return x, y, step


def bfs():
    init()
    while q:
        rx, ry, bx, by, count = q.popleft() # 현재 위치와 move count
        # print(rx, ry, bx, by, count)
        if count > 10:
            break

        for i in range(4):
            nrx, nry, rstep = move(rx, ry, dx[i], dy[i])    # 빨간 구슬 다음위치, 움직임 횟수
            nbx, nby, bstep = move(bx, by, dx[i], dy[i])    # 파란 구슬 다음위치, 움직임 횟수

            # 파란 구슬의 다음 위치가 구멍이 아니라면
            if arr[nbx][nby] != 'O':
                # 빨간 구슬의 다음 위치가 구멍이면 성공
                if arr[nrx][nry] == 'O':
                    print(count)
                    return
                # 빨간 구슬과 파란 구슬의 다음 위치가 같다면 더 많이 움직인 구슬을 한칸 뒤로 움직인다.
                if nrx == nbx and nry == nby:
                    if rstep > bstep:
                        nrx -= dx[i]
                        nry -= dy[i]
                    else:
                        nbx -= dx[i]
                        nby -= dy[i]
                # 방문하지 않은 좌표인 경우만
                if not visited[nrx][nry][nbx][nby]:
                    # print('rstep = ', rstep, 'bstep = ', bstep)
                    visited[nrx][nry][nbx][nby] = True          # 방문 체크
                    q.append((nrx, nry, nbx, nby, count + 1))   # 큐에 삽입
    print(-1)

bfs()