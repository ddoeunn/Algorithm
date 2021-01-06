"""
https://www.acmicpc.net/problem/14499
N * M 지도
(x, y) 주사위가 놓여진 좌표
    가장 처음에는 주사위 모든 면에 0이 적혀 있음.
주사위는 동(1) 서(2) 북(3) 남(4) 방향으로 굴릴 수 있으며 굴리는 방향 K개가 주어진다.

주사위를 굴렸을 때, 이동한 칸에 쓰여 있는 수가 0이면
    주사위 바닥면에 쓰여 있는 수가 칸에 복사된다.
주사위를 굴렸을 때, 이동한 칸에 쓰여 있는 수가 0이 아니면
    칸에 쓰여 있는 수가 주사위 바닥에 복사되고, 칸에 쓰여진 수는 0이 된다.

주사위가 이동할 때마다 상단에 쓰여 있는 값을 구해라.
"""

N, M, x, y, K = map(int, input().split())
arr = []
for _ in range(N):
    arr.append(list(map(int, input().split())))

directions = list(map(int, input().split()))    # 주사위 이동 명령어

dx = [0, 0, 0, -1, 1]   # x축 이동 방향
dy = [0, 1, -1, 0, 0]   # y축 이동 방향

# top = (1, 1) bottom = (3, 1) 그 외 (0, 1) (1, 0) (1, 2) (2, 1)
dice = [[0] * 3 for _ in range(4)]

# 주사위를 굴리고 주사위 값(top, bottom, ... ) 바꾸기
def move(direction):
    if direction == 1: # 동
        dice[1][0], dice[1][1], dice[1][2], dice[3][1] = dice[3][1], dice[1][0], dice[1][1], dice[1][2]

    elif direction == 2: # 서
        dice[1][0], dice[1][1], dice[1][2], dice[3][1] = dice[1][1], dice[1][2], dice[3][1], dice[1][0]

    elif direction == 3: # 북
        dice[0][1], dice[1][1], dice[2][1], dice[3][1] = dice[1][1], dice[2][1], dice[3][1], dice[0][1]
    else: # 남
        dice[0][1], dice[1][1], dice[2][1], dice[3][1] = dice[3][1], dice[0][1], dice[1][1], dice[2][1]


for dir in directions:

    # 이동 좌표
    nx = x + dx[dir]
    ny = y + dy[dir]

    if nx < 0 or nx >= N or ny < 0 or ny >= M:
        continue

    move(dir) # 주사위 굴리기

    if arr[nx][ny] == 0:
        arr[nx][ny] = dice[3][1]
    else:
        dice[3][1] = arr[nx][ny]
        arr[nx][ny] = 0

    x, y = nx, ny
    print(dice[1][1])