"""
https://www.acmicpc.net/problem/17144
R X C 격자판이 주어진다.
    6 <= R, C <= 50
공기청정기가 설치된 곳은 A_{r,c}가 -1이고, 나머지 값은 미세먼지의 양이다.
1초 동안 아래 적힌 일이 순서대로 일어난다.
1. 미세먼지가 확산된다. 확산은 미세먼지가 있는 모든 칸에서 동시에 일어난다.
    (r, c)에 있는 미세먼지는 인접한 네 방향으로 확산된다.
    인접한 방향에 공기청정기가 있거나, 칸이 없으면 그 방향으로는 확산이 일어나지 않는다.
    확산되는 양은 Ar,c/5이고 소수점은 버린다.
    (r, c)에 남은 미세먼지의 양은 Ar,c - (Ar,c/5)×(확산된 방향의 개수) 이다.
2. 공기청정기가 작동한다.
    공기청정기에서는 바람이 나온다.
    위쪽 공기청정기의 바람은 반시계방향으로 순환하고, 아래쪽 공기청정기의 바람은 시계방향으로 순환한다.
    바람이 불면 미세먼지가 바람의 방향대로 모두 한 칸씩 이동한다.
    공기청정기에서 부는 바람은 미세먼지가 없는 바람이고, 공기청정기로 들어간 미세먼지는 모두 정화된다.

? 방의 정보가 주어졌을 때, T초(1 <= T <= 1000)가 지난 후 구사과의 방에 남아있는 미세먼지의 양은 ?
"""


from copy import deepcopy

R, C, T = map(int, input().split())
data = []
for i in range(R):
    data.append(list(map(int, input().split())))
    if data[i][0] == -1:
        idx2 = i

idx1 = idx2 - 1 # 공기청정기가 있는 위치
map_tmp = [[0] * C for _ in range(R)] # 미세먼지 확산 결과를 저장 할 2차원 배열
map_tmp[idx1][0] = -1
map_tmp[idx2][0] = -1

direction1 = [(0, 1), (-1, 0), (0, -1), (1, 0)] # 공기청정기 위쪽 순환 방향
direction2 = [(0, 1), (1, 0), (0, -1), (-1, 0)] # 공기청정기 아래쪽 순환 방향


def spread_dirt(x, y):
    # 미세먼지 확산 함수
    count = 0
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 상하좌우로 확산
    dirt = data[x][y] // 5

    for i in range(4):
        move = moves[i]
        nx = x + move[0]
        ny = y + move[1]

        if nx >= 0 and nx < R and ny >= 0 and ny < C and data[nx][ny] != -1:
            map_tmp[nx][ny] += dirt
            count += 1
    map_tmp[x][y] += (data[x][y] - dirt * count)


def circulate(x, y, direction):
    data[x][y] = 0
    for i in range(4):
        dir = direction[i]
        while True:
            nx = x + dir[0]
            ny = y + dir[1]
            if nx >= 0 and nx < R and ny >= 0 and ny < C and data[nx][ny] != -1:
                data[nx][ny] = map_tmp[x][y]
                x, y = nx, ny
            else:
                break


for _ in range(T):
    # 미세먼지 확산
    for i in range(R):
        for j in range(C):
            if data[i][j] > 0:
                spread_dirt(i, j)

    data = deepcopy(map_tmp)
    # 공기청정기 작동
    circulate(idx1, 1, direction1)
    circulate(idx2, 1, direction2)
    # 초기화
    map_tmp = [[0] * C for _ in range(R)]
    map_tmp[idx1][0] = -1
    map_tmp[idx2][0] = -1

# 남아 있는 미세먼지 계산
count = 0
for i in range(R):
    for j in range(C):
        if data[i][j] > 0:
            count += data[i][j]
print(count)
