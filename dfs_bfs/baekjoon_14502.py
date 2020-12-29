"""
https://www.acmicpc.net/problem/14502
N X M 연구소
    0 = 빈칸
    1 = 벽
    2 = 바이러스 (상/하/좌/우 인접한 빈 칸으로 퍼질 수 있음)
새로 세울 수 있는 벽의 개수는 3개 (꼭 3개를 모두 세워야 함)
벽을 3개 세운 뒤, 바이러스가 퍼질 수 없는 곳을 안전 영역이라고 한다.
연구소의 지도가 주어졌을 때 얻을 수 있는 안전 영역 크기의 최댓값을 구하는 프로그램을 작성하시오.

2 <= #2 <= 10
3 <= N, M <= 8
9 <= N X M <=64
벽을 세울 수 있는 모든 가능한 경우의 수를 다 확인.
    최악의 경우 (8*8 map에서 바이러스가 2개, 나머지가 모두 빈칸인 경우) 64C3 조합
"""


N, M = map(int, input().split())
data = []
for _ in range(N):
    data.append(list(map(int, input().split())))

map_tmp = [[0] * M for _ in range(N)]
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
answer = 0

def virus(x, y):
    # 바이러스 퍼뜨리기 (재귀호출)
    for i in range(4):
        move = moves[i]
        nx = x + move[0]
        ny = y + move[1]
        if nx >= 0 and nx < N and ny >= 0 and ny < M:
            if map_tmp[nx][ny] == 0:
                map_tmp[nx][ny] = 2
                virus(nx, ny)


def get_safety():
    # 안전 영역 계산
    score = 0
    for i in range(N):
        for j in range(M):
            if map_tmp[i][j] == 0:
                score += 1
    return score


def dfs(count):
    global answer

    # 벽을 3개 모두 세웠을 때
    if count == 3:
        for i in range(N):
            for j in range(M):
                map_tmp[i][j] = data[i][j]

        # 바이러스 퍼뜨리기
        for i in range(N):
            for j in range(M):
                if map_tmp[i][j] == 2:
                    virus(i, j)

        # 안전영역 계산
        score = get_safety()
        if answer < score:
            answer = score
        return

    # 벽 세우기
    for i in range(N):
        for j in range(M):
            if data[i][j] == 0:
                data[i][j] = 1
                count += 1
                dfs(count)
                data[i][j] = 0
                count -= 1

dfs(0)
print(answer)

