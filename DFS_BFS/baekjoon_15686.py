"""
https://www.acmicpc.net/problem/15686
N*N 도시
    0 = 빈칸
    1 = 집
    2 = 치킨집

치킨거리 = 집과 가장 가까운 치킨집 사이의 거리
도시 치킨거리 = 모든 집의 치킨거리 합
거리 = |r_1 - r_2| - |c_1 - c_2|

도시의 치킨거리를 가장 작게 하는 치킨집을 최대 M개만 남겨놓고 나머지는 폐업시킨다.
폐업 시키지 않을 치킨집을 최대 M개 골랐을 때, 도시 치킨거리의 최솟값은?

2 <= N <= 50
1 <= M <= 13
M <= 치킨집 개수 <= 13
"""

N, M = map(int, input().split())
arr = []

for i in range(N):
    arr.append(list(map(int, input().split())))

chicken = []
house = []
for i in range(N):
    for j in range(N):
        if arr[i][j] == 2:
            chicken.append((i, j))  # 치킨집 좌표 저장
        elif arr[i][j] == 1:
            house.append((i, j))  # 집 좌표 저장

num_c = len(chicken)
num_h = len(house)

visited = [False] * num_c  # 폐점하지 않고 남긴다 = visited True check
answer = float('inf')


def dfs(idx, count):
    global answer
    if count == M:
        # M개가 선택되면 도시의 치킨거리를 계산한다.
        total_dist = 0
        for i in range(num_h):
            min_dist = float('inf')
            for j in range(num_c):
                if visited[j]:
                    d = abs(house[i][0] - chicken[j][0]) + abs(house[i][1] - chicken[j][1])
                    min_dist = min(min_dist, d)

            total_dist += min_dist

        answer = min(answer, total_dist)

    if idx == num_c:
        return

    # 폐점하지 않을 경우
    visited[idx] = True
    dfs(idx + 1, count + 1)
    # 폐점할 경우
    visited[idx] = False
    dfs(idx + 1, count)


dfs(0, 0)
print(answer)
