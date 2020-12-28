"""
https://www.acmicpc.net/problem/17140
크기가 3×3인 배열 A에 대해 1초가 지날때마다 배열에 다음 연산이 적용된다.
    R 연산: 배열 A의 모든 행에 대해서 정렬을 수행한다. 행의 개수 >= 열의 개수인 경우에 적용
    C 연산: 배열 A의 모든 열에 대해서 정렬을 수행한다. 행의 개수 < 열의 개수인 경우에 적용
정렬
    한 행 또는 열에 있는 수를 정렬하려면, 각각의 수가 몇 번 나왔는지 알아야 한다.
    그 다음, 수의 등장 횟수가 커지는 순으로, 그러한 것이 여러가지면 수가 커지는 순으로 정렬.
    그 다음에는 배열 A에 정렬된 결과를 다시 넣어야 한다.
    정렬된 결과를 배열에 넣을 때는, 수와 등장 횟수를 모두 넣으며, 순서는 수가 먼저이다.
배열 A에 들어있는 수와 r, c, k가 주어졌을 때,
A[r][c]에 들어있는 값이 k가 되기 위한 최소 시간은?
"""

from collections import Counter

# Input
R, C, K = map(int, input().split())
A = []
for i in range(3):
    A.append(list(map(int, input().split())))

# 정렬 함수
def get_sorted(arr):
    sorted_list = []
    table = list(sorted(Counter(arr).items(), key=lambda x: (x[1], x[0])))
    for num, cnt in table:
        # 수를 정렬할 때 0은 무시해야 한다.
        if num != 0:
            sorted_list.append(num) # 수
            sorted_list.append(cnt) # 등장횟수
    return sorted_list

num_cols = 3
num_rows = 3
count = 0

while True:
    new_A = [] # 연산 수행 후 결과 저장

    if num_rows > R - 1 and num_cols > C - 1 and A[R - 1][C - 1] == K:
        break
    elif count == 100:
        count = -1
        break

    if num_rows >= num_cols:
        # R 연산
        count += 1
        for i in range(min(num_rows, 100)):
            new_row = get_sorted(A[i]) # 정렬된 행
            len_col = len(new_row)
            new_A.append(new_row)
            if num_cols < len_col:
                num_cols = len_col
        # 크기가 가장 큰 행을 기준으로 0 채우기
        for row in new_A:
            if len(row) < num_cols:
                for _ in range(num_cols - len(row)):
                    row.append(0)
        A = new_A

    else:
        # C 연산
        count += 1
        B = list(map(list, zip(*A))) # transpose
        for i in range(min(num_cols, 100)):
            new_col = get_sorted(B[i]) # 정렬된 열
            len_row = len(new_col)
            new_A.append(new_col)
            if num_rows < len_row:
                num_rows = len_row
        # 크기가 가장 큰 열을 기준으로 0 채우기
        for col in new_A:
            if len(col) < num_rows:
                for _ in range(num_rows - len(col)):
                    col.append(0)
        A = list(map(list, zip(*new_A)))

print(count)



