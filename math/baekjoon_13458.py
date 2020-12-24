"""
https://www.acmicpc.net/problem/13458
시험장 N개
i번 시험장에는 A_i명의 응시자가 있음 (i=1, ... , N)
총감독관은 한 시험장에서 감시할 수 있는 응시자의 수가 B명
부감독관은 한 시험장에서 감시할 수 있는 응시자의 수가 C명
각각의 시험장에 총감독관은 오직 1명만 있어야 하고, 부감독관은 여러 명 있어도 된다.
각 시험장마다 응시생들을 모두 감시하기 위해 필요한 감독관 수의 최솟값은?
"""

N = int(input())
data = list(map(int, input().split()))
B, C = map(int, input().split())
count = 0

for i in range(N):
    # 각 시험장에 총감독관 한명씩 배치
    count += 1
    data[i] = data[i] - B

    # 남은 응시자 수에 대해 부감독관 배치
    if data[i] > 0:
        count += data[i] // C
        if data[i] % C != 0:
            count += 1

print(count)
