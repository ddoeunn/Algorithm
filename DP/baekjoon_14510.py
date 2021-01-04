"""
https://www.acmicpc.net/problem/14501
오늘부터 N+1일째 되는 날 퇴사를 하기 위해서, 남은 N일 동안 최대한 많은 상담을 하려고 한다.
각각의 상담은 상담을 완료하는데 걸리는 기간 T_i와 상담을 했을 때 받을 수 있는 금액 P_i로 이루어져 있다.
상담을 적절히 했을 때, 백준이가 얻을 수 있는 최대 수익을 구하는 프로그램을 작성하시오.

1 <= N <= 15
1 <= T_i <= 5
1 <= P_i <= 1000
"""

N = int(input())
times = []
pays = []

for _ in range(N):
    t, p = map(int, input().split())
    times.append(t)
    pays.append(p)

moneys = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    # 최대 수익을 맨 마지막 일부터 계산
    next_day = i + times[i]
    if next_day > N:
        moneys[i] = moneys[i + 1]
    else:
        # max(내일 일을 했을 때 얻을 수 있는 최대 수익, 오늘 일을 했을 때 얻을 수 있는 최대 수익)
        moneys[i] = max(moneys[i + 1], pays[i] + moneys[next_day])

print(moneys[0])

# 다른 풀이
for i in range(N - 1, -1, -1):
    money = pays[i]
    next_day = i + times[i]
    if next_day > N:
        moneys[i] = 0
    elif next_day == N:
        moneys[i] = money
    else:
        money += max(moneys[next_day:N])
        moneys[i] = money

print(max(moneys))
