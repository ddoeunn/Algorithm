# Algorithm for Coding Test


---
## Contents
* [Greedy Algorithm](https://github.com/ddoeunn/Algorithm#1-greedy-algorithm)
* [Implementation](https://github.com/ddoeunn/Algorithm#2-implementation)
* [Search Algorithm : DFS / BFS](https://github.com/ddoeunn/Algorithm#3-search-algorithm--dfs--bfs)
  + [Data Structure : Stack / Queue](https://github.com/ddoeunn/Algorithm#data-structure--stack--queue)
  + [Recursive Function](https://github.com/ddoeunn/Algorithm#recursive-function)
  * [DFS](https://github.com/ddoeunn/Algorithm#31-dfs)
  * [BFS](https://github.com/ddoeunn/Algorithm#32-bfs)
* [Sorting Algorithm](https://github.com/ddoeunn/Algorithm#4-sorting-algorithm)
  + [Selection Sort](https://github.com/ddoeunn/Algorithm#41-%EC%84%A0%ED%83%9D%EC%A0%95%EB%A0%AC)
  + [Insert Sort](https://github.com/ddoeunn/Algorithm#42-%EC%82%BD%EC%9E%85%EC%A0%95%EB%A0%AC)
  + [Quick Sort](https://github.com/ddoeunn/Algorithm#43-%ED%80%B5%EC%A0%95%EB%A0%AC)


---
# 1. Greedy Algorithm
* 그리디 알고리즘(탐욕법)은 현재 상황에서 지금 당장 좋은 것만 고르는 방법이다.
* 단순히 가장 좋아 보이는 것을 반복적으로 선택해도 최적의 해를 구할 수 있는지 검토하는 것이 중요하다.
* 일반적인 상황에서 그리디 알고리즘은 최적의 해를 보장할 수 없을 때가 많지만, 코딩 테스트에서는 탐욕법으로 얻은 해가 최적의 해가 되는 상황에 한하여 출제된다.
* 예제
> 거스름돈으로 사용할 500원, 100원, 50원, 10원짜리 동전이 무한히 존재한다고 가정한다. 손님에게 거슬러 주어야 할 돈이 N원일 때 거슬러 주어야 할 동전의 최소 개수를 구하시오. 단, 거슬러 줘야 할 돈 N은 항상 10의 배수이다.

``` python
N = int(input())
count = 0
coins = [500, 100, 50, 10]

for coin in coins:
    count += N // coin
    N %= coin

print(count)
```


---
# 2. Implementation
* 구현 유형의 문제는 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제를 지칭한다.
* 시뮬레이션, 완전 탐색 유형과 비슷한 점이 많다.
* 예제
> 여행가 A는 1X1 크기의 정사각형으로 나누어져 있는 NxN 크기의 정사각형 공간 위에 서있다. 가장 왼쪽 위 좌표는 (1, 1)이며, 가장 오른쪽 아래 좌표는 (N, N)에 해당한다. A는 상/하/좌/우 방향으로 이동할 수 있으며, 시작 좌표는 항상 (1, 1)이다. 계획서에는 L(왼)/R(오른)/U(아래)/D(위) 문자가 적혀있으며 각 방향으로 한 칸씩 이동함을 의미한다. 이 때 NxN 크기의 정사각형 공간을 벗어나는 움직임은 무시된다. A가 최종적으로 도착할 지점의 좌표를 공백을 기준으로 구분하여 출력하라.

```python
N = int(input())
plans = input().split()
x, y = 1, 1

# L, R, U, D 이동방향
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]
move_types = ['L', 'R', 'U', 'D']

for plan in plans:
    for i in range(len(move_types)):
        if plan == move_types[i]:
            nx = x + dx[i]
            ny = y + dy[i]
    # 공간을 벗어나는 경우 무시
    if nx < 1 or ny < 1 or nx > N or ny > N:
        continue
    x, y = nx, ny

print(x, y)
```

---
# 3. Search Algorithm : DFS / BFS
* 탐색(search)이란 많은 양의 데이터 중에서 원하는 데이터를 찾는 과정이다.
* 대표적인 그래프 탐색 알고리즘으로는 DFS(Depth First Search; 깊이 우선 탐색)와 BFS(Breadth First Search; 너비 우선 탐색)가 있다.

---
## Data Structure : Stack / Queue
* 스택 (Stack)
  + 선입후출 : First In Last Out
  + 먼저 들어 온 데이터가 나중에 나감
  + 파이썬에서는 리스트를 사용하여 구현

``` python
stack = []

# 삽입(5)-삽입(8)-삽입(3)-삭제()-삽입(7)-삽입(10)-삭제()
stack.append(5)
stack.append(8)
stack.append(3)
stack.pop()
stack.append(7)
stack.append(10)
stack.pop()

print(stack[::-1])  # 최상단 원소부터 출력 (7, 8, 5)
print(stack)        # 최하단 원소부터 출력 (5, 8, 7)
```

* 큐 (Queue)
  + 선입선출 : First In First Out
  + 먼저 들어 온 데이터가 먼저 나감
  + 파이썬에서는 deque 라이브러리를 사용하여 구현


``` python
from collections import deque

queue = deque()

# 삽입(5)-삽입(8)-삽입(3)-삭제()-삽입(7)-삽입(10)-삭제()
queue.append(5)
queue.append(8)
queue.append(3)
queue.popleft()
queue.append(7)
queue.append(10)
queue.popleft()

print(queue)    # 먼저 들어 온 순서대로 출력 (3, 7, 19)
queue.reverse() # 역순으로 바꾸기
print(queue)    # 나중에 들어 온 순서대로 출력 (10, 7, 3)
```

---
## Recursive Function
* 재귀함수(recursive function)란 자기 자신을 다시 호출하는 함수이다.
* 재귀함수를 잘 활용하면 복잡한 알고리즘을 간결하게 작성할 수 있다.
* 재귀함수를 사용할 때는 재귀함수 종료 조건을 반드시 명시해야 한다.
  + 종료 조건을 제대로 명시하지 않으면 함수가 무한히 호출될 수 있다.
* 모든 재귀함수는 반복문을 이용하여 동일한 기능을 구현할 수 있다.
  + 재귀함수가 반복문보다 유리한 경우도 있고 불리한 경우도 있기 때문에 어떤 방법이 문제를 풀 때 더 유리한지 잘 파악해야 한다.

``` python
# Factorial n! = n * (n-1) * ... * 1
def factorial_iterative(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n-1)

print(factorial_iterative(5)) # 120
print(factorial_recursive(5)) # 120
```


* 컴퓨터가 함수를 연속적으로 호출하면 컴퓨터 메모리 내부의 스택 프레임에 쌓인다.
  + 따라서 스택을 사용해야할 때 구현상 스택 라이브러리 대신에 재귀함수를 이용하는 경우가 많다.
* 예제

``` python
# 유클리드 호제법
# 두 자연수 A, B(A>B)에 대하여 A를 B로 나눈 나머지를 R이라고 하자.
# 이 때 A와 B의 최대공약수는 B와 R의 최대공약수와 같다.

def gcd(a, b):
    if a % b == 0:
        return b
    else:
        return gcd(b, a % b)

print(gcd(192, 162))
```

---
## 3.1 DFS
* 깊이 우선 탐색 (Depth First Search)
* 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘이다.
* 스택 또는 재귀함수를 이용하여 구현한다.
> 1. 탐색 시작 노드를 스택에 삽입하고 방문 처리를 한다.
> 2. 스택의 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있으면 그 노드를 스택에 넣고 방문 처리한다. 방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 꺼낸다.
> 3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다.
* 예제 (시작노드=1, 방문기준: 번호가 낮은 인접 노드부터)

<p align="center">
<img src="https://github.com/ddoeunn/Algorithm/blob/master/img/graph.png?raw=true" alt="graph"  width="300">
</p>


``` python
def dfs(graph, v, visited):
    # 현재 노드를 방문 처리
    visited[v] = True
    print(v, end=' ')
    # 현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]
# 각 노드가 방문된 정보를 표현
visited = [False] * 9
dfs(graph, 1, visited) # 1, 2, 7, 6, 8, 3, 4, 5
```

---
## 3.2 BFS
* 너비 우선 탐색 (Breadth First Search)
* 그래프에서 가까운 노드부터 우선적으로 탐색하는 알고리즘이다.
* 큐를 이용하여 구현한다.
> 1. 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다.
> 2. 큐에서 노드를 꺼낸 뒤에 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리를 한다.
> 3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복한다.
* 예제 (시작노드=1, 방문기준: 번호가 낮은 인접 노드부터)

<p align="center">
<img src="https://github.com/ddoeunn/Algorithm/blob/master/img/graph.png?raw=true" alt="graph"  width="300">
</p>


``` python
from collections import deque

def bfs(graph, start, visited):
    queue = deque([start])
    # 현재 노드를 방문 처리
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        print(v, end=' ')
        # 아직 방문하지 않은 인접한 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
]
# 각 노드가 방문된 정보를 표현
visited = [False] * 9
bfs(graph, 1, visited) # 1, 2, 3, 8, 7, 4, 5, 6
```

---
# 4. Sorting Algorithm
* 정렬(sorting)이란 데이터를 특정한 기준에 따라 순서대로 나열하는 것이다.
* 일반적으로 문제 상황에 따라서 적절한 정렬 알고리즘이 공식처럼 사용된다.
* 선택정렬 / 삽입정렬 / 퀵정렬

---
## 4.1 선택정렬
* 처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 바꾸는 것을 반복
* 데이터 개수 N번 만큼 가장 작은 수를 찾아서 맨 앞으로 보내야 한다.
  + 전체 연산 횟수 = N + (N-1) + ... + 2
  + 시간복잡도 = O(N^2)

``` python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(len(array)):
    min_idx = i
    for j in range(i+1, len(array)):
        if array[min_idx] > array[j]:
            min_idx = j
    array[i], array[min_idx] = array[min_idx], array[i]
    print(array)

    # [0, 5, 9, 7, 3, 1, 6, 2, 4, 8]
    # [0, 1, 9, 7, 3, 5, 6, 2, 4, 8]
    # [0, 1, 2, 7, 3, 5, 6, 9, 4, 8]
    # [0, 1, 2, 3, 7, 5, 6, 9, 4, 8]
    # [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]
    # [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]
    # [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]
    # [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```  

---
## 4.2 삽입정렬
* 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입하여 정렬
* 현재 데이터가 거의 정렬되어 있는 상태라면 매우 빠르게 동작한다.
* 선택정렬과 마찬가지로 반복문이 두 번 중첩되어 사용된다.
  + 시간복잡도 = O(N^2)
  + 최선의 경우 시간복잡도 = O(N)

``` python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(array)):
    for j in range(i, 0, -1):
        if array[j] < array[j-1]:
            array[j], array[j-1] = array[j-1], array[j]
        else:
            break
    print(array)

    # [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]
    # [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]
    # [0, 5, 7, 9, 3, 1, 6, 2, 4, 8]
    # [0, 3, 5, 7, 9, 1, 6, 2, 4, 8]
    # [0, 1, 3, 5, 7, 9, 6, 2, 4, 8]
    # [0, 1, 3, 5, 6, 7, 9, 2, 4, 8]
    # [0, 1, 2, 3, 5, 6, 7, 9, 4, 8]
    # [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---
## 4.3 퀵정렬
* 기준 데이터(pivot)를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
* 가장 기본적인 퀵정렬은 첫 번째 데이터를 기준 데이터로 설정한다.
* 병합정렬과 더불어 대부분의 프로그래밍 언어의 정렬 라이브러리의 근간이 되는 알고리즘이다.
* 시간복잡도
  + 평균적으로 O(NlogN)
  + 최악의 경우 O(N^2)  
  : 첫번째 원소를 피벗으로 삼을 때, 이미 정렬된 배열에 대해서 퀵 정렬을 수행하는 경우

> 1. 왼쪽에서부터 피벗보다 큰 데이터를 선택하고 오른쪽에서부터 피벗보다 작은 데이터를 선택한 후 이 두 데이터의 위치를 서로 변경한다.
> 2. 다시 피벗보다 큰 데이터와 작은 데이터를 각각 찾는다. 찾은 뒤에는 두 값의 위치를 서로 변경한다.
> 3. 왼쪽에서부터 찾는 값과 오른쪽에서부터 찾는 값의 위치가 서로 엇갈릴 때까지 2를 반복한다.
> 4. 두 값이 엇갈린 경우 작은 데이터와 피벗의 위치를 서로 변경한다. 피벗의 왼쪽에는 피벗보다 작은 데이터가 있고, 피벗의 오른쪽에는 피벗보다 큰 데이터가 있는 상태이다. 이 작업을 분할(Divide)이라고 한다.
> 5. 왼쪽 리스트와 오른쪽 리스트에 대해서 1~4번을 수행하는 과정을 반복한다.

<p align="center">
<img src="https://github.com/ddoeunn/Algorithm/blob/master/img/quick sort.png?raw=true" alt="quick sort"  width="350">
</p>



``` python
array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array, start, end):
    if start >= end: # 원소가 1개인 경우
        return
    pivot = start
    left = start + 1
    right = end
    while left <= right:
        # 피벗보다 큰 데이터를 찾을 때까지
        while left <= end and array[left] <= array[pivot]:
            left += 1
        # 피멋보다 작은 데이터를 찾을 때까지
        while right > start and array[right] >= array[pivot]:
            right -= 1
        # 엇갈렸다면 작은 데이터와 피벗을 교체
        if left > right:
            array[right], array[pivot] = array[pivot], array[right]
        # 엇갈리지 않았다면 큰 데이터와 작은 데이터를 교체
        else:
            array[left], array[right] = array[right], array[left]

    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
    quick_sort(array, start, right-1)
    quick_sort(array, right+1, end)

quick_sort(array, 0, len(array)-1)
print(array)
```
