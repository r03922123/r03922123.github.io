---
title: Shortest paths beyond Dijkstra: Bellman–Ford, Floyd–Warshall and A*
summary: Three shortest-path algorithms for when Dijkstra is the wrong tool. Bellman–Ford as a DP over edge counts (negative edges, negative cycles, at most k stops), Floyd–Warshall as a DP over allowed middle nodes, and A* with a heuristic that never overestimates.
group: Graphs
order: 4
tags: Bellman–Ford · Floyd–Warshall · A* · shortest path · dynamic programming
---
[Dijkstra](../dijkstra/) is the default for shortest paths, but it rests on two assumptions: weights are non-negative, and you want distances from one source. Drop the first and you need **Bellman–Ford**. Want every pair at once on a small graph and **Floyd–Warshall** is three nested loops. Keep both assumptions but care about only one target, and **A*** lets a good guess about the remaining distance steer the search toward it.

Bellman–Ford and Floyd–Warshall are both [dynamic programming](../dynamic-programming/); they differ only in what the subproblem limits. One limits the number of edges a path may use, the other limits which nodes it may pass through.

## Bellman–Ford: a DP over edge counts
Let \(d_k(v)\) be the length of the shortest path from the source to \(v\) that uses **at most** \(k\) edges. A path with at most \(k\) edges either uses at most \(k - 1\), or its last edge is some \((u, v)\):

\[
d_k(v) = \min\Big(d_{k-1}(v),\ \min_{(u,v) \in E} \big(d_{k-1}(u) + w(u,v)\big)\Big), \qquad d_0(s) = 0,\ d_0(v) = \infty \text{ otherwise.}
\]

If there is no negative cycle reachable from the source, every shortest path can be taken simple, so it has at most \(n - 1\) edges and \(d_{n-1}\) is the answer. Each round scans all \(m\) edges: \(O(nm)\) time, \(O(n)\) space.

In practice the rounds are run **in place**: one array, relaxed edge by edge. That can only make values smaller faster (an update within a round may be reused in the same round), and after round \(k\) every value is still at most \(d_k(v)\) and still the length of a real path, so the \(n - 1\) bound holds. Two cheap improvements follow:

- **Early exit:** if a whole round changes nothing, nothing ever will.
- **Negative-cycle test:** after \(n - 1\) rounds, if some edge can still be relaxed, a negative cycle is reachable from the source and "shortest" is undefined.

```python
import heapq
import random

INF = float("inf")


def bellman_ford(n, edges, src):
    """Distances from src, or None if a negative cycle is reachable from src."""
    dist = [INF] * n
    dist[src] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:          # inf + w stays inf, so unreachable u never relaxes
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None                         # still improving after n - 1 rounds
    return dist


assert bellman_ford(4, [(0, 1, 4), (0, 2, 5), (2, 1, -3), (1, 3, 2)], 0) == [0, 2, 5, 4]
assert bellman_ford(3, [(0, 1, 1), (1, 2, -2), (2, 1, 1)], 0) is None        # cycle 1 -> 2 -> 1 has weight -1
assert bellman_ford(3, [(1, 2, -2), (2, 1, 1)], 0) == [0, INF, INF]          # same cycle, but unreachable
```

A queue-driven variant, often called **SPFA**, only rescans edges out of nodes whose distance just changed. It is frequently faster on sparse random graphs but has the same \(O(nm)\) worst case, and detecting a negative cycle needs extra bookkeeping (counting how many edges each current path uses), so the round-based version above is the safer default.

## At most k stops: keep the rounds honest
[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) asks for the cheapest route with at most \(k\) intermediate stops, i.e. at most \(k + 1\) flights. That is exactly \(d_{k+1}(\text{dst})\), so run \(k + 1\) rounds. Here the in-place shortcut is **wrong**: an update made earlier in a round could be extended by a later edge in the same round, sneaking in an extra flight. Each round must read only the previous round's array.

```python
def find_cheapest_price(n, flights, src, dst, k):
    cost = [INF] * n
    cost[src] = 0
    for _ in range(k + 1):                      # round r allows paths of at most r flights
        nxt = cost[:]
        for u, v, p in flights:
            if cost[u] + p < nxt[v]:            # read cost (last round), write nxt
                nxt[v] = cost[u] + p
        cost = nxt
    return -1 if cost[dst] == INF else cost[dst]


assert find_cheapest_price(4, [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]], 0, 3, 1) == 700
assert find_cheapest_price(3, [[0, 1, 100], [1, 2, 100], [0, 2, 500]], 0, 2, 1) == 200
assert find_cheapest_price(3, [[0, 1, 100], [1, 2, 100], [0, 2, 500]], 0, 2, 0) == 500


def cheapest_brute(n, flights, src, dst, k):    # try every route of at most k + 1 flights
    best = INF

    def go(u, spent, used):
        nonlocal best
        if u == dst:
            best = min(best, spent)
        if used == k + 1:
            return
        for a, b, p in flights:
            if a == u:
                go(b, spent + p, used + 1)

    go(src, 0, 0)
    return -1 if best == INF else best


for _ in range(300):
    n = random.randint(2, 6)
    flights = [[u, v, random.randint(1, 20)] for u in range(n) for v in range(n) if u != v and random.random() < 0.35]
    src, dst, k = random.sample(range(n), 2) + [random.randint(0, 3)]
    assert find_cheapest_price(n, flights, src, dst, k) == cheapest_brute(n, flights, src, dst, k)
```

The time is \(O(k \cdot m)\). Dijkstra over states `(city, flights used)` also works, but the rounds are shorter to write and need no heap.

## Floyd–Warshall: a DP over middle nodes
For distances between **all pairs**, let \(d_k(i, j)\) be the shortest path from \(i\) to \(j\) whose intermediate nodes all come from \(\{0, \dots, k - 1\}\). Allowing node \(k\) as well, the best path either avoids it or goes through it exactly once:

\[
d_{k+1}(i, j) = \min\big(d_k(i, j),\ d_k(i, k) + d_k(k, j)\big)
\]

The base case is \(d_0(i, i) = 0\), \(d_0(i, j) = w(i, j)\) for an edge (the smallest one, if there are parallel edges) and \(\infty\) otherwise. The update can be done in place: row \(k\) and column \(k\) do not change during round \(k\), since \(d(i,k) + d(k,k)\) cannot beat \(d(i,k)\) when \(d(k,k) \ge 0\). That leaves one \(n \times n\) table, \(O(n^3)\) time and \(O(n^2)\) space. The **outer** loop must be the middle node; putting it inside is a classic bug.

Negative edges are fine. A negative cycle shows up as a negative diagonal entry \(d(i, i) < 0\).

```python
def floyd_warshall(n, edges):
    d = [[INF] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    for u, v, w in edges:
        d[u][v] = min(d[u][v], w)               # keep the cheapest parallel edge
    for k in range(n):                          # middle node outermost
        dk = d[k]
        for i in range(n):
            dik = d[i][k]
            if dik == INF:
                continue
            row = d[i]
            for j in range(n):
                if dik + dk[j] < row[j]:
                    row[j] = dik + dk[j]
    return d
```

[1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/) needs, for every city, how many others are within the threshold, with ties going to the largest index. With \(n \le 100\), all pairs is one Floyd–Warshall.

```python
def find_the_city(n, edges, threshold):
    both_ways = [(u, v, w) for u, v, w in edges] + [(v, u, w) for u, v, w in edges]
    d = floyd_warshall(n, both_ways)
    reach = [sum(1 for j in range(n) if j != i and d[i][j] <= threshold) for i in range(n)]
    return min(range(n), key=lambda i: (reach[i], -i))


assert find_the_city(4, [[0, 1, 3], [1, 2, 1], [1, 3, 4], [2, 3, 1]], 4) == 3
assert find_the_city(5, [[0, 1, 2], [0, 4, 8], [1, 2, 3], [1, 4, 2], [2, 3, 1], [3, 4, 1]], 2) == 0
```

## Checking them against each other
On random graphs with non-negative weights, Bellman–Ford from every source, Floyd–Warshall and Dijkstra must agree exactly. With negative weights Dijkstra drops out, but the other two still have to agree, including on whether a negative cycle is reachable: Bellman–Ford from `s` fails exactly when some node `v` is reachable from `s` and sits on a negative cycle, which Floyd–Warshall reports as \(d(s, v) < \infty\) and \(d(v, v) < 0\).

```python
def dijkstra(n, edges, src):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    dist = [INF] * n
    dist[src], heap = 0, [(0, src)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        for v, w in graph[u]:
            if du + w < dist[v]:
                dist[v] = du + w
                heapq.heappush(heap, (dist[v], v))
    return dist


def random_edges(n, low, high):
    return [(u, v, random.randint(low, high))
            for u in range(n) for v in range(n) if random.random() < 0.3]


for _ in range(300):                             # non-negative: all three agree
    n = random.randint(1, 8)
    edges = random_edges(n, 0, 9)
    fw = floyd_warshall(n, edges)
    for s in range(n):
        assert bellman_ford(n, edges, s) == fw[s] == dijkstra(n, edges, s)

for _ in range(300):                             # negative edges: Bellman-Ford vs Floyd-Warshall
    n = random.randint(1, 7)
    edges = random_edges(n, -3, 9)
    fw = floyd_warshall(n, edges)
    for s in range(n):
        bf = bellman_ford(n, edges, s)
        bad = any(fw[s][v] < INF and fw[v][v] < 0 for v in range(n))
        assert (bf is None) == bad
        if not bad:
            assert bf == fw[s]

for _ in range(300):                             # 1334 via Floyd-Warshall vs Dijkstra from every city
    n = random.randint(2, 8)
    edges = [[u, v, random.randint(1, 6)] for u in range(n) for v in range(u + 1, n) if random.random() < 0.4]
    t = random.randint(1, 10)
    both = [(u, v, w) for u, v, w in edges] + [(v, u, w) for u, v, w in edges]
    reach = [sum(1 for j, x in enumerate(dijkstra(n, both, i)) if j != i and x <= t) for i in range(n)]
    best = min(reach)
    assert find_the_city(n, edges, t) == max(i for i in range(n) if reach[i] == best)
```

(A self-loop with negative weight counts as a negative cycle, and the random graphs include self-loops, so both detectors get exercised. The last loop also checks 1334, including its largest-index tie rule, against Dijkstra run from every city.)

## A*: Dijkstra with a sense of direction
Dijkstra expands nodes in order of \(g(v)\), the distance travelled so far, so it spreads out evenly in every direction. A* expands in order of

\[
f(v) = g(v) + h(v),
\]

where \(h(v)\) estimates the remaining distance to the goal. Two properties of \(h\) matter:

- **Admissible:** \(h(v)\) never exceeds the true remaining distance. Then the first time the goal is popped, its cost is optimal.
- **Consistent:** \(h(u) \le w(u, v) + h(v)\) for every edge, with \(h(\text{goal}) = 0\). Then \(f\) never decreases along a path, so, as in Dijkstra, a node is final the first time it is popped and can be skipped afterwards. Consistent implies admissible.

With \(h = 0\), A* is Dijkstra. The better \(h\) approximates the truth, the fewer nodes it expands. On grids the standard choices come from how you are allowed to move, with unit step cost:

| Moves | Heuristic | Formula |
| --- | --- | --- |
| 4 directions | Manhattan | \(\lvert dx \rvert + \lvert dy \rvert\) |
| 8 directions, diagonal costs 1 | Chebyshev | \(\max(\lvert dx \rvert, \lvert dy \rvert)\) |
| 8 directions, diagonal costs \(\sqrt 2\) | Octile | \(\lvert dx \rvert + \lvert dy \rvert + (\sqrt 2 - 2)\min(\lvert dx \rvert, \lvert dy \rvert)\) |
| any angle | Euclidean | \(\sqrt{dx^2 + dy^2}\) |

Using Manhattan distance on an 8-direction grid overestimates (a diagonal step reduces it by 2), so it is not admissible and can return a longer path.

[1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/) moves in 8 directions with unit cost, so Chebyshev distance to the bottom-right corner is consistent. The answer counts cells, so a path of \(g\) steps has length \(g + 1\).

```python
def shortest_path_binary_matrix(grid):
    n = len(grid)
    if grid[0][0] or grid[n - 1][n - 1]:
        return -1
    h = lambda r, c: max(n - 1 - r, n - 1 - c)
    g = {(0, 0): 0}
    heap = [(h(0, 0), 0, 0, 0)]                 # (f, g, row, col)
    done = set()
    while heap:
        _, steps, r, c = heapq.heappop(heap)
        if (r, c) == (n - 1, n - 1):
            return steps + 1                    # count cells, not moves
        if (r, c) in done:
            continue
        done.add((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and not grid[nr][nc] and steps + 1 < g.get((nr, nc), INF):
                    g[nr, nc] = steps + 1
                    heapq.heappush(heap, (steps + 1 + h(nr, nc), steps + 1, nr, nc))
    return -1


assert shortest_path_binary_matrix([[0, 1], [1, 0]]) == 2
assert shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4
assert shortest_path_binary_matrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]) == -1
assert shortest_path_binary_matrix([[0]]) == 1                              # start is the goal: one cell
assert shortest_path_binary_matrix([[1]]) == -1
assert shortest_path_binary_matrix([[0, 0], [0, 1]]) == -1                  # goal blocked


from collections import deque


def binary_matrix_bfs(grid):                    # reference: plain BFS
    n = len(grid)
    if grid[0][0] or grid[n - 1][n - 1]:
        return -1
    dist = {(0, 0): 1}
    q = deque([(0, 0)])
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and not grid[nr][nc] and (nr, nc) not in dist:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist.get((n - 1, n - 1), -1)


for _ in range(500):
    n = random.randint(1, 8)
    grid = [[int(random.random() < 0.35) for _ in range(n)] for _ in range(n)]
    assert shortest_path_binary_matrix(grid) == binary_matrix_bfs(grid)
```

With unit costs, plain [BFS](../bfs/) already runs in \(O(n^2)\), so A* wins nothing asymptotically here; its payoff is on large open maps where most of the grid lies away from the goal. Worst case it is still Dijkstra's \(O(m \log n)\).

## Which one to use
| Situation | Algorithm | Time |
| --- | --- | --- |
| Unweighted, or all weights equal | [BFS](../bfs/) | \(O(n + m)\) |
| Weights 0 or 1 | 0–1 BFS with a deque | \(O(n + m)\) |
| Weighted DAG, any signs | relax in [topological order](../topological-sort/) | \(O(n + m)\) |
| Non-negative weights, one source | [Dijkstra](../dijkstra/) | \(O(m \log n)\) |
| One source and one target, good heuristic | A* | at most Dijkstra's |
| Negative weights, or detect a negative cycle | Bellman–Ford | \(O(nm)\) |
| At most k edges | Bellman–Ford, k rounds with a copy | \(O(km)\) |
| All pairs, small \(n\) | Floyd–Warshall | \(O(n^3)\) |
| All pairs, sparse and non-negative | Dijkstra from every node | \(O(nm \log n)\) |

Two rules of thumb: reach for Floyd–Warshall when \(n\) is a few hundred at most and you need many pairwise distances, and reach for Bellman–Ford when the problem caps the number of edges, even if all weights are positive.

## Practice
- [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)
- [1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)
- [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
- [1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/) (Floyd–Warshall on booleans: transitive closure)
- [2976. Minimum Cost to Convert String I](https://leetcode.com/problems/minimum-cost-to-convert-string-i/) (Floyd–Warshall over 26 letters)
- [1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/)
- [2642. Design Graph With Shortest Path Calculator](https://leetcode.com/problems/design-graph-with-shortest-path-calculator/)
- [1697. Checking Existence of Edge Length Limited Paths](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/)
- [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/) (minimise the largest step: the same bottleneck idea)
- [1102. Path With Maximum Minimum Value](https://leetcode.com/problems/path-with-maximum-minimum-value/) (bottleneck path: a Dijkstra variant, or [union–find](../union-find/))

From R. Bellman, "On a Routing Problem" (1958), and L. R. Ford Jr., "Network Flow Theory" (1956); R. W. Floyd, "Algorithm 97: Shortest Path" (1962), and S. Warshall, "A Theorem on Boolean Matrices" (1962); P. E. Hart, N. J. Nilsson and B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" (1968). The grid heuristics follow Amit Patel's notes on A* heuristics ([theory.stanford.edu/~amitp/GameProgramming](https://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html)).
