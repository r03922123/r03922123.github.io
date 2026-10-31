---
title: Dijkstra's algorithm, three ways
summary: Single-source shortest paths with non-negative weights. Why settling the closest node is safe, the heap version with lazy deletion, the O(n²) array version for dense graphs, and variants that change what "distance" means.
group: Graphs
order: 3
tags: Dijkstra · shortest path · heap
---
Dijkstra's algorithm finds shortest distances from one source when every edge weight is **non-negative**. It grows a set of settled nodes, always settling the unsettled node with the smallest tentative distance, and relaxing its outgoing edges.

## Why settling the closest node is safe
Suppose `u` has the smallest tentative distance among unsettled nodes. Any other path to `u` must leave the settled set through some unsettled node `x`, which is already at least as far as `u`, and the remaining edges cannot reduce the total because they are non-negative. So `u`'s tentative distance is final. A single negative edge breaks this argument; then use [Bellman–Ford](../shortest-paths/).

## With a heap and lazy deletion
The textbook version needs a priority queue that can lower a key. Python's `heapq` cannot, so push a new entry whenever a distance improves, and skip stale entries when popped. The heap holds at most \(m\) entries, so the total is \(O((n + m) \log n)\).

```python
import heapq


def dijkstra(n, edges, src):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    dist = [float("inf")] * n
    dist[src], heap = 0, [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                            # stale entry: u was settled with a smaller d
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return dist


def network_delay_time(times, n, k):            # 743: time for a signal from k to reach all nodes
    dist = dijkstra(n, [(u - 1, v - 1, w) for u, v, w in times], k - 1)
    return -1 if max(dist) == float("inf") else max(dist)


assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
assert network_delay_time([[1, 2, 1]], 2, 2) == -1
```

## Dense graphs: an array instead of a heap
When \(m\) is close to \(n^2\), scanning an array for the minimum is \(O(n)\) per step and \(O(n^2)\) overall, which beats \(O(n^2 \log n)\) with a heap. This matters for complete graphs, such as points where every pair is connected.

```python
def dijkstra_dense(weight, src):                 # weight[u][v], inf if no edge
    n = len(weight)
    dist, done = [float("inf")] * n, [False] * n
    dist[src] = 0
    for _ in range(n):
        u = min((v for v in range(n) if not done[v]), key=dist.__getitem__)
        if dist[u] == float("inf"):
            break
        done[u] = True
        for v in range(n):
            if dist[u] + weight[u][v] < dist[v]:
                dist[v] = dist[u] + weight[u][v]
    return dist


import random
INF = float("inf")
for _ in range(200):
    n = random.randint(1, 7)
    edges = [(u, v, random.randint(0, 9)) for u in range(n) for v in range(n) if u != v and random.random() < 0.4]
    w = [[INF] * n for _ in range(n)]
    for u, v, c in edges:
        w[u][v] = min(w[u][v], c)
    assert dijkstra(n, edges, 0) == dijkstra_dense(w, 0)
```

## Changing what distance means
The algorithm works for any path cost that never decreases as the path grows, not just sums.

- **Bottleneck paths.** [1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/): the cost of a path is its largest single step. Replace `d + w` with `max(d, w)`.

```python
def minimum_effort_path(heights):
    rows, cols = len(heights), len(heights[0])
    best = [[float("inf")] * cols for _ in range(rows)]
    best[0][0], heap = 0, [(0, 0, 0)]
    while heap:
        effort, r, c = heapq.heappop(heap)
        if (r, c) == (rows - 1, cols - 1):
            return effort
        if effort > best[r][c]:
            continue
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols:
                e = max(effort, abs(heights[nr][nc] - heights[r][c]))
                if e < best[nr][nc]:
                    best[nr][nc] = e
                    heapq.heappush(heap, (e, nr, nc))


assert minimum_effort_path([[1, 2, 2], [3, 8, 2], [5, 3, 5]]) == 2
assert minimum_effort_path([[1, 2, 1, 1, 1], [1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 1, 1, 2, 1]]) == 0
```

- **Extra state.** If the cost depends on more than the node (stops used, fuel left), make the state `(node, extra)` as in [BFS over states](../bfs/).
- **Several sources or targets.** Run it from each relevant source and combine: [2203. Minimum Weighted Subgraph With the Required Paths](https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/) runs it three times, once on the reversed graph, and minimizes over the meeting node.

## Practice
- [2203. Minimum Weighted Subgraph With the Required Paths](https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/)
- [2290. Minimum Obstacle Removal to Reach Corner](https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/)
- [505. The Maze II](https://leetcode.com/problems/the-maze-ii/)
- [2642. Design Graph With Shortest Path Calculator](https://leetcode.com/problems/design-graph-with-shortest-path-calculator/)
- [787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

From E. W. Dijkstra, "A Note on Two Problems in Connexion with Graphs" (1959).
