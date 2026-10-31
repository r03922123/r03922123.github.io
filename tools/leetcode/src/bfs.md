---
title: BFS: shortest paths by layers, and from both ends
summary: Breadth-first search finds shortest paths when every step costs the same. The layer-by-layer template, BFS over a state space larger than the grid, 0-1 BFS for two edge costs, and bidirectional search.
group: Graphs
order: 1
tags: BFS · shortest path · graphs
---
Breadth-first search explores a graph in rings: everything one step from the source, then everything two steps away, and so on. The first time it reaches a node is therefore along a shortest path, provided every edge costs the same. For weighted edges, see [Dijkstra](../dijkstra/).

## Layer by layer
Processing one whole frontier at a time makes the distance explicit, which is convenient when the answer is "the number of steps". Mark nodes as seen when they are **added**, not when they are popped, or the same node can enter the queue many times.

```python
from collections import deque


def bfs_layers(start, neighbors, goal):
    if start == goal:
        return 0
    seen, frontier, steps = {start}, [start], 0
    while frontier:
        steps += 1
        nxt = []
        for node in frontier:
            for nb in neighbors(node):
                if nb == goal:
                    return steps
                if nb not in seen:
                    seen.add(nb)                # mark on enqueue
                    nxt.append(nb)
        frontier = nxt
    return -1


grid = ["..#", ".#.", "..."]
def grid_neighbors(cell):
    r, c = cell
    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
        if 0 <= nr < 3 and 0 <= nc < 3 and grid[nr][nc] == ".":
            yield nr, nc


assert bfs_layers((0, 0), grid_neighbors, (1, 2)) == 5
assert bfs_layers((0, 0), grid_neighbors, (0, 0)) == 0
```

## The state is more than the position
When the rules depend on something besides where you are, put it in the node. [1293. Shortest Path in a Grid with Obstacles Elimination](https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/) lets you remove up to `k` walls, so the state is `(row, col, walls_left)`: the same cell reached with more removals left is a different, better state.

```python
def shortest_path_with_removals(grid, k):
    rows, cols = len(grid), len(grid[0])
    if k >= rows + cols - 2:                    # enough removals to walk straight there
        return rows + cols - 2
    start = (0, 0, k)
    seen, q = {start}, deque([(start, 0)])
    while q:
        (r, c, left), dist = q.popleft()
        if (r, c) == (rows - 1, cols - 1):
            return dist
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols:
                state = (nr, nc, left - grid[nr][nc])
                if state[2] >= 0 and state not in seen:
                    seen.add(state)
                    q.append((state, dist + 1))
    return -1


assert shortest_path_with_removals([[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 1], [0, 0, 0]], 1) == 6
assert shortest_path_with_removals([[0, 1, 1], [1, 1, 1], [1, 0, 0]], 1) == -1
```

The state space is \(rows \times cols \times (k+1)\), and BFS visits each state once.

## 0-1 BFS: two edge costs
If edges cost either 0 or 1, a deque replaces Dijkstra's heap: push 0-cost neighbors to the **front** and 1-cost ones to the back, and nodes still come out in distance order. [1368. Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/): following a cell's arrow is free, going any other way costs 1.

```python
def min_cost(grid):
    rows, cols = len(grid), len(grid[0])
    moves = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    dist = [[float("inf")] * cols for _ in range(rows)]
    dist[0][0], dq = 0, deque([(0, 0)])
    while dq:
        r, c = dq.popleft()
        for sign, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            cost = 0 if grid[r][c] == sign else 1
            if 0 <= nr < rows and 0 <= nc < cols and dist[r][c] + cost < dist[nr][nc]:
                dist[nr][nc] = dist[r][c] + cost
                if cost == 0:
                    dq.appendleft((nr, nc))
                else:
                    dq.append((nr, nc))
    return dist[-1][-1]


assert min_cost([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1], [2, 2, 2, 2]]) == 3
assert min_cost([[1, 1, 3], [3, 2, 2], [1, 1, 4]]) == 0
```

## Bidirectional BFS
With branching factor \(b\) and answer depth \(L\), BFS explores about \(b^L\) nodes. Searching from both ends and stopping when the frontiers meet explores about \(2 b^{L/2}\), an enormous saving. Always expand the **smaller** frontier. [127. Word Ladder](https://leetcode.com/problems/word-ladder/): words are nodes, one changed letter is an edge, and wildcard patterns like `h*t` index the neighbors.

```python
from collections import defaultdict


def ladder_length(begin, end, word_list):
    words = set(word_list)
    if end not in words:
        return 0
    buckets = defaultdict(list)
    for w in words | {begin}:
        for i in range(len(w)):
            buckets[w[:i] + "*" + w[i + 1:]].append(w)
    front, back, seen, length = {begin}, {end}, {begin, end}, 1
    while front and back:
        if len(front) > len(back):
            front, back = back, front           # expand the smaller side
        length += 1
        nxt = set()
        for w in front:
            for i in range(len(w)):
                for nb in buckets[w[:i] + "*" + w[i + 1:]]:
                    if nb in back:
                        return length
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
        front = nxt
    return 0


assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
```

## Practice
- [1654. Minimum Jumps to Reach Home](https://leetcode.com/problems/minimum-jumps-to-reach-home/)
- [1494. Parallel Courses II](https://leetcode.com/problems/parallel-courses-ii/) (BFS over bitmask states)
- [752. Open the Lock](https://leetcode.com/problems/open-the-lock/)
- [1102. Path With Maximum Minimum Value](https://leetcode.com/problems/path-with-maximum-minimum-value/)
- [2290. Minimum Obstacle Removal to Reach Corner](https://leetcode.com/problems/minimum-obstacle-removal-to-reach-corner/) (0-1 BFS)
