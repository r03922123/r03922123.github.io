---
title: DFS: reachability, components and connectivity
summary: Depth-first search as the tool for "what can I reach?". An iterative template, counting components, testing whether a directed graph is strongly connected with one reversed pass, and finding bridges with low-link values.
group: Graphs
order: 2
tags: DFS · connectivity · graphs
---
Depth-first search follows one path as far as it goes before backing up. It does not find shortest paths (that is [BFS](../bfs/)), but it answers **reachability** questions: what can I reach from here, is everything connected, which edges are essential. Its other gift is structure: the order in which DFS finishes nodes underlies topological sorting, strongly connected components and bridges.

## An iterative template
Recursion is the natural way to write DFS, but Python's recursion limit (1000 by default) makes it fragile on large graphs. An explicit stack has no such limit.

```python
def reachable(graph, start):
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        for nb in graph.get(node, ()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


g = {0: [1], 1: [2], 2: [0], 3: [2]}
assert reachable(g, 0) == {0, 1, 2}
assert reachable(g, 3) == {0, 1, 2, 3}
```

## Counting components
Start a DFS from every node not yet seen; each start finds one component. [200. Number of Islands](https://leetcode.com/problems/number-of-islands/):

```python
def num_islands(grid):
    rows, cols = len(grid), len(grid[0])
    seen, count = set(), 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in seen:
                count += 1
                seen.add((r, c))
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1" and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
    return count


assert num_islands(["11000", "11000", "00100", "00011"]) == 3
```

For components that grow over time, [union–find](../union-find/) is the better tool.

## Is a directed graph strongly connected?
In an undirected graph, one DFS from any node that reaches everything proves the graph connected. A directed graph needs more: every node must reach every other. Two passes suffice. From any node \(s\), DFS must reach every node (so \(s\) reaches all); then DFS on the graph with **every edge reversed** must also reach every node (so all reach \(s\)). Any \(u\) can then reach any \(v\) through \(s\).

```python
def strongly_connected(n, edges):
    forward, backward = {}, {}
    for a, b in edges:
        forward.setdefault(a, []).append(b)
        backward.setdefault(b, []).append(a)
    return len(reachable(forward, 0)) == n and len(reachable(backward, 0)) == n


assert strongly_connected(3, [(0, 1), (1, 2), (2, 0)])
assert not strongly_connected(3, [(0, 1), (1, 2)])
```

## Bridges: edges whose removal disconnects the graph
[1192. Critical Connections in a Network](https://leetcode.com/problems/critical-connections-in-a-network/). Number nodes in DFS order (`disc`), and let `low[v]` be the smallest number reachable from `v`'s subtree using at most one non-tree edge. A tree edge `(u, v)` is a bridge exactly when `low[v] > disc[u]`: nothing under `v` can get back to `u` or above without it. The iterative version keeps, for each stack frame, an iterator over the remaining neighbors.

```python
def critical_connections(n, connections):
    graph = [[] for _ in range(n)]
    for a, b in connections:
        graph[a].append(b)
        graph[b].append(a)
    disc, low, bridges, clock = [-1] * n, [0] * n, [], 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = clock
        clock += 1
        stack = [(root, -1, iter(graph[root]))]
        while stack:
            node, parent, it = stack[-1]
            nb = next(it, None)
            if nb is None:                      # node finished: report to its parent
                stack.pop()
                if parent != -1:
                    low[parent] = min(low[parent], low[node])
                    if low[node] > disc[parent]:
                        bridges.append(sorted((parent, node)))
            elif nb == parent:
                continue
            elif disc[nb] == -1:                # tree edge: descend
                disc[nb] = low[nb] = clock
                clock += 1
                stack.append((nb, node, iter(graph[nb])))
            else:                               # back edge
                low[node] = min(low[node], disc[nb])
    return sorted(bridges)


def brute(n, connections):
    out = []
    for skip in connections:
        rest = {}
        for e in connections:
            if e is not skip:
                rest.setdefault(e[0], []).append(e[1])
                rest.setdefault(e[1], []).append(e[0])
        if len(reachable(rest, 0)) < n:
            out.append(sorted(skip))
    return sorted(out)


assert critical_connections(4, [[0, 1], [1, 2], [2, 0], [1, 3]]) == [[1, 3]]
import random
for _ in range(200):
    n = random.randint(2, 7)
    edges = [[i, random.randrange(i)] for i in range(1, n)]          # a random tree keeps it connected
    edges += [list(e) for e in {tuple(sorted(random.sample(range(n), 2))) for _ in range(random.randint(0, 4))}
              if list(e) not in edges and list(e)[::-1] not in edges]
    assert critical_connections(n, edges) == brute(n, edges)
```

`skip` is a parent **node**, which is wrong for graphs with parallel edges; LeetCode's inputs have none.

## Practice
- [1778. Shortest Path in a Hidden Grid](https://leetcode.com/problems/shortest-path-in-a-hidden-grid/) (DFS to map, BFS to measure)
- [1916. Count Ways to Build Rooms in an Ant Colony](https://leetcode.com/problems/count-ways-to-build-rooms-in-an-ant-colony/)
- [1240. Tiling a Rectangle with the Fewest Squares](https://leetcode.com/problems/tiling-a-rectangle-with-the-fewest-squares/) (DFS with pruning)
- [1368. Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/)
- [1102. Path With Maximum Minimum Value](https://leetcode.com/problems/path-with-maximum-minimum-value/)

Bridge finding with low-link values is due to R. E. Tarjan, "A note on finding the bridges of a graph" (1974).
