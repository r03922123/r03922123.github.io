---
title: Topological sort: ordering a DAG
summary: Why a directed graph can be lined up so every edge points forward exactly when it has no cycle, then the two standard algorithms (Kahn's queue of ready nodes and DFS finishing order), each detecting cycles for free, and two harder uses: two-level ordering and longest paths in a DAG.
group: Graphs
order: 5
tags: topological sort · DAG · graphs
---
Dependencies make a directed graph: an edge \(u \to v\) says "\(u\) must come before \(v\)". A **topological order** lists the nodes so that every edge points forward. Course plans, build systems and spreadsheet recalculation all need one. This post shows when such an order exists and how to find one in linear time. Both algorithms also detect a cycle, the one case in which no order exists.

## When does an order exist?
A **DAG** is a directed graph with no directed cycle. Number the nodes \(v_1, \dots, v_n\) in some order; the order is topological if every edge \((v_i, v_j)\) has \(i < j\).

**Claim.** A directed graph \(G\) has a topological order if and only if \(G\) is a DAG.

**Order implies no cycle.** Suppose, for contradiction, that \(G\) has both a topological order and a cycle \(C\). Let \(v_i\) be the node of \(C\) with the smallest index in that order. Going around \(C\), some node \(v_j\) of \(C\) has the edge \((v_j, v_i)\), and \(j \ge i\) because \(v_i\) has the smallest index on \(C\) (equality only for a self-loop). So this edge does not point forward, which a topological order does not allow. Hence no cycle exists.

**No cycle implies an order.** Induct on \(n\). First, a DAG always has a node with no incoming edge: if every node had one, we could keep walking backwards along incoming edges forever. With only \(n\) nodes, the walk would repeat a node, and that is a cycle. Take such a source \(v\). Deleting \(v\) cannot create a cycle, so \(G - v\) is a DAG with \(n - 1\) nodes and has a topological order by induction. Put \(v\) in front of it. Every edge at \(v\) leaves \(v\), so it points forward, and the result is a topological order of \(G\).

The second half is already an algorithm: repeatedly remove a node with no incoming edges.

## Kahn's algorithm
Keep the **in-degree** of every node, the number of edges still pointing into it. Every node at in-degree zero is ready, so put it in a queue. Each pop goes to the output, and its out-edges are deleted by decrementing their targets' in-degrees. Any target that hits zero becomes ready.

If the graph has a cycle, no node on that cycle ever reaches in-degree zero, so the output stops short of \(n\) nodes. **Checking `len(order) == n` is the cycle test.** Every node is pushed and popped at most once and every edge is looked at once, so the cost is \(O(V + E)\) time and \(O(V + E)\) space for the adjacency lists.

```python
import random
from collections import deque


def kahn(n, edges):
    """Topological order of nodes 0..n-1, or None if there is a cycle."""
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(v for v in range(n) if indeg[v] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else None


def is_topological(n, edges, order):
    if order is None or sorted(order) != list(range(n)):
        return False
    pos = {v: i for i, v in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in edges)


def random_dag(n, m):
    perm = list(range(n))
    random.shuffle(perm)                       # hidden order: edges go forward in perm
    edges = set()
    for _ in range(m if n > 1 else 0):
        i, j = sorted(random.sample(range(n), 2))
        edges.add((perm[i], perm[j]))
    return list(edges)


def has_cycle_brute(n, edges):
    reach = [[False] * n for _ in range(n)]    # transitive closure, Floyd–Warshall style
    for u, v in edges:
        reach[u][v] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True
    return any(reach[i][i] for i in range(n))


for _ in range(300):
    n = random.randint(2, 9)
    dag = random_dag(n, random.randint(0, 20))
    assert is_topological(n, dag, kahn(n, dag))
    noisy = dag + [(random.randrange(n), random.randrange(n)) for _ in range(random.randint(0, 3))]
    assert (kahn(n, noisy) is None) == has_cycle_brute(n, noisy)

assert kahn(0, []) == []
assert kahn(1, []) == [0]
assert kahn(1, [(0, 0)]) is None                    # a self-loop is a cycle
assert kahn(3, [(0, 1), (0, 1), (1, 2)]) == [0, 1, 2]  # duplicate edges count twice in in-degree
assert kahn(4, [(1, 0), (3, 2)]) in ([1, 3, 0, 2], [3, 1, 2, 0], [1, 3, 2, 0], [3, 1, 0, 2])
```

The random DAGs are made by drawing edges forward along a hidden shuffled order, so they are acyclic by construction. Adding a few random edges, self-loops included, sometimes closes a cycle, and the transitive closure confirms each verdict.

[207. Course Schedule](https://leetcode.com/problems/course-schedule/) and [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) are this function. A prerequisite pair `[a, b]` means `b` comes before `a`, so the edge is `b -> a`.

```python
def can_finish(num_courses, prerequisites):
    return kahn(num_courses, [(b, a) for a, b in prerequisites]) is not None


def find_order(num_courses, prerequisites):
    return kahn(num_courses, [(b, a) for a, b in prerequisites]) or []


assert can_finish(2, [[1, 0]]) is True
assert can_finish(2, [[1, 0], [0, 1]]) is False
assert find_order(2, [[1, 0]]) == [0, 1]
assert find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]) in ([0, 1, 2, 3], [0, 2, 1, 3])
assert find_order(1, []) == [0]
assert find_order(2, [[1, 0], [0, 1]]) == []
```

Two variations come straight out of the queue. Replace the deque with a heap and you get the **lexicographically smallest** order. Process the queue one layer at a time and the number of layers is the minimum number of rounds when every ready task can run in parallel.

## DFS finishing order
Depth-first search gives a second algorithm. When the DFS **finishes** a node \(u\) (every out-neighbour fully explored), every node reachable from \(u\) has already finished, unless the search is still in the middle of it, and that only happens on a cycle. So reverse finishing order is a topological order.

Three colours keep track. White means unvisited, grey means on the current DFS path, black means finished. Reaching a grey node means an edge back into the current path, which closes a cycle. A black node is finished and safe to skip. This is the directed-graph version of the back-edge idea from [DFS](../dfs/). The grey state is what makes it work: in a directed graph, reaching an already-finished node is harmless, so a plain visited flag would report false cycles. It also runs in \(O(V + E)\).

```python
import sys

sys.setrecursionlimit(10000)


def dfs_topo(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    WHITE, GREY, BLACK = 0, 1, 2
    color, finished = [WHITE] * n, []

    def visit(u):
        color[u] = GREY
        for v in adj[u]:
            if color[v] == GREY:
                return False                   # back edge: cycle
            if color[v] == WHITE and not visit(v):
                return False
        color[u] = BLACK
        finished.append(u)
        return True

    for s in range(n):
        if color[s] == WHITE and not visit(s):
            return None
    return finished[::-1]


for _ in range(300):
    n = random.randint(2, 9)
    dag = random_dag(n, random.randint(0, 20))
    assert is_topological(n, dag, dfs_topo(n, dag))
    noisy = dag + [(random.randrange(n), random.randrange(n)) for _ in range(random.randint(0, 3))]
    assert (dfs_topo(n, noisy) is None) == has_cycle_brute(n, noisy)

assert dfs_topo(0, []) == []
assert dfs_topo(1, [(0, 0)]) is None
assert dfs_topo(3, [(0, 1), (0, 2), (1, 2)]) == [0, 1, 2]   # 2 is reached again while black: no cycle
```

Which one to use? Kahn's needs no recursion and handles the layer and smallest-order variations. The DFS version fits better when you are already doing a DFS, and the same finishing order is the first pass of Kosaraju's strongly connected components algorithm. For deep graphs in Python, Kahn's avoids the recursion limit.

## Two levels of order: 1203
[1203. Sort Items by Groups Respecting Dependencies](https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/): items belong to groups (or to none, marked `-1`), the items of a group must be contiguous, and `before_items[i]` lists the items that must precede item `i`. Return any valid order, or `[]` if none exists.

Contiguity means groups behave like super-nodes. Give each ungrouped item a fresh group of its own. An item dependency \(j \to i\) across different groups becomes a group dependency \(g(j) \to g(i)\). Then sort twice:

1. Topologically sort the **groups** by the group edges.
2. Topologically sort **all items** by the item edges.
3. Bucket the items by group, keeping their item order, and print the buckets in group order.

A within-group dependency is respected because each bucket keeps the item order. A cross-group one is respected because the group order puts \(g(j)\) first. If either sort finds a cycle, the answer is empty. Building and sorting both graphs is \(O(n + m + E)\) for \(E\) dependency pairs.

```python
from itertools import permutations


def sort_items(n, m, group, before_items):
    group = group[:]
    for i in range(n):
        if group[i] == -1:
            group[i] = m                       # a private group for each loner
            m += 1
    item_edges, group_edges = [], set()
    for i in range(n):
        for j in before_items[i]:
            item_edges.append((j, i))
            if group[j] != group[i]:
                group_edges.add((group[j], group[i]))
    item_order = kahn(n, item_edges)
    group_order = kahn(m, list(group_edges))
    if item_order is None or group_order is None:
        return []
    buckets = [[] for _ in range(m)]
    for i in item_order:
        buckets[group[i]].append(i)
    return [i for g in group_order for i in buckets[g]]


def valid_1203(n, group, before_items, order):
    if sorted(order) != list(range(n)):
        return False
    pos = {v: k for k, v in enumerate(order)}
    if any(pos[j] > pos[i] for i in range(n) for j in before_items[i]):
        return False
    for g in set(group) - {-1}:
        spots = sorted(pos[i] for i in range(n) if group[i] == g)
        if spots[-1] - spots[0] != len(spots) - 1:
            return False
    return True


g1 = [-1, -1, 1, 0, 0, 1, 0, -1]
ok = sort_items(8, 2, g1, [[], [6], [5], [6], [3, 6], [], [], []])
assert valid_1203(8, g1, [[], [6], [5], [6], [3, 6], [], [], []], ok)
assert sort_items(8, 2, g1, [[], [6], [5], [6], [3], [], [4], []]) == []

for _ in range(150):
    n, m = random.randint(1, 6), random.randint(1, 3)
    grp = [random.randint(-1, m - 1) for _ in range(n)]
    before = [random.sample([j for j in range(n) if j != i], random.randint(0, min(2, n - 1)))
              for i in range(n)]
    got = sort_items(n, m, grp, before)
    feasible = any(valid_1203(n, grp, before, list(p)) for p in permutations(range(n)))
    assert (got != []) == feasible
    assert got == [] or valid_1203(n, grp, before, got)
```

The brute force tries every permutation of up to six items, so it checks both that the answers are correct and that `[]` is returned only when no valid order exists.

## Longest path in a DAG
Shortest and longest paths are easy on a DAG: relax edges in topological order. By the time a node is popped, every path into it has been accounted for. Longest path is NP-hard on general graphs, but here it is one pass. [2050. Parallel Courses III](https://leetcode.com/problems/parallel-courses-iii/) asks for the minimum total months when course `i` takes `time[i - 1]`, courses run in parallel, and prerequisites must finish first. A course finishes at its own duration plus the latest finish among its prerequisites. The answer is the largest finish time, which is the heaviest path in the DAG. The cost is \(O(V + E)\).

```python
from functools import lru_cache


def minimum_time(n, relations, time):
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in relations:
        adj[a - 1].append(b - 1)
        indeg[b - 1] += 1
    start = [0] * n                            # earliest start = latest prerequisite finish
    q = deque(v for v in range(n) if indeg[v] == 0)
    best = 0
    while q:
        u = q.popleft()
        done = start[u] + time[u]
        best = max(best, done)
        for v in adj[u]:
            start[v] = max(start[v], done)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return best


def minimum_time_brute(n, relations, time):
    preds = [[] for _ in range(n)]
    for a, b in relations:
        preds[b - 1].append(a - 1)

    @lru_cache(None)
    def finish(v):
        return time[v] + max((finish(p) for p in preds[v]), default=0)

    return max(finish(v) for v in range(n))


assert minimum_time(3, [[1, 3], [2, 3]], [3, 2, 5]) == 8
assert minimum_time(5, [[1, 5], [2, 5], [3, 5], [3, 4], [4, 5]], [1, 2, 3, 4, 5]) == 12

for _ in range(200):
    n = random.randint(1, 10)
    rel = [[u + 1, v + 1] for u, v in random_dag(n, random.randint(0, 15))]
    t = [random.randint(1, 9) for _ in range(n)]
    assert minimum_time(n, rel, t) == minimum_time_brute(n, rel, t)
```

The brute force is the same recurrence written top-down with memoisation. That is [dynamic programming](../dynamic-programming/) on a DAG, and the topological order just picks an order in which to fill the table.

## Peeling leaves of a tree
Kahn's idea also works on undirected trees: repeatedly remove a leaf (a node of degree 1). Each leaf is final once removed, so it can hand its accumulated value to its one remaining neighbour. [2872. Maximum Number of K-Divisible Components](https://leetcode.com/problems/maximum-number-of-k-divisible-components/) peels leaves and cuts off a component whenever a subtree sum is divisible by `k`. [2440. Create Components With Same Value](https://leetcode.com/problems/create-components-with-same-value/) runs the same peel once for each candidate component sum, and [310. Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/) peels until one or two centre nodes remain.

## Practice
- [207. Course Schedule](https://leetcode.com/problems/course-schedule/)
- [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [1203. Sort Items by Groups Respecting Dependencies](https://leetcode.com/problems/sort-items-by-groups-respecting-dependencies/)
- [2322. Minimum Score After Removals on a Tree](https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/) (subtree XORs and entry/exit times on a rooted tree)
- [2440. Create Components With Same Value](https://leetcode.com/problems/create-components-with-same-value/)
- [2872. Maximum Number of K-Divisible Components](https://leetcode.com/problems/maximum-number-of-k-divisible-components/)
- [2050. Parallel Courses III](https://leetcode.com/problems/parallel-courses-iii/)
- [269. Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) (edges from the first differing letter of adjacent words)
- [802. Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/) (Kahn's on the reversed graph)
- [1857. Largest Color Value in a Directed Graph](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/)
- [2392. Build a Matrix With Conditions](https://leetcode.com/problems/build-a-matrix-with-conditions/)
- [444. Sequence Reconstruction](https://leetcode.com/problems/sequence-reconstruction/) (the order is unique exactly when the queue never holds two nodes)

Kahn's algorithm is from A. B. Kahn, "Topological sorting of large networks", Communications of the ACM 5(11), 1962.
