---
title: Union–find: merging groups in nearly constant time
summary: A forest of parent pointers that merges sets and answers "same group?" in amortized near-O(1). Path halving and union by size, then connectivity, cycle detection, and processing events in reverse.
group: Trees and range queries
order: 4
tags: union find · disjoint sets · graphs
---
Union–find (a disjoint-set forest) maintains a partition of items into groups under two operations: **union** two groups, and **find** which group an item is in. It cannot split groups, but it does the two things it does almost in constant time, which makes it the tool for incremental connectivity.

## The structure
Each item points to a parent; a group is a tree, identified by its root. Two refinements keep the trees flat:

- **Union by size:** hang the smaller tree under the larger, so a tree of size \(s\) has height at most \(\log_2 s\).
- **Path halving:** during `find`, point every other node on the path at its grandparent, flattening the path for next time.

Together they give an amortized cost of \(O(\alpha(n))\) per operation, where \(\alpha\) is the inverse Ackermann function, below 5 for any \(n\) that fits in the universe.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.groups = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False                        # already in the same group
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra                    # smaller tree under the larger
        self.size[ra] += self.size[rb]
        self.groups -= 1
        return True


import random
n = 200
uf, label = UnionFind(n), list(range(n))        # label: a naive reference partition
for _ in range(400):
    a, b = random.randrange(n), random.randrange(n)
    uf.union(a, b)
    la, lb = label[a], label[b]
    label = [la if x == lb else x for x in label]
    x, y = random.randrange(n), random.randrange(n)
    assert (uf.find(x) == uf.find(y)) == (label[x] == label[y])
assert uf.groups == len(set(label))
```

`union` returning whether it merged anything is the trick behind the next two problems.

## Counting components
[547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/): union every connected pair; the remaining group count is the answer.

```python
def find_circle_num(is_connected):
    n = len(is_connected)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j]:
                uf.union(i, j)
    return uf.groups


assert find_circle_num([[1, 1, 0], [1, 1, 0], [0, 0, 1]]) == 2
assert find_circle_num([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3
```

## Detecting a cycle
[684. Redundant Connection](https://leetcode.com/problems/redundant-connection/): adding edges one by one, the first edge whose endpoints are already connected closes a cycle. That is exactly when `union` returns `False`. Kruskal's minimum spanning tree uses the same test to skip edges (see [graphs](../graph-algorithms/)).

```python
def find_redundant_connection(edges):
    uf = UnionFind(len(edges) + 1)
    for a, b in edges:
        if not uf.union(a, b):
            return [a, b]


assert find_redundant_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
assert find_redundant_connection([[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]) == [1, 4]
```

## Running time backwards
Union–find cannot delete. When a problem **removes** things over time, reverse it: start from the final state and add them back. [1970. Last Day Where You Can Still Cross](https://leetcode.com/problems/last-day-where-you-can-still-cross/) floods one cell per day and asks for the last day you can walk from the top row to the bottom row. Backwards, it un-floods cells one by one; the first moment two virtual nodes, `TOP` and `BOTTOM`, become connected answers the question.

```python
def latest_day_to_cross(row, col, cells):
    TOP, BOTTOM = row * col, row * col + 1
    uf, land = UnionFind(row * col + 2), [[False] * col for _ in range(row)]
    for day in range(len(cells) - 1, -1, -1):
        r, c = cells[day][0] - 1, cells[day][1] - 1
        land[r][c] = True
        cell = r * col + c
        if r == 0:
            uf.union(cell, TOP)
        if r == row - 1:
            uf.union(cell, BOTTOM)
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < row and 0 <= nc < col and land[nr][nc]:
                uf.union(cell, nr * col + nc)
        if uf.find(TOP) == uf.find(BOTTOM):
            return day                          # cells[day] is still water on that day
    return 0


assert latest_day_to_cross(2, 2, [[1, 1], [2, 1], [1, 2], [2, 2]]) == 2
assert latest_day_to_cross(2, 2, [[1, 1], [1, 2], [2, 1], [2, 2]]) == 1
assert latest_day_to_cross(3, 3, [[1, 2], [2, 1], [3, 3], [2, 2], [1, 1], [1, 3], [2, 3], [3, 2], [3, 1]]) == 3
```

Forward, the same problem is a [binary search](../binary-search/) over days with a BFS per check, \(O(rc \log(rc))\); backwards with union–find it is one pass.

## Practice
- [2382. Maximum Segment Sum After Removals](https://leetcode.com/problems/maximum-segment-sum-after-removals/) (reverse time again)
- [2092. Find All People With Secret](https://leetcode.com/problems/find-all-people-with-secret/)
- [1102. Path With Maximum Minimum Value](https://leetcode.com/problems/path-with-maximum-minimum-value/)
- [1632. Rank Transform of a Matrix](https://leetcode.com/problems/rank-transform-of-a-matrix/)
- [721. Accounts Merge](https://leetcode.com/problems/accounts-merge/)

The \(O(\alpha(n))\) bound is due to R. E. Tarjan, "Efficiency of a Good But Not Linear Set Union Algorithm" (1975).
