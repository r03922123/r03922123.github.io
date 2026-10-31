---
title: Bitmask DP: dynamic programming over subsets
summary: When n is around 20, a subset fits in one integer and "which items are used" becomes a DP index. Three archetypes (ordering, grouping, assignment), Held–Karp for the travelling salesman, BFS over (node, mask), and submask enumeration in O(3^n).
group: Dynamic programming, bits and games
order: 3
tags: dynamic programming · bitmask · subsets · TSP
---
Some problems ask for the best way to order, group or match a small set of things, and the only honest description of "where you are" is **which items have been used so far**. Trying every permutation costs \(n!\); remembering only the set of used items costs \(2^n\). At \(n = 15\) that is \(1.3 \times 10^{12}\) against about 33,000, and the second number is the one that fits in a DP table.

The trick is that a subset of \(\{0, \dots, n-1\}\) is just an \(n\)-bit integer, so the table can be a plain list indexed by masks. The constraint that tells you to reach for this is a suspiciously small bound: \(n \le 12\), \(16\) or \(20\).

## Subsets as integers
Bit \(i\) of `mask` says whether item \(i\) is in the set. Every set operation becomes one machine operation:

| Operation | Expression |
| --- | --- |
| empty set, full set | `0`, `(1 << n) - 1` |
| is `i` in the set? | `mask >> i & 1` |
| remove `i` | `mask & ~(1 << i)` |
| size of the set | `mask.bit_count()` (Python 3.10+), or `bin(mask).count("1")` |
| lowest set bit (as a one-bit mask) | `mask & -mask` |

Adding `i` is the bitwise OR, `mask | 1 << i`.

Counting masks upward from `0` to `2^n - 1` visits every subset **after all of its proper subsets**, because removing a bit always makes the number smaller. That single fact means a bottom-up loop `for mask in range(1 << n)` is already in a valid topological order for any recurrence that goes from a set to a smaller one.

```python
import random
from functools import cache
from itertools import permutations, product


def popcount(x):
    return bin(x).count("1")


def members(mask):
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


assert members(0b1011) == [0, 1, 3]
assert 0b1011 & -0b1011 == 0b0001 and popcount(0b1011) == 3
```

The rest of the post is three shapes of problem that all use `mask` as "items already consumed", then two classic applications and a counting trick.

## Archetype 1: orderings
When the answer is a permutation, build it one position at a time. If the mask holds the items already placed, the next position is simply `popcount(mask)`, so the mask alone is the whole state.

[526. Beautiful Arrangement](https://leetcode.com/problems/beautiful-arrangement/) counts permutations of \(1..n\) where, at every 1-based position \(i\), the value divides \(i\) or \(i\) divides the value. Let `ways[mask]` be the number of valid ways to fill positions \(1..|mask|\) using exactly the values in `mask`. To extend, choose a value not yet used that is compatible with the next position.

```python
def count_arrangement(n):
    ways = [0] * (1 << n)
    ways[0] = 1
    for mask in range(1 << n):
        if not ways[mask]:
            continue
        pos = popcount(mask) + 1                     # next 1-based position to fill
        for v in range(n):                           # bit v stands for the value v + 1
            if not mask >> v & 1 and (pos % (v + 1) == 0 or (v + 1) % pos == 0):
                ways[mask | 1 << v] += ways[mask]
    return ways[-1]


assert count_arrangement(2) == 2
assert count_arrangement(1) == 1


def count_arrangement_brute(n):
    return sum(all(p % i == 0 or i % p == 0 for i, p in enumerate(perm, 1))
               for perm in permutations(range(1, n + 1)))


for n in range(1, 9):
    assert count_arrangement(n) == count_arrangement_brute(n)
```

There are \(2^n\) states and \(n\) choices from each: \(O(n \cdot 2^n)\) time, \(O(2^n)\) space. A permutation search would revisit the same set of used values once for every order in which they were placed; the table collapses those orders into one entry.

## Archetype 2: grouping into equal parts
[698. Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/) asks whether `nums` splits into `k` groups of equal sum \(T = \text{sum} / k\). Fill the groups one after another, never leaving a group half-done before starting the next. Then for any set of used numbers, the total is fixed, so the current group's fill level is just `total(mask) % T`. The state is again only the mask.

`fill[mask]` is that fill level when `mask` can be reached legally, and `-1` otherwise. A number may be added only if it does not overflow the current group.

```python
def can_partition_k_subsets(nums, k):
    total = sum(nums)
    if total % k:
        return False
    target, n = total // k, len(nums)
    fill = [-1] * (1 << n)
    fill[0] = 0
    for mask in range(1 << n):
        if fill[mask] < 0:
            continue
        for i in range(n):
            if not mask >> i & 1 and fill[mask] + nums[i] <= target:
                fill[mask | 1 << i] = (fill[mask] + nums[i]) % target
    return fill[-1] == 0


assert can_partition_k_subsets([4, 3, 2, 3, 5, 2, 1], 4) is True
assert can_partition_k_subsets([1, 2, 3, 4], 3) is False
assert can_partition_k_subsets([7], 1) is True
assert can_partition_k_subsets([2, 2, 2, 2, 3, 3], 2) is True        # 2+2+3 twice
assert can_partition_k_subsets([5, 1, 1, 1], 2) is False            # 5 cannot share a group


def can_partition_brute(nums, k):
    for labels in product(range(k), repeat=len(nums)):
        sums = [0] * k
        for x, g in zip(nums, labels):
            sums[g] += x
        if len(set(sums)) == 1:
            return True
    return False


for _ in range(300):
    nums = [random.randint(1, 6) for _ in range(random.randint(1, 7))]
    k = random.randint(1, 3)
    assert can_partition_k_subsets(nums, k) == can_partition_brute(nums, k)
```

\(O(n \cdot 2^n)\). The modulo is what makes the archetype work: a group that reaches exactly \(T\) wraps to \(0\), and the next number starts the next group. Sorting the numbers or pruning duplicates helps a backtracking solution, but the DP needs neither.

## Archetype 3: assignment
Here two sets meet: people and hats, workers and bikes, jobs and machines. Iterate over the **larger** side one item at a time, and keep the **smaller** side as the mask. The DP is then a 0/1 knapsack in disguise: for each item, either skip it or give it to one still-free member of the mask.

[1434. Number of Ways to Wear Different Hats to Each Other](https://leetcode.com/problems/number-of-ways-to-wear-different-hats-to-each-other/) has at most 10 people and 40 hats, and asks for the number of ways to give every person a distinct hat they like. Indexing by hats with a 10-bit mask of people who already have one gives \(40 \cdot 2^{10}\) states; the other way round would need \(2^{40}\).

`ways[mask]` counts ways to dress exactly the people in `mask` using the hats seen so far. Processing a new hat, each state either ignores it or hands it to a person in the mask who likes it. Walking masks **downward** reads only entries that have not yet been updated for this hat, so no hat is used twice (the same reason a 0/1 knapsack loops capacity downward; see [knapsack](../knapsack/)).

```python
def number_ways(hats):
    mod, n = 10**9 + 7, len(hats)
    likers = [[] for _ in range(41)]
    for person, liked in enumerate(hats):
        for h in liked:
            likers[h].append(person)
    ways = [0] * (1 << n)
    ways[0] = 1
    for h in range(1, 41):
        for mask in range((1 << n) - 1, 0, -1):       # downward: hat h used at most once
            for person in likers[h]:
                if mask >> person & 1:
                    ways[mask] = (ways[mask] + ways[mask ^ 1 << person]) % mod
    return ways[-1]


assert number_ways([[3, 4], [4, 5], [5]]) == 1
assert number_ways([[3, 5, 1], [3, 5]]) == 4
assert number_ways([[1, 2, 3, 4]] * 4) == 24


def number_ways_brute(hats):
    return sum(len(set(pick)) == len(pick) for pick in product(*hats))


for _ in range(200):
    hats = [random.sample(range(1, 9), random.randint(1, 4)) for _ in range(random.randint(1, 5))]
    assert number_ways(hats) == number_ways_brute(hats)
```

\(O(H \cdot 2^n \cdot n)\) for \(H = 40\) hats and \(n\) people.

When every item on one side must be matched, the outer counter disappears again: hand out that side in a fixed order and let the mask record what has been taken from the other side. In [1066. Campus Bikes II](https://leetcode.com/problems/campus-bikes-ii/) there are \(m \le n \le 10\) workers and bikes and every worker gets a bike, so the mask holds the bikes already taken, and worker number `popcount(mask)` is the one choosing next, exactly as in archetype 1:

```python
def assign_bikes(workers, bikes):
    m, n = len(workers), len(bikes)

    def dist(w, b):
        return abs(workers[w][0] - bikes[b][0]) + abs(workers[w][1] - bikes[b][1])

    INF = float("inf")
    best = [INF] * (1 << n)
    best[0] = 0
    answer = INF
    for mask in range(1 << n):
        if best[mask] == INF:
            continue
        w = popcount(mask)                           # the next worker to receive a bike
        if w == m:
            answer = min(answer, best[mask])
            continue
        for b in range(n):
            if not mask >> b & 1:
                nxt = mask | 1 << b
                best[nxt] = min(best[nxt], best[mask] + dist(w, b))
    return answer


assert assign_bikes([[0, 0], [2, 1]], [[1, 2], [3, 3]]) == 6
assert assign_bikes([[0, 0], [1, 1], [2, 0]], [[1, 0], [2, 2], [2, 1]]) == 4
assert assign_bikes([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
                    [[0, 999], [1, 999], [2, 999], [3, 999], [4, 999]]) == 4995
assert assign_bikes([[0, 0]], [[5, 5], [1, 0], [9, 9]]) == 1         # fewer workers than bikes


for _ in range(200):
    n = random.randint(1, 6)
    m = random.randint(1, n)
    pts = lambda c: [[random.randint(0, 20), random.randint(0, 20)] for _ in range(c)]
    workers, bikes = pts(m), pts(n)
    brute = min(sum(abs(workers[w][0] - bikes[b][0]) + abs(workers[w][1] - bikes[b][1])
                    for w, b in enumerate(choice))
                for choice in permutations(range(n), m))
    assert assign_bikes(workers, bikes) == brute
```

\(O(n \cdot 2^n)\), against \(n! / (n - m)!\) for trying every choice. The same template solves [1879. Minimum XOR Sum of Two Arrays](https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/) with `nums1[i] ^ nums2[j]` as the cost and \(m = n\).

## Held–Karp: the travelling salesman
Sometimes the set of used items is not enough, because the cost of the next step depends on **where you are standing**. In the travelling salesman problem, the cost of adding city \(v\) depends on the last city visited, so the state becomes a pair: `cost[mask][v]` is the cheapest path that starts at city 0, visits exactly the cities in `mask`, and ends at \(v \in\) `mask`.

\[
\text{cost}[S][v] = \min_{u \in S \setminus \{v\}} \big( \text{cost}[S \setminus \{v\}][u] + d(u, v) \big) \quad (0 \in S,\ v \ne 0), \qquad \text{cost}[\{0\}][0] = 0
\]

The tour closes with \(\min_v \text{cost}[\text{all}][v] + d(v, 0)\). Fixing city 0 as the start is free, since a tour is a cycle. To recover the tour itself, walk backwards from the best final city, at each step finding the predecessor that achieves the minimum.

```python
def held_karp(d):
    n = len(d)
    INF = float("inf")
    cost = [[INF] * n for _ in range(1 << n)]
    cost[1][0] = 0
    for mask in range(1, 1 << n, 2):                 # odd masks: city 0 is always included
        for u in range(n):
            if cost[mask][u] == INF:
                continue
            for v in range(n):
                if not mask >> v & 1:
                    nxt = mask | 1 << v
                    cost[nxt][v] = min(cost[nxt][v], cost[mask][u] + d[u][v])
    full = (1 << n) - 1
    if n == 1:
        return 0, [0, 0]
    last = min(range(1, n), key=lambda v: cost[full][v] + d[v][0])
    best = cost[full][last] + d[last][0]

    tour, mask, v = [0], full, last                  # walk back through the table
    while v:
        tour.append(v)
        prev = mask ^ 1 << v
        v = next(u for u in range(n) if prev >> u & 1 and cost[prev][u] + d[u][v] == cost[mask][v])
        mask = prev
    tour.append(0)
    return best, tour[::-1]


def tour_length(d, tour):
    return sum(d[a][b] for a, b in zip(tour, tour[1:]))


assert held_karp([[0]]) == (0, [0, 0])
assert held_karp([[0, 3], [5, 0]]) == (8, [0, 1, 0])
assert held_karp([[0, 1, 9], [9, 0, 1], [1, 9, 0]])[0] == 3            # asymmetric: 0 -> 1 -> 2 -> 0


for _ in range(100):
    n = random.randint(2, 7)
    d = [[0 if i == j else random.randint(1, 30) for j in range(n)] for i in range(n)]   # may be asymmetric
    best, tour = held_karp(d)
    brute = min(tour_length(d, (0,) + p + (0,)) for p in permutations(range(1, n)))
    assert best == brute and tour_length(d, tour) == best and sorted(tour[1:]) == list(range(n))
```

This is \(O(n^2 2^n)\) time and \(O(n 2^n)\) space. At \(n = 20\) that is about \(4 \times 10^8\) basic steps, slow in Python but feasible in a compiled language, while \(19!\) is about \(1.2 \times 10^{17}\). It remains the best known exact worst-case bound for general TSP, though the problem is NP-hard, so nobody expects a polynomial one.

## Shortest path visiting all nodes
[847. Shortest Path Visiting All Nodes](https://leetcode.com/problems/shortest-path-visiting-all-nodes/) is the unweighted cousin: start anywhere, end anywhere, revisits allowed, and find the fewest edges that touch every node. Because every edge costs 1, the Held–Karp table can be filled by [BFS](../bfs/) over states `(node, visited_mask)`. Start from all \(n\) states `(v, 1 << v)` at once; the first time any state with a full mask comes off the queue, its distance is the answer.

```python
from collections import deque


def shortest_path_length(graph):
    n = len(graph)
    full = (1 << n) - 1
    seen = [[False] * (1 << n) for _ in range(n)]
    queue = deque()
    for v in range(n):
        seen[v][1 << v] = True
        queue.append((v, 1 << v, 0))
    while queue:
        v, mask, steps = queue.popleft()
        if mask == full:
            return steps
        for u in graph[v]:
            nxt = mask | 1 << u
            if not seen[u][nxt]:
                seen[u][nxt] = True
                queue.append((u, nxt, steps + 1))


assert shortest_path_length([[1, 2, 3], [0], [0], [0]]) == 4
assert shortest_path_length([[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]]) == 4
assert shortest_path_length([[]]) == 0                                # a single node is already covered
assert shortest_path_length([[1], [0, 2], [1, 3], [2]]) == 3          # a path: walk it end to end
```

There are \(n 2^n\) states and each scans the adjacency list, so \(O(n^2 2^n)\) in the worst case. As a cross-check, revisiting means "walk from \(a\) to \(b\)" really costs the shortest-path distance between them, so the answer equals the best ordering of the nodes under all-pairs distances:

```python
def shortest_path_brute(graph):
    n = len(graph)
    dist = [[0 if i == j else float("inf") for j in range(n)] for i in range(n)]
    for v in range(n):
        for u in graph[v]:
            dist[v][u] = 1
    for k in range(n):                                # Floyd–Warshall
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return min(sum(dist[a][b] for a, b in zip(p, p[1:])) for p in permutations(range(n)))


for _ in range(150):
    n = random.randint(1, 7)
    order = list(range(n))
    random.shuffle(order)
    edges = {(order[i - 1], order[i]) for i in range(1, n)}    # a random spanning tree keeps it connected
    edges |= {(i, j) for i in range(n) for j in range(i + 1, n) if random.random() < 0.3}
    graph = [[] for _ in range(n)]
    for a, b in edges:
        if a != b and b not in graph[a]:
            graph[a].append(b)
            graph[b].append(a)
    assert shortest_path_length(graph) == shortest_path_brute(graph)
```

## Enumerating submasks in O(3^n)
Partitioning problems with **unequal** parts need a different move: "the last group is some subset of what remains". That requires looping over every submask of a mask. The idiom `sub = (sub - 1) & mask` steps through them in decreasing order: subtracting 1 clears the lowest set bit and sets every bit below it, and the `& mask` throws away the bits outside the mask.

Looping over the submasks of every mask costs much less than \(4^n\). Each element is, independently, outside the mask, in the mask but not the submask, or in both, so the number of (mask, submask) pairs is exactly

\[
\sum_{k=0}^{n} \binom{n}{k} 2^k = 3^n.
\]

```python
def submasks(mask):
    sub = mask
    while True:
        yield sub
        if sub == 0:
            return
        sub = (sub - 1) & mask


assert list(submasks(0b101)) == [0b101, 0b100, 0b001, 0]
assert sum(1 for m in range(1 << 8) for _ in submasks(m)) == 3 ** 8
```

[1723. Find Minimum Time to Finish All Jobs](https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/) splits `jobs` among `k` workers to minimize the busiest worker's total. Precompute `load[mask]`, the total of the jobs in `mask` (one addition per mask using the lowest bit). Then `best[mask]` is the minimum possible maximum load when the jobs in `mask` go to the workers handled so far; adding one more worker means giving them some submask:

```python
def minimum_time_required(jobs, k):
    n = len(jobs)
    load = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        load[mask] = load[mask ^ low] + jobs[low.bit_length() - 1]
    best = load[:]                                    # one worker takes everything
    for _ in range(k - 1):
        nxt = best[:]
        for mask in range(1, 1 << n):
            for sub in submasks(mask):                # sub goes to the new worker
                nxt[mask] = min(nxt[mask], max(best[mask ^ sub], load[sub]))
        best = nxt
    return best[-1]


assert minimum_time_required([3, 2, 3], 3) == 3
assert minimum_time_required([1, 2, 4, 7, 8], 2) == 11
assert minimum_time_required([5], 1) == 5
assert minimum_time_required([4, 4, 4], 3) == 4                      # one job each


for _ in range(200):
    jobs = [random.randint(1, 10) for _ in range(random.randint(1, 6))]
    k = random.randint(1, 4)
    brute = min(max(sum(j for j, w in zip(jobs, who) if w == g) for g in range(k))
                for who in product(range(k), repeat=len(jobs)))
    assert minimum_time_required(jobs, k) == brute
```

\(O(k \cdot 3^n)\). At \(n = 12\) that is about half a million submask steps per worker, which is why the constraint is 12 here rather than 20.

## Practice
- [1494. Parallel Courses II](https://leetcode.com/problems/parallel-courses-ii/) (submasks of the currently available courses)
- [638. Shopping Offers](https://leetcode.com/problems/shopping-offers/) (the state is a vector of small counts, the same idea in a mixed radix)
- [2172. Maximum AND Sum of Array](https://leetcode.com/problems/maximum-and-sum-of-array/) (each slot holds two numbers: use a base-3 mask, or duplicate every slot)
- [1879. Minimum XOR Sum of Two Arrays](https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/)
- [943. Find the Shortest Superstring](https://leetcode.com/problems/find-the-shortest-superstring/) (Held–Karp with overlap costs)
- [1986. Minimum Number of Work Sessions to Finish the Tasks](https://leetcode.com/problems/minimum-number-of-work-sessions-to-finish-the-tasks/)
- [2305. Fair Distribution of Cookies](https://leetcode.com/problems/fair-distribution-of-cookies/)
- [1349. Maximum Students Taking Exam](https://leetcode.com/problems/maximum-students-taking-exam/) (a mask per row, see also [bit manipulation](../bit-manipulation/))

The subset recurrence for the travelling salesman is due independently to R. Bellman, "Dynamic Programming Treatment of the Travelling Salesman Problem", Journal of the ACM (1962), and M. Held and R. M. Karp, "A Dynamic Programming Approach to Sequencing Problems", Journal of SIAM (1962). The split into ordering, grouping and assignment archetypes follows the LeetCode Discuss article "Dynamic programming on subsets with examples explained".
