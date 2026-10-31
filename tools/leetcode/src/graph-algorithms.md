---
title: Spanning trees, Euler paths and matchings
summary: Three classic graph algorithms beyond shortest paths. Minimum spanning trees from the cut and cycle properties (Kruskal, Prim, and the O(n²) dense Prim), Hierholzer's algorithm for Euler paths, and Kuhn's augmenting paths for bipartite matching.
group: Graphs
order: 6
tags: minimum spanning tree · Euler path · bipartite matching
---
Shortest paths get their own posts ([BFS](../bfs/), [Dijkstra](../dijkstra/), [the rest](../shortest-paths/)). This one collects three other questions that come up again and again on graphs, each with a short, greedy-looking algorithm whose correctness rests on one clean idea:

- **Minimum spanning tree:** the cheapest set of edges that connects every node. The idea is the cut property.
- **Euler path:** a walk that uses every edge exactly once. The idea is that you can splice detours into a path after the fact.
- **Bipartite matching:** the largest set of pairs with nobody in two pairs. The idea is the augmenting path.

## Minimum spanning trees
A spanning tree of a connected graph with \(n\) nodes is a set of \(n - 1\) edges that connects them all with no cycle. Given edge weights, we want the one with the smallest total. Two facts about cuts and cycles do all the work. A **cut** splits the nodes into two sides, \(S\) and the rest; an edge **crosses** the cut if its endpoints are on different sides.

**Cut property.** Let \(A\) be a set of edges contained in some MST, and take any cut that no edge of \(A\) crosses. Then \(A\) plus a lightest edge \(e\) crossing the cut is still contained in some MST. Proof: take an MST \(T \supseteq A\). If \(e \in T\) we are done. Otherwise adding \(e\) to \(T\) creates a cycle, and a cycle that crosses the cut once must cross it again, at some edge \(f\) of \(T\); \(f\) is not in \(A\), because no edge of \(A\) crosses the cut. Swapping, \(T + e - f\) is still a spanning tree, still contains \(A\), and since \(w(e) \le w(f)\) it costs no more than \(T\). So it is also an MST. (With \(A\) empty this says: a lightest edge across any cut is in some MST.) Both algorithms below grow \(A\) one edge at a time, and this form of the property is what keeps "\(A\) is part of some MST" true at every step.

**Cycle property.** For any cycle, if one edge \(f\) on it is strictly heavier than all the others, \(f\) is in no MST. Proof: if an MST \(T\) contained \(f\), removing \(f\) would split \(T\) into two pieces, a cut. The rest of the cycle connects the two endpoints of \(f\), so some other cycle edge \(e\) crosses that cut, and \(T - f + e\) is a spanning tree strictly cheaper than \(T\), a contradiction.

With ties in the weights the minimum spanning tree need not be unique, which is why the statements say "some MST" and "strictly heavier". With all weights distinct, the MST is unique and both properties hold for it.

### Kruskal: lightest edges first
Sort the edges by weight and take each one unless it would close a cycle. Each accepted edge \((u, v)\) is a lightest edge crossing the cut "\(u\)'s current component versus everything else": no taken edge crosses that cut, and any lighter crossing edge would have been examined earlier and accepted, merging the components. So the cut property says taking it is safe. Skipped edges need no argument of their own, since the accepted edges end up spanning; with distinct weights the cycle property adds that each skipped edge, the heaviest on the cycle it would close, is in no MST at all. "Would it close a cycle?" is exactly the question [union–find](../union-find/) answers.

Sorting dominates: \(O(E \log E)\) time, plus \(O(E\,\alpha(V))\) for the union–find; space is \(O(V)\) for the union–find plus \(O(E)\) for the sorted copy of the edges. [1135. Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) is the textbook statement, with \(-1\) if the cities cannot all be connected.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def kruskal(n, edges):
    """edges: (u, v, w) with 0-based nodes. Returns the MST weight, or -1 if disconnected."""
    uf, total, used = UnionFind(n), 0, 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if uf.union(u, v):
            total += w
            used += 1
            if used == n - 1:
                break
    return total if used == n - 1 else -1


def minimum_cost(n, connections):
    return kruskal(n, [(u - 1, v - 1, w) for u, v, w in connections])


assert minimum_cost(3, [[1, 2, 5], [1, 3, 6], [2, 3, 1]]) == 6
assert minimum_cost(4, [[1, 2, 3], [3, 4, 4]]) == -1
assert minimum_cost(1, []) == 0                                 # one city, nothing to connect
assert minimum_cost(3, [[1, 2, 1], [2, 3, 1], [1, 3, 1]]) == 2  # all ties: any two edges
```

### Prim: grow one tree
Start from any node and repeatedly add the lightest edge leaving the tree built so far. That edge is the lightest across the cut "tree versus the rest", so the cut property justifies every step. A min-[heap](../heap/) holds candidate edges as `(weight, node)`; stale entries for nodes already in the tree are skipped when popped, the same lazy deletion as in Dijkstra.

Each edge is pushed at most twice, so this is \(O(E \log E) = O(E \log V)\) time and \(O(V + E)\) space. (With a heap that supports decrease-key it is \(O(E \log V)\) too; a Fibonacci heap brings it to \(O(E + V \log V)\).)

```python
import heapq
import itertools
import random


def prim(n, edges):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))
    in_tree, total, heap = [False] * n, 0, [(0, 0)]
    added = 0
    while heap and added < n:
        w, u = heapq.heappop(heap)
        if in_tree[u]:
            continue                        # stale entry
        in_tree[u] = True
        total += w
        added += 1
        for wv, v in adj[u]:
            if not in_tree[v]:
                heapq.heappush(heap, (wv, v))
    return total if added == n else -1


def mst_brute(n, edges):
    """Try every set of n - 1 edges; keep the cheapest one that forms a spanning tree."""
    best = -1
    for subset in itertools.combinations(edges, n - 1):
        uf = UnionFind(n)
        if all(uf.union(u, v) for u, v, _ in subset):
            cost = sum(w for _, _, w in subset)
            best = cost if best < 0 else min(best, cost)
    return best if n > 1 else 0


for _ in range(400):
    n = random.randint(1, 6)
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if random.random() < 0.6]
    edges = [(u, v, random.randint(1, 9)) for u, v in pairs]   # ties on purpose
    expected = mst_brute(n, edges)
    assert kruskal(n, edges) == prim(n, edges) == expected

assert prim(1, []) == 0 and prim(3, [(0, 1, 4)]) == -1

for _ in range(30):                                            # bigger graphs: the two must agree
    n = random.randint(2, 200)
    edges = [(random.randrange(n), random.randrange(n), random.randint(1, 1000)) for _ in range(3 * n)]
    assert kruskal(n, edges) == prim(n, edges)
```

Kruskal and Prim may pick different trees when weights tie, but the totals always agree, which is what the cross-check asserts. Self-loops and parallel edges in the random graphs are harmless to both.

### Dense graphs: Prim in O(n²) with no heap
[1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/) has an edge between every pair of points, weighted by Manhattan distance, so \(E \approx n^2/2\). Kruskal would sort all of them, \(O(n^2 \log n)\). Prim with a heap pushes up to \(n^2\) entries, also \(O(n^2 \log n)\).

Better: keep an array `best[v]`, the cheapest edge from the tree to each outside node. Each step scans the array for its minimum, adds that node, and updates `best` from it. That is \(n\) steps of \(O(n)\) each, \(O(n^2)\) time and \(O(n)\) space, and the edges are never materialized.

```python
def min_cost_connect_points(points):
    n = len(points)
    best = [float("inf")] * n
    best[0] = 0
    in_tree, total = [False] * n, 0
    for _ in range(n):
        u = min((v for v in range(n) if not in_tree[v]), key=best.__getitem__)
        in_tree[u] = True
        total += best[u]
        ux, uy = points[u]
        for v in range(n):
            if not in_tree[v]:
                d = abs(ux - points[v][0]) + abs(uy - points[v][1])
                if d < best[v]:
                    best[v] = d
    return total


assert min_cost_connect_points([[0, 0], [2, 2], [3, 10], [5, 2], [7, 0]]) == 20
assert min_cost_connect_points([[3, 12], [-2, 5], [-4, 1]]) == 18

for _ in range(100):
    pts = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(random.randint(1, 30))]
    full = [(i, j, abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]))
            for i in range(len(pts)) for j in range(i + 1, len(pts))]
    assert min_cost_connect_points(pts) == max(0, kruskal(len(pts), full))
```

A trick worth remembering for MST problems: a "virtual node" turns a per-node cost into an edge. In [1168. Optimize Water Distribution in a Village](https://leetcode.com/problems/optimize-water-distribution-in-a-village/), digging a well at house \(i\) for cost \(c_i\) becomes an edge from a new node 0 to house \(i\) of weight \(c_i\), and the answer is a plain MST.

## Euler paths: every edge exactly once
An **Euler path** uses each edge exactly once; an **Euler circuit** is one that ends where it started. For a directed graph whose edges all lie in one connected piece, an Euler circuit exists exactly when every node has in-degree equal to out-degree, and an Euler path exists exactly when that holds for all nodes except a start with one extra outgoing edge and an end with one extra incoming edge. (For undirected graphs: zero or two nodes of odd degree.)

### Hierholzer's algorithm
Walk from the start, using up edges, until you get stuck. In a graph that has an Euler path you can only get stuck at the end node, but you may have skipped some edges. Those skipped edges form detours that start and end at nodes already on the walk, and each can be spliced in where it begins.

The neat implementation is a DFS that appends a node to the answer only **after** all its outgoing edges are used up, then reverses the answer at the end. A node that gets stuck early lands at the end of the reversed list, and detours explored later from earlier nodes end up spliced before it. Every edge is consumed once: \(O(E)\) time and \(O(V + E)\) space. The version below uses an explicit stack, so long paths do not hit Python's recursion limit.

[332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/) asks for the Euler path from `"JFK"` that is smallest in lexical order. Always leave by the alphabetically smallest unused ticket. Greedy here can only go wrong by walking into a dead end too early, and the post-order splicing fixes exactly that: the dead-end branch is pushed to the back of the itinerary, where it belongs. Sorting the adjacency lists in reverse lets us pop the smallest from the end, \(O(E \log E)\) total.

```python
from collections import defaultdict


def find_itinerary(tickets):
    adj = defaultdict(list)
    for a, b in tickets:
        adj[a].append(b)
    for a in adj:
        adj[a].sort(reverse=True)           # smallest destination at the end
    stack, route = ["JFK"], []
    while stack:
        u = stack[-1]
        if adj[u]:
            stack.append(adj[u].pop())      # follow the smallest unused ticket
        else:
            route.append(stack.pop())       # stuck: u is finished
    return route[::-1]


assert find_itinerary([["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]) == \
    ["JFK", "MUC", "LHR", "SFO", "SJC"]
assert find_itinerary([["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], ["ATL", "JFK"], ["ATL", "SFO"]]) == \
    ["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"]
assert find_itinerary([["JFK", "KUL"], ["JFK", "NRT"], ["NRT", "JFK"]]) == \
    ["JFK", "NRT", "JFK", "KUL"]            # greedy KUL first is a dead end; splicing fixes it


def itinerary_brute(tickets):
    """Every ordering of the tickets that chains from JFK; return the smallest route."""
    best = None
    for order in itertools.permutations(tickets):
        if order[0][0] == "JFK" and all(order[i][1] == order[i + 1][0] for i in range(len(order) - 1)):
            route = ["JFK"] + [b for _, b in order]
            best = route if best is None or route < best else best
    return best


for _ in range(300):
    names, at, tickets = ["JFK", "AAA", "BBB", "CCC"], "JFK", []
    for _ in range(random.randint(1, 6)):   # a random walk guarantees a valid itinerary exists
        nxt = random.choice(names)
        tickets.append([at, nxt])
        at = nxt
    random.shuffle(tickets)
    assert find_itinerary(tickets) == itinerary_brute(tickets)
```

When the start is not given, as in [2097. Valid Arrangement of Pairs](https://leetcode.com/problems/valid-arrangement-of-pairs/), pick the node whose out-degree exceeds its in-degree by one, or any node with an outgoing edge if all are balanced, and run the same loop. For a route through a set of nodes rather than edges, the problem changes completely: visiting every **node** once is the travelling salesman problem, which needs [bitmask DP](../bitmask-dp/).

## Bipartite matching with augmenting paths
In a bipartite graph the nodes split into a left side and a right side, and every edge joins the two sides. A **matching** is a set of edges with no shared endpoint; we want the largest one.

Greedy pairing fails: matching the first left node to its first neighbor may block a better arrangement. The fix is the **augmenting path**: a path that starts at an unmatched left node, ends at an unmatched right node, and alternates between non-matching and matching edges. Flipping every edge on it (matched becomes unmatched and vice versa) grows the matching by one. Berge's theorem says a matching is maximum exactly when no augmenting path exists, so we can keep searching until none is left.

### Kuhn's algorithm
For each left node \(u\), run a DFS looking for an augmenting path: try each right neighbor \(v\) not yet visited in this search; if \(v\) is free, or if the left node currently holding \(v\) can be re-matched elsewhere (recursively), give \(v\) to \(u\). With adjacency lists each search is \(O(V + E)\), since every right node is entered at most once, and there is one search per left node: \(O(VE)\) time, \(O(V)\) extra space. With the grid below, scanning a boy's row costs \(n\) whether or not the cells are 1, so a search is \(O(mn)\) and the whole thing \(O(m^2 n)\). Hopcroft–Karp improves this to \(O(E\sqrt{V})\) by finding many shortest augmenting paths at once, but Kuhn's is short and enough for LeetCode sizes.

[1820. Maximum Number of Accepted Invitations](https://leetcode.com/problems/maximum-number-of-accepted-invitations/) gives a grid where `grid[i][j] == 1` means boy `i` may invite girl `j`, each person in at most one pair. That is a bipartite matching with boys on the left and girls on the right.

```python
def maximum_invitations(grid):
    m, n = len(grid), len(grid[0])
    owner = [-1] * n                        # owner[girl] = the boy matched to her

    def augment(boy, seen):
        for girl in range(n):
            if grid[boy][girl] and not seen[girl]:
                seen[girl] = True
                if owner[girl] == -1 or augment(owner[girl], seen):
                    owner[girl] = boy
                    return True
        return False

    return sum(augment(boy, [False] * n) for boy in range(m))


assert maximum_invitations([[1, 1, 1], [1, 0, 1], [0, 0, 1]]) == 3
assert maximum_invitations([[1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]]) == 3
assert maximum_invitations([[1, 1], [1, 0]]) == 2   # boy 0 takes girl 0 first, then gives her up
assert maximum_invitations([[0, 0], [0, 0]]) == 0


def matching_brute(grid):
    """Each boy either stays out or takes a free girl he may invite."""
    m, n = len(grid), len(grid[0])

    def go(boy, used):
        if boy == m:
            return 0
        best = go(boy + 1, used)
        for girl in range(n):
            if grid[boy][girl] and not used >> girl & 1:
                best = max(best, 1 + go(boy + 1, used | 1 << girl))
        return best

    return go(0, 0)


for _ in range(400):
    m, n = random.randint(1, 6), random.randint(1, 6)
    grid = [[int(random.random() < 0.4) for _ in range(n)] for _ in range(m)]
    assert maximum_invitations(grid) == matching_brute(grid)
```

The `seen` array is fresh for each boy and shared across the recursion of that one search: a girl who could not be freed once in this search cannot be freed later in it either, which is what keeps each search linear in the number of edges.

## Practice
- [1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/) (rerun Kruskal with an edge forced in or left out)
- [1697. Checking Existence of Edge Length Limited Paths](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/) (sort queries and edges together, Kruskal-style)
- [1168. Optimize Water Distribution in a Village](https://leetcode.com/problems/optimize-water-distribution-in-a-village/)
- [1135. Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/)
- [332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/)
- [2097. Valid Arrangement of Pairs](https://leetcode.com/problems/valid-arrangement-of-pairs/)
- [753. Cracking the Safe](https://leetcode.com/problems/cracking-the-safe/) (an Euler circuit on a de Bruijn graph)
- [1820. Maximum Number of Accepted Invitations](https://leetcode.com/problems/maximum-number-of-accepted-invitations/)
- [1066. Campus Bikes II](https://leetcode.com/problems/campus-bikes-ii/) (a minimum-cost matching; small enough for [bitmask DP](../bitmask-dp/))
- [2077. Paths in Maze That Lead to Same Room](https://leetcode.com/problems/paths-in-maze-that-lead-to-same-room/) (counting triangles; orient each edge from lower to higher degree)

The proofs of the cut and cycle properties follow the MIT 6.046J (Spring 2015) lecture notes on minimum spanning trees. The algorithms are from J. B. Kruskal, "On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem" (1956); R. C. Prim, "Shortest Connection Networks and Some Generalizations" (1957), anticipated by V. Jarník (1930); C. Hierholzer, "Über die Möglichkeit, einen Linienzug ohne Wiederholung und ohne Unterbrechung zu umfahren" (1873); C. Berge, "Two Theorems in Graph Theory" (1957); and H. W. Kuhn, "The Hungarian Method for the Assignment Problem" (1955).
