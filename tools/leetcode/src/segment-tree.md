---
title: Range queries: Fenwick trees, segment trees, lazy propagation
summary: Three structures for arrays that change while you query them. A Fenwick tree for prefix sums, an iterative segment tree for any associative operation, and lazy propagation for updating whole ranges, each checked against brute force.
group: Trees and range queries
order: 3
tags: segment tree · Fenwick tree · range queries
---
Prefix sums answer range-sum queries in \(O(1)\), but one update to the array forces an \(O(n)\) rebuild. When updates and queries interleave, you need a structure where **both** are \(O(\log n)\). Pick the simplest one that covers the operations you need:

| Need | Structure |
|---|---|
| Point update, prefix or range **sum** | Fenwick tree |
| Point update, range query of any associative op (sum, min, max, gcd) | Segment tree |
| **Range** update and range query | Segment tree with lazy propagation |

## Fenwick tree (binary indexed tree)
Store, at 1-based index `i`, the sum of a block of length `i & -i` (the lowest set bit of `i`) ending at `i`. A prefix sum hops down by removing the lowest set bit; an update hops up by adding it. Each takes at most \(\log_2 n\) hops.

```python
class Fenwick:
    def __init__(self, n):
        self.tree = [0] * (n + 1)

    def add(self, i, delta):                    # a[i] += delta, i is 0-based
        i += 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i                         # next block that covers i

    def prefix(self, i):                        # sum of a[0:i]
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i                         # drop the lowest set bit
        return total

    def range_sum(self, lo, hi):                # sum of a[lo:hi]
        return self.prefix(hi) - self.prefix(lo)


import random
a = [random.randint(-9, 9) for _ in range(50)]
fw = Fenwick(len(a))
for i, x in enumerate(a):
    fw.add(i, x)
for _ in range(2000):
    i = random.randrange(len(a))
    delta = random.randint(-5, 5)
    a[i] += delta
    fw.add(i, delta)
    lo, hi = sorted(random.sample(range(len(a) + 1), 2))
    assert fw.range_sum(lo, hi) == sum(a[lo:hi])
```

A classic use is counting inversions or "smaller elements after self" ([315](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)): scan from the right, query how many smaller values were seen, then add the current value. Values are first compressed to ranks so the tree stays small.

```python
def count_smaller(nums):
    rank = {v: r for r, v in enumerate(sorted(set(nums)))}
    fw, out = Fenwick(len(rank)), []
    for x in reversed(nums):
        out.append(fw.prefix(rank[x]))          # values with a smaller rank, seen to the right
        fw.add(rank[x], 1)
    return out[::-1]


assert count_smaller([5, 2, 6, 1]) == [2, 1, 1, 0]
assert count_smaller([-1, -1]) == [0, 0]
```

## Segment tree
For any **associative** operation, a segment tree stores the answer for each node's range. The iterative version is compact: leaves live at `n..2n-1`, and node `i` combines children `2i` and `2i+1`. A query on `[lo, hi)` climbs from both ends toward the root, taking a node whenever it lies fully inside the range.

```python
class SegmentTree:
    def __init__(self, values, op=min, identity=float("inf")):
        self.n, self.op, self.identity = len(values), op, identity
        self.t = [identity] * self.n + list(values)
        for i in range(self.n - 1, 0, -1):
            self.t[i] = op(self.t[2 * i], self.t[2 * i + 1])

    def update(self, i, value):                 # a[i] = value
        i += self.n
        self.t[i] = value
        while i > 1:
            i //= 2
            self.t[i] = self.op(self.t[2 * i], self.t[2 * i + 1])

    def query(self, lo, hi):                    # op over a[lo:hi]
        res_left = res_right = self.identity
        lo, hi = lo + self.n, hi + self.n
        while lo < hi:
            if lo & 1:                          # lo is a right child: take it, step past
                res_left = self.op(res_left, self.t[lo])
                lo += 1
            if hi & 1:                          # hi is exclusive: take its left sibling
                hi -= 1
                res_right = self.op(self.t[hi], res_right)
            lo, hi = lo // 2, hi // 2
        return self.op(res_left, res_right)


a = [random.randint(0, 99) for _ in range(37)]  # not a power of two, on purpose
st = SegmentTree(a)
for _ in range(2000):
    i = random.randrange(len(a))
    a[i] = random.randint(0, 99)
    st.update(i, a[i])
    lo, hi = sorted(random.sample(range(len(a) + 1), 2))
    assert st.query(lo, hi) == min(a[lo:hi])
```

Keeping the left and right partial results separate means it is correct even for operations that are associative but not commutative, such as matrix products or string concatenation.

## Lazy propagation: updating whole ranges
"Add 5 to every element in `[lo, hi)`" would touch \(O(n)\) leaves. Lazy propagation records the pending addition at the highest nodes that cover the range and pushes it to the children only when a later operation needs to look inside. The recursive form is easiest to get right:

```python
class LazySegmentTree:                          # range add, range sum
    def __init__(self, n):
        self.n = n
        self.sum = [0] * (4 * n)
        self.lazy = [0] * (4 * n)

    def _push(self, node, lo, mid, hi):
        if self.lazy[node]:
            for child, length in ((2 * node, mid - lo), (2 * node + 1, hi - mid)):
                self.sum[child] += self.lazy[node] * length
                self.lazy[child] += self.lazy[node]
            self.lazy[node] = 0

    def add(self, l, r, delta, node=1, lo=0, hi=None):     # a[l:r] += delta
        hi = self.n if hi is None else hi
        if r <= lo or hi <= l:
            return
        if l <= lo and hi <= r:                 # fully covered: record and stop
            self.sum[node] += delta * (hi - lo)
            self.lazy[node] += delta
            return
        mid = (lo + hi) // 2
        self._push(node, lo, mid, hi)
        self.add(l, r, delta, 2 * node, lo, mid)
        self.add(l, r, delta, 2 * node + 1, mid, hi)
        self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]

    def query(self, l, r, node=1, lo=0, hi=None):          # sum of a[l:r]
        hi = self.n if hi is None else hi
        if r <= lo or hi <= l:
            return 0
        if l <= lo and hi <= r:
            return self.sum[node]
        mid = (lo + hi) // 2
        self._push(node, lo, mid, hi)
        return self.query(l, r, 2 * node, lo, mid) + self.query(l, r, 2 * node + 1, mid, hi)


a = [0] * 41
lz = LazySegmentTree(len(a))
for _ in range(2000):
    l, r = sorted(random.sample(range(len(a) + 1), 2))
    if random.random() < 0.5:
        delta = random.randint(-5, 5)
        lz.add(l, r, delta)
        for i in range(l, r):
            a[i] += delta
    else:
        assert lz.query(l, r) == sum(a[l:r])
```

For range add with only **point** queries, a Fenwick tree over the difference array is enough: add `delta` at `l` and `-delta` at `r`, and the value at `i` is a prefix sum. Range assignment needs a different lazy tag (a pending "set to v" that overrides older adds).

## Practice
- [307. Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)
- [2179. Count Good Triplets in an Array](https://leetcode.com/problems/count-good-triplets-in-an-array/)
- [2407. Longest Increasing Subsequence II](https://leetcode.com/problems/longest-increasing-subsequence-ii/) (segment tree over values, max)
- [2286. Booking Concert Tickets in Groups](https://leetcode.com/problems/booking-concert-tickets-in-groups/)
- [2158. Amount of New Area Painted Each Day](https://leetcode.com/problems/amount-of-new-area-painted-each-day/)
- [2031. Count Subarrays With More Ones Than Zeros](https://leetcode.com/problems/count-subarrays-with-more-ones-than-zeros/)

The Fenwick tree is from P. M. Fenwick, "A New Data Structure for Cumulative Frequency Tables" (1994).
