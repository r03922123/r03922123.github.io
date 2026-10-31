---
title: Heaps: from an array to four patterns
summary: A binary heap built from scratch on a plain list, then the four ways heaps show up in problems — top k, two heaps for a running median, greedy with a heap of options, and k-way merging.
group: Stacks, queues, heaps and hashing
order: 3
tags: heap · priority queue · greedy
---
A heap gives you the smallest item in \(O(1)\) and removes or inserts in \(O(\log n)\). That is all it does, and it is exactly what a surprising number of greedy algorithms need: "repeatedly take the best available option".

## A heap is a list
A binary min-heap is a complete binary tree stored level by level in a list, with each parent no larger than its children. With 0-based indices, the node at `i` has children `2i + 1` and `2i + 2` and parent `(i - 1) // 2`, so no pointers are needed.

- **Push:** append at the end, then swap upward while smaller than the parent.
- **Pop:** take the root, move the last item to the root, then swap downward with the smaller child while larger than it.

Each walks one root-to-leaf path, \(O(\log n)\).

```python
class MinHeap:
    def __init__(self):
        self.a = []

    def push(self, x):
        a = self.a
        a.append(x)
        i = len(a) - 1
        while i and a[(i - 1) // 2] > a[i]:          # sift up
            a[(i - 1) // 2], a[i] = a[i], a[(i - 1) // 2]
            i = (i - 1) // 2

    def pop(self):
        a = self.a
        top, last = a[0], a.pop()
        if a:
            a[0], i = last, 0
            while True:                               # sift down
                small = i
                for c in (2 * i + 1, 2 * i + 2):
                    if c < len(a) and a[c] < a[small]:
                        small = c
                if small == i:
                    break
                a[i], a[small] = a[small], a[i]
                i = small
        return top


import random
h, xs = MinHeap(), [random.randint(0, 50) for _ in range(200)]
for x in xs:
    h.push(x)
assert [h.pop() for _ in xs] == sorted(xs)
```

Building a heap from \(n\) items by sifting down from the last parent is \(O(n)\), not \(O(n \log n)\), because most nodes are near the bottom and travel only a short way; Python's `heapq.heapify` does this. From here on the code uses `heapq`, which is a min-heap on a list: negate values for a max-heap.

## Pattern 1: top k
Keep a heap of the best `k` seen so far, with the **worst of them** at the root, so a new item only has to beat the root. [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/) orders by frequency descending, then alphabetically. For small `k`, the tuple `(-count, word)` sorts exactly that way:

```python
import heapq
from collections import Counter


def top_k_frequent_words(words, k):
    return [w for _, w in heapq.nsmallest(k, ((-c, w) for w, c in Counter(words).items()))]


assert top_k_frequent_words(["i", "love", "leetcode", "i", "love", "coding"], 2) == ["i", "love"]
assert top_k_frequent_words(["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], 4) == \
    ["the", "is", "sunny", "day"]
```

`nsmallest` keeps a size-`k` heap internally: \(O(n \log k)\). See [quickselect](../quick-select/) and [bucket sort](../bucket-sort/) for linear alternatives.

## Pattern 2: two heaps for a running median
[295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/): a max-heap holds the lower half, a min-heap the upper half, sized equal or with the lower half one larger. The median is read off the two roots.

```python
class MedianFinder:
    def __init__(self):
        self.low, self.high = [], []                 # low is a max-heap via negation

    def add(self, x):
        heapq.heappush(self.low, -x)
        heapq.heappush(self.high, -heapq.heappop(self.low))   # largest of low moves up
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2


mf, seen = MedianFinder(), []
for x in [random.randint(0, 100) for _ in range(300)]:
    mf.add(x)
    seen.append(x)
    s, m = sorted(seen), len(seen)
    assert mf.median() == (s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2)
```

## Pattern 3: greedy with a heap of unlocked options
Sort what becomes available over time, and keep the options already available in a heap ordered by value. [502. IPO](https://leetcode.com/problems/ipo/): with capital `w`, pick at most `k` projects; each needs some capital and pays a profit. Always taking the most profitable affordable project is optimal, since profit only ever unlocks more.

```python
def find_maximized_capital(k, w, profits, capital):
    projects = sorted(zip(capital, profits))
    available, i = [], 0                         # max-heap of affordable profits
    for _ in range(k):
        while i < len(projects) and projects[i][0] <= w:
            heapq.heappush(available, -projects[i][1])
            i += 1
        if not available:
            break
        w -= heapq.heappop(available)
    return w


assert find_maximized_capital(2, 0, [1, 2, 3], [0, 1, 1]) == 4
assert find_maximized_capital(3, 0, [1, 2, 3], [0, 1, 2]) == 6
```

[871. Minimum Number of Refueling Stops](https://leetcode.com/problems/minimum-number-of-refueling-stops/) has the same shape: stations you have driven past are options, and you refuel from the biggest only when you must.

## Pattern 4: k-way merge
To merge `k` sorted lists, keep one cursor per list in a heap keyed by the current value; pop the smallest, advance that list. \(O(N \log k)\) for \(N\) items in total ([23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)).

```python
def merge_k_sorted(lists):
    heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(heap)
    out = []
    while heap:
        value, i, j = heapq.heappop(heap)
        out.append(value)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j + 1], i, j + 1))   # i breaks ties, never compares lists
    return out


assert merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6]]) == [1, 1, 2, 3, 4, 4, 5, 6]
assert merge_k_sorted([[], []]) == []
```

[759. Employee Free Time](https://leetcode.com/problems/employee-free-time/) is this merge with gap detection; [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) keeps a heap of end times, one per room in use (or see the [sweep line](../sweep-line/) version).

## Practice
- [621. Task Scheduler](https://leetcode.com/problems/task-scheduler/)
- [767. Reorganize String](https://leetcode.com/problems/reorganize-string/)
- [1882. Process Tasks Using Servers](https://leetcode.com/problems/process-tasks-using-servers/)
- [2034. Stock Price Fluctuation](https://leetcode.com/problems/stock-price-fluctuation/) (lazy deletion)
- [1851. Minimum Interval to Include Each Query](https://leetcode.com/problems/minimum-interval-to-include-each-query/)
- [2163. Minimum Difference in Sums After Removal of Elements](https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/)
