---
title: Monotonic queue: the window maximum in O(1)
summary: A deque kept in sorted order gives the maximum of a sliding window at every step in amortized O(1). Sliding window maximum, then the harder use, shortest subarray with sum at least k over prefix sums.
group: Stacks, queues, heaps and hashing
order: 2
tags: deque · sliding window · prefix sum
---
A [monotonic stack](../monotonic-stack/) finds the nearest larger element. Add the ability to drop elements from the **front** when they fall out of a window, and you get a monotonic queue: a deque whose values decrease from front to back, so the front is always the current window's maximum.

## Why it works
When a new element arrives, every element behind it in the deque that is **smaller** can never be a window maximum again: the new element is larger and will stay in the window longer. So pop them from the back. What remains is a decreasing sequence of candidates, each of which could still be the maximum of some future window. Each index enters and leaves once, so a window of any size costs amortized \(O(1)\) per step.

## Sliding window maximum
[239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/): the maximum of every window of size `k`. The deque stores indices, so the front can be expired by position.

```python
from collections import deque


def max_sliding_window(nums, k):
    dq, out = deque(), []                       # indices; nums[dq] decreasing front to back
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:         # dominated: older and not larger
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                      # front fell out of the window
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_sliding_window([1], 1) == [1]
import random
for _ in range(500):
    xs = [random.randint(-5, 5) for _ in range(random.randint(1, 12))]
    kk = random.randint(1, len(xs))
    assert max_sliding_window(xs, kk) == [max(xs[i:i + kk]) for i in range(len(xs) - kk + 1)]
```

A heap also works, in \(O(n \log n)\) with lazy deletion; the deque is \(O(n)\) and simpler.

## Shortest subarray with sum at least k
[862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/) allows **negative** numbers, which breaks the [sliding window](../sliding-window/): adding an element can lower the sum. Switch to prefix sums \(P\), where the subarray `(i, j]` sums to \(P_j - P_i\). For each `j` we want the **latest** `i` with \(P_i \le P_j - k\). Two observations make the deque work:

1. **Pop from the front while it satisfies the condition.** Once a start `i` gives a valid subarray ending at `j`, any later end would only be longer, so `i` has done its best and can go.
2. **Pop from the back while \(P_{\text{back}} \ge P_j\).** A start with a larger prefix and an earlier position is worse than `j` in both ways: `j` would give a larger sum and a shorter subarray.

So the deque holds increasing prefix sums, front to back.

```python
from itertools import accumulate


def shortest_subarray(nums, k):
    prefix = list(accumulate(nums, initial=0))
    dq, best = deque(), len(nums) + 1
    for j, pj in enumerate(prefix):
        while dq and pj - prefix[dq[0]] >= k:   # observation 1
            best = min(best, j - dq.popleft())
        while dq and prefix[dq[-1]] >= pj:      # observation 2
            dq.pop()
        dq.append(j)
    return best if best <= len(nums) else -1


def brute(nums, k):
    lengths = [j - i for i in range(len(nums)) for j in range(i + 1, len(nums) + 1) if sum(nums[i:j]) >= k]
    return min(lengths, default=-1)


assert shortest_subarray([1], 1) == 1
assert shortest_subarray([1, 2], 4) == -1
assert shortest_subarray([2, -1, 2], 3) == 3
for _ in range(500):
    xs = [random.randint(-4, 6) for _ in range(random.randint(1, 10))]
    kk = random.randint(1, 10)
    assert shortest_subarray(xs, kk) == brute(xs, kk)
```

## When to reach for it
- You need a max or min over a window that slides, at every step.
- A dynamic program reads the best value from a sliding range of earlier states, \(dp_j = \max_{j-k \le i < j} dp_i + \ldots\), as in [1696. Jump Game VI](https://leetcode.com/problems/jump-game-vi/). The deque drops that inner max from \(O(k)\) to \(O(1)\).

## Practice
- [1696. Jump Game VI](https://leetcode.com/problems/jump-game-vi/)
- [1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit](https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/) (two deques, one for max and one for min)
- [1499. Max Value of Equation](https://leetcode.com/problems/max-value-of-equation/)
- [1687. Delivering Boxes from Storage to Ports](https://leetcode.com/problems/delivering-boxes-from-storage-to-ports/)
