---
title: Monotonic stack: next greater, previous smaller
summary: A stack kept in sorted order answers "nearest larger or smaller element" for every position in one pass. The template, why it is O(n), and its two classic uses, daily temperatures and the largest rectangle.
group: Stacks, queues, heaps and hashing
order: 1
tags: stack · arrays · amortized analysis
---
For each element, "where is the next element to the right that is greater?" naively takes \(O(n)\) per element. A monotonic stack answers it for **all** elements in one pass. The stack holds indices still **waiting** for their answer, and their values are kept in decreasing order. When a new element arrives that is greater than the top, it is the answer for the top: pop it, record, repeat.

## Why the order is decreasing
Suppose two waiting indices \(i < j\) with \(A_i \ge A_j\). Any future element greater than \(A_i\) is also greater than \(A_j\), so \(A_j\) is answered first. Waiting elements therefore sit in decreasing order, and an element only leaves when something larger arrives. Each index is pushed once and popped at most once, so the whole scan is \(O(n)\) even though there is a loop inside the loop.

## The template
One pass gives two answers. When `i` pops `top`, `i` is `top`'s **next greater**. After the pops, whatever remains on top is `i`'s **previous greater-or-equal** element, because everything smaller than `A[i]` was just removed.

```python
def next_greater_and_previous_ge(a):
    n = len(a)
    next_greater = [-1] * n          # index of the first later element > a[i]
    prev_ge = [-1] * n               # index of the nearest earlier element >= a[i]
    stack = []                       # indices whose values are decreasing
    for i, x in enumerate(a):
        while stack and a[stack[-1]] < x:
            next_greater[stack.pop()] = i
        if stack:
            prev_ge[i] = stack[-1]
        stack.append(i)
    return next_greater, prev_ge


def brute(a):
    n = len(a)
    nxt = [next((j for j in range(i + 1, n) if a[j] > a[i]), -1) for i in range(n)]
    prv = [next((j for j in range(i - 1, -1, -1) if a[j] >= a[i]), -1) for i in range(n)]
    return nxt, prv


assert next_greater_and_previous_ge([2, 1, 2, 4, 3]) == ([3, 2, 3, -1, -1], [-1, 0, 0, -1, 3])
import random
for _ in range(500):
    xs = [random.randint(0, 5) for _ in range(random.randint(0, 10))]
    assert next_greater_and_previous_ge(xs) == brute(xs)
```

Flip the comparison to get next **smaller** with an increasing stack. Whether ties pop (`<` versus `<=`) decides whether "greater" means strict, which matters when counting.

## Daily temperatures
[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/): for each day, how many days until a warmer one? That is the next-greater distance.

```python
def daily_temperatures(temps):
    out, stack = [0] * len(temps), []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            out[j] = i - j
        stack.append(i)
    return out


assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
```

## Largest rectangle in a histogram
[84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/): every maximal rectangle is as tall as some bar and spans as far as bars stay at least that tall, so from the previous smaller bar to the next smaller bar. With an **increasing** stack, a bar's next smaller is the element that pops it, and its previous smaller is the element beneath it on the stack. A sentinel height 0 at the end flushes the stack.

```python
def largest_rectangle_area(heights):
    best, stack = 0, []                          # indices with increasing heights
    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            left = stack[-1] if stack else -1    # previous smaller bar
            best = max(best, height * (i - left - 1))
        stack.append(i)
    return best


def brute_area(hs):
    return max((min(hs[i:j + 1]) * (j - i + 1) for i in range(len(hs)) for j in range(i, len(hs))), default=0)


assert largest_rectangle_area([2, 1, 5, 6, 2, 3]) == 10
assert largest_rectangle_area([2, 4]) == 4
for _ in range(500):
    hs = [random.randint(0, 6) for _ in range(random.randint(0, 9))]
    assert largest_rectangle_area(hs) == brute_area(hs)
```

[85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/) runs this on each row of a binary matrix, with heights counting consecutive ones above.

## Contribution counting
Questions like "sum of the minimum of every subarray" ([907](https://leetcode.com/problems/sum-of-subarray-minimums/)) ask how many subarrays each element is the minimum of. With previous-smaller at `L` and next-smaller-or-equal at `R`, element `i` is the minimum of `(i - L) * (R - i)` subarrays. Mixing a strict and a non-strict side counts each subarray exactly once even with ties.

## Practice
- [907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/)
- [1673. Find the Most Competitive Subsequence](https://leetcode.com/problems/find-the-most-competitive-subsequence/)
- [1130. Minimum Cost Tree From Leaf Values](https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/)
- [1504. Count Submatrices With All Ones](https://leetcode.com/problems/count-submatrices-with-all-ones/)
- [2281. Sum of Total Strength of Wizards](https://leetcode.com/problems/sum-of-total-strength-of-wizards/)
- [1776. Car Fleet II](https://leetcode.com/problems/car-fleet-ii/)

For the sliding-window version of the same idea, see the [monotonic queue](../monotonic-queue/).
