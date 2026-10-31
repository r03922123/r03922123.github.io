---
title: Binary search on the answer
summary: What bisect_left and bisect_right count, and two loop templates — the first k where a predicate turns true, the last k where it is still true — applied to problems that never mention sorting.
group: Sorting, searching and strings
order: 1
tags: binary search · bisect · monotone predicates
---
Binary search is usually taught as "find x in a sorted array". Its more useful form is: given a **monotone predicate** \(f(k)\), one that is false up to some point and true after it, find the boundary in \(O(\log n)\) evaluations. The array never has to exist. If you can ask "is \(k\) enough?" and the answer only flips once, you can binary search \(k\).

## What bisect counts
Python's `bisect` module is the cleanest way to think about boundaries in sorted data:

- `bisect_left(a, x)` is the number of items **less than** `x`, which is also the first index where `x` could go.
- `bisect_right(a, x)` is the number of items **less than or equal to** `x`.

```python
from bisect import bisect_left, bisect_right

a = [1, 3, 3, 3, 7]
assert bisect_left(a, 3) == 1 and bisect_right(a, 3) == 4    # three 3s occupy a[1:4]
assert bisect_right(a, 3) - bisect_left(a, 3) == a.count(3)
assert bisect_left(a, 5) == bisect_right(a, 5) == 4            # absent: both give the insertion point
```

So "first and last position of x" ([34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)) is two calls, and "how many values fall in [lo, hi]" is `bisect_right(a, hi) - bisect_left(a, lo)`.

## Template 1: the first k where f(k) is true
The search space is `[lo, hi]` and looks like `F F F T T T`. Keep the invariant that the answer is inside `[lo, hi]`; when `f(mid)` is true, `mid` might be the answer, so keep it.

```python
def first_true(lo, hi, f):
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid              # mid could be the answer
        else:
            lo = mid + 1          # mid is definitely not
    return lo
```

[875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/): the smallest eating speed that finishes all piles within `h` hours. Faster is never worse, so "speed k is enough" is monotone:

```python
def min_eating_speed(piles, h):
    return first_true(1, max(piles), lambda k: sum((p + k - 1) // k for p in piles) <= h)


assert min_eating_speed([3, 6, 7, 11], 8) == 4
assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
```

## Template 2: the last k where f(k) is still true
Now the space looks like `T T T F F F`. The midpoint must round **up**, or `lo = mid` would loop forever when `hi = lo + 1`.

```python
def last_true(lo, hi, f):
    while lo < hi:
        mid = (lo + hi + 1) // 2  # round up so the range always shrinks
        if f(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo
```

[1802. Maximum Value at a Given Index in a Bounded Array](https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/): build an array of positive integers, adjacent values differing by at most 1, total at most `max_sum`; maximize `nums[index]`. For a peak value `v`, the cheapest array slopes down by 1 on each side and stays at 1 once it gets there, and a bigger peak never costs less, so "peak v fits" is monotone:

```python
def max_value(n, index, max_sum):
    def side(v, length):          # cheapest sum of `length` values sloping down from v - 1, floored at 1
        if v - 1 >= length:
            return (v - 1 + v - length) * length // 2
        return (v - 1) * v // 2 + (length - (v - 1))
    return last_true(1, max_sum, lambda v: v + side(v, index) + side(v, n - index - 1) <= max_sum)


def brute(n, index, max_sum):
    best = 0
    for v in range(1, max_sum + 1):
        if sum(max(v - abs(i - index), 1) for i in range(n)) <= max_sum:
            best = v
    return best


assert max_value(4, 2, 6) == 2
assert max_value(6, 1, 10) == 3
for n in range(1, 7):
    for index in range(n):
        for s in range(n, 25):
            assert max_value(n, index, s) == brute(n, index, s)
```

## Checklist
1. Name the predicate and argue it flips only once.
2. Choose bounds that surely contain the answer.
3. First-true rounds down and keeps `hi = mid`; last-true rounds up and keeps `lo = mid`.
4. The cost is \(O(\log(\text{range}) \times \text{cost of } f)\), which is why \(f\) can afford an \(O(n)\) scan, or even a BFS as in [1970. Last Day Where You Can Still Cross](https://leetcode.com/problems/last-day-where-you-can-still-cross/).

## Practice
- [1539. Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/)
- [1970. Last Day Where You Can Still Cross](https://leetcode.com/problems/last-day-where-you-can-still-cross/)
- [1102. Path With Maximum Minimum Value](https://leetcode.com/problems/path-with-maximum-minimum-value/)
- [1055. Shortest Way to Form String](https://leetcode.com/problems/shortest-way-to-form-string/)
- [410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)
