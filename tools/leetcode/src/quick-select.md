---
title: Quickselect: the kth element in linear expected time
summary: Partition like quicksort but recurse into one side only. Why that is O(n) on average, why repeated values break the naive version, and how it compares with a heap.
group: Sorting, searching and strings
order: 3
tags: selection · partitioning · randomization
---
To find the \(k\)th largest element ([215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)) you do not need a sorted array, only the element that **would** be at position \(n - k\) if it were sorted. Quickselect partitions around a pivot like quicksort, then throws away the side that cannot contain that position.

## Why it is linear
After a partition, the target is on one side, so only that side is searched. With a random pivot, each partition shrinks the range to about three quarters on average, so the expected total work is a geometric series:

\[ n + \tfrac{3}{4}n + \left(\tfrac{3}{4}\right)^2 n + \cdots = 4n = O(n). \]

The worst case is still \(O(n^2)\) if the pivot is always extreme; a random pivot makes that astronomically unlikely. (Median-of-medians guarantees \(O(n)\) worst case, at a constant factor nobody pays in practice.)

## The duplicate trap
The usual two-way (Lomuto) partition puts every element equal to the pivot on one side. On an array of identical values each partition peels off a single element, and the run is quadratic. This is exactly the kind of input that makes a quickselect time out on LeetCode. Partitioning three ways, `< pivot`, `== pivot`, `> pivot`, makes the middle block a single step: if the target index falls inside it, the pivot **is** the answer.

```python
import random


def kth_largest(nums, k):
    nums, target = list(nums), len(nums) - k        # index of the answer in ascending order
    lo, hi = 0, len(nums)                           # search nums[lo:hi]
    while True:
        pivot = nums[random.randrange(lo, hi)]
        lt, i, gt = lo, lo, hi                      # three-way partition (Dutch flag)
        while i < gt:
            if nums[i] < pivot:
                nums[lt], nums[i] = nums[i], nums[lt]
                lt, i = lt + 1, i + 1
            elif nums[i] > pivot:
                gt -= 1
                nums[i], nums[gt] = nums[gt], nums[i]
            else:
                i += 1
        if target < lt:
            hi = lt
        elif target >= gt:
            lo = gt
        else:
            return pivot                            # target is inside the block equal to the pivot


assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
assert kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
assert kth_largest([1] * 100_000, 50_000) == 1      # the input that sinks a two-way partition
for _ in range(500):
    xs = [random.randint(0, 9) for _ in range(random.randint(1, 15))]
    kk = random.randint(1, len(xs))
    assert kth_largest(xs, kk) == sorted(xs)[-kk]
```

The loop replaces recursion, so no call stack builds up.

## Quickselect or a heap?
A min-heap of size \(k\) gives the same answer in \(O(n \log k)\) time and \(O(k)\) space:

```python
import heapq


def kth_largest_heap(nums, k):
    return heapq.nlargest(k, nums)[-1]


assert kth_largest_heap([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
```

- Use **quickselect** when the data is in memory, you may reorder it, and you want the fastest expected time.
- Use a **heap** when the data streams in, when you must not modify the input, or when \(k\) is small; its time is also guaranteed, not expected. See [heaps](../heap/).

## Practice
- [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
- [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) (also a [bucket sort](../bucket-sort/))
- [324. Wiggle Sort II](https://leetcode.com/problems/wiggle-sort-ii/) (select the median, then three-way partition)
