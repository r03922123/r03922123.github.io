---
title: Counting pairs with i < j: merge sort and sorted containers
summary: Problems that count pairs (i, j) with i < j and some inequality between values. Counting during merge sort, and the sorted-list alternative, applied to reverse pairs and a shifted-difference variant.
group: Sorting, searching and strings
order: 5
tags: inversions · merge sort · binary search
---
A family of problems asks how many pairs \(i < j\) satisfy an inequality between `nums[i]` and `nums[j]`. The textbook member counts **inversions**, pairs that are out of order. Checking every pair is \(O(n^2)\). Two standard \(O(n \log n)\) approaches: count during a merge sort, or scan once while keeping the earlier values in a sorted structure.

## Counting during merge sort
In merge sort, just before merging, the left half holds earlier indices and the right half later ones, and **each half is sorted**. So for every element of the right half, the left-half elements that pair with it form a contiguous run, found with a pointer that only moves forward. Count first, then merge.

[493. Reverse Pairs](https://leetcode.com/problems/reverse-pairs/) counts pairs with `nums[i] > 2 * nums[j]`:

```python
def reverse_pairs(nums):
    def sort(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, cl = sort(a[:mid])
        right, cr = sort(a[mid:])
        count, i = cl + cr, 0
        for x in right:                       # both halves sorted: i only moves forward
            while i < len(left) and left[i] <= 2 * x:
                i += 1
            count += len(left) - i            # left[i:] are all > 2x
        merged, p, q = [], 0, 0
        while p < len(left) and q < len(right):
            if left[p] <= right[q]:
                merged.append(left[p]); p += 1
            else:
                merged.append(right[q]); q += 1
        return merged + left[p:] + right[q:], count
    return sort(list(nums))[1]


def brute(nums):
    return sum(nums[i] > 2 * nums[j] for i in range(len(nums)) for j in range(i + 1, len(nums)))


assert reverse_pairs([1, 3, 2, 3, 1]) == 2
assert reverse_pairs([2, 4, 3, 5, 1]) == 3
import random
for _ in range(400):
    xs = [random.randint(-6, 6) for _ in range(random.randint(0, 12))]
    assert reverse_pairs(xs) == brute(xs)
```

The counting pass is separate from the merge because the condition (`> 2x`) is not the sort order. For plain inversions they coincide and you can count inside the merge.

## A sorted structure over the prefix
Alternatively, scan left to right and keep every earlier value in a sorted list; for each new value, binary search how many earlier values pair with it. With a plain Python list, insertion is \(O(n)\), so this is only fast with a balanced structure, such as `sortedcontainers.SortedList` (available on LeetCode) or a [Fenwick tree](../segment-tree/) over compressed values. Here is the shape with the standard library, correct but quadratic in the worst case because of `insort`:

```python
from bisect import bisect_right, insort


def number_of_pairs(nums1, nums2, diff):
    """2426: count i < j with nums1[i] - nums1[j] <= nums2[i] - nums2[j] + diff."""
    seen, total = [], 0
    for a, b in zip(nums1, nums2):
        c = a - b                             # rewrite: c[i] <= c[j] + diff, with c = nums1 - nums2
        total += bisect_right(seen, c + diff)
        insort(seen, c)                       # SortedList.add would make this O(log n)
    return total


assert number_of_pairs([3, 2, 5], [2, 2, 1], 1) == 3
assert number_of_pairs([3, -1], [-2, 2], -1) == 0
```

The rewrite is the real trick in [2426. Number of Pairs Satisfying Inequality](https://leetcode.com/problems/number-of-pairs-satisfying-inequality/): move every `i` term to one side and every `j` term to the other, and a two-array condition becomes a one-array prefix count.

## Practice
- [315. Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
- [327. Count of Range Sum](https://leetcode.com/problems/count-of-range-sum/)
- [2426. Number of Pairs Satisfying Inequality](https://leetcode.com/problems/number-of-pairs-satisfying-inequality/)
- [775. Global and Local Inversions](https://leetcode.com/problems/global-and-local-inversions/)
