---
title: Merge sort, quicksort and the Dutch flag partition
summary: The two comparison sorts worth writing from memory, what each guarantees, and the three-way partition that keeps quicksort fast when values repeat.
group: Sorting, searching and strings
order: 2
tags: sorting · divide and conquer · partitioning
---
You will rarely write a sort in practice; `sorted` is a tuned Timsort. But the two classic sorts are templates for a whole family of problems: merge sort's merge step counts inversions and merges sorted streams, and quicksort's partition step drives selection and in-place grouping. [912. Sort an Array](https://leetcode.com/problems/sort-an-array/) is the test bed.

## Merge sort
Split in half, sort each half, merge. Every level of recursion does \(O(n)\) work merging, and there are \(\log_2 n\) levels, so it is \(O(n \log n)\) in **every** case. It is stable (equal items keep their order) because the merge takes from the left half on ties. The cost is \(O(n)\) extra space for the buffer.

```python
def merge_sort(nums):
    nums, aux = list(nums), [0] * len(nums)

    def sort(lo, hi):                       # sorts nums[lo..hi] inclusive
        if lo >= hi:
            return
        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid + 1, hi)
        if nums[mid] <= nums[mid + 1]:      # halves already in order: skip the merge
            return
        aux[lo:hi + 1] = nums[lo:hi + 1]
        i, j = lo, mid + 1
        for k in range(lo, hi + 1):
            if i > mid or (j <= hi and aux[j] < aux[i]):   # take right only if strictly smaller: stable
                nums[k], j = aux[j], j + 1
            else:
                nums[k], i = aux[i], i + 1

    sort(0, len(nums) - 1)
    return nums


import random
for _ in range(300):
    xs = [random.randint(-5, 5) for _ in range(random.randint(0, 20))]
    assert merge_sort(xs) == sorted(xs)
```

The early exit when `nums[mid] <= nums[mid + 1]` makes already-sorted input linear, a refinement from Sedgewick and Wayne's *Algorithms*.

## Quicksort
Pick a pivot, partition so smaller items come first, recurse on both sides. It sorts in place and is usually fastest in practice, but its \(O(n \log n)\) is only **expected**: a bad pivot every time gives \(O(n^2)\). A random pivot makes that vanishingly unlikely for any input order.

There is a second trap. With a two-way partition, items **equal** to the pivot all land on one side, so an array of identical values still degrades to \(O(n^2)\), whatever the pivot. The fix is to partition three ways.

## The Dutch national flag partition
Dijkstra's problem: arrange red, white and blue items in order with one pass and no extra space ([75. Sort Colors](https://leetcode.com/problems/sort-colors/)). With a pivot value, it is "less than, equal to, greater than". Keep four regions and shrink the unknown one by one each step:

```plaintext
[ < pivot | == pivot | unknown | > pivot ]
 0     lt  lt     i  i     gt  gt      n
```

```python
def dutch_flag(nums, pivot, lo=0, hi=None):
    """Partition nums[lo:hi] in place; return (lt, gt) so nums[lt:gt] == pivot."""
    hi = len(nums) if hi is None else hi
    lt, i, gt = lo, lo, hi
    while i < gt:
        if nums[i] < pivot:
            nums[lt], nums[i] = nums[i], nums[lt]
            lt, i = lt + 1, i + 1
        elif nums[i] > pivot:
            gt -= 1
            nums[i], nums[gt] = nums[gt], nums[i]   # do not advance i: the swapped-in item is unknown
        else:
            i += 1
    return lt, gt


colors = [2, 0, 2, 1, 1, 0]
dutch_flag(colors, 1)
assert colors == [0, 0, 1, 1, 2, 2]
```

With it, quicksort recurses only on the strictly-smaller and strictly-larger parts, and repeated values cost nothing extra:

```python
def quick_sort(nums):
    nums = list(nums)

    def sort(lo, hi):                        # sorts nums[lo:hi]
        if hi - lo <= 1:
            return
        lt, gt = dutch_flag(nums, nums[random.randrange(lo, hi)], lo, hi)
        sort(lo, lt)
        sort(gt, hi)

    sort(0, len(nums))
    return nums


for _ in range(300):
    xs = [random.randint(-5, 5) for _ in range(random.randint(0, 20))]
    assert quick_sort(xs) == sorted(xs)
assert quick_sort([7] * 50_000) == [7] * 50_000   # one pass, no deep recursion
```

## Which to reach for
| | Merge sort | Quicksort (3-way, random pivot) |
|---|---|---|
| Time | \(O(n \log n)\) always | \(O(n \log n)\) expected |
| Extra space | \(O(n)\) | \(O(\log n)\) stack |
| Stable | Yes | No |
| Reuse | Counting inversions, external sorting | [Selection](../quick-select/), in-place grouping |

## Practice
- [75. Sort Colors](https://leetcode.com/problems/sort-colors/)
- [148. Sort List](https://leetcode.com/problems/sort-list/) (merge sort on a linked list)
- [2171. Removing Minimum Number of Magic Beans](https://leetcode.com/problems/removing-minimum-number-of-magic-beans/)
- [1326. Minimum Number of Taps to Open to Water a Garden](https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/)
- [1152. Analyze User Website Visit Pattern](https://leetcode.com/problems/analyze-user-website-visit-pattern/)
