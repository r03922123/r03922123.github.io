---
title: Two pointers: move the side that limits you
summary: Two indices walking toward each other over sorted or bounded data. Three sum, container with most water, and trapping rain water, each reduced to one rule about which pointer moves.
group: Two pointers, windows and prefix sums
order: 1
tags: two pointers · arrays · sorting
---
Many array problems ask for the best pair, or every pair, that satisfies a condition. Checking all pairs costs \(O(n^2)\). Two pointers get it down to \(O(n)\) when the data has an order you can exploit: put one pointer at each end, and at every step move the pointer that **cannot possibly be part of a better answer**. The whole skill is proving which side that is.

## The rule
At each step there are two candidates to discard, the element under the left pointer or the one under the right. If you can show that one of them has already been paired with everything that could make it useful, drop it and move on. Each step drops one element, so the scan ends after at most \(n\) steps.

## Three sum
[15. 3Sum](https://leetcode.com/problems/3sum/) asks for all unique triples that sum to zero. Sort the array, fix the smallest element `nums[i]`, and look for a pair summing to `-nums[i]` in the rest with two pointers. If the pair's sum is too small, the left element is too small for **every** remaining right partner, since they only get smaller as `hi` moves left, so `lo` moves right. Symmetrically, a sum that is too large moves `hi` left.

```python
def three_sum(nums):
    nums = sorted(nums)
    n, out = len(nums), []
    for i in range(n - 2):
        if i and nums[i] == nums[i - 1]:      # skip duplicate anchors
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                out.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                while lo < hi and nums[lo] == nums[lo - 1]:   # skip duplicate partners
                    lo += 1
    return out


assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
```
Sorting costs \(O(n \log n)\) and each anchor scans once, so the total is \(O(n^2)\), down from \(O(n^3)\).

## Container with most water
[11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/): pick two walls to hold the most water. The area is `width × min(height[lo], height[hi])`. Moving the **taller** wall inward can never help: the width shrinks and the height is still capped by the shorter wall. So the shorter wall has met its best partner already, and it is the one to drop.

```python
def max_area(height):
    lo, hi, best = 0, len(height) - 1, 0
    while lo < hi:
        best = max(best, (hi - lo) * min(height[lo], height[hi]))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return best


assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
assert max_area([1, 1]) == 1
```

## Trapping rain water
[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/): water above a bar is `min(tallest bar to its left, tallest to its right) − its height`. Keep the running maximum seen from each end. Whichever side has the **lower** maximum is settled: its water level is fixed by its own maximum, because a taller wall is guaranteed on the other side. Add that bar's water and move that pointer.

```python
def trap(height):
    lo, hi = 0, len(height) - 1
    left_max = right_max = water = 0
    while lo < hi:
        if height[lo] < height[hi]:
            left_max = max(left_max, height[lo])
            water += left_max - height[lo]
            lo += 1
        else:
            right_max = max(right_max, height[hi])
            water += right_max - height[hi]
            hi -= 1
    return water


def trap_brute(height):
    return sum(max(0, min(max(height[:i + 1]), max(height[i:])) - h) for i, h in enumerate(height))


assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
assert trap([4, 2, 0, 3, 2, 5]) == 9
import random
for _ in range(500):
    hs = [random.randint(0, 6) for _ in range(random.randint(0, 12))]
    assert trap(hs) == trap_brute(hs)
```
\(O(n)\) time and \(O(1)\) extra space, against \(O(n)\) space for the version with prefix and suffix maximum arrays.

## When it applies
- The data is sorted, or can be sorted without losing the answer (three sum, but not "return original indices" unless you keep them).
- A local argument tells you which end is dominated: the shorter wall, the smaller sum, the lower maximum.
- If instead both pointers move the **same** direction and you care about a contiguous stretch, it is a [sliding window](../sliding-window/).

## Practice
- [838. Push Dominoes](https://leetcode.com/problems/push-dominoes/)
- [777. Swap Adjacent in LR String](https://leetcode.com/problems/swap-adjacent-in-lr-string/)
- [1055. Shortest Way to Form String](https://leetcode.com/problems/shortest-way-to-form-string/)
- [727. Minimum Window Subsequence](https://leetcode.com/problems/minimum-window-subsequence/)
- [1782. Count Pairs Of Nodes](https://leetcode.com/problems/count-pairs-of-nodes/)
