---
title: Prefix state map: subarrays as differences of prefixes
summary: Store each prefix's state in a hash map, and a subarray becomes a pair of prefixes. Counting subarrays with sum k, nice subarrays, and the longest substring with even vowel counts using a bitmask.
group: Two pointers, windows and prefix sums
order: 3
tags: prefix sum · hash map · bitmask
---
A subarray `nums[i:j]` is the difference of two prefixes: everything before `j` minus everything before `i`. If a question about the subarray can be phrased as a relation between two **prefix states**, then a single left-to-right pass with a hash map of states seen so far answers it in \(O(n)\).

This works where sliding windows fail: negative numbers, or conditions that are not monotone, because nothing here relies on the window shrinking.

## Counting: how many earlier prefixes complete this one
For a subarray ending at `j` to sum to `k`, we need an earlier prefix with sum `prefix_j - k`. Count prefix sums in a map, and add how many times the complement has appeared. The map starts with `{0: 1}`, the empty prefix, so subarrays starting at index 0 are counted. [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/):

```python
from collections import Counter


def subarray_sum(nums, k):
    seen, prefix, total = Counter({0: 1}), 0, 0
    for x in nums:
        prefix += x
        total += seen[prefix - k]           # earlier prefixes that leave exactly k
        seen[prefix] += 1
    return total


assert subarray_sum([1, 1, 1], 2) == 2
assert subarray_sum([1, -1, 0], 0) == 3     # negatives: a sliding window would get this wrong
```

The prefix does not have to be a sum. In [1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/) it is the number of odd values so far, and a "nice" subarray has exactly `k` odds:

```python
def number_of_subarrays(nums, k):
    seen, odds, total = Counter({0: 1}), 0, 0
    for x in nums:
        odds += x & 1
        total += seen[odds - k]
        seen[odds] += 1
    return total


assert number_of_subarrays([1, 1, 2, 1, 1], 3) == 2
assert number_of_subarrays([2, 2, 2, 1, 2, 2, 1, 2, 2, 2], 2) == 16
```

## Longest: remember the first time each state appeared
For the **longest** subarray, store the earliest index where each state occurred; the longest subarray ending at `j` with matching endpoints starts right after that first occurrence.

When the condition is about **parity** of several counts, the state is a bitmask: bit `b` flips each time item `b` appears. Two prefixes with the same mask differ by an even count of every item. [1371. Find the Longest Substring Containing Vowels in Even Counts](https://leetcode.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/):

```python
def find_the_longest_substring(s):
    bit = {v: 1 << i for i, v in enumerate("aeiou")}
    first, mask, best = {0: -1}, 0, 0       # the empty prefix sits at index -1
    for j, ch in enumerate(s):
        mask ^= bit.get(ch, 0)
        if mask in first:
            best = max(best, j - first[mask])
        else:
            first[mask] = j
    return best


def brute(s):
    return max((j - i for i in range(len(s) + 1) for j in range(i, len(s) + 1)
                if all(s[i:j].count(v) % 2 == 0 for v in "aeiou")), default=0)


assert find_the_longest_substring("eleetminicoworoep") == 13
assert find_the_longest_substring("bcbcbc") == 6
import random
for _ in range(300):
    t = "".join(random.choice("aeb") for _ in range(random.randint(0, 9)))
    assert find_the_longest_substring(t) == brute(t)
```

There are only 32 masks, so the map could be an array of size 32.

## Variations
- **Allow one odd count**, as in [1542. Find Longest Awesome Substring](https://leetcode.com/problems/find-longest-awesome-substring/): also look up each mask with one bit flipped, ten extra lookups per step.
- **Turn the condition into a sum.** Map each value to `+1` or `-1` and ask for a prefix difference: equal numbers of 0s and 1s in [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/), or "more tiring days than not" in [1124. Longest Well-Performing Interval](https://leetcode.com/problems/longest-well-performing-interval/).

## Practice
- [525. Contiguous Array](https://leetcode.com/problems/contiguous-array/)
- [1124. Longest Well-Performing Interval](https://leetcode.com/problems/longest-well-performing-interval/)
- [1542. Find Longest Awesome Substring](https://leetcode.com/problems/find-longest-awesome-substring/)
- [1177. Can Make Palindrome from Substring](https://leetcode.com/problems/can-make-palindrome-from-substring/)
- [2488. Count Subarrays With Median K](https://leetcode.com/problems/count-subarrays-with-median-k/)
- [2031. Count Subarrays With More Ones Than Zeros](https://leetcode.com/problems/count-subarrays-with-more-ones-than-zeros/)
