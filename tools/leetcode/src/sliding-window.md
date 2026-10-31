---
title: Sliding window: three templates and the exactly-k trick
summary: A window over a contiguous stretch that grows on the right and shrinks on the left. Templates for the shortest valid window, the longest valid window, and counting windows with exactly k of something.
group: Two pointers, windows and prefix sums
order: 2
tags: sliding window · two pointers · strings
---
A sliding window answers questions about **contiguous** subarrays or substrings: the shortest one that contains something, the longest one that avoids something, how many satisfy a condition. Both ends only move right, so each element enters and leaves the window at most once and the scan is \(O(n)\).

It works when validity is **monotone**: if a window is valid, every window containing it is valid too (or every window inside it is). That is what makes it safe to move `lo` without looking back.

## Shortest valid window
Grow `hi` until the window is valid, then shrink `lo` while it stays valid, recording the best each time. [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) asks for the shortest substring of `s` containing every character of `t`, with multiplicity.

```python
from collections import Counter


def min_window(s, t):
    need, missing = Counter(t), len(t)       # missing = characters of t not yet covered
    lo, best = 0, (0, float("inf"))
    for hi, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:                  # valid: shrink while it stays valid
            if hi - lo < best[1] - best[0]:
                best = (lo, hi)
            need[s[lo]] += 1
            if need[s[lo]] > 0:
                missing += 1
            lo += 1
    return "" if best[1] == float("inf") else s[best[0]:best[1] + 1]


assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "aa") == ""
```

## Longest valid window
Grow `hi` every step; when the window becomes invalid, shrink `lo` until it is valid again. [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/):

```python
def length_of_longest_substring(s):
    seen, lo, best = {}, 0, 0
    for hi, ch in enumerate(s):
        if seen.get(ch, -1) >= lo:          # ch repeats inside the window: jump lo past it
            lo = seen[ch] + 1
        seen[ch] = hi
        best = max(best, hi - lo + 1)
    return best


assert length_of_longest_substring("abcabcbb") == 3
assert length_of_longest_substring("pwwkew") == 3
```

### A window that never shrinks
When only the **length** of the longest window matters, the window never needs to get smaller: once it reaches length \(L\), you only care about finding a valid window of length \(L + 1\). So move `lo` by **one** step when invalid, instead of a `while`. The window slides at constant size, grows when it can, and its final size is the answer. In [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) this also lets `top` stay stale: it only needs to be correct when the window grows.

```python
def character_replacement(s, k):
    count, lo, top = Counter(), 0, 0
    for hi, ch in enumerate(s):
        count[ch] += 1
        top = max(top, count[ch])           # most frequent char ever seen in a window this size
        if hi - lo + 1 - top > k:           # invalid: slide, do not shrink
            count[s[lo]] -= 1
            lo += 1
    return len(s) - lo


assert character_replacement("ABAB", 2) == 4
assert character_replacement("AABABBA", 1) == 4
```

## Counting windows with exactly k
"Exactly k" is not monotone: adding an element can take a window from valid to invalid and back. "At most k" **is** monotone, and for each right end `hi` the valid windows ending there are exactly those starting in `[lo, hi]`, which is `hi - lo + 1` of them. So count exactly-k as a difference:

\[ \text{exactly}(k) = \text{atMost}(k) - \text{atMost}(k - 1). \]

[992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/):

```python
def subarrays_with_k_distinct(nums, k):
    def at_most(k):
        count, lo, total = Counter(), 0, 0
        for hi, x in enumerate(nums):
            count[x] += 1
            while len(count) > k:
                count[nums[lo]] -= 1
                if count[nums[lo]] == 0:
                    del count[nums[lo]]
                lo += 1
            total += hi - lo + 1             # windows [lo..hi], [lo+1..hi], ..., [hi..hi]
        return total
    return at_most(k) - at_most(k - 1)


def brute(nums, k):
    return sum(len(set(nums[i:j + 1])) == k for i in range(len(nums)) for j in range(i, len(nums)))


assert subarrays_with_k_distinct([1, 2, 1, 2, 3], 2) == 7
assert subarrays_with_k_distinct([1, 2, 1, 3, 4], 3) == 3
import random
for _ in range(300):
    xs = [random.randint(1, 4) for _ in range(random.randint(1, 10))]
    kk = random.randint(1, 4)
    assert subarrays_with_k_distinct(xs, kk) == brute(xs, kk)
```

### At least k, counted directly
For "at least", validity is monotone the other way: once `[lo, hi]` is valid, every extension to the right is valid too. While the window is valid, every start in `[0, lo]` works for this `hi`, so after shrinking to the first invalid `lo`, add `lo`. [1358. Number of Substrings Containing All Three Characters](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/):

```python
def number_of_substrings(s):
    count, lo, total = Counter(), 0, 0
    for ch in s:
        count[ch] += 1
        while len(count) == 3:              # valid: shrink until it is not
            count[s[lo]] -= 1
            if count[s[lo]] == 0:
                del count[s[lo]]
            lo += 1
        total += lo                         # starts 0..lo-1 all give a valid window ending here
    return total


assert number_of_substrings("abcabc") == 10
assert number_of_substrings("aaacb") == 3
```

## Practice
- [3306. Count of Substrings Containing Every Vowel and K Consonants II](https://leetcode.com/problems/count-of-substrings-containing-every-vowel-and-k-consonants-ii/)
- [1838. Frequency of the Most Frequent Element](https://leetcode.com/problems/frequency-of-the-most-frequent-element/)
- [1888. Minimum Number of Flips to Make the Binary String Alternating](https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/)
- [2009. Minimum Number of Operations to Make Array Continuous](https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/)
- [1610. Maximum Number of Visible Points](https://leetcode.com/problems/maximum-number-of-visible-points/)
- [1499. Max Value of Equation](https://leetcode.com/problems/max-value-of-equation/), which pairs a window with a [monotonic queue](../monotonic-queue/)

The templates follow the structure of Liu Zhenglai's [sliding-window notes](https://liuzhenglaichn.gitbook.io/algorithm/array/sliding-window).
