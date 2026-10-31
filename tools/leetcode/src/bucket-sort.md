---
title: Bucket sort by frequency
summary: When the sort key is a count bounded by n, index buckets by the key instead of comparing. Top k frequent elements and sorting characters by frequency in linear time.
group: Sorting, searching and strings
order: 4
tags: bucket sort · counting · hash map
---
Comparison sorts cannot beat \(O(n \log n)\). But when the key you sort by is a small integer, you do not need comparisons: put each item in the bucket numbered by its key, then read the buckets in order. A frequency is the ideal key, because no item can appear more than \(n\) times, so \(n + 1\) buckets always suffice.

## The pattern
1. Count with a hash map: \(O(n)\).
2. Put each distinct item in `bucket[count]`: \(O(\text{distinct})\).
3. Walk the buckets from high to low and take what you need: \(O(n)\).

## Top k frequent elements
[347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) asks for better than \(O(n \log n)\), which rules out sorting the counts.

```python
from collections import Counter


def top_k_frequent(nums, k):
    buckets = [[] for _ in range(len(nums) + 1)]     # buckets[f] = values seen exactly f times
    for value, freq in Counter(nums).items():
        buckets[freq].append(value)
    out = []
    for freq in range(len(nums), 0, -1):
        for value in buckets[freq]:
            out.append(value)
            if len(out) == k:
                return out
    return out


assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
assert top_k_frequent([1], 1) == [1]
```

\(O(n)\) time and space. A heap of size \(k\) gives \(O(n \log k)\) and wins when the distinct values are far fewer than \(n\) and memory matters; [quickselect](../quick-select/) on the counts gives \(O(n)\) expected.

## Sort characters by frequency
[451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/): rebuild the string with the most frequent characters first. Only frequencies up to the maximum can occur, so size the buckets by that.

```python
def frequency_sort(s):
    count = Counter(s)
    buckets = [[] for _ in range(max(count.values(), default=0) + 1)]
    for ch, freq in count.items():
        buckets[freq].append(ch)
    return "".join(ch * freq for freq in range(len(buckets) - 1, 0, -1) for ch in buckets[freq])


out = frequency_sort("tree")
assert out in ("eert", "eetr")
assert frequency_sort("Aabb") in ("bbAa", "bbaA")
assert frequency_sort("") == ""
```

## Beyond frequencies
The same idea works for any bounded integer key: sort ages, scores or digits by counting (counting sort), or sort by one digit at a time to handle larger keys (radix sort). [164. Maximum Gap](https://leetcode.com/problems/maximum-gap/) uses buckets of equal **width** instead, where the pigeonhole principle guarantees the answer lies between buckets, never inside one.

## Practice
- [2333. Minimum Sum of Squared Difference](https://leetcode.com/problems/minimum-sum-of-squared-difference/)
- [692. Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)
- [164. Maximum Gap](https://leetcode.com/problems/maximum-gap/)
