---
title: Majority vote and the sieve of Eratosthenes
summary: Two small classics. Boyer–Moore voting finds a majority element in one pass and O(1) space, and generalizes to elements above n/3. The sieve finds every prime below n in O(n log log n).
group: Two pointers, windows and prefix sums
order: 5
tags: arrays · counting · number theory
---
Two short algorithms that come up often enough to know by heart. Both replace an obvious approach (count everything, test every number) with an argument that makes most of the work unnecessary.

## Boyer–Moore majority vote
Given an array where one value appears more than \(n/2\) times, find it in one pass with \(O(1)\) extra space ([169. Majority Element](https://leetcode.com/problems/majority-element/)). Keep a candidate and a counter. A matching element adds one vote, a different one cancels one. When the counter hits zero, the next element becomes the candidate.

Why it works: think of each cancellation as removing a pair of **different** values. Every pair removed contains at most one copy of the majority, so removing pairs can never use up all its copies; whatever survives at the end must be the majority. Cut the array wherever the counter returns to zero, and each finished segment has no majority of its own, so the majority must dominate the final segment:

```plaintext
[7, 7, 5, 7, 5, 1 | 5, 7 | 5, 5, 7, 7 | 7, 7, 7, 7]
```

```python
def majority_element(nums):
    candidate, count = None, 0
    for x in nums:
        if count == 0:
            candidate = x
        count += 1 if x == candidate else -1
    return candidate


assert majority_element([3, 2, 3]) == 3
assert majority_element([2, 2, 1, 1, 1, 2, 2]) == 2
assert majority_element([7, 7, 5, 7, 5, 1, 5, 7, 5, 5, 7, 7, 7, 7, 7, 7]) == 7
```

If a majority is not guaranteed, the candidate is only a candidate: a second pass must count it.

### More than n/3
At most two values can each appear more than \(n/3\) times ([229. Majority Element II](https://leetcode.com/problems/majority-element-ii/)). Keep two candidates; an element that matches neither cancels one vote from **both**, which removes a triple of distinct values. Then verify both candidates with a count.

```python
def majority_element_ii(nums):
    c1, c2, n1, n2 = None, None, 0, 0
    for x in nums:
        if x == c1:
            n1 += 1
        elif x == c2:
            n2 += 1
        elif n1 == 0:
            c1, n1 = x, 1
        elif n2 == 0:
            c2, n2 = x, 1
        else:
            n1, n2 = n1 - 1, n2 - 1
    return sorted(c for c in {c1, c2} if c is not None and nums.count(c) > len(nums) // 3)


assert majority_element_ii([3, 2, 3]) == [3]
assert majority_element_ii([1, 2]) == [1, 2]
import random
from collections import Counter
for _ in range(500):
    xs = [random.randint(1, 4) for _ in range(random.randint(1, 12))]
    assert majority_element_ii(xs) == sorted(v for v, c in Counter(xs).items() if c > len(xs) // 3)
```

The same idea with \(k - 1\) counters finds every value above \(n/k\) (the Misra–Gries summary).

## Sieve of Eratosthenes
To find every prime below \(n\), cross out multiples instead of testing each number. Start crossing out at \(p^2\): any smaller multiple \(m \cdot p\) with \(m < p\) was already crossed out by a smaller prime factor of \(m\). For the same reason the outer loop can stop at \(\sqrt{n}\). [204. Count Primes](https://leetcode.com/problems/count-primes/):

```python
def count_primes(n):
    if n < 3:
        return 0
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(n ** 0.5) + 1):
        if is_prime[p]:
            is_prime[p * p::p] = [False] * len(range(p * p, n, p))   # cross out p², p² + p, ...
    return sum(is_prime)


assert count_primes(10) == 4                # 2, 3, 5, 7
assert count_primes(0) == 0 and count_primes(2) == 0
assert count_primes(100) == 25
```

The total work is \(\sum_{p \le n} n/p = O(n \log \log n)\). The slice assignment does the inner loop in C, which matters at \(n = 5 \times 10^6\).

## Practice
- [229. Majority Element II](https://leetcode.com/problems/majority-element-ii/)
- [1157. Online Majority Element In Subarray](https://leetcode.com/problems/online-majority-element-in-subarray/)
- [2523. Closest Prime Numbers in Range](https://leetcode.com/problems/closest-prime-numbers-in-range/)

The voting algorithm is due to R. S. Boyer and J S. Moore, "MJRTY: A Fast Majority Vote Algorithm" (1991); the generalization to \(n/k\) to J. Misra and D. Gries, "Finding Repeated Elements" (1982).
