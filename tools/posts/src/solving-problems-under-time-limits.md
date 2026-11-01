---
title: Solving algorithm problems under a time limit
summary: A repeatable method for an algorithm problem against the clock: pin the problem down, let the constraints set the target complexity, write the brute force first, find what it repeats, name the pattern, list edge cases before coding, and test with a stress test against the brute force. One problem worked end to end, including a tempting wrong pattern that the stress test catches.
group: Coding and data
order: 1
tags: problem solving · complexity · testing · algorithms
---
With a clock running, most failed attempts are not a lack of knowledge. They are skipped steps: solving a slightly different problem than the one asked, chasing a clever idea for twenty minutes with nothing written, or finishing a correct idea that breaks on an empty list. A fixed sequence of steps removes those failures, and frees attention for the actual problem.

The method here has six steps: understand, size, brute force, optimize, code, test. Each is shown on one problem, worked end to end. For the patterns themselves (sliding windows, heaps, graph searches and so on), the [LeetCode patterns](/leetcode/) series has a short post on each.

## A time budget
Decide the split before starting, so a stuck step is noticed rather than felt. For a timed block, for example a 45-minute contest problem or practice session:

| Step | Time | Output |
|---|---|---|
| Understand | ~5 min | the problem in your own words, constraints, one worked example |
| Size and design | ~10 min | brute force with its complexity, then the better idea and its complexity |
| Edge cases | ~2 min | a written list, before any code |
| Code | ~15 min | the solution, with the invariant written as a comment |
| Test | ~8 min | a hand trace, the edge cases, a stress test |
| Reserve | ~5 min | for the bug you did not expect |

The reserve matters. Plans that use every minute end with untested code.

## Step 1: understand the problem
Restate it in your own words and settle every ambiguity: are inputs sorted, can values be negative or repeated, is the answer a count, a list or one example, can the input be empty? Then work one small example by hand. If you cannot answer a three-element input by hand, you are not ready to design an algorithm for it.

The running example: given a list of integers `nums` and an integer `k`, count the contiguous, non-empty subarrays whose sum is exactly `k`. The list has up to \(2 \times 10^4\) elements, each between -1000 and 1000 ([560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)). By hand, `nums = [1, 2, 3]` with `k = 3` has two such subarrays, `[1, 2]` and `[3]`. Note the facts that will matter later: values can be **negative**, and overlapping subarrays count separately.

## Step 2: let the constraints set the target
The input size tells you which complexities are even possible. A useful rule of thumb is about \(10^8\) simple steps per second in a compiled language; Python manages roughly ten times fewer. Rather than memorize a table, compute it:

```python
import math


def largest_n(cost, budget=1e8):
    """Largest n with cost(n) <= budget (cost increasing in n), by doubling then binary search."""
    hi = 1
    while cost(hi * 2) <= budget:
        hi *= 2
    lo, hi = hi, hi * 2                         # cost(lo) fits, cost(hi) does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if cost(mid) <= budget else (lo, mid)
    return lo


limits = {
    "n!": largest_n(math.factorial),
    "2^n * n": largest_n(lambda n: 2 ** n * n),
    "n^3": largest_n(lambda n: n ** 3),
    "n^2": largest_n(lambda n: n ** 2),
    "n log n": largest_n(lambda n: n * math.log2(n)),
}
assert limits["n!"] == 11 and limits["2^n * n"] == 22 and limits["n^3"] == 464
assert limits["n^2"] == 10_000 and 4_000_000 < limits["n log n"] < 5_000_000
```

| Largest n (about \(10^8\) steps) | Affordable complexity | Typical approaches |
|---|---|---|
| 11 | \(O(n!)\) | permutations, backtracking |
| 22 | \(O(2^n n)\) | subsets, [bitmask DP](/leetcode/bitmask-dp/) |
| ~460 | \(O(n^3)\) | interval DP, all-pairs shortest paths |
| 10,000 | \(O(n^2)\) | DP over pairs, comparing all pairs |
| ~4.5 million | \(O(n \log n)\) | [sorting](/leetcode/sorting/), [heaps](/leetcode/heap/), [binary search](/leetcode/binary-search/) |
| ~\(10^8\) | \(O(n)\) | one pass, a hash map, [two pointers](/leetcode/two-pointers/) |
| larger | \(O(\log n)\), \(O(1)\) | binary search on the answer, a formula |

For the running example, \(n = 2 \times 10^4\) gives \(n^2 / 2 = 2 \times 10^8\) pairs of endpoints: about two seconds of work even in a compiled language, and tens of seconds in Python. The target is \(O(n \log n)\) or better.

The table also works in reverse. If \(n \le 20\), the intended solution is probably exponential, and searching for a polynomial one wastes the clock.

## Step 3: brute force first
Write down the most direct correct solution and its complexity before looking for a better one. It does three jobs: it proves you understood the problem, it is a fallback you can finish if the better idea never comes, and it becomes the reference that tests the fast version.

```python
def count_brute(nums, k):
    """Every (start, end) pair, with a running sum per start: O(n^2) time, O(1) space."""
    count = 0
    for i in range(len(nums)):
        total = 0
        for j in range(i, len(nums)):
            total += nums[j]
            count += total == k
    return count


assert count_brute([1, 2, 3], 3) == 2
```

The fully naive version re-adds every subarray from scratch and costs \(O(n^3)\); keeping a running sum per start already removes one factor of \(n\). That move, keeping work from one iteration for the next, is the start of the next step.

## Step 4: find what the brute force repeats
Look at the inner loop and ask what it recomputes or searches for. Here, for a fixed end `j`, it scans every start `i` looking for a subarray sum of `k`. Let \(P_j\) be the sum of the first \(j\) elements, with \(P_0 = 0\). Then the sum of `nums[i:j]` is \(P_j - P_i\), so the inner loop is really asking:

\[ \#\{\, i < j : P_i = P_j - k \,\}. \]

A scan that asks "how many earlier values equal this one" is a lookup, and lookups are what hash maps are for. Keep a count of every prefix sum seen so far, and each step costs expected \(O(1)\): \(O(n)\) expected time for \(O(n)\) extra space. The pattern has a name, a [prefix state map](/leetcode/prefix-state-map/), and naming it is useful because it brings its known details with it: seed the map with the empty prefix, and add to the map after the lookup, not before.

Common signals and the patterns they usually point to:

| What the problem says | Pattern to try first |
|---|---|
| sorted input, or a monotone yes/no question | [binary search](/leetcode/binary-search/), [two pointers](/leetcode/two-pointers/) |
| contiguous range, and growing the range only pushes the condition one way (e.g. positive values) | [sliding window](/leetcode/sliding-window/) |
| contiguous range, values can be negative | [prefix sums with a hash map](/leetcode/prefix-state-map/) |
| next greater or smaller element | [monotonic stack](/leetcode/monotonic-stack/) |
| k best items, or repeatedly take the best option | [heap](/leetcode/heap/) |
| fewest steps in an unweighted graph | [BFS](/leetcode/bfs/) |
| tasks with prerequisites | [topological sort](/leetcode/topological-sort/) |
| groups that merge, connectivity queries | [union-find](/leetcode/union-find/) |
| best value over choices with overlapping subproblems | [dynamic programming](/leetcode/dynamic-programming/) |

State the trade explicitly: here \(O(n)\) memory buys a factor of \(n\) in time, which is a good trade unless memory is the stated constraint. And the fastest solution is not always the right one: if \(n\) were at most 1,000, the brute force would be fast enough and harder to get wrong.

### The tempting wrong pattern
"Contiguous subarray with a target sum" looks like a sliding window, and a window solution is short:

```python
def count_window(nums, k):
    """Sliding window. Correct only when every value is positive."""
    count = lo = total = 0
    for hi, x in enumerate(nums):
        total += x
        while total > k and lo <= hi:            # assumes shrinking always lowers the sum
            total -= nums[lo]
            lo += 1
        if total == k and lo <= hi:
            count += 1                            # assumes at most one valid start per end
    return count


assert count_window([1, 2, 3], 3) == 2           # passes the example...
```

A window relies on monotonicity: adding an element must raise the sum and removing one must lower it, so each end has at most one matching start. Negative values (and even zeros) break both assumptions. The example passes anyway, which is exactly why a pattern is chosen for its preconditions and not for its look. Step 6 shows how to catch this mechanically.

## Step 5: edge cases, then code
List edge cases before writing code, while you still think about the problem rather than your implementation. A general checklist:

- empty input, if the constraints allow it; a single element
- all elements equal; many duplicates
- zeros and negative numbers
- extreme values: in Java or C++, sums of many large values overflow a 32-bit integer (Python's own integers do not overflow, but NumPy's fixed-width arrays do)
- already sorted, and reverse sorted, input
- for graphs: cycles, self-loops, disconnected parts; for strings: one character, all the same character

For this problem: `k = 0`, a subarray that starts at index 0 (which needs the empty prefix in the map), all zeros (every one of the \(n(n+1)/2\) subarrays counts), and negatives. Now the code, with the invariant written as a comment so that a reader, including you in ten minutes, can check each line against it:

```python
from collections import Counter


def count_fast(nums, k):
    """Prefix sums plus a hash map: O(n) time, O(n) space."""
    seen = Counter({0: 1})        # invariant: seen[s] = number of prefixes P_0..P_j equal to s
    prefix = count = 0
    for x in nums:
        prefix += x               # prefix is now P_{j+1}
        count += seen[prefix - k] # subarrays ending here that sum to k
        seen[prefix] += 1         # only now add P_{j+1}, so a subarray is never empty
    return count
```

## Step 6: test in three passes
**First, trace the example by hand** through the code, not through your memory of the idea. For `[1, 2, 3]` and `k = 3`:

| x | prefix | looks up | found | count |
|---|---|---|---|---|
| 1 | 1 | -2 | 0 | 0 |
| 2 | 3 | 0 | 1 | 1 |
| 3 | 6 | 3 | 1 | 2 |

**Second, run the edge cases** from the list, with answers worked out by hand:

```python
assert count_fast([1, 2, 3], 3) == 2
assert count_fast([], 0) == 0                     # empty
assert count_fast([5], 5) == 1 and count_fast([5], 0) == 0
assert count_fast([0, 0, 0], 0) == 6              # all n(n+1)/2 subarrays
assert count_fast([1, -1, 1, -1], 0) == 4         # negatives
assert count_fast([3, 4, 7, 2, -3, 1, 4, 2], 7) == 4
for nums, k in [([], 0), ([5], 5), ([0, 0, 0], 0), ([1, -1, 1, -1], 0), ([3, 4, 7, 2, -3, 1, 4, 2], 7)]:
    assert count_brute(nums, k) == count_fast(nums, k)   # the hand answers agree with the brute force
```

**Third, stress test** the fast solution against the brute force on many random small inputs. Keep values in a narrow range so that interesting cases (zeros, equal sums, cancellations) happen often, and try sizes from smallest to largest, so the first failure found is already a small one that is easy to trace:

```python
import random


def stress(fast, slow, make_case, max_size=8, trials=300, seed=0):
    """Return the first input where fast and slow disagree, smallest sizes first, or None."""
    rng = random.Random(seed)
    for size in range(max_size + 1):
        for _ in range(trials):
            case = make_case(rng, size)
            if fast(*case) != slow(*case):
                return case
    return None


def small_ints(lo, hi):
    return lambda rng, n: ([rng.randint(lo, hi) for _ in range(n)], rng.randint(lo, hi))


assert stress(count_fast, count_brute, small_ints(-3, 3)) is None      # agrees everywhere
assert stress(count_window, count_brute, small_ints(1, 5)) is None     # fine on positive values

bad = stress(count_window, count_brute, small_ints(-3, 3))
nums, k = bad
assert len(nums) <= 2                                                  # a tiny counterexample
assert count_window(nums, k) != count_brute(nums, k)
assert count_window([0, 0], 0) == 2 and count_brute([0, 0], 0) == 3    # zeros alone break it
assert count_fast(*bad) == count_brute(*bad)
```

The stress test caught the window solution on an input of at most two elements, and found no disagreement at all on positive values, which matches its real precondition. A stress test is evidence, not proof: the argument for positive values is the monotonicity above. The brute force from step 3 is what makes this check possible.

A last check is that the fast solution is fast at the full size:

```python
import time

rng = random.Random(1)
big = [rng.randint(-1000, 1000) for _ in range(20_000)]
start = time.perf_counter()
count_fast(big, 0)
assert time.perf_counter() - start < 1.0          # O(n): milliseconds at n = 20,000
assert count_fast([0] * 20_000, 0) == 20_000 * 20_001 // 2
```

## When stuck
If the better idea has not come by the end of the design budget, write the brute force cleanly and finish it. A correct slow solution beats an unfinished fast one, and often the optimization becomes visible once the slow code is on the page. While writing it, try these:

- **Solve a smaller or easier version:** all values positive, \(k = 0\), a sorted input. Then ask what breaks when the restriction is lifted.
- **Ask what the inner loop searches for.** A search for an equal value suggests a hash map; for the nearest larger value, a stack; for the best value, a heap.
- **Sort first,** if the order of the input does not matter to the answer.
- **Binary search on the answer** when "can it be done with budget x?" is monotone in x and easier than finding the best x.
- **Run the problem backwards,** or from both ends, or count the complement.

## Pitfalls
- **Coding before the problem is pinned down.** A clean solution to the wrong question is worth nothing. Settle the ambiguities in step 1.
- **Choosing a pattern by its look.** Check its preconditions against the constraints, as the window example showed.
- **Silent assumptions.** Write the plan as comments before the code, and the invariant next to the loop. A written trail shows where reasoning went wrong, and makes the solution easier to review later.
- **Testing only the example.** The given example is usually the case the naive idea already handles. The edge-case list and the stress test are where bugs show up.
- **No time left to test.** Keep the reserve. If coding runs long, cut features, not testing.

The understand, plan, carry out, look back structure is George Pólya's, from How to Solve It (1945); stress testing against a brute force is standard practice in competitive programming. The problem is [LeetCode 560](https://leetcode.com/problems/subarray-sum-equals-k/), and the [prefix state map](/leetcode/prefix-state-map/) post covers its pattern in more depth.
