---
title: Knapsack: choosing items under a budget
summary: 0/1 and unbounded knapsack written three ways (memoized recursion, a full table, one rolling row), why the inner loop runs backwards for one and forwards for the other, how to read the chosen items back out, and four LeetCode problems that are knapsack in disguise.
group: Dynamic programming, bits and games
order: 2
tags: dynamic programming · knapsack · subset sum
---
You have a bag that holds weight \(W\) and a pile of items, each with a weight and a value. Which items should go in to make the total value as large as possible without going over? Trying every subset costs \(2^n\). Knapsack DP avoids that by noticing that once you have decided about the first few items, all that matters about those decisions is **how much room they left**. So the state is "which item am I looking at, and how much room is left", and there are only \(n \cdot (W + 1)\) of those.

Two versions come up again and again. In **0/1 knapsack** each item is taken at most once. In **unbounded knapsack** each item can be taken any number of times (coins, tile sizes, ingredients). The recurrences differ by one index, and in the space-optimized code that index turns into the direction of a loop.

## 0/1 knapsack: the recurrence
Let \(f(i, c)\) be the best value you can get from items \(i, i+1, \dots, n-1\) with capacity \(c\) still free. For item \(i\) with weight \(w_i\) and value \(v_i\) there are two choices: skip it, or take it if it fits.

\[
f(i, c) = \max\bigl(f(i+1, c),\; f(i+1, c - w_i) + v_i\bigr), \qquad f(n, c) = 0
\]

The second term is only allowed when \(w_i \le c\). The answer is \(f(0, W)\). There are \(O(nW)\) states, each computed in \(O(1)\), so the time is \(O(nW)\). That is pseudo-polynomial: polynomial in the numeric value \(W\), but exponential in the number of bits needed to write \(W\) down. This is no accident, since the decision version of knapsack is NP-complete; the DP only helps when capacities are small integers.

### Top-down
The recurrence translated directly, with a cache. The cache holds \(O(nW)\) entries and the recursion goes \(n\) deep.

```python
import random
from functools import lru_cache
from itertools import product


def knapsack_01_memo(weights, values, cap):
    n = len(weights)

    @lru_cache(maxsize=None)
    def f(i, c):
        if i == n:
            return 0
        best = f(i + 1, c)                                   # skip item i
        if weights[i] <= c:
            best = max(best, f(i + 1, c - weights[i]) + values[i])   # take it
        return best

    return f(0, cap)


def knapsack_brute(weights, values, cap):
    best = 0
    for pick in product((0, 1), repeat=len(weights)):        # every subset
        if sum(p * w for p, w in zip(pick, weights)) <= cap:
            best = max(best, sum(p * v for p, v in zip(pick, values)))
    return best


assert knapsack_01_memo([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9  # items of weight 3 and 4
```

### Bottom-up
Row \(i\) depends only on row \(i + 1\), so fill the table from the last item backwards. Within a row, any order of \(c\) works, since every value read comes from the row below.

```python
def knapsack_01_table(weights, values, cap):
    n = len(weights)
    f = [[0] * (cap + 1) for _ in range(n + 1)]              # row n is the base case
    for i in range(n - 1, -1, -1):
        w, v = weights[i], values[i]
        for c in range(cap + 1):
            f[i][c] = f[i + 1][c]
            if w <= c:
                f[i][c] = max(f[i][c], f[i + 1][c - w] + v)
    return f
```

### Recovering the chosen items
The table also says which items to take. Start at \(f(0, W)\). If \(f(i, c) \ne f(i+1, c)\), skipping item \(i\) would have lost value, so item \(i\) was taken: record it and subtract its weight. Otherwise skip it. Walk \(i\) forward from \(0\) to \(n - 1\). This needs the full table; the one-row version below throws away exactly the information this walk uses.

```python
def knapsack_01_items(weights, values, cap):
    f = knapsack_01_table(weights, values, cap)
    chosen, c = [], cap
    for i in range(len(weights)):
        if f[i][c] != f[i + 1][c]:
            chosen.append(i)
            c -= weights[i]
    return f[0][cap], chosen


best, chosen = knapsack_01_items([1, 3, 4, 5], [1, 4, 5, 7], 7)
assert best == 9 and chosen == [1, 2]
```

### One row, iterated backwards
Since row \(i\) reads only row \(i + 1\), you can keep a single list `dp` and overwrite it in place. At the moment you update `dp[c]`, you want `dp[c]` and `dp[c - w]` to still hold the old row (the values without item \(i\)). Index `c - w` is smaller than `c`. If `c` runs **downwards**, `dp[c - w]` has not been touched yet in this pass, so it is still the old value. If `c` ran upwards, `dp[c - w]` might already include item \(i\), and adding \(v\) again would take the item twice. Space drops to \(O(W)\).

```python
def knapsack_01(weights, values, cap):
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):                      # high to low: each item once
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[cap]


for _ in range(300):
    n = random.randint(0, 8)
    ws = [random.randint(0, 10) for _ in range(n)]           # zero weights allowed
    vs = [random.randint(0, 20) for _ in range(n)]
    cap = random.randint(0, 30)
    expect = knapsack_brute(ws, vs, cap)
    assert knapsack_01_memo(ws, vs, cap) == knapsack_01(ws, vs, cap) == expect
    best, chosen = knapsack_01_items(ws, vs, cap)
    assert best == expect == sum(vs[i] for i in chosen)
    assert sum(ws[i] for i in chosen) <= cap

assert knapsack_01([], [], 5) == 0 and knapsack_01([3], [9], 2) == 0
assert knapsack_01([0, 2], [4, 3], 1) == 4                  # a weightless item is always worth taking
```

## Unbounded knapsack: the same loop, forwards
If items can be reused, taking item \(i\) does not move you on to item \(i + 1\); you may take it again:

\[
f(i, c) = \max\bigl(f(i+1, c),\; f(i, c - w_i) + v_i\bigr)
\]

The only change is \(f(i, \cdot)\) instead of \(f(i+1, \cdot)\) in the second term. In one row that means you want `dp[c - w]` to already include item \(i\), which is exactly what an **upward** loop gives you. The loop direction is the whole difference between the two problems.

```python
def knapsack_unbounded(weights, values, cap):
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(w, cap + 1):                          # low to high: reuse allowed
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[cap]


def unbounded_brute(weights, values, cap):
    def go(i, c):                                            # try every count of item i
        if i == len(weights):
            return 0
        return max(go(i + 1, c - k * weights[i]) + k * values[i]
                   for k in range(c // weights[i] + 1))
    return go(0, cap)


for _ in range(200):
    n = random.randint(0, 4)
    ws = [random.randint(1, 8) for _ in range(n)]
    vs = [random.randint(0, 15) for _ in range(n)]
    cap = random.randint(0, 20)
    assert knapsack_unbounded(ws, vs, cap) == unbounded_brute(ws, vs, cap)
```

## 416. Partition Equal Subset Sum
[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/): can the array be split into two parts with equal sums? Equivalently, is there a subset summing to half the total? That is 0/1 knapsack where the "value" is a boolean: `dp[s]` says whether some subset of the items seen so far sums to `s`. Odd totals fail immediately. Time \(O(n \cdot \text{target})\), space \(O(\text{target})\).

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = [True] + [False] * target                           # the empty subset sums to 0
    for x in nums:
        for s in range(target, x - 1, -1):                   # 0/1, so downwards
            dp[s] = dp[s] or dp[s - x]
    return dp[target]


assert can_partition([1, 5, 11, 5])
assert not can_partition([1, 2, 3, 5])
for _ in range(300):
    nums = [random.randint(1, 12) for _ in range(random.randint(1, 10))]
    brute = any(2 * sum(x for p, x in zip(pick, nums) if p) == sum(nums)
                for pick in product((0, 1), repeat=len(nums)))
    assert can_partition(nums) == brute
```

The same row fits in one Python integer: bit `s` set means sum `s` is reachable, and taking item `x` is `bits |= bits << x`. Python then does the whole inner loop as one big-integer shift, a machine word at a time; see [bit manipulation](../bit-manipulation/).

## 322. Coin Change
[322. Coin Change](https://leetcode.com/problems/coin-change/): the fewest coins that make `amount`, with unlimited coins of each denomination, or \(-1\). This is unbounded knapsack with "max value" replaced by "min count": `dp[a] = min(dp[a], dp[a - coin] + 1)`, looping upwards. Unreachable amounts stay at infinity. Time \(O(\text{amount} \cdot k)\) for \(k\) coin types.

```python
def coin_change(coins, amount):
    INF = float("inf")
    dp = [0] + [INF] * amount
    for coin in coins:
        for a in range(coin, amount + 1):                    # unbounded, so upwards
            dp[a] = min(dp[a], dp[a - coin] + 1)
    return dp[amount] if dp[amount] < INF else -1


assert coin_change([1, 2, 5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0


def coin_change_brute(coins, amount):                        # BFS over amounts left
    frontier, seen, steps = {amount}, {amount}, 0
    while frontier:
        if 0 in frontier:
            return steps
        frontier = {a - c for a in frontier for c in coins if a - c >= 0} - seen
        seen |= frontier
        steps += 1
    return -1


for _ in range(300):
    coins = random.sample(range(1, 15), random.randint(1, 4))
    amount = random.randint(0, 60)
    assert coin_change(coins, amount) == coin_change_brute(coins, amount)
```

Greedy (always take the largest coin that fits) is not safe: with coins `[1, 3, 4]` and amount 6 it picks 4, 1, 1, while 3, 3 uses two coins.

## 518. Coin Change II, and why loop order counts
[518. Coin Change II](https://leetcode.com/problems/coin-change-ii/): the number of **combinations** of coins summing to `amount`. Replace min with `+=`: `dp[a] += dp[a - coin]`, starting from `dp[0] = 1`.

There is a second ordering question here, between the two loops. With **coins on the outside**, each pass adds one denomination to the set allowed so far, so every combination is built in one fixed order (all 1s, then all 2s, ...) and counted once. Put **amounts on the outside** and every coin is tried as the last coin at every amount, so `1 + 2` and `2 + 1` are counted separately: that counts ordered sequences, which is [377. Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/).

```python
def change(amount, coins):                                   # combinations
    dp = [1] + [0] * amount
    for coin in coins:                                       # coins outer
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    return dp[amount]


def combination_sum4(nums, target):                          # ordered sequences
    dp = [1] + [0] * target
    for a in range(1, target + 1):                           # amounts outer
        for x in nums:
            if x <= a:
                dp[a] += dp[a - x]
    return dp[target]


assert change(5, [1, 2, 5]) == 4
assert change(3, [2]) == 0
assert change(10, [10]) == 1
assert combination_sum4([1, 2, 3], 4) == 7
assert combination_sum4([9], 3) == 0


def change_brute(amount, coins):                             # choose a count per coin
    if not coins:
        return int(amount == 0)
    return sum(change_brute(amount - k * coins[0], coins[1:])
               for k in range(amount // coins[0] + 1))


def sequences_brute(nums, target):
    return 1 if target == 0 else sum(sequences_brute(nums, target - x) for x in nums if x <= target)


for _ in range(200):
    coins = random.sample(range(1, 10), random.randint(1, 4))
    amount = random.randint(0, 25)
    assert change(amount, coins) == change_brute(amount, coins)
    assert combination_sum4(coins, amount % 15) == sequences_brute(coins, amount % 15)
```

## 494. Target Sum
[494. Target Sum](https://leetcode.com/problems/target-sum/): put `+` or `-` in front of every number and count the ways to reach `target`. Let \(P\) be the sum of the numbers with a plus and \(T\) the total. The minus side sums to \(T - P\), so

\[
P - (T - P) = \text{target} \quad\Longleftrightarrow\quad P = \frac{T + \text{target}}{2}
\]

So the question becomes: how many subsets sum to \(P\)? If \(T + \text{target}\) is odd or negative, or \(|\text{target}| > T\), the answer is 0. Otherwise it is counting 0/1 knapsack, inner loop downwards. Zeros need no special care: a zero doubles every count, matching the two signs it can take.

```python
def find_target_sum_ways(nums, target):
    total = sum(nums)
    if abs(target) > total or (total + target) % 2:
        return 0
    p = (total + target) // 2
    dp = [1] + [0] * p
    for x in nums:
        for s in range(p, x - 1, -1):
            dp[s] += dp[s - x]
    return dp[p]


assert find_target_sum_ways([1, 1, 1, 1, 1], 3) == 5
assert find_target_sum_ways([1], 1) == 1
for _ in range(300):
    nums = [random.randint(0, 6) for _ in range(random.randint(1, 9))]
    target = random.randint(-15, 15)
    brute = sum(sum(s * x for s, x in zip(signs, nums)) == target
                for signs in product((1, -1), repeat=len(nums)))
    assert find_target_sum_ways(nums, target) == brute
```

## Recognizing a knapsack
The tell is a set of items, a numeric budget or target that is a small integer, and a per-item decision. Ask two questions: can an item be reused (loop direction), and does order matter when counting (loop nesting). The general framework of states and transitions is in [dynamic programming](../dynamic-programming/); when the "budget" is a set of used items instead of a number, see [bitmask DP](../bitmask-dp/).

## Practice
- [2291. Maximum Profit From Trading Stocks](https://leetcode.com/problems/maximum-profit-from-trading-stocks/) (plain 0/1 knapsack)
- [2209. Minimum White Tiles After Covering With Carpets](https://leetcode.com/problems/minimum-white-tiles-after-covering-with-carpets/) (a budget of carpets as the second index)
- [2463. Minimum Total Distance Traveled](https://leetcode.com/problems/minimum-total-distance-traveled/) (each factory is a bounded item group)
- [1911. Maximum Alternating Subsequence Sum](https://leetcode.com/problems/maximum-alternating-subsequence-sum/) (take-or-skip DP with a parity state)
- [474. Ones and Zeroes](https://leetcode.com/problems/ones-and-zeroes/) (two capacities)
- [1049. Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/) (subset sum in disguise)
- [279. Perfect Squares](https://leetcode.com/problems/perfect-squares/) (Coin Change with squares)
- [879. Profitable Schemes](https://leetcode.com/problems/profitable-schemes/)
- [1155. Number of Dice Rolls With Target Sum](https://leetcode.com/problems/number-of-dice-rolls-with-target-sum/) (group knapsack)

The suffix-subproblem formulation and the backtracking step follow MIT 6.006 Introduction to Algorithms (Fall 2011), Lecture 21 (Dynamic Programming III, which covers knapsack). The LeetCode Discuss study guide "Thief with a knapsack: a series of crimes" and William Fiset's Algorithms repository on GitHub cover the same ground. Dynamic programming itself is from R. Bellman, "Dynamic Programming", Princeton University Press (1957).
