---
title: Dynamic programming: SRTBOT, subsequences and stock state machines
summary: A six-step recipe for writing any DP, then four classics built with it — longest common subsequence, palindromic subsequences, the whole stock-trading family as one state machine, and Kadane's maximum subarray.
group: Dynamic programming, bits and games
order: 1
tags: dynamic programming · memoization · state machine
---
Dynamic programming is recursion that refuses to solve the same problem twice. You describe the answer in terms of answers to smaller versions of the same question, then either cache the recursive calls (**top-down**, memoization) or fill a table in an order where every entry's dependencies are already known (**bottom-up**). Both compute the same numbers; the hard part is never the caching, it is choosing the smaller questions.

## The SRTBOT recipe
MIT's 6.006 breaks the design of a DP into six checks, abbreviated SRTBOT:

1. **Subproblems:** a family of questions, usually indexed by a prefix, a suffix, or a substring of the input, plus whatever extra state the problem needs.
2. **Relation:** how one subproblem's answer is built from strictly smaller ones. A good habit is to "guess" the first decision (take this item or not, match these characters or not) and take the best over all guesses.
3. **Topological order:** the dependencies must form a DAG, and the bottom-up loops must visit it in that order.
4. **Base cases:** the subproblems with no smaller ones beneath them.
5. **Original problem:** which subproblem (or combination of them) is the answer.
6. **Time:** number of subproblems times work per subproblem.

Writing the six lines down before any code catches most DP bugs: a relation that refers to a larger subproblem, a base case that is missing, or an answer read from the wrong cell.

## Longest common subsequence
[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/). Run the recipe on strings `s` (length \(m\)) and `t` (length \(n\)).

- **S:** \(L(i, j)\) is the LCS length of the suffixes `s[i:]` and `t[j:]`.
- **R:** if the first characters match, using them is never worse; otherwise one of them is not in the answer, so drop either and keep the better:

\[ L(i, j) = \begin{cases} 1 + L(i+1, j+1) & s_i = t_j \\ \max\bigl(L(i+1, j),\ L(i, j+1)\bigr) & \text{otherwise.} \end{cases} \]

- **T:** every call increases \(i\) or \(j\), so loop \(i\) and \(j\) downward.
- **B:** \(L(m, j) = L(i, n) = 0\): an empty suffix has nothing in common.
- **O:** \(L(0, 0)\).
- **T:** \((m+1)(n+1)\) subproblems at \(O(1)\) each, so \(O(mn)\).

The top-down version is the recurrence typed in, with a cache in front of it:

```python
import random
from functools import lru_cache
from itertools import combinations, product


def lcs_top_down(s, t):
    @lru_cache(maxsize=None)
    def L(i, j):
        if i == len(s) or j == len(t):
            return 0
        if s[i] == t[j]:
            return 1 + L(i + 1, j + 1)
        return max(L(i + 1, j), L(i, j + 1))
    return L(0, 0)


assert lcs_top_down("abcde", "ace") == 3
assert lcs_top_down("abc", "abc") == 3
assert lcs_top_down("abc", "def") == 0
```

It recurses up to \(m + n\) deep, which overflows Python's default limit at the problem's 1000-character inputs. The bottom-up table avoids that, and because row `i` only reads row `i + 1`, one row is enough. The single subtle cell is the diagonal \(L(i+1, j+1)\), which the row has already overwritten by the time we need it, so it is saved in a variable first:

```python
def lcs(s, t):
    n = len(t)
    row = [0] * (n + 1)                  # row[j] = L(i + 1, j) before the update, L(i, j) after
    for i in range(len(s) - 1, -1, -1):
        diag = 0                         # L(i + 1, n)
        for j in range(n - 1, -1, -1):
            below = row[j]               # L(i + 1, j), about to be overwritten
            row[j] = 1 + diag if s[i] == t[j] else max(below, row[j + 1])
            diag = below
    return row[0]


def is_subsequence(sub, s):
    it = iter(s)
    return all(c in it for c in sub)


def lcs_brute(s, t):
    for size in range(min(len(s), len(t)), -1, -1):
        if any(is_subsequence(c, t) for c in combinations(s, size)):
            return size


assert lcs("abcde", "ace") == 3 and lcs("abc", "abc") == 3 and lcs("abc", "def") == 0
for _ in range(300):
    s = "".join(random.choices("abc", k=random.randint(0, 8)))
    t = "".join(random.choices("abc", k=random.randint(0, 8)))
    assert lcs(s, t) == lcs_top_down(s, t) == lcs_brute(s, t)
```

Space drops to \(O(n)\); time stays \(O(mn)\). The same table, with "match" meaning equal characters and "mismatch" resetting to zero, gives longest common **substring** ([718. Maximum Length of Repeated Subarray](https://leetcode.com/problems/maximum-length-of-repeated-subarray/)), and with three choices per cell gives edit distance ([72. Edit Distance](https://leetcode.com/problems/edit-distance/)).

## Palindromes: subproblems on substrings
When an answer depends on both ends of the string, the subproblem is a substring `s[lo..hi]` instead of a suffix. [516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/):

- **S:** \(P(lo, hi)\) is the answer for `s[lo..hi]` inclusive.
- **R:** if the end characters match, both belong to the palindrome, \(P = 2 + P(lo+1, hi-1)\); otherwise drop one end, \(P = \max(P(lo+1, hi), P(lo, hi-1))\).
- **T:** a substring depends only on shorter ones, so iterate `lo` downward and `hi` upward from `lo`.
- **B:** \(P(lo, lo) = 1\), and an empty range is 0.
- **O:** \(P(0, n-1)\). **Time:** \(O(n^2)\).

(It is also the LCS of `s` with its reverse, a nice check on both functions.)

```python
def longest_palindrome_subseq(s):
    n = len(s)
    P = [[0] * n for _ in range(n)]
    for lo in range(n - 1, -1, -1):
        P[lo][lo] = 1
        for hi in range(lo + 1, n):
            if s[lo] == s[hi]:
                P[lo][hi] = 2 + P[lo + 1][hi - 1]      # an empty middle is the 0 below the diagonal
            else:
                P[lo][hi] = max(P[lo + 1][hi], P[lo][hi - 1])
    return P[0][n - 1]


def lps_brute(s):
    return max(len(c) for size in range(len(s) + 1) for c in combinations(s, size) if c == c[::-1])


assert longest_palindrome_subseq("bbbab") == 4
assert longest_palindrome_subseq("cbbd") == 2
for _ in range(300):
    s = "".join(random.choices("ab", k=random.randint(1, 10)))
    assert longest_palindrome_subseq(s) == lps_brute(s) == lcs(s, s[::-1])
```

For contiguous palindromes ([5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)) the table holds booleans: `s[lo..hi]` is a palindrome when its ends match and its inside, of length below 2 or already known, is one. The same loop order applies.

```python
def longest_palindrome(s):
    n, best = len(s), (0, 0)
    pal = [[False] * n for _ in range(n)]
    for lo in range(n - 1, -1, -1):
        for hi in range(lo, n):
            pal[lo][hi] = s[lo] == s[hi] and (hi - lo < 3 or pal[lo + 1][hi - 1])
            if pal[lo][hi] and hi - lo > best[1] - best[0]:
                best = (lo, hi)
    return s[best[0]:best[1] + 1]


assert longest_palindrome("babad") in ("bab", "aba")
assert longest_palindrome("cbbd") == "bb"
assert longest_palindrome("a") == "a"
for _ in range(300):
    s = "".join(random.choices("ab", k=random.randint(1, 12)))
    got = longest_palindrome(s)
    assert got in s and got == got[::-1]
    assert len(got) == max(j - i for i in range(len(s)) for j in range(i + 1, len(s) + 1)
                           if s[i:j] == s[i:j][::-1])
```

\(O(n^2)\) time and space; expanding around each of the \(2n - 1\) centres is the same time in \(O(1)\) space, and Manacher's algorithm is linear.

## The stock family as a state machine
Six problems ask for the best profit from a price list under different rules: one trade (121), unlimited trades (122), at most two (123) or \(k\) (188), a one-day cooldown after selling (309), or a fee per trade (714). They are one DP. On each day you are in a state, and the subproblem is "the best cash you can have at the end of day \(i\) in that state":

- \(\text{cash}_i[t]\): not holding a share, having bought at most \(t\) times;
- \(\text{hold}_i[t]\): holding a share, having bought at most \(t\) times.

Each day you rest, sell, or buy:

\[ \text{cash}_i[t] = \max\bigl(\text{cash}_{i-1}[t],\ \text{hold}_{i-1}[t] + p_i - \text{fee}\bigr), \qquad \text{hold}_i[t] = \max\bigl(\text{hold}_{i-1}[t],\ \text{cash}_{i-1-c}[t-1] - p_i\bigr) \]

where \(c\) is the cooldown length: a buy may only follow a state from \(c\) days earlier. Base cases: before day 0, all cash states are 0 and holding is impossible (\(-\infty\)). The answer is \(\text{cash}[k]\) on the last day. Unlimited trades are just \(k = \lfloor n/2 \rfloor\), since a buy and its sell use two different days.

```python
NEG = float("-inf")


def stock_profit(prices, k=None, fee=0, cooldown=0):
    n = len(prices)
    k = n // 2 if k is None else min(k, n // 2)
    past = [[0] * (k + 1) for _ in range(cooldown + 1)]   # cash arrays of the last few days
    hold = [NEG] * (k + 1)
    for p in prices:
        cash, buy_from = past[-1], past[-1 - cooldown]
        hold = [NEG] + [max(hold[t], buy_from[t - 1] - p) for t in range(1, k + 1)]
        new_cash = [max(cash[t], hold[t] + p - fee) for t in range(k + 1)]
        past = past[1:] + [new_cash]
    return past[-1][k]
```

That is \(O(nk)\) time and \(O((c+1)k)\) space, which for unlimited trades means \(O(n^2)\) time because \(k\) becomes \(n/2\); the specialisations below bring that back to \(O(n)\). One subtlety: `hold` is computed before `new_cash`, so a sale on day \(i\) uses a share bought on day \(i\) at worst, which is a no-op with zero fee and a loss with a fee, so it never changes the answer.

To trust it, compare with honest brute force: try every string of actions (buy, sell, rest) and simulate the rules.

```python
def stock_brute(prices, k=None, fee=0, cooldown=0):
    best = 0
    for plan in product("bsr", repeat=len(prices)):
        cash, holding, buys, wait, ok = 0, False, 0, 0, True
        for p, a in zip(prices, plan):
            if a == "b":
                if holding or wait or (k is not None and buys == k):
                    ok = False
                    break
                cash, holding, buys = cash - p, True, buys + 1
            elif a == "s":
                if not holding:
                    ok = False
                    break
                cash, holding, wait = cash + p - fee, False, cooldown + 1
            wait = max(0, wait - 1)
        if ok and not holding:
            best = max(best, cash)
    return best


assert stock_profit([7, 1, 5, 3, 6, 4], k=1) == 5                    # 121
assert stock_profit([7, 6, 4, 3, 1], k=1) == 0
assert stock_profit([7, 1, 5, 3, 6, 4]) == 7                         # 122
assert stock_profit([1, 2, 3, 4, 5]) == 4
assert stock_profit([7, 6, 4, 3, 1]) == 0
assert stock_profit([3, 3, 5, 0, 0, 3, 1, 4], k=2) == 6              # 123
assert stock_profit([1, 2, 3, 4, 5], k=2) == 4
assert stock_profit([7, 6, 4, 3, 1], k=2) == 0
assert stock_profit([2, 4, 1], k=2) == 2                             # 188
assert stock_profit([3, 2, 6, 5, 0, 3], k=2) == 7
assert stock_profit([1, 2, 3, 0, 2], cooldown=1) == 3                # 309
assert stock_profit([1], cooldown=1) == 0
assert stock_profit([1, 3, 2, 8, 4, 9], fee=2) == 8                  # 714
assert stock_profit([1, 3, 7, 5, 10, 3], fee=3) == 6

for _ in range(150):
    prices = [random.randint(0, 9) for _ in range(random.randint(0, 7))]
    for args in ({"k": 1}, {"k": 2}, {}, {"cooldown": 1}, {"fee": 2}):
        assert stock_profit(prices, **args) == stock_brute(prices, **args), (prices, args)
```

Each special case then shrinks to a couple of variables. With \(k = 1\), \(\text{cash}[0]\) is always 0, so the buy is simply \(-p\). With unlimited trades the \(t\) index disappears altogether (\(t\) and \(t-1\) are the same "unlimited"), leaving one `cash` and one `hold`; cooldown keeps one extra day of `cash` around:

```python
def max_profit_one(prices):                  # 121
    cash, hold = 0, NEG
    for p in prices:
        cash, hold = max(cash, hold + p), max(hold, -p)
    return cash


def max_profit_unlimited(prices, fee=0):     # 122, 714
    cash, hold = 0, NEG
    for p in prices:
        cash = max(cash, hold + p - fee)
        hold = max(hold, cash - p)           # today's cash: selling and rebuying today never helps
    return cash


def max_profit_cooldown(prices):             # 309
    cash, prev_cash, hold = 0, 0, NEG
    for p in prices:
        cash, prev_cash, hold = max(cash, hold + p), cash, max(hold, prev_cash - p)
    return cash


for _ in range(300):
    prices = [random.randint(0, 20) for _ in range(random.randint(0, 12))]
    assert max_profit_one(prices) == stock_profit(prices, k=1)
    assert max_profit_unlimited(prices) == stock_profit(prices)
    assert max_profit_unlimited(prices, fee=3) == stock_profit(prices, fee=3)
    assert max_profit_cooldown(prices) == stock_profit(prices, cooldown=1)
```

All three are \(O(n)\) time and \(O(1)\) space.

## Kadane's maximum subarray
[53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) is a one-state DP. **S:** \(E(i)\), the best sum of a subarray **ending** at \(i\). **R:** either extend the best one ending at \(i-1\) or start fresh, \(E(i) = \max(x_i, E(i-1) + x_i)\). **O:** \(\max_i E(i)\), because the best subarray ends somewhere. \(O(n)\) time, \(O(1)\) space.

The answer must be seeded with the first element, not 0: the subarray is non-empty, so an all-negative array's answer is its largest element. (Seeding with 0 instead solves the variant that allows the empty subarray.)

```python
def max_subarray(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([5, 4, -1, 7, 8]) == 23
assert max_subarray([-3, -1, -2]) == -1      # all negative: not 0
for _ in range(300):
    nums = [random.randint(-10, 10) for _ in range(random.randint(1, 15))]
    assert max_subarray(nums) == max(sum(nums[i:j]) for i in range(len(nums)) for j in range(i + 1, len(nums) + 1))
```

"Best ending here" is the subproblem behind many one-pass DPs: [1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/) keeps the best `values[i] + i` seen so far, and [918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/) runs Kadane for both the maximum and the minimum. When the extra state is a capacity, the DP becomes a [knapsack](../knapsack/); when it is a set, it becomes [bitmask DP](../bitmask-dp/).

## Practice
- [1014. Best Sightseeing Pair](https://leetcode.com/problems/best-sightseeing-pair/)
- [1031. Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)
- [1039. Minimum Score Triangulation of Polygon](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/) (interval DP, like 516)
- [1055. Shortest Way to Form String](https://leetcode.com/problems/shortest-way-to-form-string/)
- [1062. Longest Repeating Substring](https://leetcode.com/problems/longest-repeating-substring/) (the substring table of `s` against itself, keeping only cells with \(i < j\))
- [72. Edit Distance](https://leetcode.com/problems/edit-distance/)
- [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)
- [1312. Minimum Insertion Steps to Make a String Palindrome](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)
- [123. Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) and [188. Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)
- [309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) and [714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)
- [152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/) (Kadane with two states)

The SRTBOT framework is from MIT 6.006 Introduction to Algorithms, Spring 2020 (E. Demaine, J. Ku, J. Solomon). Dynamic programming itself is from R. Bellman, "Dynamic Programming", Princeton University Press (1957); Kadane's algorithm was popularised by J. Bentley, "Programming Pearls: Algorithm Design Techniques", Communications of the ACM (1984).
