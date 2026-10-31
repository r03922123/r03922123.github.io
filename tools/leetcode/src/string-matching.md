---
title: String matching with KMP and the Z-function
summary: Two linear-time tools that reuse what a failed comparison already revealed. The KMP failure table, the Z-array, and how each finds a pattern or scores a string against its own prefixes.
group: Sorting, searching and strings
order: 7
tags: strings · KMP · Z-function
---
The naive way to find a pattern of length \(m\) in a text of length \(n\) restarts after every mismatch, \(O(nm)\) in the worst case (try finding `aaab` in `aaaa…a`). Both algorithms here are \(O(n + m)\), and both win the same way: when a comparison fails, they already know how much of the pattern still matches, so they never re-read the text.

## KMP: the failure table
For each prefix `p[:i+1]` of the pattern, let `lps[i]` be the length of its longest proper prefix that is also a suffix. If we have matched `j` characters and the next one fails, the last `lps[j-1]` characters we matched are also the pattern's first `lps[j-1]` characters, so we can continue from there instead of from zero.

```plaintext
pattern  a b a b a c a
lps      0 0 1 2 3 0 1
```

Building the table is the same algorithm run on the pattern against itself:

```python
def build_lps(p):
    lps, j = [0] * len(p), 0                # j = length of the current matched prefix
    for i in range(1, len(p)):
        while j and p[i] != p[j]:
            j = lps[j - 1]                  # fall back to the next shorter border
        if p[i] == p[j]:
            j += 1
        lps[i] = j
    return lps


def kmp_find(text, p):
    if not p:
        return 0
    lps, j = build_lps(p), 0
    for i, ch in enumerate(text):
        while j and ch != p[j]:
            j = lps[j - 1]
        if ch == p[j]:
            j += 1
        if j == len(p):
            return i - len(p) + 1
    return -1


assert build_lps("ababaca") == [0, 0, 1, 2, 3, 0, 1]
assert kmp_find("sadbutsad", "sad") == 0
assert kmp_find("aaaaaaaab", "aaab") == 5
import random
for _ in range(500):
    t = "".join(random.choice("ab") for _ in range(random.randint(0, 14)))
    p = "".join(random.choice("ab") for _ in range(random.randint(0, 4)))
    assert kmp_find(t, p) == t.find(p)
```

Why linear: `j` increases at most once per text character, and every fallback decreases it, so the total number of fallbacks is at most \(n\).

## The Z-function
`z[i]` is the length of the longest substring starting at `i` that matches a **prefix** of the string. The algorithm keeps the rightmost window `[lo, hi)` known to match a prefix; for a position inside it, the mirrored position `i - lo` already tells us a lower bound, and only characters beyond `hi` are compared fresh.

```python
def z_function(s):
    n = len(s)
    z, lo, hi = [0] * n, 0, 0               # s[lo:hi] matches the prefix s[0:hi-lo]
    for i in range(1, n):
        if i < hi:
            z[i] = min(hi - i, z[i - lo])   # reuse the mirrored value, capped at the window's end
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > hi:
            lo, hi = i, i + z[i]
    return z


def z_brute(s):
    out = [0] * len(s)
    for i in range(1, len(s)):
        while i + out[i] < len(s) and s[out[i]] == s[i + out[i]]:
            out[i] += 1
    return out


assert z_function("aabxaab") == [0, 1, 0, 0, 3, 1, 0]
for _ in range(500):
    t = "".join(random.choice("ab") for _ in range(random.randint(0, 14)))
    assert z_function(t) == z_brute(t)
```

To search, run it on `pattern + sep + text` with a separator that occurs in neither: a match is any position where `z[i]` equals the pattern length.

[2223. Sum of Scores of Built Strings](https://leetcode.com/problems/sum-of-scores-of-built-strings/) asks, for every suffix, the length of its common prefix with the whole string. That is the Z-array by definition, plus \(n\) for the string itself:

```python
def sum_scores(s):
    return sum(z_function(s)) + len(s)


assert sum_scores("babab") == 9
assert sum_scores("azbazbzaz") == 14
```

## Which to use
Both are linear and equally short. KMP streams over the text without storing it, which suits online matching. The Z-array answers "how much of the prefix matches here" for every position at once, which suits scoring problems like 2223. For comparing many substrings, a [rolling hash](../rolling-hash/) is simpler.

## Practice
- [459. Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/)
- [214. Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/)
- [1392. Longest Happy Prefix](https://leetcode.com/problems/longest-happy-prefix/)
- [758. Bold Words in String](https://leetcode.com/problems/bold-words-in-string/)
- [2223. Sum of Scores of Built Strings](https://leetcode.com/problems/sum-of-scores-of-built-strings/)

KMP is from D. E. Knuth, J. H. Morris and V. R. Pratt, "Fast Pattern Matching in Strings" (1977).
