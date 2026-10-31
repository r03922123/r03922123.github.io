---
title: Rolling hash (Rabin–Karp)
summary: Hash every length-m window of a string in O(1) per step by sliding a polynomial hash. Substring search, then the real use, comparing many substrings at once, with binary search on length.
group: Sorting, searching and strings
order: 6
tags: hashing · strings · binary search
---
Comparing two strings of length \(m\) costs \(O(m)\). A rolling hash makes comparing two **substrings** cost \(O(1)\), after \(O(n)\) preprocessing, by giving every window a number that can be updated as the window slides. Equal strings always have equal hashes; unequal ones collide only with small probability, and a collision can be ruled out with a direct comparison.

## The polynomial hash
Treat the string as digits in base \(B\), modulo a large prime \(M\):

\[ h(s_0 s_1 \ldots s_{m-1}) = \left(s_0 B^{m-1} + s_1 B^{m-2} + \cdots + s_{m-1}\right) \bmod M. \]

Sliding one step right removes the leading digit and appends a new one:

\[ h_{\text{next}} = \left((h - s_{\text{out}} B^{m-1}) \cdot B + s_{\text{in}}\right) \bmod M. \]

Precompute \(B^{m-1} \bmod M\) once, and each step is a few multiplications. Use a large prime \(M\) (here \(2^{61} - 1\)) and a random base, so no fixed input can be built to force collisions.

## Substring search
[28. Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/): compare the pattern's hash with each window's, and confirm matches directly.

```python
import random

MOD = (1 << 61) - 1
BASE = random.randrange(256, MOD - 1)


def find(text, pattern):
    m, n = len(pattern), len(text)
    if m == 0:
        return 0
    if m > n:
        return -1
    top = pow(BASE, m - 1, MOD)                     # weight of the leading character
    hp = hw = 0
    for i in range(m):
        hp = (hp * BASE + ord(pattern[i])) % MOD
        hw = (hw * BASE + ord(text[i])) % MOD
    for i in range(n - m + 1):
        if hw == hp and text[i:i + m] == pattern:   # confirm: rules out a collision
            return i
        if i + m < n:
            hw = ((hw - ord(text[i]) * top) * BASE + ord(text[i + m])) % MOD
    return -1


assert find("sadbutsad", "sad") == 0
assert find("leetcode", "leeto") == -1
for _ in range(500):
    t = "".join(random.choice("ab") for _ in range(random.randint(0, 12)))
    p = "".join(random.choice("ab") for _ in range(random.randint(0, 4)))
    assert find(t, p) == t.find(p)
```

Expected \(O(n + m)\). For one pattern, [KMP](../string-matching/) gives the same bound with no probability involved; the rolling hash earns its place when many substrings must be compared.

## Where it shines: many substrings at once
With prefix hashes, the hash of **any** substring is \(O(1)\):

\[ h(s[i:j]) = \left(P_j - P_i \cdot B^{j-i}\right) \bmod M, \qquad P_{k+1} = (P_k \cdot B + s_k) \bmod M. \]

[1044. Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/): if a repeated substring of length \(L\) exists, one of length \(L - 1\) does too, so the length can be [binary searched](../binary-search/), and each check hashes all windows of that length into a set.

```python
def longest_dup_substring(s):
    n = len(s)
    prefix, power = [0] * (n + 1), [1] * (n + 1)
    for i, ch in enumerate(s):
        prefix[i + 1] = (prefix[i] * BASE + ord(ch)) % MOD
        power[i + 1] = power[i] * BASE % MOD

    def repeated(length):                            # start of a repeated substring, or -1
        seen = {}
        for i in range(n - length + 1):
            h = (prefix[i + length] - prefix[i] * power[length]) % MOD
            if h in seen and s[seen[h]:seen[h] + length] == s[i:i + length]:
                return i
            seen.setdefault(h, i)
        return -1

    lo, hi, best = 1, n - 1, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        start = repeated(mid)
        if start >= 0:
            best, lo = s[start:start + mid], mid + 1
        else:
            hi = mid - 1
    return best


def brute(s):
    return max((s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1) if s.find(s[i:j], i + 1) != -1),
               key=len, default="")


assert longest_dup_substring("banana") == "ana"
assert longest_dup_substring("abcd") == ""
for _ in range(300):
    t = "".join(random.choice("ab") for _ in range(random.randint(1, 10)))
    assert len(longest_dup_substring(t)) == len(brute(t))
```

\(O(n \log n)\) expected; a suffix array solves it deterministically but is far more code.

## Practice
- [187. Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/)
- [1554. Strings Differ by One Character](https://leetcode.com/problems/strings-differ-by-one-character/)
- [2261. K Divisible Elements Subarrays](https://leetcode.com/problems/k-divisible-elements-subarrays/)
- [1923. Longest Common Subpath](https://leetcode.com/problems/longest-common-subpath/)

Rabin–Karp is from R. M. Karp and M. O. Rabin, "Efficient randomized pattern-matching algorithms" (1987).
