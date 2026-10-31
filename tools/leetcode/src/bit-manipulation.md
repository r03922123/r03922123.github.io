---
title: Bit manipulation: identities, XOR tricks and counting bits
summary: The handful of two's-complement identities that power most bit tricks (lowest set bit, clearing it, popcount, parity, rounding to powers of two, reversing bits, Gosper's hack), then XOR for the Single Number family and a DP for counting bits.
group: Dynamic programming, bits and games
order: 4
tags: bit manipulation · xor · two's complement
---
An integer is a row of switches. Most bit tricks come down to a few identities about how adding, subtracting and negating flip those switches. Learn them once and a whole family of problems turns into one-liners: finding the odd one out, counting set bits, reversing a word, listing subsets in order.

This post goes through those identities, uses them to build popcount, parity, power-of-two rounding and bit reversal, and then solves the classic XOR and counting problems. Almost every snippet is checked against a slow, obvious version on thousands of random inputs.

## Two's complement, and what Python changes
In two's complement, \(-x\) is `~x + 1`: flip every bit, then add one. Since `x + ~x` is all ones, which is \(-1\), this follows directly. Python's integers are unbounded, but its bitwise operators act as if every integer had an **infinite** two's-complement expansion: a negative number has infinitely many leading ones. So `x & -x` and `~x` mean the same thing as in C. What changes is overflow, because there is none. When LeetCode expects a 32-bit answer, you mask with `0xFFFFFFFF` to get the unsigned value, and convert back to signed when bit 31 is set.

```python
import random

MASK32 = 0xFFFFFFFF


def to_unsigned32(x):
    return x & MASK32


def to_signed32(u):
    u &= MASK32
    return u - (1 << 32) if u >> 31 else u


assert to_unsigned32(-1) == 0xFFFFFFFF
assert to_signed32(0xFFFFFFFF) == -1 and to_signed32(0x7FFFFFFF) == 2**31 - 1
for _ in range(2000):
    x = random.randint(-2**31, 2**31 - 1)
    assert to_signed32(to_unsigned32(x)) == x
    assert -x == ~x + 1
```

## Core identities
Subtracting one from \(x\) turns its lowest set bit off and every zero below it on, leaving the higher bits alone. Negation does the opposite: it flips every bit above the lowest set bit and leaves that bit, and the zeros below it, as they were. Two identities follow:

- `x & (x - 1)` **clears** the lowest set bit. It is zero exactly when \(x\) is a power of two (or zero).
- `x & -x` **isolates** the lowest set bit. For \(x = 12 = 1100_2\), \(-x\) ends in `...10100`, and the AND is \(100_2 = 4\).

Setting, clearing and toggling bit \(k\) use the mask `1 << k` with OR, AND-NOT and XOR. All of these are \(O(1)\) word operations.

```python
def lowest_set_bit(x):
    return x & -x


def clear_lowest_set_bit(x):
    return x & (x - 1)


def is_power_of_two(x):
    return x > 0 and x & (x - 1) == 0


def set_bit(x, k):
    return x | (1 << k)


def clear_bit(x, k):
    return x & ~(1 << k)


def toggle_bit(x, k):
    return x ^ (1 << k)


for _ in range(5000):
    x = random.randint(1, 2**64)
    s = bin(x)[2:]
    low = len(s) - 1 - s.rindex("1")                  # index of the lowest 1, the slow way
    assert lowest_set_bit(x) == 1 << low
    assert clear_lowest_set_bit(x) == x - (1 << low)
    assert is_power_of_two(x) == (s.count("1") == 1)
    k = random.randrange(70)
    assert (set_bit(x, k) >> k) & 1 == 1 and (clear_bit(x, k) >> k) & 1 == 0
    assert toggle_bit(toggle_bit(x, k), k) == x
assert not is_power_of_two(0) and not is_power_of_two(-8)
assert clear_lowest_set_bit(0) == 0 and lowest_set_bit(0) == 0
assert lowest_set_bit(-12) == 4 and clear_lowest_set_bit(-12) == -16   # works on negatives too
```

## Popcount and parity
Clearing the lowest set bit until nothing is left counts the ones in \(O(s)\) steps, where \(s\) is the number of set bits, not the word width. This is [191. Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/). In everyday Python, `bin(x).count("1")` (or `int.bit_count` from 3.10) is faster still, because it runs in C.

The **parity** of a word is 1 if it has an odd number of ones. It equals the XOR of all its bits, and XOR is associative, so a 64-bit word can be folded in half: XOR the top 32 bits onto the bottom 32, then 16 onto 16, and so on. After six folds, bit 0 holds the answer, so this takes \(O(\log w)\) operations for a \(w\)-bit word. The other standard speedup is a lookup table: precompute the parity of every 16-bit value once, then XOR four table entries together.

```python
def hamming_weight(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


assert hamming_weight(0b1011) == 3
assert hamming_weight(0) == 0
assert hamming_weight(128) == 1
assert hamming_weight(2147483645) == 30
assert hamming_weight(4294967293) == 31


def parity_fold(x):                                  # x is a 64-bit unsigned word
    for shift in (32, 16, 8, 4, 2, 1):
        x ^= x >> shift
    return x & 1


PARITY16 = [0] * (1 << 16)
for v in range(1, 1 << 16):
    PARITY16[v] = PARITY16[v >> 1] ^ (v & 1)         # parity of v = parity of v without its last bit, XOR that bit


def parity_table(x):
    return (PARITY16[x & 0xFFFF] ^ PARITY16[(x >> 16) & 0xFFFF]
            ^ PARITY16[(x >> 32) & 0xFFFF] ^ PARITY16[x >> 48])


for _ in range(5000):
    x = random.getrandbits(64)
    ones = bin(x).count("1")
    assert hamming_weight(x) == ones
    assert parity_fold(x) == parity_table(x) == ones % 2
```

## Rounding to a power of two
To round down, copy the highest set bit into every position below it. OR-ing `x >> 1` makes the top two bits ones, `x >> 2` makes four, and so on until every bit below the top one is set. The smeared value is \(2^{m+1} - 1\), where \(m\) is the index of the top bit, so adding one and shifting right by one gives \(2^m\).

To round up, subtract one first, then smear and add one. Subtracting first means an exact power of two maps to itself rather than to the next one. Zero needs care here: \(0 - 1 = -1\) is all ones, so the smear does nothing and the answer would come out as 0. Treat inputs up to 1 as rounding up to 1. In Python, `x.bit_length()` gives the same answers directly, and it is used below as the cross-check.

```python
def smear_right(x):                                  # x < 2**64
    for shift in (1, 2, 4, 8, 16, 32):
        x |= x >> shift
    return x


def round_down_pow2(x):                              # largest power of two <= x, for x >= 1
    return (smear_right(x) + 1) >> 1


def round_up_pow2(x):                                # smallest power of two >= x
    if x <= 1:
        return 1
    return smear_right(x - 1) + 1


assert [round_up_pow2(x) for x in range(9)] == [1, 1, 2, 4, 4, 8, 8, 8, 8]
assert [round_down_pow2(x) for x in range(1, 9)] == [1, 2, 2, 4, 4, 4, 4, 8]
for _ in range(5000):
    x = random.randint(1, 2**63)
    assert round_down_pow2(x) == 1 << (x.bit_length() - 1)
    assert round_up_pow2(x) == 1 << (x - 1).bit_length()
```

## Reversing bits
[190. Reverse Bits](https://leetcode.com/problems/reverse-bits/) reverses an unsigned 32-bit word. Shifting bits out of `n` and into the answer one at a time takes 32 steps. A faster version works by divide and conquer. Swap the two 16-bit halves, then swap the bytes inside each half, then the nibbles inside each byte, then the pairs, then the single bits. Each level is one mask-shift-OR, so that is 5 levels for 32 bits, or \(\log_2 w\) in general. The masks pick the left and right member of every pair at that level: `0x55555555` is `0101...`, `0x33333333` is `0011...`, `0x0F0F0F0F` is `00001111...`.

```python
def reverse_bits_loop(n):
    out = 0
    for _ in range(32):
        out = (out << 1) | (n & 1)
        n >>= 1
    return out


def reverse_bits(n):
    n = ((n >> 16) | (n << 16)) & MASK32
    n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8)
    n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4)
    n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2)
    n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1)
    return n


assert reverse_bits(43261596) == 964176192
assert reverse_bits(4294967293) == 3221225471
assert reverse_bits(2147483644) == 1073741822
assert reverse_bits(0) == 0 and reverse_bits(MASK32) == MASK32 and reverse_bits(1) == 1 << 31
for _ in range(5000):
    n = random.getrandbits(32)
    assert reverse_bits(n) == reverse_bits_loop(n) == int(f"{n:032b}"[::-1], 2)
```

The `& MASK32` on the first line is defensive rather than essential. Since nothing overflows in Python, `n << 16` does leave junk in bits 32 to 47, but every term on the next line is masked to 32 bits first, so the junk is thrown away there. Where unmasked growth really bites is a shifted or added term that is never masked afterwards, as in the carry loop of [371. Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/), which loops forever on inputs such as -1 and 1 unless each step is masked to 32 bits. That is the usual way these tricks go wrong in Python. If the function is called many times, a third option is a table of reversed bytes: reverse each of the four bytes by lookup and put them back in the opposite order.

## Generating k-subsets with Gosper's hack
Write a subset of \(\{0, \dots, n-1\}\) as an \(n\)-bit mask. Gosper's hack steps from one mask with \(k\) ones to the next larger mask with exactly \(k\) ones, so starting from `(1 << k) - 1` it lists all \(\binom{n}{k}\) subsets in increasing order, \(O(1)\) word operations each. The step works like this:

1. `c = x & -x` is the lowest set bit.
2. `r = x + c` adds it in. The lowest run of ones carries into the next zero, so that run collapses to a single one, one place higher.
3. The run had \(L\) ones and now has one, so \(L - 1\) ones must come back, as low as possible. `(r ^ x)` is the old run plus the new bit, a block of \(L + 1\) ones starting at the position \(p\) of `c`. Shifting right by 2 and then dividing by \(c = 2^p\) moves the block right by \(p + 2\) places in total, so two of its ones fall off the bottom and the other \(L - 1\) land at bit 0.

Take \(x = 0011_2\). Then \(c = 1\) and \(r = 0100_2\). \(r \oplus x = 0111_2\), shifted right by 2 is \(1\), divided by 1 is \(1\), and the next mask is \(0101_2\). [2397. Maximum Rows Covered by Columns](https://leetcode.com/problems/maximum-rows-covered-by-columns/) is a direct use: try every set of `numSelect` columns.

```python
from itertools import combinations


def next_same_popcount(x):                           # x > 0
    c = x & -x
    r = x + c
    return r | (((r ^ x) >> 2) // c)


def k_subsets(n, k):
    if k == 0:
        yield 0                                      # the step divides by c, so k = 0 is handled apart
        return
    mask = (1 << k) - 1
    while mask < 1 << n:
        yield mask
        mask = next_same_popcount(mask)


for n in range(0, 11):
    for k in range(0, n + 1):
        expected = sorted(sum(1 << i for i in combo) for combo in combinations(range(n), k))
        assert list(k_subsets(n, k)) == expected


def maximum_rows(matrix, num_select):
    cols = len(matrix[0])
    rows = [sum(v << j for j, v in enumerate(row)) for row in matrix]   # each row as a column mask
    return max(sum(r & ~chosen == 0 for r in rows) for chosen in k_subsets(cols, num_select))


assert maximum_rows([[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]], 2) == 3
assert maximum_rows([[1], [0]], 1) == 2


def maximum_rows_brute(matrix, num_select):
    best = 0
    for chosen in combinations(range(len(matrix[0])), num_select):
        best = max(best, sum(all(j in chosen for j, v in enumerate(row) if v) for row in matrix))
    return best


for _ in range(300):
    m, n = random.randint(1, 6), random.randint(1, 8)
    matrix = [[random.randint(0, 1) for _ in range(n)] for _ in range(m)]
    k = random.randint(1, n)
    assert maximum_rows(matrix, k) == maximum_rows_brute(matrix, k)
```

## XOR tricks: the Single Number family
XOR is addition mod 2 in each bit position. It is associative and commutative, \(a \oplus a = 0\), and \(a \oplus 0 = a\). XOR a whole list together and every value that appears twice cancels out.

- [136. Single Number](https://leetcode.com/problems/single-number/): every value appears twice except one. XOR everything. \(O(n)\) time, \(O(1)\) space.
- [260. Single Number III](https://leetcode.com/problems/single-number-iii/): two values appear once. XOR everything to get \(d = a \oplus b \ne 0\). Any set bit of \(d\) is a position where \(a\) and \(b\) differ, and `d & -d` picks the lowest one. Split the numbers on that bit and XOR each side. Python's infinite sign bits mean this works for negative numbers with no masking.
- [137. Single Number II](https://leetcode.com/problems/single-number-ii/): every value appears three times except one. Now each bit position must count mod 3, which takes two bits of state. `ones` holds the positions seen once (mod 3) and `twos` the positions seen twice. Each update is a small per-bit state machine, 0 → 1 → 2 → 0, run on all positions in parallel.

```python
from collections import Counter
from functools import reduce
from operator import xor


def single_number(nums):
    return reduce(xor, nums, 0)


def single_number_iii(nums):
    d = reduce(xor, nums, 0)
    low = d & -d
    a = reduce(xor, (x for x in nums if x & low), 0)
    return [a, a ^ d]


def single_number_ii(nums):
    ones = twos = 0
    for x in nums:
        ones = (ones ^ x) & ~twos
        twos = (twos ^ x) & ~ones
    return ones


def single_number_ii_counting(nums):                 # the same idea written out per bit, 32-bit signed
    u = 0
    for bit in range(32):
        if sum((x >> bit) & 1 for x in nums) % 3:
            u |= 1 << bit
    return to_signed32(u)


assert single_number([2, 2, 1]) == 1 and single_number([4, 1, 2, 1, 2]) == 4 and single_number([1]) == 1
assert sorted(single_number_iii([1, 2, 1, 3, 2, 5])) == [3, 5]
assert sorted(single_number_iii([-1, 0])) == [-1, 0]
assert sorted(single_number_iii([0, 1])) == [0, 1]
for f in (single_number_ii, single_number_ii_counting):
    assert f([2, 2, 3, 2]) == 3 and f([0, 1, 0, 1, 0, 1, 99]) == 99

for _ in range(300):
    vals = random.sample(range(-2**31, 2**31), 6)
    once, rest = vals[:2], vals[2:]
    pairs = once[:1] + rest * 2
    random.shuffle(pairs)
    assert single_number(pairs) == once[0]
    both = once + rest * 2
    random.shuffle(both)
    assert sorted(single_number_iii(both)) == sorted(once)
    triples = once[:1] + rest * 3
    random.shuffle(triples)
    brute = [v for v, c in Counter(triples).items() if c == 1][0]
    assert single_number_ii(triples) == single_number_ii_counting(triples) == brute
```

The plain `single_number_ii` needs no 32-bit mask, because the operations act on every bit position the same way, including Python's infinite run of sign bits. The counting version only looks at 32 positions, so it has to rebuild the sign by hand. That is where `to_signed32` comes in.

## Counting bits for every number up to n
[338. Counting Bits](https://leetcode.com/problems/counting-bits/) asks for the popcount of every \(i \le n\) in \(O(n)\). Each value can reuse one computed earlier. Dropping the last bit gives \(\text{bits}[i] = \text{bits}[i \gg 1] + (i \mathbin{\&} 1)\). Dropping the lowest set bit gives \(\text{bits}[i] = \text{bits}[i \mathbin{\&} (i-1)] + 1\). Both are [dynamic programming](../dynamic-programming/) over a smaller index, and the 16-bit parity table above was built the same way.

```python
def count_bits(n):
    bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits[i] = bits[i & (i - 1)] + 1
    return bits


assert count_bits(2) == [0, 1, 1]
assert count_bits(5) == [0, 1, 1, 2, 1, 2]
table = count_bits(1 << 14)
assert all(table[i] == bin(i).count("1") == table[i >> 1] + (i & 1) for i in range(len(table)))
```

The same "mask as a set" view drives [bitmask DP](../bitmask-dp/), where a mask of visited items indexes the DP table, and the XOR-of-prefixes idea appears in the [prefix state map](../prefix-state-map/) post.

## Practice
- [137. Single Number II](https://leetcode.com/problems/single-number-ii/)
- [201. Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/): the answer is the common binary prefix of the endpoints. Strip the lowest set bit of `right` until it is no larger than `left`.
- [2401. Longest Nice Subarray](https://leetcode.com/problems/longest-nice-subarray/): a [sliding window](../sliding-window/) whose OR is the union of bits in use.
- [2354. Number of Excellent Pairs](https://leetcode.com/problems/number-of-excellent-pairs/): popcount(a OR b) + popcount(a AND b) = popcount(a) + popcount(b).
- [2322. Minimum Score After Removals on a Tree](https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/): subtree XORs plus entry and exit times.
- [371. Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/): XOR adds without carries, and AND shifted left is the carry. In Python, mask to 32 bits.
- [231. Power of Two](https://leetcode.com/problems/power-of-two/)
- [268. Missing Number](https://leetcode.com/problems/missing-number/)
- [461. Hamming Distance](https://leetcode.com/problems/hamming-distance/)
- [2397. Maximum Rows Covered by Columns](https://leetcode.com/problems/maximum-rows-covered-by-columns/)
- [3304. Find the K-th Character in String Game I](https://leetcode.com/problems/find-the-k-th-character-in-string-game-i/) and [3307. Find the K-th Character in String Game II](https://leetcode.com/problems/find-the-k-th-character-in-string-game-ii/)
- [3315. Construct the Minimum Bitwise Array II](https://leetcode.com/problems/construct-the-minimum-bitwise-array-ii/)

The parity and bit-reversal material follows the primitive-types chapter of A. Aziz, T.-H. Lee and A. Prakash, Elements of Programming Interviews; rounding to a power of two is treated in the "Power-of-2 Boundaries" chapter of H. S. Warren Jr., Hacker's Delight. Clearing the lowest set bit to count ones is due to P. Wegner, "A technique for counting ones in a binary computer", Communications of the ACM (1960). The next-subset step is item 175 of M. Beeler, R. W. Gosper and R. Schroeppel, HAKMEM, MIT AI Memo 239 (1972). Hacker's Delight collects many more.
