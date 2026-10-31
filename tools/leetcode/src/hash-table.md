---
title: Hash tables from scratch: chaining, hash functions, resizing
summary: Build a hash map with separate chaining. The division, multiplication and universal hash functions, an operator-precedence bug that quietly changes which bits you keep, and why doubling keeps operations O(1) on average.
group: Stacks, queues, heaps and hashing
order: 4
tags: hashing · design · data structures
---
[705. Design HashSet](https://leetcode.com/problems/design-hashset/) and [706. Design HashMap](https://leetcode.com/problems/design-hashmap/) ask you to build what `dict` gives you for free. The point is to see the three decisions inside every hash table: how keys map to slots, what happens when two keys share a slot, and when to grow.

## Hash functions for integer keys
A hash function maps a key to one of \(m\) slots. A good one spreads keys evenly, including keys with patterns, such as all multiples of 1024.

- **Division:** \(h(k) = k \bmod m\). Simple, but with \(m = 2^r\) it keeps only the low \(r\) bits, so keys that differ only in high bits all collide. Choose \(m\) prime, away from powers of 2.
- **Multiplication:** with \(m = 2^r\) and \(w\)-bit words, \(h(k) = \big((a \cdot k) \bmod 2^w\big) \gg (w - r)\) for an odd constant \(a\). Multiplying mixes every bit of \(k\) into the **high** bits of the product, and the shift keeps exactly those \(r\) bits.
- **Universal:** \(h(k) = \big((a k + b) \bmod p\big) \bmod m\), with prime \(p\) larger than any key and \(a, b\) chosen at random. For any two distinct keys, the collision probability is at most about \(1/m\) **over the random choice**, so no fixed input can be bad for every run.

### A precedence trap
In Python, `>>` binds tighter than `&`. So this line, a natural transcription of the multiplication method, does not do what it says:

```python
w, r, a = 32, 10, 2654435769                    # a: an odd constant near 2^32 / golden ratio


def mult_hash_wrong(k):
    return (a * k) & (1 << w) - 1 >> (w - r)    # parsed as (a*k) & (((1<<w) - 1) >> (w-r)): low bits


def mult_hash(k):
    return ((a * k) & ((1 << w) - 1)) >> (w - r)  # high r bits of the w-bit product


keys = range(0, 1024 * 64, 1024)                # multiples of 1024: patterned keys
assert len({mult_hash(k) for k in keys}) > 50   # spread over many slots
assert len({mult_hash_wrong(k) for k in keys}) == 1   # low bits of a*k are all zero: one slot
```

Both versions return numbers in range, so nothing crashes; the wrong one simply keeps the low bits, and patterned keys pile into one slot. Parenthesize bit arithmetic explicitly.

## Separate chaining and resizing
Each slot holds a small list of `(key, value)` pairs. With \(n\) items in \(m\) slots, the **load factor** \(\alpha = n/m\) is the average chain length, and an operation costs \(O(1 + \alpha)\). Keep \(\alpha\) bounded by doubling \(m\) and rehashing whenever it passes a threshold. A single resize costs \(O(n)\), but doubling means it happens after \(n\) cheap inserts, so the cost averages out to \(O(1)\) per insert.

```python
import random


class MyHashMap:
    P = (1 << 61) - 1                           # prime larger than any key

    def __init__(self):
        self.m, self.n = 8, 0
        self.a, self.b = random.randrange(1, self.P), random.randrange(self.P)
        self.slots = [[] for _ in range(self.m)]

    def _slot(self, key):
        return self.slots[((self.a * key + self.b) % self.P) % self.m]   # universal hashing

    def put(self, key, value):
        slot = self._slot(key)
        for i, (k, _) in enumerate(slot):
            if k == key:
                slot[i] = (key, value)
                return
        slot.append((key, value))
        self.n += 1
        if self.n > 0.75 * self.m:              # load factor too high: double and rehash
            items = [kv for s in self.slots for kv in s]
            self.m *= 2
            self.slots = [[] for _ in range(self.m)]
            for k, v in items:
                self._slot(k).append((k, v))

    def get(self, key):
        return next((v for k, v in self._slot(key) if k == key), -1)

    def remove(self, key):
        slot = self._slot(key)
        for i, (k, _) in enumerate(slot):
            if k == key:
                slot.pop(i)
                self.n -= 1
                return


hm, ref = MyHashMap(), {}
for _ in range(20_000):                         # random operations, checked against dict
    key = random.randrange(2000)
    op = random.random()
    if op < 0.5:
        hm.put(key, op)
        ref[key] = op
    elif op < 0.75:
        hm.remove(key)
        ref.pop(key, None)
    assert hm.get(key) == ref.get(key, -1)
assert hm.n == len(ref) and hm.n <= 0.75 * hm.m
assert max(len(s) for s in hm.slots) <= 10     # chains stay short
```

The final asserts are the point: after twenty thousand random operations the map agrees with `dict` on every lookup, the load factor is bounded, and no chain grew long.

## Chaining versus open addressing
Python's own `dict` uses **open addressing**: no lists, collisions probe other slots in the same array. It is more cache-friendly, but deletion needs tombstones and performance falls off sharply as the table fills, so it resizes earlier. Chaining degrades gracefully and is the easier one to implement correctly.

## Practice
- [705. Design HashSet](https://leetcode.com/problems/design-hashset/)
- [706. Design HashMap](https://leetcode.com/problems/design-hashmap/)
- [362. Design Hit Counter](https://leetcode.com/problems/design-hit-counter/)
- [2296. Design a Text Editor](https://leetcode.com/problems/design-a-text-editor/)
- [631. Design Excel Sum Formula](https://leetcode.com/problems/design-excel-sum-formula/)

The hash functions follow the presentation in MIT 6.006 (Fall 2011), lecture 8, "Hashing with Chaining".
