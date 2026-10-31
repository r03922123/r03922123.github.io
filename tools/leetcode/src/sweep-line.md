---
title: Sweep line: turn intervals into events
summary: Replace each interval with a +change at its start and a −change at its end, then sweep in order. Difference arrays, sorted events, and answering point queries with binary search.
group: Two pointers, windows and prefix sums
order: 4
tags: sweep line · difference array · intervals
---
Interval problems ask what is true **at each moment**: how many meetings overlap, how many passengers are in the car, how many flowers are in bloom. Scanning every interval for every moment is slow. Instead, record only where things change: each interval `[start, end)` with value `delta` becomes two events, `+delta` at `start` and `-delta` at `end`. A running sum over the events in time order is the value at each moment.

## Three ways to store the events
1. **A difference array**, when times are small integers: `diff[start] += delta`, `diff[end] -= delta`, then a prefix sum. \(O(n + T)\).
2. **A sorted list of events**, when times are large or real-valued: \(O(n \log n)\) for the sort.
3. **A hash map then sorted keys**, when many events share a time and you want to merge them first.

At equal times, ends must come before starts for half-open intervals, so a meeting ending at 10 frees its room for one starting at 10. Storing `-delta` and sorting by `(time, delta)` does this automatically.

## Difference array
[1094. Car Pooling](https://leetcode.com/problems/car-pooling/): passengers get on at `start` and off at `end`; can a car of a given capacity take every trip?

```python
from itertools import accumulate


def car_pooling(trips, capacity):
    diff = [0] * 1001                        # stops are at most 1000
    for passengers, start, end in trips:
        diff[start] += passengers
        diff[end] -= passengers              # they are off before anyone boards at `end`
    return max(accumulate(diff)) <= capacity


assert car_pooling([[2, 1, 5], [3, 3, 7]], 4) is False
assert car_pooling([[2, 1, 5], [3, 3, 7]], 5) is True
```

## Sorted events
The minimum number of rooms for a set of meetings is the maximum overlap at any moment ([253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)). Times can be large, so sort events instead of allocating an array:

```python
def min_meeting_rooms(intervals):
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    rooms = best = 0
    for _, delta in events:                  # at equal times, -1 sorts before +1
        rooms += delta
        best = max(best, rooms)
    return best


def brute(intervals):
    times = {t for iv in intervals for t in iv}
    return max((sum(s <= t < e for s, e in intervals) for t in times), default=0)


assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
assert min_meeting_rooms([[7, 10], [2, 4]]) == 1
assert min_meeting_rooms([[1, 5], [5, 9]]) == 1   # back-to-back meetings share a room
import random
for _ in range(300):
    ivs = []
    for _ in range(random.randint(1, 8)):
        s = random.randint(0, 20)
        ivs.append([s, s + random.randint(1, 6)])
    assert min_meeting_rooms(ivs) == brute(ivs)
```

The same count answers [2406. Divide Intervals Into Minimum Number of Groups](https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups/), with closed intervals: there, `end + 1` is the release time.

## Point queries with binary search
Sometimes the question is the value at given **query** times. [2251. Number of Flowers in Full Bloom](https://leetcode.com/problems/number-of-flowers-in-full-bloom/): flowers bloom on the closed interval `[start, end]`. The number in bloom at time `t` is (flowers started by `t`) minus (flowers ended before `t`), and both counts are binary searches over sorted arrays:

```python
from bisect import bisect_left, bisect_right


def full_bloom_flowers(flowers, people):
    starts = sorted(s for s, _ in flowers)
    ends = sorted(e for _, e in flowers)
    return [bisect_right(starts, t) - bisect_left(ends, t) for t in people]


assert full_bloom_flowers([[1, 6], [3, 7], [9, 12], [4, 13]], [2, 3, 7, 11]) == [1, 2, 2, 2]
assert full_bloom_flowers([[1, 10], [3, 3]], [3, 3, 2]) == [2, 2, 1]
```

`bisect_right` counts starts at or before `t`; `bisect_left` counts ends strictly before `t`, so a flower ending at `t` still counts.

## When the value is not a sum
If you need the **maximum** active height rather than a count, as in [218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/), a running sum is not enough: sweep the events and keep the active heights in a heap with lazy deletion (see [heaps](../heap/)).

## Practice
- [1109. Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/)
- [1854. Maximum Population Year](https://leetcode.com/problems/maximum-population-year/)
- [1943. Describe the Painting](https://leetcode.com/problems/describe-the-painting/)
- [2381. Shifting Letters II](https://leetcode.com/problems/shifting-letters-ii/)
- [1589. Maximum Sum Obtained of Any Permutation](https://leetcode.com/problems/maximum-sum-obtained-of-any-permutation/)
- [732. My Calendar III](https://leetcode.com/problems/my-calendar-iii/)
- [1851. Minimum Interval to Include Each Query](https://leetcode.com/problems/minimum-interval-to-include-each-query/)
- [218. The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)
