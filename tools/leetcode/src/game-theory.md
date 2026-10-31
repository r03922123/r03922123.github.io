---
title: Game theory: minimax on the score difference
summary: Two players, perfect information, both playing their best. One recurrence on "my score minus yours" solves the stone-game family (877, 486, 1406, 1140), and a win/lose labelling of positions explains Nim, with a short look at Sprague–Grundy numbers.
group: Dynamic programming, bits and games
order: 5
tags: game theory · minimax · dynamic programming · nim
---
A two-player game where both sides see everything and both play perfectly looks like it needs two recursions, one for each player, passing turns back and forth. It does not. The trick is to stop tracking whose turn it is and instead ask, from any position: **how far ahead can the player about to move finish?** The opponent faces the same question from the next position, and since the game is symmetric, it is the same function.

Write \(f(s)\) for the best final value of (mover's score minus the other player's score) from state \(s\). If a move gains \(g\) points and leads to \(s'\), the opponent then gets ahead by \(f(s')\), so the mover ends up ahead by \(g - f(s')\). Taking the best move:

\[
f(s) = \max_{\text{moves } s \to s'} \bigl( g(s \to s') - f(s') \bigr), \qquad f(\text{end}) = 0.
\]

This is minimax, computed by backward induction from the end of the game, folded into a single sign flip. Unrolled over a game, the alternating subtractions telescope into exactly the quantity we want: \(a_0 - (b_0 - (a_1 - (b_1 - \dots))) = (a_0 + a_1 + \dots) - (b_0 + b_1 + \dots)\). Once \(f\) is known, the first player wins if \(f > 0\), ties at \(0\), and their own score is \((\text{total} + f) / 2\) when every point goes to someone. The rest is choosing the state, which is a [dynamic programming](../dynamic-programming/) question like any other.

## A reference: exhaustive minimax
Every DP below is checked against a referee that knows nothing clever: it plays out every line of the game, keeps both scores explicitly, and lets each player pick the move that is best for them. It is exponential and only usable on tiny inputs, which is exactly what a cross-check needs.

```python
import random
from functools import lru_cache
from itertools import accumulate


def play_ends(piles, alice=0, bob=0, alice_turn=True):
    """Take from either end of the row. Returns (alice, bob) under optimal play."""
    if not piles:
        return alice, bob
    results = []
    for taken, rest in ((piles[0], piles[1:]), (piles[-1], piles[:-1])):
        if alice_turn:
            results.append(play_ends(rest, alice + taken, bob, False))
        else:
            results.append(play_ends(rest, alice, bob + taken, True))
    if alice_turn:
        return max(results, key=lambda r: r[0] - r[1])
    return max(results, key=lambda r: r[1] - r[0])


assert play_ends([1, 5, 2]) == (3, 5)
```

Each player maximises their own lead, which is the same as minimising the other's; that is the "min" in minimax.

## Predict the Winner: an interval of piles
[486. Predict the Winner](https://leetcode.com/problems/predict-the-winner/): players alternately take a number from either end of an array; does player 1 finish with at least as much as player 2? The remaining numbers are always a contiguous slice `nums[i..j]`, so the state is the pair \((i, j)\):

\[
f(i, j) = \max\bigl(\text{nums}_i - f(i+1, j),\; \text{nums}_j - f(i, j-1)\bigr), \qquad f(i, i) = \text{nums}_i .
\]

Fill it by increasing slice length, \(O(n^2)\) time. A single row suffices because \(f(i, j)\) only reads slices one shorter, so space is \(O(n)\).

```python
def predict_the_winner(nums):
    n = len(nums)
    f = nums[:]                                  # length-1 slices: f[i] = f(i, i)
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            f[i] = max(nums[i] - f[i + 1], nums[j] - f[i])   # f[i+1] is (i+1, j), f[i] is (i, j-1)
    return f[0] >= 0


assert predict_the_winner([1, 5, 2]) is False
assert predict_the_winner([1, 5, 233, 7]) is True
assert predict_the_winner([0]) is True           # a tie counts as a win for player 1
assert predict_the_winner([2, 2]) is True

for _ in range(300):
    nums = [random.randint(0, 20) for _ in range(random.randint(1, 8))]
    a, b = play_ends(nums)
    assert predict_the_winner(nums) == (a >= b)
```

The in-place update works because, before overwriting, `f[i]` still holds slice \((i, j-1)\) from the previous length and `f[i + 1]` already holds \((i+1, j)\).

## Stone Game: the first player cannot lose
[877. Stone Game](https://leetcode.com/problems/stone-game/) is the same game with two extra promises: the number of piles is **even** and the total is **odd**. The DP above answers it, but the promises make it unnecessary.

Colour the piles by index parity: positions \(0, 2, 4, \dots\) are "even", \(1, 3, 5, \dots\) "odd". With an even number of piles, the two ends always have different colours at the start of Alice's turn. If Alice takes an even pile, the ends Bob sees are both odd; whatever he takes exposes an even pile again. So Alice can force herself to collect **all** even-indexed piles, or symmetrically all odd-indexed ones. The total is odd, so one colour sum is strictly larger, and she picks that colour before her first move. Optimal play can only do at least as well, so the answer is always `True`.

```python
def stone_game(piles):
    return True                                  # the colouring strategy above


def stone_game_dp(piles):
    n = len(piles)
    f = piles[:]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            f[i] = max(piles[i] - f[i + 1], piles[i + length - 1] - f[i])
    return f[0] > 0


assert stone_game([5, 3, 4, 5]) and stone_game_dp([5, 3, 4, 5])
assert stone_game([3, 7, 2, 3]) and stone_game_dp([3, 7, 2, 3])

for _ in range(300):
    piles = [random.randint(1, 15) for _ in range(2 * random.randint(1, 4))]
    if sum(piles) % 2 == 0:
        piles[0] += 1                            # keep the promise: total is odd
    a, b = play_ends(piles)
    assert a > b and stone_game_dp(piles)
```

Note the strategy argument does not say colour play is optimal, only that it already wins. The DP's margin is often larger. Pairing arguments like this one are worth looking for whenever a problem guarantees structure that seems too convenient.

## Stone Game III: take one to three from the front
[1406. Stone Game III](https://leetcode.com/problems/stone-game-iii/): stones in a row carry values (possibly negative); each turn takes the first 1, 2 or 3 remaining stones. Report `"Alice"`, `"Bob"` or `"Tie"`. Now the state is just the index \(i\) of the first remaining stone:

\[
f(i) = \max_{k \in \{1,2,3\},\, i + k \le n} \Bigl( \textstyle\sum_{t=i}^{i+k-1} v_t - f(i+k) \Bigr), \qquad f(n) = 0.
\]

Negative values matter: the mover **must** take at least one stone, so \(f\) can be negative and must not be initialised to 0 when taking the max. Computed right to left, this is \(O(n)\) time, and only three later values are ever read.

```python
def stone_game_iii(values):
    n = len(values)
    f = [0] * (n + 4)
    for i in range(n - 1, -1, -1):
        best, taken = float("-inf"), 0
        for k in range(1, 4):
            if i + k > n:
                break
            taken += values[i + k - 1]
            best = max(best, taken - f[i + k])
        f[i] = best
    return "Alice" if f[0] > 0 else "Bob" if f[0] < 0 else "Tie"


def play_front(values, alice=0, bob=0, alice_turn=True):
    """Referee for 'take the first 1, 2 or 3 stones'."""
    if not values:
        return alice, bob
    results = []
    for k in range(1, min(3, len(values)) + 1):
        g = sum(values[:k])
        if alice_turn:
            results.append(play_front(values[k:], alice + g, bob, False))
        else:
            results.append(play_front(values[k:], alice, bob + g, True))
    sign = 1 if alice_turn else -1
    return max(results, key=lambda r: sign * (r[0] - r[1]))


assert stone_game_iii([1, 2, 3, 7]) == "Bob"
assert stone_game_iii([1, 2, 3, -9]) == "Alice"
assert stone_game_iii([1, 2, 3, 6]) == "Tie"
assert stone_game_iii([-1]) == "Bob"               # forced to take a negative stone
assert stone_game_iii([-1, -2, -3]) == "Tie"       # Alice takes -1, -2; Bob is left with -3

for _ in range(200):
    v = [random.randint(-10, 10) for _ in range(random.randint(1, 9))]
    a, b = play_front(v)
    assert stone_game_iii(v) == ("Alice" if a > b else "Bob" if a < b else "Tie")
```

## Stone Game II: a state with a memory
[1140. Stone Game II](https://leetcode.com/problems/stone-game-ii/): each turn takes the first \(X\) remaining piles with \(1 \le X \le 2M\), and then \(M\) becomes \(\max(M, X)\), starting from \(M = 1\). Return Alice's total. The move set depends on history, so the state must carry it: \((i, M)\).

This one asks for a score, not a margin, and there is an equally short version of the recurrence that works directly in stones. Let \(S_i\) be the sum of piles from \(i\) to the end. Whatever the mover does not get from that suffix, the opponent gets, so

\[
\text{best}(i, M) = \max_{1 \le X \le 2M} \bigl( S_i - \text{best}(i + X, \max(M, X)) \bigr),
\]

with \(\text{best}(i, M) = S_i\) as soon as \(i + 2M \ge n\): taking everything is allowed and, with positive piles, optimal. This is the difference recurrence in disguise, since \(\text{best} = (S_i + f)/2\). There are \(O(n^2)\) states (\(M\) never exceeds \(n\)) and \(O(n)\) moves each, so \(O(n^3)\) time, fine for \(n \le 100\).

```python
def stone_game_ii(piles):
    n = len(piles)
    suffix = list(accumulate(reversed(piles)))[::-1] + [0]

    @lru_cache(maxsize=None)
    def best(i, m):
        if i + 2 * m >= n:
            return suffix[i]
        return max(suffix[i] - best(i + x, max(m, x)) for x in range(1, 2 * m + 1))

    return best(0, 1)


def play_ii(piles, m=1, alice=0, bob=0, alice_turn=True):
    if not piles:
        return alice, bob
    results = []
    for x in range(1, min(2 * m, len(piles)) + 1):
        g = sum(piles[:x])
        nxt = max(m, x)
        if alice_turn:
            results.append(play_ii(piles[x:], nxt, alice + g, bob, False))
        else:
            results.append(play_ii(piles[x:], nxt, alice, bob + g, True))
    sign = 1 if alice_turn else -1
    return max(results, key=lambda r: sign * (r[0] - r[1]))


assert stone_game_ii([2, 7, 9, 4, 4]) == 10
assert stone_game_ii([1, 2, 3, 4, 5, 100]) == 104
assert stone_game_ii([7]) == 7 and stone_game_ii([3, 9]) == 12

for _ in range(200):
    piles = [random.randint(1, 20) for _ in range(random.randint(1, 9))]
    assert stone_game_ii(piles) == play_ii(piles)[0]
```

## Win/lose states and Nim
Many games have no score, only a winner: the player who cannot move loses. Then \(f\) collapses to a boolean. A position is **losing** if every move leads to a winning position for the opponent (including the case of no moves at all), and **winning** if some move leads to a losing one.

[292. Nim Game](https://leetcode.com/problems/nim-game/): one heap of \(n\) stones, remove 1 to 3, whoever takes the last stone wins. Labelling small \(n\) by the rule above gives L, W, W, W, L, W, W, W, ... The pattern holds because from a multiple of 4 every move lands on a non-multiple, and from a non-multiple you can always step down to one. So the answer is `n % 4 != 0`, in \(O(1)\).

```python
def can_win_nim(n):
    return n % 4 != 0


win = [False] * 201                              # win[0]: no move left, mover lost
for n in range(1, 201):
    win[n] = any(not win[n - k] for k in (1, 2, 3) if k <= n)
assert all(win[n] == can_win_nim(n) for n in range(1, 201))
assert can_win_nim(4) is False and can_win_nim(1) is True and can_win_nim(2) is True
```

With several heaps, where a move takes any positive number from one heap, Bouton's theorem says the mover loses exactly when the XOR of the heap sizes is 0. The Sprague–Grundy theorem extends this to any sum of independent impartial games (both players have the same moves, and whoever cannot move loses): give each position the **Grundy number** \(g = \operatorname{mex}\) of the Grundy numbers of its options (the smallest non-negative integer not among them). A position is losing iff \(g = 0\), and a sum of games has Grundy number equal to the XOR of its parts. For the 1-to-3 subtraction game, \(g(n) = n \bmod 4\), which reproduces 292. The check below confirms the XOR rule for sums of such heaps against a direct win/lose search on the combined position.

```python
def grundy_numbers(limit, moves):
    g = [0] * (limit + 1)
    for n in range(1, limit + 1):
        options = {g[n - k] for k in moves if k <= n}
        g[n] = next(x for x in range(len(options) + 1) if x not in options)   # mex
    return g


g = grundy_numbers(30, (1, 2, 3))
assert g == [n % 4 for n in range(31)]


@lru_cache(maxsize=None)
def mover_wins(heaps):
    for h in range(len(heaps)):
        for k in (1, 2, 3):
            if k <= heaps[h]:
                nxt = tuple(sorted(heaps[:h] + (heaps[h] - k,) + heaps[h + 1:]))
                if not mover_wins(nxt):
                    return True
    return False


for _ in range(200):
    heaps = tuple(sorted(random.randint(0, 9) for _ in range(random.randint(1, 3))))
    x = 0
    for h in heaps:
        x ^= g[h]
    assert mover_wins(heaps) == (x != 0)
```

When a problem is a single game with a small state space, the win/lose labelling (often over a [bitmask](../bitmask-dp/) of used items, as in 464) is all you need; Grundy numbers earn their keep when the position splits into independent pieces.

## Practice
- [1140. Stone Game II](https://leetcode.com/problems/stone-game-ii/)
- [877. Stone Game](https://leetcode.com/problems/stone-game/)
- [486. Predict the Winner](https://leetcode.com/problems/predict-the-winner/)
- [1406. Stone Game III](https://leetcode.com/problems/stone-game-iii/)
- [292. Nim Game](https://leetcode.com/problems/nim-game/)
- [1690. Stone Game VII](https://leetcode.com/problems/stone-game-vii/) (interval DP on the margin)
- [1510. Stone Game IV](https://leetcode.com/problems/stone-game-iv/) (win/lose states)
- [1025. Divisor Game](https://leetcode.com/problems/divisor-game/) (a parity argument again)
- [464. Can I Win](https://leetcode.com/problems/can-i-win/) (win/lose over a bitmask)
- [1872. Stone Game VIII](https://leetcode.com/problems/stone-game-viii/) (prefix sums plus a running max)
- [1563. Stone Game V](https://leetcode.com/problems/stone-game-v/) (interval DP; despite the name, only one player chooses)

Backward induction on finite games of perfect information goes back to E. Zermelo, "Über eine Anwendung der Mengenlehre auf die Theorie des Schachspiels" (1913); the minimax theorem to J. von Neumann, "Zur Theorie der Gesellschaftsspiele", Mathematische Annalen 100 (1928). Nim's XOR rule is from C. L. Bouton, "Nim, a game with a complete mathematical theory", Annals of Mathematics 3 (1901–02); Grundy numbers are due independently to R. Sprague (1935) and P. M. Grundy (1939).
