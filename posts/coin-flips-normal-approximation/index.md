# Coin flips and the normal approximation: a binomial tail by hand

A fair coin is flipped 900 times. What is the probability of getting **480 or more heads**? It looks like a question for a computer, yet the quick answer of 2.5% is close to the exact 2.46%.

The exact answer is a sum of 421 binomial terms, each with a coefficient like \(\binom{900}{480}\), a number with more than 250 digits. Nobody sums that by hand. But 900 flips is a lot of flips, and the count of heads from many independent flips is very close to normally distributed. Once the problem is a normal one, it takes a square root, one subtraction and a fact you already know about bell curves.

I solve the question that way, then check the answer against the exact binomial and ask the more useful questions: how far off is the quick answer, why, and when does the shortcut stop working? The quick answer turns out to be close partly by luck, since two errors cancel.

## Why a bell curve shows up
The number of heads is \(X \sim \text{Binomial}(n, p)\), the sum of \(n\) independent coin flips that are each 1 (heads) with probability \(p\) and 0 otherwise. Its mean and variance follow from adding up the flips (see [Bernoulli and binomial](../bernoulli-binomial/) for the derivation):

\[ \mathbb{E}[X] = np, \qquad \text{Var}(X) = np(1 - p). \]

The central limit theorem says a sum of many independent, identically distributed terms with finite variance is approximately normal once you centre it and scale it. For the binomial this special case is older than the general theorem: Abraham de Moivre worked it out for fair coins in 1733, and Laplace later extended it to any \(p\), which is why it is often called the de Moivre-Laplace theorem. In practice it means

\[ X \approx \mathcal{N}\big(np,\; np(1 - p)\big) \quad \text{when } np(1 - p) \text{ is not small.} \]

A common rule of thumb asks for \(np\) and \(n(1-p)\) both at least 5 or 10. Treat that as a minimum, not a guarantee; the last sections show where it is not enough.

## Solving it in three lines
**Step 1: centre and spread.** With \(n = 900\) and \(p = 1/2\):

\[ \mu = 900 \times \tfrac12 = 450, \qquad \sigma^2 = 900 \times \tfrac12 \times \tfrac12 = 225, \qquad \sigma = 15. \]

For a fair coin \(\sigma = \sqrt{n}/2\), so when \(n\) is a perfect square the standard deviation is a whole number. That is what makes this kind of question solvable in your head, and it is worth noticing which flip counts have this property: 100, 400, 900, 1600, 2500.

**Step 2: standardize.** 480 heads is 30 above the mean, which is

\[ z = \frac{480 - 450}{15} = 2 \]

standard deviations.

**Step 3: read the tail.** The 68-95-99.7 rule says about 95% of a normal distribution lies within two standard deviations of the mean. The remaining 5% is split evenly between the two tails by symmetry, and you only want the upper one: roughly **2.5%**.

That is a fine answer to give quickly. It is also not quite the number the normal distribution gives, and the normal distribution is not quite the binomial. Both gaps are worth understanding.

```python
import math
import numpy as np
from scipy import stats

n, p, k = 900, 0.5, 480
mu, sigma = n * p, math.sqrt(n * p * (1 - p))
assert mu == 450 and sigma == 15 == math.sqrt(n) / 2
z = (k - mu) / sigma
assert z == 2

# the exact answer: P(X >= 480) = sum of the PMF from 480 to 900, in exact integer arithmetic
exact = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
assert math.isclose(exact, stats.binom.sf(k - 1, n, p), rel_tol=1e-9)
assert round(exact, 4) == 0.0246
assert len(str(math.comb(n, k))) > 250                 # the coefficients are enormous

empirical_rule = (1 - 0.95) / 2
assert math.isclose(empirical_rule, 0.025)
```

## Two errors that cancel
The quick answer of 2.5% is within 0.05 percentage points of the exact 2.46%. That looks like the normal approximation is excellent, but it is partly a coincidence, because two separate errors point in opposite directions.

**The 95% is rounded.** The share of a normal distribution within two standard deviations is 95.45%, not 95%. The "95" in the 68-95-99.7 rule really belongs to \(z = 1.96\). Using the true value, the upper tail beyond \(z = 2\) is

\[ P(Z \ge 2) = \frac{1 - 0.9545}{2} \approx 2.28\%, \]

which is further from the exact answer than the rough 2.5% was.

**A discrete count is not a continuous curve.** The binomial puts its probability on the integers 0, 1, ..., 900. If you draw it as a histogram, the bar for 480 covers the interval from 479.5 to 480.5. "At least 480 heads" therefore corresponds to the normal area from **479.5** upward, not from 480. Cutting at 480 throws away half of the 480 bar, which is about 0.2 percentage points here. This half-unit shift is the **continuity correction**:

\[ P(X \ge 480) \approx P\!\left(Z \ge \frac{479.5 - 450}{15}\right) = P(Z \ge 1.967) \approx 2.46\%. \]

With the correction the normal answer matches the exact one to about 0.003 percentage points. So the order of quality is: corrected normal (excellent), the 68-95-99.7 shortcut (good by luck), uncorrected normal (a little low). The table collects them.

| method | value of \(P(X \ge 480)\) |
|---|---|
| exact binomial sum | 2.458% |
| normal with continuity correction, cut at 479.5 | 2.461% |
| 68-95-99.7 rule, \((1 - 0.95)/2\) | 2.5% |
| normal without correction, cut at 480 | 2.275% |

```python
uncorrected = stats.norm.sf((k - mu) / sigma)
corrected = stats.norm.sf((k - 0.5 - mu) / sigma)
assert round(uncorrected, 5) == 0.02275
assert math.isclose(stats.norm.cdf(2) - stats.norm.cdf(-2), 0.9545, abs_tol=1e-4)
assert abs(corrected - exact) < 4e-5                    # about 0.003 percentage points
assert abs(empirical_rule - exact) < abs(uncorrected - exact)   # the rough answer wins, by luck
assert uncorrected < exact < corrected < empirical_rule

# the half-unit really is half of the bar at 480
half_bar = stats.binom.pmf(k, n, p) / 2
assert math.isclose(corrected - uncorrected, half_bar, rel_tol=0.05)
```

The correction matters most when \(\sigma\) is small, since half a unit is then a bigger slice of a standard deviation. With 100 flips and "at least 60 heads", \(z\) is again exactly 2, but now the exact tail is 2.84%, the uncorrected normal still says 2.28%, and only the corrected version (2.87%) gets close. With 2,500 flips and at least 1,300 heads, also \(z = 2\), the uncorrected error shrinks to about 0.1 percentage points.

```python
for flips, heads in [(100, 60), (900, 480), (2500, 1300)]:
    m, s = flips / 2, math.sqrt(flips) / 2
    assert (heads - m) / s == 2                         # every case sits exactly 2 sd out
    ex = stats.binom.sf(heads - 1, flips, 0.5)
    err_plain = abs(stats.norm.sf((heads - m) / s) - ex)
    err_cc = abs(stats.norm.sf((heads - 0.5 - m) / s) - ex)
    assert err_cc < err_plain / 10                      # the correction wins by an order of magnitude
    print(f"{flips:5d} flips: exact {ex:.4%}  plain error {err_plain:.4%}  corrected error {err_cc:.4%}")
```

## Simulating it
A third check needs no formulas at all: flip the coin. Half a million simulated runs of 900 flips should land on at least 480 heads about 2.46% of the time, and the spread of the head counts should come out at 15.

```python
rng = np.random.default_rng(7)
heads = rng.binomial(n, p, size=500_000)
sim = (heads >= k).mean()
se = math.sqrt(exact * (1 - exact) / heads.size)
assert abs(sim - exact) < 4 * se
assert abs(heads.std() - 15) < 0.1

# and the histogram is close to the bell curve at every count within 3 sd, not only at 480
counts = np.arange(400, 501)
assert np.abs(stats.binom.pmf(counts, n, p) - stats.norm.pdf(counts, mu, sigma)).max() < 2e-4
```

## Working the other way: how many heads is surprising?
Often the useful question runs backwards: given \(n\) flips of a coin you believe is fair, where should alarm start? By the normal approximation about 2.3% of runs land two or more standard deviations above the mean, and about 0.13% land three or more above. (Because the count is discrete, the exact share at a whole-number threshold is a little higher: 2.46% rather than 2.28% at 900 flips, as computed above.) So with a fair coin

\[ \text{two sd above} = \frac{n}{2} + \sqrt{n}, \qquad \text{three sd above} = \frac{n}{2} + 1.5\sqrt{n}. \]

For 900 flips those are 480 and 495 heads. Notice what scales: the margin grows with \(\sqrt{n}\) while the number of flips grows with \(n\), so as a **proportion** the threshold shrinks toward one half. Two standard deviations is 53.3% heads at 900 flips but only 51% at 10,000 flips.

The same arithmetic answers a common evaluation question. A binary classifier is scored on 2,500 balanced test examples and gets 52% right. Is that better than guessing? A guesser's correct count is \(\text{Binomial}(2500, 1/2)\) with mean 1,250 and \(\sigma = 25\); 1,300 correct is \(z = 2\), and a guesser does that well about 2.4% of the time. Suggestive, not conclusive, especially if several models were tried and this is the best one. [Is this coin fair?](../is-this-coin-fair/) and [p-values](../p-values/) take that line of reasoning further.

```python
def two_sd_threshold(flips):
    return flips / 2 + math.sqrt(flips)


assert two_sd_threshold(900) == 480
assert 900 / 2 + 1.5 * math.sqrt(900) == 495
assert round(two_sd_threshold(900) / 900, 3) == 0.533
assert two_sd_threshold(10_000) / 10_000 == 0.51

assert round(stats.binom.sf(1299, 2500, 0.5), 3) == 0.024
assert round(stats.norm.sf(3), 4) == 0.0013
assert math.isclose(stats.binom.sf(494, 900, 0.5), stats.norm.sf(2.967), rel_tol=0.01)
```

## Where the approximation breaks
**Far tails.** The approximation is good in absolute terms everywhere, but tails are tiny, so the right measure there is the **relative** error. With 900 flips the corrected normal is within 1% of the exact tail at three standard deviations and within 6% at five, which is fine for a sanity check. Without the correction it is about 10% low throughout. With only 100 flips the corrected normal is about 23% too high at four standard deviations and 70% too high at five. Anyone computing the probability of a rare event, a one-in-a-million false alarm say, should use the exact binomial tail rather than a bell curve.

**Skewed coins.** Everything above used \(p = 1/2\), where the binomial is perfectly symmetric. When \(p\) is small the binomial is lopsided, with a long right tail, while the normal stays symmetric. Take 500 trials with a 2% success rate. The mean is \(np = 10\), so even the stricter form of the rule of thumb is met, and \(\sigma \approx 3.13\). Yet for 20 or more successes, about 3.2 standard deviations out, the exact tail is 0.31% while the corrected normal gives 0.12%, too low by a factor of more than two. On the other side the normal errs the opposite way: it puts 3.9% on 4 or fewer successes, against an exact 2.8%. A Poisson with \(\lambda = np = 10\) gets the right tail within about 10%, and it is the better approximation whenever \(p\) is small; see [the Poisson distribution](../poisson-distribution/).

```python
# far tails: relative error of the corrected normal at 3, 4 and 5 sd above the mean (900 flips)
for zz in (3, 4, 5):
    kk = int(mu + zz * sigma)
    ex = stats.binom.sf(kk - 1, n, p)
    approx = stats.norm.sf((kk - 0.5 - mu) / sigma)
    assert abs(approx / ex - 1) < {3: 0.01, 4: 0.03, 5: 0.06}[zz]
    assert -0.12 < stats.norm.sf(zz) / ex - 1 < -0.09      # uncorrected: about 10% low

# with only 100 flips the corrected tail drifts far off
assert round(stats.norm.sf(69.5, 50, 5) / stats.binom.sf(69, 100, 0.5) - 1, 2) in (0.22, 0.23)   # 4 sd
assert round(stats.norm.sf(74.5, 50, 5) / stats.binom.sf(74, 100, 0.5) - 1, 2) == 0.70            # 5 sd

# a skewed binomial that passes the rule of thumb: n = 500, p = 0.02, np = 10
n2, p2 = 500, 0.02
m2, s2 = n2 * p2, math.sqrt(n2 * p2 * (1 - p2))
assert m2 >= 10 and n2 * (1 - p2) >= 10 and round(s2, 2) == 3.13
ex_tail = stats.binom.sf(19, n2, p2)                   # P(X >= 20)
normal_tail = stats.norm.sf(19.5, m2, s2)
poisson_tail = stats.poisson.sf(19, m2)
assert round(ex_tail, 4) == 0.0031 and round(normal_tail, 4) == 0.0012
assert ex_tail > 2 * normal_tail                       # normal badly underestimates the long tail
assert abs(poisson_tail / ex_tail - 1) < 0.12          # Poisson is within about 10%
ex_left, normal_left = stats.binom.cdf(4, n2, p2), stats.norm.cdf(4.5, m2, s2)
assert round(ex_left, 3) == 0.028 and round(normal_left, 3) == 0.039   # and overstates the short one
```

## A bound that always holds
The normal approximation is an estimate with no guarantee attached. When you need a guarantee, Hoeffding's inequality (Wassily Hoeffding, 1963) gives one for any number of flips: for the share of heads \(\hat p = X/n\) and any \(t > 0\),

\[ P(\hat p - p \ge t) \le e^{-2nt^2}. \]

For this question \(t = 30/900 = 1/30\), so the bound is \(e^{-2} \approx 13.5\%\). That is true, but more than five times larger than the real 2.46%. Bounds like this are what learning theory uses, because they hold for every \(n\) and every distribution of bounded variables; for a numerical answer to a specific question, the corrected normal or the exact sum is far sharper.

```python
t = 30 / 900
hoeffding = math.exp(-2 * n * t ** 2)
assert math.isclose(hoeffding, math.exp(-2))
assert hoeffding > exact and hoeffding / exact > 5

# the bound holds for every n and every threshold we try, including small n where the normal is poor
for flips in (10, 30, 100, 900):
    for heads_needed in range(flips // 2 + 1, flips + 1):
        tt = heads_needed / flips - 0.5
        assert stats.binom.sf(heads_needed - 1, flips, 0.5) <= math.exp(-2 * flips * tt ** 2) + 1e-12
```

## Summary
- **The recipe:** for \(X \sim \text{Binomial}(n, p)\), use \(\mu = np\) and \(\sigma = \sqrt{np(1-p)}\), standardize, and read the tail of a standard normal. For a fair coin \(\sigma = \sqrt{n}/2\).
- **Tails to remember:** \(P(Z \ge 1) \approx 16\%\), \(P(Z \ge 2) \approx 2.3\%\), \(P(Z \ge 3) \approx 0.13\%\). The "95" in 68-95-99.7 is really 95.45%.
- **Continuity correction:** cut at 479.5, not 480, when approximating "at least 480". It matters most when \(\sigma\) is small.
- **Know when to stop:** use the exact binomial for far tails and the Poisson when \(p\) is small; use Hoeffding when you need a bound that cannot fail.

The approximation checks cover fair coins at 100, 900 and 2,500 flips and one skewed case, 500 trials at 2%, so they show where the approximation fails, not an error rate for every \(n\) and \(p\). The rule of thumb is the open part: the skewed case meets it and the normal is still off by a factor of more than two, so for a new case the exact tail, or the Poisson when \(p\) is small, is the check to run.

The [normal distribution](../normal-distribution/) post covers the bell curve itself, and the [probability cheatsheet](../probability-cheatsheet/) has these tail values on one page. The question at the top of this post was inspired by a coin-flipping probability problem on InterviewQuery, rewritten here with different numbers.
