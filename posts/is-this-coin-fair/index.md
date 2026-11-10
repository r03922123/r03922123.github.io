# Is this coin fair? Testing a proportion from ten flips to a million

A simulated coin is flipped 10 times and comes up tails 8 times. Is it broken? The question is small, but answering it properly uses most of the machinery behind every A/B test and every "is this metric really different" check: a null hypothesis, a p-value, the difference between "not significant" and "fine", and the number of samples it takes before a test can tell you anything.

This post works the example by hand, checks each number by brute force and against SciPy, then scales it up to large samples and looks at it from the Bayesian side.

## The question, made precise
Let \(p\) be the probability of heads. "Fair" is the **null hypothesis** \(H_0: p = 1/2\). We have no reason to expect bias in one direction, so the alternative is two-sided, \(H_1: p \ne 1/2\). The data are summarized by the number of heads \(X\), which under \(H_0\) follows \(\text{Binomial}(10, 1/2)\) (see [Bernoulli and binomial](../bernoulli-binomial/)). We saw \(X = 2\).

The **p-value** is the probability, if the coin were fair, of a result at least as surprising as the one we saw. "At least as surprising" here means 2 or fewer heads, or, symmetrically, 2 or fewer tails:

\[ P(X \le 2) + P(X \ge 8) = 2 \cdot \frac{\binom{10}{0} + \binom{10}{1} + \binom{10}{2}}{2^{10}} = 2 \cdot \frac{1 + 10 + 45}{1024} = \frac{112}{1024} \approx 0.109. \]

At the usual threshold \(\alpha = 0.05\), that is not small enough to reject fairness. A fair coin gives a split this lopsided about one time in nine.

## The exact test, by brute force
Nothing about this needs a formula. A fair coin makes all \(2^{10} = 1024\) sequences equally likely, so the p-value is just the fraction of sequences at least as extreme as ours. The code counts them and compares with `scipy.stats.binomtest`, which runs the same exact test.

```python
import itertools
import math
import numpy as np
from scipy import stats

n, heads = 10, 2


def extremeness(k, n):
    return abs(k - n / 2)                  # distance from the expected count


seqs = list(itertools.product((0, 1), repeat=n))
as_extreme = sum(extremeness(sum(s), n) >= extremeness(heads, n) for s in seqs)
p_brute = as_extreme / len(seqs)

assert as_extreme == 112
assert math.isclose(p_brute, 112 / 1024)
assert math.isclose(p_brute, stats.binomtest(heads, n, 0.5).pvalue)
assert p_brute > 0.05                      # not enough to call the coin biased
```

A note on "two-sided". Doubling the smaller one-sided tail is unambiguous here because \(\text{Binomial}(n, 1/2)\) is symmetric. For a null like \(p = 0.3\) it is not, and raw doubling can even give a p-value above 1. Doubling and capping at 1 is one legitimate convention (the "central" p-value). The other common one, the default in SciPy and in R's `binom.test`, adds up every outcome whose probability under \(H_0\) is no larger than that of the observed one. For a symmetric null the two agree; for a skewed one they can differ:

```python
def two_sided_exact(k, n, p0):
    pmf = stats.binom.pmf(np.arange(n + 1), n, p0)
    return pmf[pmf <= pmf[k] * (1 + 1e-7)].sum()        # small tolerance for float ties


for k in range(n + 1):
    assert math.isclose(two_sided_exact(k, n, 0.5), stats.binomtest(k, n, 0.5).pvalue)
    assert math.isclose(two_sided_exact(k, n, 0.3), stats.binomtest(k, n, 0.3).pvalue)

def doubled(k, n, p0):
    return 2 * min(stats.binom.cdf(k, n, p0), stats.binom.sf(k - 1, n, p0))


for k in range(n + 1):                                  # symmetric null: both conventions agree
    assert math.isclose(min(1.0, doubled(k, n, 0.5)), two_sided_exact(k, n, 0.5))
assert doubled(3, n, 0.3) > 1                           # raw doubling overshoots, so cap it at 1
assert not math.isclose(doubled(1, n, 0.3), two_sided_exact(1, n, 0.3))   # skewed null: they differ
```

## "Not significant" is not "fair"
Failing to reject \(H_0\) says the data are consistent with a fair coin. It does not say the coin is fair, and with 10 flips that distinction is most of the story. Two numbers make it concrete.

**Ten flips can barely reject anything.** The only outcomes with a two-sided p-value of 0.05 or below are 0, 1, 9 and 10 heads. Because the binomial is discrete, the real false-alarm rate of this test is not 5% but \(22/1024 \approx 2.1\%\). And if the coin truly landed heads 80% of the time, the test would catch it only about 38% of the time. If heads came up 60% of the time it would catch it less than 5% of the time.

**The confidence interval is enormous.** The Clopper–Pearson interval, the exact interval built by inverting two one-sided binomial tests (and so, at \(p_0 = 1/2\), agreeing with the test above about whether 1/2 is plausible), puts \(p\) between about 0.03 and 0.56 at 95% confidence. The data are consistent with a fair coin and also with one that lands heads 3% of the time. That interval is a more honest summary than "p = 0.11".

```python
pvals = np.array([stats.binomtest(k, n, 0.5).pvalue for k in range(n + 1)])
reject = pvals <= 0.05
assert list(np.flatnonzero(reject)) == [0, 1, 9, 10]


def power(n, p_true, alpha=0.05):
    ks = np.arange(n + 1)
    rej = np.array([stats.binomtest(int(k), n, 0.5).pvalue <= alpha for k in ks])
    return stats.binom.pmf(ks, n, p_true)[rej].sum()


assert math.isclose(power(10, 0.5), 22 / 1024)          # actual size, well under 5%
assert round(power(10, 0.8), 2) == 0.38
assert power(10, 0.6) < 0.05

ci = stats.binomtest(heads, n, 0.5).proportion_ci(confidence_level=0.95, method="exact")
assert round(ci.low, 3) == 0.025 and round(ci.high, 2) == 0.56
assert ci.low < 0.5 < ci.high                            # consistent with fair, and with very unfair
for k in range(n + 1):                                   # the interval contains 1/2 exactly when the test does not reject
    ci_k = stats.binomtest(k, n, 0.5).proportion_ci(confidence_level=0.95, method="exact")
    assert (ci_k.low <= 0.5 <= ci_k.high) == (not reject[k])
```

## How many flips would it take?
To decide in advance, fix the smallest bias you care about and the chance you want of detecting it. Say a coin landing heads 60% of the time matters, and we want 80% **power** at \(\alpha = 0.05\). The normal approximation gives the classic sample-size formula:

\[ n \approx \left( \frac{z_{1 - \alpha/2} \sqrt{p_0 (1 - p_0)} + z_{1 - \beta} \sqrt{p_1 (1 - p_1)}}{p_1 - p_0} \right)^2 = \left( \frac{1.96 \cdot 0.5 + 0.84 \cdot 0.49}{0.1} \right)^2 \approx 194. \]

About two hundred flips, twenty times what we have. The exact test adds a wrinkle: as \(n\) grows its power does not rise smoothly but zigzags, because the rejection region can only grow one whole count at a time. At \(n = 194\) the exact test has only about 76% power, while at 190 it has 79%. Its power first reaches 80% at \(n = 199\), falls back below at 200, and stays above only from \(n = 210\) on. In practice you round up and check the exact power at the chosen \(n\), here by computing it for every \(n\) and by simulating 20,000 experiments at \(n = 210\).

```python
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
p0, p1 = 0.5, 0.6
n_needed = ((z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
assert math.ceil(n_needed) == 194

assert round(power(194, 0.6), 2) == 0.76 and round(power(190, 0.6), 2) == 0.79   # not monotone in n
first = next(m for m in range(10, 400) if power(m, 0.6) >= 0.80)
assert first == 199 and power(200, 0.6) < 0.80
assert power(209, 0.6) < 0.80 and all(power(m, 0.6) >= 0.80 for m in range(210, 260))

rng = np.random.default_rng(0)
m = 210
rej_region = np.array([stats.binomtest(k, m, 0.5).pvalue <= 0.05 for k in range(m + 1)])
sim_heads = rng.binomial(m, p1, size=20_000)
assert abs(rej_region[sim_heads].mean() - power(m, p1)) < 0.01
```

## Large samples: the normal approximation
With many flips, \(X\) is close to normal with mean \(n p_0\) and standard deviation \(\sqrt{n p_0 (1 - p_0)}\), so the test becomes a **one-sample z-test for a proportion**:

\[ z = \frac{X - n p_0}{\sqrt{n p_0 (1 - p_0)}}. \]

It is one-sample because there is a single coin compared against a fixed value; the two-sample version compares two groups, as in an A/B test. For 65 tails in 100 flips, \(z = (65 - 50)/5 = 3\) and the two-sided p-value is about 0.0027, against an exact 0.0035. Shifting the count half a step toward the mean, the **continuity correction**, gives \(z = 2.9\) and 0.0037, much closer. For 80 tails in 100, \(z = 6\) and the p-value is around \(2 \times 10^{-9}\): overwhelming evidence of bias.

Two related facts. The chi-squared goodness-of-fit test on the counts (heads, tails) gives exactly the same p-value as the uncorrected z-test, since its statistic is \(z^2\). And the old excuse for the approximation, that exact binomial sums are tedious, no longer applies: the exact test on a million flips takes a fraction of a second. The approximation is still worth knowing because it explains the \(\sqrt{n}\) behaviour and gives closed-form sample sizes. [Coin flips and the normal approximation](../coin-flips-normal-approximation/) looks at when it holds.

```python
def z_test(k, n, p0=0.5, correction=False):
    diff = k - n * p0
    if correction:
        diff -= math.copysign(0.5, diff)
    z = diff / math.sqrt(n * p0 * (1 - p0))
    return z, 2 * stats.norm.sf(abs(z))


exact65 = stats.binomtest(65, 100, 0.5).pvalue
z, p_plain = z_test(65, 100)
_, p_cc = z_test(65, 100, correction=True)
assert z == 3.0 and round(p_plain, 4) == 0.0027 and round(exact65, 4) == 0.0035
assert abs(p_cc - exact65) < abs(p_plain - exact65)       # the correction helps

z80, p80 = z_test(80, 100)
exact80 = stats.binomtest(80, 100, 0.5).pvalue
assert z80 == 6.0 and round(p80, 10) == 2.0e-9 and round(exact80, 10) == 1.1e-9   # normal is off by ~2x out here

chi2 = stats.chisquare([65, 35], [50, 50])
assert math.isclose(chi2.statistic, z ** 2) and math.isclose(chi2.pvalue, p_plain)

big = stats.binomtest(500_800, 1_000_000, 0.5)            # exact, and instant
assert 0.10 < big.pvalue < 0.12
assert stats.binomtest(502_000, 1_000_000, 0.5).pvalue < 1e-4   # a 0.2-point bias is unmistakable at this n
```

## A Bayesian reading
A p-value answers "how surprising is this data if the coin is fair?", not "how likely is the coin to be fair?". The second question needs a prior. Put a uniform prior on \(p\); after 2 heads and 8 tails the posterior is \(\text{Beta}(3, 9)\), and the probability that the coin favours tails is about 0.97. That sounds decisive, but it only says which way the coin leans if it leans at all.

To weigh "exactly fair" against "some unknown bias", compare how well each hypothesis predicted the sequence we saw. A fair coin gave it probability \(2^{-10}\); the uniform-bias model gave it \(\int_0^1 p^2 (1-p)^8 \, dp = 2!\,8!/11!\). The ratio, the **Bayes factor**, is about 2.1 in favour of bias: weak evidence, which agrees with the p-value's shrug. [Beta uncertainty](../beta-uncertainty/) and [MLE and MAP](../statistical-estimation-mle-map/) take this further.

```python
posterior = stats.beta(1 + heads, 1 + n - heads)
assert round(posterior.cdf(0.5), 2) == 0.97              # P(p < 1/2 | data)

evidence_fair = 0.5 ** n
evidence_biased = math.factorial(heads) * math.factorial(n - heads) / math.factorial(n + 1)
from scipy.integrate import quad
numeric, _ = quad(lambda q: q ** heads * (1 - q) ** (n - heads), 0, 1)
assert math.isclose(evidence_biased, numeric, rel_tol=1e-9)
assert round(evidence_biased / evidence_fair, 1) == 2.1
```

## What a count cannot see
Every test so far looks only at the number of heads. A simulator that outputs H, T, H, T, ... passes them perfectly: exactly half heads, p-value 1. A fair coin also needs **independent** flips. The Wald–Wolfowitz **runs test** checks this by counting runs, maximal blocks of identical outcomes. With \(n_1\) heads and \(n_2\) tails in independent order, the number of runs has mean \(2 n_1 n_2 / n + 1\). Alternating sequences have too many runs; "sticky" ones that tend to repeat the last outcome have too few.

```python
def runs_test(seq):
    seq = np.asarray(seq)
    n1, n2 = int(seq.sum()), int(len(seq) - seq.sum())
    runs = 1 + int((seq[1:] != seq[:-1]).sum())
    mean = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mean) / math.sqrt(var)
    return runs, 2 * stats.norm.sf(abs(z))


alternating = [i % 2 for i in range(200)]
assert stats.binomtest(sum(alternating), 200, 0.5).pvalue == 1.0   # the count test is satisfied
assert runs_test(alternating)[1] < 1e-10                          # the runs test is not

sticky = [0]
for _ in range(199):                                     # repeat the last flip 80% of the time
    sticky.append(sticky[-1] if rng.random() < 0.8 else 1 - sticky[-1])
assert runs_test(sticky)[1] < 1e-6

pv_fair = [runs_test(rng.integers(0, 2, 200))[1] for _ in range(2000)]
assert abs(np.mean(np.array(pv_fair) < 0.05) - 0.05) < 0.02       # about 5% false alarms on a fair coin
```

## Which test, and pitfalls
| situation | use |
|---|---|
| any \(n\), one coin vs a fixed \(p_0\) | exact binomial test, with a Clopper–Pearson or Wilson interval |
| large \(n\), want a formula or a sample size | one-sample z-test, ideally with continuity correction |
| more than two outcomes (a die, a categorical simulator) | chi-squared goodness of fit, or an exact multinomial test for small counts |
| two coins or two variants | two-sample test for proportions; see [A/B testing](../ab-testing/) |
| order matters, not just counts | runs test, or autocorrelation of the sequence |
| you want a probability about \(p\) itself | Beta posterior |

- **Decide the number of flips before flipping.** Flipping until the p-value dips below 0.05 and then stopping makes a false alarm far more likely than 5%, because every peek is another chance to cross the line. More on reading them in [P-values](../p-values/).
- **Report the interval, not only the verdict.** "p = 0.11" hides that the data allow anything from a heavily biased coin to a fair one.
- **With a huge \(n\), everything is significant.** A million flips detect a bias of 0.2 percentage points that may not matter for any purpose. Ask how large a bias would matter, then check whether the interval rules it in or out.
- **Testing many coins means many chances of a false alarm.** Check 100 fair coins at \(\alpha = 0.05\) and expect around five to look biased by chance alone (somewhat fewer with a conservative exact test).
- **Rules of thumb such as \(n \ge 30\) are not a licence for the normal approximation.** What matters is that \(n p_0\) and \(n (1 - p_0)\) are both comfortably large; far in the tails even large \(n\) leaves the approximation off by a sizable factor (above, \(2.0 \times 10^{-9}\) against an exact \(1.1 \times 10^{-9}\)). When in doubt, run the exact test.

The tests here are textbook material. The exact interval is from Clopper and Pearson (1934), the score interval from Wilson (1927), and the runs test from Wald and Wolfowitz (1940).
