# P-values: what they measure and how they mislead

Model A gets 850 of 1,000 test examples right and model B gets 830. Is A better, or did it get lucky on this particular test set? A p-value answers a narrower question than that: if the two models were really equally good, how often would a gap this large show up anyway? If the answer is "rarely", the gap is hard to put down to luck.

This post defines the p-value, checks the definition by simulation, uses it to compare two models properly, and then works through the common misreadings, each with a simulation that shows how big the error is.

## The definition
Pick a **null hypothesis** \(H_0\), the boring explanation ("the models are equally accurate"), and a **test statistic** \(T\) that tends to be large when \(H_0\) is wrong. Having observed \(T = t_{\text{obs}}\), the p-value is

\[ p = P(T \ge t_{\text{obs}} \mid H_0). \]

It is the probability, computed as if \(H_0\) were true, of a result at least as extreme as the one in hand. What counts as "extreme" comes from the alternative: for a one-sided question it is one tail, for a two-sided question about a statistic centred on zero, you put it in absolute value, so both tails count.

Nothing in the formula is a probability that \(H_0\) is true; that would need a prior, which the definition never uses. Nor does it need a closed form. Below, 50 measured improvements have known noise \(\sigma = 1\) and mean 0.3. Simulating the null world many times and counting how often its statistic beats ours gives the same number as the normal tail formula.

```python
import math
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
n, sigma, observed_mean = 50, 1.0, 0.3
t_obs = observed_mean / (sigma / math.sqrt(n))             # z statistic, about 2.12

null_means = rng.normal(0, sigma, size=(100_000, n)).mean(axis=1)
t_null = null_means / (sigma / math.sqrt(n))
p_sim = np.mean(t_null >= t_obs)                           # one-sided: only large values count
p_formula = stats.norm.sf(t_obs)

assert round(p_formula, 3) == 0.017
assert abs(p_sim - p_formula) < 0.002
p_two_sided = np.mean(np.abs(t_null) >= abs(t_obs))        # two-sided: both tails count
assert abs(p_two_sided - 2 * p_formula) < 0.003
```

## Why "reject when p ≤ α" works
Fix a threshold \(\alpha\) before looking, and reject \(H_0\) when \(p \le \alpha\). The promise is that if \(H_0\) is true you raise a false alarm with probability at most \(\alpha\). The reason is a neat fact: when \(T\) is continuous and \(H_0\) fully specifies its distribution, the p-value is **uniformly distributed** on \([0, 1]\) under \(H_0\).

The argument is two lines. Write \(S(t) = P(T \ge t \mid H_0)\), so \(p = S(T)\). The function \(S\) is decreasing, so \(p \le \alpha\) happens exactly when \(T\) lands in the upper tail that has probability \(\alpha\):

\[ P(p \le \alpha \mid H_0) = P\big(T \ge S^{-1}(\alpha) \mid H_0\big) = \alpha. \]

So \(\alpha\) is the false positive rate of the procedure, chosen in advance, not a property of any single result.

For discrete statistics (counts, ranks) the p-value can only take a few values, and the equality becomes an inequality: \(P(p \le \alpha) \le \alpha\). The test is **conservative**. The same happens with a composite null such as "the effect is at most zero", where the guarantee holds at the boundary and the test is conservative everywhere else.

```python
a = rng.normal(size=(20_000, 30))
b = rng.normal(size=(20_000, 30))                          # same distribution: H0 is true
p_null = stats.ttest_ind(a, b, axis=1).pvalue

assert abs(np.mean(p_null <= 0.05) - 0.05) < 0.005         # false alarms at the advertised rate
assert stats.kstest(p_null, "uniform").pvalue > 0.01       # consistent with Uniform(0, 1)
for alpha in (0.01, 0.1, 0.5):
    assert abs(np.mean(p_null <= alpha) - alpha) < 0.01

# discrete: the exact binomial test on 20 fair flips never reaches its nominal size
m = 20
p_by_count = np.array([stats.binomtest(k, m, 0.5).pvalue for k in range(m + 1)])
pmf = stats.binom.pmf(np.arange(m + 1), m, 0.5)
for alpha in (0.01, 0.05, 0.1, 0.2):
    actual = pmf[p_by_count <= alpha].sum()
    assert actual <= alpha
assert round(pmf[p_by_count <= 0.05].sum(), 3) == 0.041
```

[Is this coin fair?](../is-this-coin-fair/) works through the discrete case in detail, including how few results a 10-flip test can ever call significant.

## Worked example: is model A better than model B?
Back to the two models. Both were scored on the same 1,000 examples, which matters: an example that is hard for A is usually hard for B too. Here they agree on 920 examples (800 both right, 120 both wrong) and disagree on 80: A alone is right on 50, B alone on 30.

The statistic is the accuracy gap. The null says that for every example, the label "A" or "B" on the two predictions is arbitrary. So the null world is simulated by **swapping the two models' results within each example** with probability one half, which flips the sign of that example's difference. Examples where they agree do not change. Counting how often a swapped gap is at least as large as the real one gives the p-value. Two details: add one to the count and to the denominator, which keeps a Monte Carlo p-value valid and never exactly zero, and use absolute values for a two-sided question.

The swapping only matters on the 80 disagreements, and each one becomes +1 or −1 with equal probability. The exact null distribution is therefore binomial, and the test is the exact form of **McNemar's test**, so SciPy's binomial test checks the simulation.

A common slip is to ignore the pairing: pool all 2,000 correct/incorrect flags, shuffle them and split them into two fake models. That treats the test set as two independent samples and loses most of the power.

```python
n_ex = 1000
a_ok = np.r_[np.ones(800), np.zeros(120), np.ones(50), np.zeros(30)].astype(np.int8)
b_ok = np.r_[np.ones(800), np.zeros(120), np.zeros(50), np.ones(30)].astype(np.int8)
order = rng.permutation(n_ex)
a_ok, b_ok = a_ok[order], b_ok[order]
d = a_ok - b_ok                                            # +1, 0 or -1 per example
gap = d.mean()
assert math.isclose(gap, 0.02)

B = 20_000
signs = rng.integers(0, 2, size=(B, n_ex), dtype=np.int8) * 2 - 1
swapped = (signs * d).mean(axis=1, dtype=np.float64)       # each example's result swapped or not
p_paired = (1 + np.sum(np.abs(swapped) >= abs(gap) - 1e-12)) / (B + 1)

only_a, only_b = int(np.sum(d == 1)), int(np.sum(d == -1))
p_mcnemar = stats.binomtest(only_b, only_a + only_b, 0.5).pvalue
se = math.sqrt(p_mcnemar * (1 - p_mcnemar) / B)
assert abs(p_paired - p_mcnemar) < 4 * se + 1 / B           # simulation matches the exact test
assert round(p_mcnemar, 3) == 0.033

pool = np.r_[a_ok, b_ok]
shuffled = rng.permuted(np.tile(pool, (5_000, 1)), axis=1)  # the slip: ignore the pairing
fake_gap = shuffled[:, :n_ex].mean(axis=1) - shuffled[:, n_ex:].mean(axis=1)
p_unpaired = (1 + np.sum(np.abs(fake_gap) >= abs(gap) - 1e-12)) / (5_000 + 1)
assert p_unpaired > 0.15 > 0.05 > p_paired                  # same data, opposite conclusions
```

The paired test says 0.03; the unpaired one says about 0.2. Same data, and only the paired test asks the right question.

## Power: what happens when the null is false
A p-value controls one error, the false positive. The other, missing a real effect, is governed by **power**, the probability of rejecting when a given alternative is true. Picture two bell curves for the statistic, one under \(H_0\) and one under the alternative, with the cutoff between them. The area of the null curve past the cutoff is \(\alpha\). The area of the alternative curve before the cutoff is the false negative rate \(\beta\), and the rest of it is the power \(1 - \beta\).

For a one-sided z-test where the true effect is \(\delta\) standard errors, the cutoff is \(z_{1-\alpha}\) and

\[ \text{power} = P(Z + \delta \ge z_{1-\alpha}) = \Phi(\delta - z_{1-\alpha}). \]

With \(\alpha = 0.05\), an effect of 2.5 standard errors gives about 80% power. Under the alternative, p-values are no longer uniform; they pile up near zero.

```python
z_crit = stats.norm.ppf(0.95)
delta = 2.5
power = stats.norm.cdf(delta - z_crit)
assert round(power, 2) == 0.80

p_alt = stats.norm.sf(delta + rng.normal(size=200_000))
assert abs(np.mean(p_alt <= 0.05) - power) < 0.005
assert np.median(p_alt) < 0.01                              # a real effect drives p toward 0
assert stats.kstest(p_alt, "uniform").pvalue < 1e-10
```

## Mistake 1: reading p as the chance the null is true
"p = 0.04, so there is a 4% chance the result is a fluke" is the most common misreading. The chance that a significant result is a false alarm depends on how many of the hypotheses you test are true to begin with, which the p-value knows nothing about. The structure is [Bayes' rule](../conditional-probability-bayes/) with the base rate playing the prior.

Suppose 1 in 10 ideas you test actually works, each with the 80% power above. Of 1,000 ideas, 900 are null and about 45 of them pass at \(\alpha = 0.05\); 100 are real and about 80 pass. So 45 of the 125 significant results, 36%, are false alarms. Among results that only just pass, with p between 0.04 and 0.05, the share is far worse: about three in four are null. A null result spreads its p-values evenly, while a real one at this power puts more than half of them below 0.01, so a borderline p-value is weak evidence.

```python
n_exp = 1_000_000
is_real = rng.random(n_exp) < 0.10
z = rng.normal(size=n_exp) + np.where(is_real, delta, 0.0)
p = stats.norm.sf(z)

sig = p <= 0.05
analytic = 0.9 * 0.05 / (0.9 * 0.05 + 0.1 * power)
assert round(analytic, 2) == 0.36
assert abs(np.mean(~is_real[sig]) - analytic) < 0.01

borderline = (p > 0.04) & (p <= 0.05)
assert np.mean(~is_real[borderline]) > 0.70                  # most "just significant" results are flukes here
assert 0.5 < np.mean(p[is_real] < 0.01) < 0.6                # real effects: more than half land below 0.01
```

## Mistake 2: confusing small p with a big effect
The statistic grows like \(\sqrt{n}\) times the effect, so with enough data any nonzero effect gives a tiny p-value, and with too little data an important effect gives a large one. A p-value answers "is there a difference?", not "how big is it?" or "does it matter?". Report the effect with a confidence interval next to the p-value.

```python
def two_prop_p(x1, n1, x2, n2):
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 2 * stats.norm.sf(abs(x1 / n1 - x2 / n2) / se)

tiny_lift_huge_n = two_prop_p(1_005_000, 10_000_000, 1_000_000, 10_000_000)   # 10.05% vs 10.00%
big_lift_small_n = two_prop_p(13, 100, 10, 100)                               # 13% vs 10%
assert tiny_lift_huge_n < 1e-3 < 0.5 < big_lift_small_n
```

## Mistake 3: peeking and stopping early
Checking an experiment every day and stopping the first time \(p \le 0.05\) is not a test at level 0.05. Each look is another chance for noise to cross the line, and the chances accumulate. With ten evenly spaced looks at data where nothing is happening, about one experiment in five gets declared a winner.

The fixes all decide the looks in advance. The crudest splits \(\alpha\) across them (0.005 per look for ten looks), which is valid but costs power. **Group sequential designs**, such as Pocock's and O'Brien and Fleming's boundaries, spend \(\alpha\) more cleverly across a fixed schedule of looks, and **always-valid** (sequential) p-values allow looking at any time. The simplest fix is a fixed sample size from a power calculation; see [A/B testing](../ab-testing/).

```python
n_sims, n_max, looks = 4_000, 1_000, np.arange(100, 1001, 100)
paths = np.cumsum(rng.normal(size=(n_sims, n_max)), axis=1)  # H0 true: mean zero
z_at_looks = paths[:, looks - 1] / np.sqrt(looks)
p_at_looks = 2 * stats.norm.sf(np.abs(z_at_looks))

fixed = np.mean(p_at_looks[:, -1] <= 0.05)                  # look once, at the planned end
peeking = np.mean((p_at_looks <= 0.05).any(axis=1))         # stop at the first p <= 0.05
split = np.mean((p_at_looks <= 0.05 / len(looks)).any(axis=1))
assert abs(fixed - 0.05) < 0.012
assert peeking > 0.15                                       # roughly 1 in 5, not 1 in 20
assert split <= 0.05
```

## Mistake 4: testing many things and reporting the hits
Test 1,000 features, segments or metrics at \(\alpha = 0.05\) and around 50 will come out significant even if none of them is real. Two standard corrections:

- **Bonferroni** tests each hypothesis at \(\alpha / m\). It bounds the chance of even one false alarm (the family-wise error rate) and is very strict when \(m\) is large.
- **Benjamini–Hochberg** bounds the false discovery rate, the expected share of false alarms among the rejections. Sort the p-values, find the largest \(k\) with \(p_{(k)} \le q k / m\), and reject the \(k\) smallest. The guarantee holds when the tests are independent or positively correlated, and it keeps far more true discoveries.

The code implements Benjamini–Hochberg, checks it against `scipy.stats.false_discovery_control`, and estimates each method's error rate over 200 repetitions with 50 real effects among 1,000 tests.

```python
def benjamini_hochberg(p, q=0.05):
    m = len(p)
    order = np.argsort(p)
    passes = p[order] <= q * np.arange(1, m + 1) / m
    k = passes.nonzero()[0].max() + 1 if passes.any() else 0   # largest passing rank, not the first failure
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True
    return reject


m, n_real, reps = 1000, 50, 200
real = np.arange(m) < n_real
stats_by_method = {"none": [], "bonferroni": [], "bh": []}
for _ in range(reps):
    p = stats.norm.sf(rng.normal(size=m) + np.where(real, 3.5, 0.0))
    bh = benjamini_hochberg(p)
    assert np.array_equal(bh, stats.false_discovery_control(p) <= 0.05)
    for name, rej in (("none", p <= 0.05), ("bonferroni", p <= 0.05 / m), ("bh", bh)):
        false = np.sum(rej & ~real)
        stats_by_method[name].append((false, false / max(rej.sum(), 1), np.sum(rej & real)))

summary = {k: np.array(v).mean(axis=0) for k, v in stats_by_method.items()}   # false alarms, FDP, true hits
assert summary["none"][0] > 40                              # ~47 false alarms per run with no correction
assert summary["bh"][1] <= 0.05 + 0.01                      # BH: false discovery rate held near q or below
assert summary["bonferroni"][0] < 0.2                       # Bonferroni: almost never a false alarm...
assert summary["bh"][2] > summary["bonferroni"][2] + 10     # ...but BH finds many more real effects
```

Use Bonferroni when a single false alarm is costly, Benjamini–Hochberg when you screen many candidates and follow up on the hits.

## How to use them
- **Decide the test before the data:** the hypothesis, the statistic, one or two sided, \(\alpha\), and the sample size. Choosing any of these after looking turns the guarantee into decoration.
- **Match the test to how the data were collected.** Two models on one test set are paired; users randomized into two arms are independent. [Simulation-based evaluation](../simulation-based-evaluation/) covers more ways to build a null world by resampling.
- **Report an effect size with an interval.** The p-value adds whether the data rule out zero; the interval says what they rule in.
- **Treat 0.05 as a convention, not a law.** p = 0.049 and p = 0.051 are the same evidence. Results near the line deserve a replication, not a headline.
- **Count every test you ran,** including the slices you looked at and dropped, and correct for them.
- **A large p-value is not evidence of no effect.** With low power it is mostly evidence of too little data.

The framing of the threshold as a false positive rate follows the hypothesis testing chapter of Berkeley's Data 102 course notes, and the peeking simulation is in the spirit of Evan Miller's "How Not To Run an A/B Test". The false discovery rate procedure is from Benjamini and Hochberg (1995), McNemar's test from McNemar (1947), and the group sequential boundaries from Pocock (1977) and O'Brien and Fleming (1979). The American Statistical Association's 2016 statement on p-values (Wasserstein and Lazar) is a short, readable summary of the misreadings above.
