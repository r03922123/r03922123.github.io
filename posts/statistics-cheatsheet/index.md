# Statistics on one page: estimates, intervals, tests and the traps

Use 1.96 with an estimated standard error at \(n = 10\) and a "95%" interval covers the truth only about 92% of the time. Check an experiment with no effect 20 times and stop at the first \(p < 0.05\), and about a quarter end up "significant". Most statistics in day-to-day ML work comes down to one question asked in different clothes: is this difference real, or is it noise? A new ranker lifts click-through rate by 0.4 points, a fine-tuned model gains two points of accuracy, a cohort spends more after a launch. This page collects the tools for answering it, in the order you usually need them. Sections are short and link to the posts that develop each idea; the distributions themselves live in the [probability cheatsheet](../probability-cheatsheet/). Most formulas and numbers are checked in code, against `scipy`, `scikit-learn` or a simulation.

## Estimates and standard errors
An estimate is a number computed from a sample, and the **standard error** (SE) is how much that number would wobble if you drew a fresh sample. Four formulas cover most cases:

- **Sample variance** divides by \(n - 1\), not \(n\): \(s^2 = \frac{1}{n-1}\sum_i (x_i - \bar{x})^2\). The sample mean sits closer to the data than the true mean does, so dividing by \(n\) runs low by a factor \((n-1)/n\). In NumPy that is `ddof=1`.
- **Mean:** \(\text{SE}(\bar{x}) = s / \sqrt{n}\). Halving the error costs four times the data.
- **Proportion:** \(\text{SE}(\hat{p}) = \sqrt{\hat{p}(1-\hat{p})/n}\), the mean formula applied to 0/1 data.
- **Difference of independent estimates:** standard errors add in quadrature, \(\text{SE}(\hat{a} - \hat{b}) = \sqrt{\text{SE}(\hat{a})^2 + \text{SE}(\hat{b})^2}\).

By the central limit theorem the estimate is roughly normal around the truth with that SE once \(n\) is large enough. How large depends on the skew: a few dozen draws for symmetric data, thousands for revenue per user.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# dividing by n - 1 makes the sample variance unbiased; dividing by n runs low by (n - 1) / n
n, sigma = 5, 2.0
samples = rng.normal(0, sigma, (200_000, n))
assert abs(samples.var(axis=1, ddof=1).mean() - sigma ** 2) < 0.03
assert abs(samples.var(axis=1, ddof=0).mean() - sigma ** 2 * (n - 1) / n) < 0.03

# the standard error of a mean is sigma / sqrt(n), even for skewed data
for size in (10, 40, 160):
    means = rng.exponential(3.0, (50_000, size)).mean(axis=1)        # sd of one draw = 3
    assert abs(means.std() - 3.0 / np.sqrt(size)) < 0.02 * 3.0 / np.sqrt(size)

# the standard error of a difference adds in quadrature
p_a, p_b, n_a, n_b = 0.10, 0.12, 4000, 3000
diffs = rng.binomial(n_b, p_b, 200_000) / n_b - rng.binomial(n_a, p_a, 200_000) / n_a
se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
assert abs(diffs.std() - se_diff) < 0.01 * se_diff
```

## Confidence intervals
The usual interval is the estimate plus or minus a quantile times its standard error:

\[
\hat{\theta} \pm q \cdot \text{SE}(\hat{\theta})
\]

For a 95% interval \(q = 1.96\) from the normal distribution, or the Student t quantile with \(n - 1\) degrees of freedom when the SE uses an estimated standard deviation from a small sample. The t quantile is larger (2.26 at \(n = 10\)) because the estimated SE is itself noisy, and it approaches 1.96 as \(n\) grows.

"95%" is a property of the **procedure**: over many repeated samples, about 95% of the intervals it produces contain the true value. It is not a 95% probability that the one interval you computed contains it; a probability statement about the parameter needs a prior and a Bayesian credible interval, as in [Beta for uncertainty](../beta-uncertainty/). Coverage can be checked, and checking exposes three common problems:

- **z instead of t at small \(n\).** Using 1.96 with an estimated SE at \(n = 10\) covers only about 92% of the time.
- **The textbook interval for a rare proportion.** \(\hat{p} \pm 1.96\,\text{SE}\) collapses to a single point when no successes are seen, so with \(p = 3\%\) and \(n = 50\) it misses badly. The Wilson interval, which inverts the test instead of plugging \(\hat{p}\) into the SE, holds up.
- **Statistics with no simple SE**, such as a median or a 95th-percentile latency. The **bootstrap** resamples the data with replacement, recomputes the statistic each time and takes the middle 95% of the results.

An interval and a test made from the same ingredients agree: a 95% interval for a difference excludes zero exactly when the two-sided test rejects at \(\alpha = 0.05\).

```python
# t versus z coverage at n = 10, by simulation
true_mu, reps, size = 5.0, 20_000, 10
data = rng.normal(true_mu, 2.0, (reps, size))
centre, se = data.mean(axis=1), data.std(axis=1, ddof=1) / np.sqrt(size)
t_q = stats.t.ppf(0.975, df=size - 1)
assert abs(t_q - 2.262) < 1e-3 and abs(stats.t.ppf(0.975, df=1000) - 1.96) < 3e-3
assert 0.94 < (np.abs(centre - true_mu) <= t_q * se).mean() < 0.96
assert (np.abs(centre - true_mu) <= 1.96 * se).mean() < 0.935
assert np.allclose(stats.t.interval(0.95, size - 1, loc=centre[0], scale=se[0]),
                   (centre[0] - t_q * se[0], centre[0] + t_q * se[0]))


def wald(k, n, z=1.96):
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


assert np.allclose(wilson(3, 50), tuple(stats.binomtest(3, 50).proportion_ci(method="wilson")), atol=1e-4)


def exact_coverage(interval, p, n):
    ks = np.arange(n + 1)
    lo, hi = interval(ks, n)
    return stats.binom(n, p).pmf(ks)[(lo <= p) & (p <= hi)].sum()   # sum P(k) over the k whose interval covers p


assert exact_coverage(wald, 0.03, 50) < 0.80
assert exact_coverage(wilson, 0.03, 50) > 0.90

# bootstrap interval for a median of skewed data: check its coverage over 400 datasets
true_median = np.exp(1.0)                                    # median of a lognormal(1, 1)
hits = 0
for _ in range(400):
    x = rng.lognormal(1.0, 1.0, 100)
    boot = np.median(x[rng.integers(0, 100, (1000, 100))], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    hits += lo <= true_median <= hi
assert 0.88 < hits / 400 < 0.99

# interval and test agree: Welch's 95% interval excludes 0 exactly when p < 0.05
for _ in range(200):
    a, b = rng.normal(0, 1, 30), rng.normal(0.4, 1.5, 40)
    res = stats.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval(0.95)
    assert (res.pvalue < 0.05) == (ci.low > 0 or ci.high < 0)
```

## The testing loop
Every frequentist test follows the same steps:

1. State a **null hypothesis** \(H_0\), usually "no difference", and an alternative.
2. Fix the significance level \(\alpha\), commonly 0.05, before looking at the data.
3. Compute a test statistic that is large when the data disagree with \(H_0\).
4. Compute the **p-value**: the probability, assuming \(H_0\) is true, of a statistic at least as extreme as the one observed.
5. Reject \(H_0\) if \(p \le \alpha\). Otherwise you have failed to reject it, which is not the same as showing it is true.

There are two ways to be wrong:

| | \(H_0\) true | \(H_0\) false |
|---|---|---|
| Reject | **Type I error**, false positive, rate \(\alpha\) | correct, probability = **power** \(= 1 - \beta\) |
| Do not reject | correct | **Type II error**, false negative, rate \(\beta\) |

"Reject when \(p \le \alpha\)" works because under the null a p-value is uniform, so it falls below \(\alpha\) exactly \(\alpha\) of the time. A p-value is not the probability that \(H_0\) is true; that needs a prior. [P-values](../p-values/) works through this and the other misreadings in detail. The 0.05 line is a convention that goes back to Fisher, not a law, and some fields use far stricter thresholds.

```python
# under the null, p-values are uniform, so P(p <= alpha) = alpha
a = rng.normal(0, 1, (10_000, 30))
b = rng.normal(0, 1, (10_000, 30))
p = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
assert abs((p <= 0.05).mean() - 0.05) < 0.006
assert abs((p <= 0.20).mean() - 0.20) < 0.012
```

## Which test for which data

| Question | Test | In `scipy.stats` |
|---|---|---|
| Is one mean different from a fixed value? | one-sample t | `ttest_1samp` |
| Do two independent groups differ in mean? | Welch's t (no equal-variance assumption) | `ttest_ind(equal_var=False)` |
| Did the same units change between two measurements? | paired t, on the differences | `ttest_rel` |
| Do two conversion rates differ? | two-proportion z, same as chi-squared on the 2×2 table | `chi2_contingency(correction=False)` |
| Are two categorical variables related? | chi-squared test of independence | `chi2_contingency` |
| Does one group tend to have larger values, with skewed data or ranks? | Mann-Whitney U | `mannwhitneyu` |
| Any statistic, with few assumptions | permutation test | `permutation_test` |

A z-test suits large samples, where the estimated SE is accurate; a t-test pays for estimating the SE from a small sample with heavier tails, and the two agree as \(n\) grows. Default to Welch's t: it costs almost nothing when variances are equal and protects you when they are not.

The costliest mistake here is ignoring pairing. When two models are scored on the same test set, most of the variation is between examples, and the paired test removes it. Below, an unpaired test misses an improvement that the paired test finds with \(p < 0.001\).

```python
# two proportions: z squared is exactly the Pearson chi-squared statistic
conv, users = np.array([410, 468]), np.array([4000, 4000])
pooled = conv.sum() / users.sum()
z = (conv[1] / users[1] - conv[0] / users[0]) / np.sqrt(pooled * (1 - pooled) * (1 / users[0] + 1 / users[1]))
chi2, p_chi, _, _ = stats.chi2_contingency(np.c_[conv, users - conv], correction=False)
assert np.isclose(z ** 2, chi2) and np.isclose(2 * stats.norm.sf(abs(z)), p_chi)

# Welch's t by hand, including its degrees of freedom, matches scipy
x, y = rng.normal(10, 2, 25), rng.normal(11, 4, 40)
vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
t = (x.mean() - y.mean()) / np.sqrt(vx + vy)
df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
res = stats.ttest_ind(x, y, equal_var=False)
assert np.isclose(t, res.statistic) and np.isclose(2 * stats.t.sf(abs(t), df), res.pvalue)

# paired data: the paired test is a one-sample test on the differences
before = rng.normal(50, 10, 30)                          # units differ a lot
after = before + rng.normal(1.0, 1.0, 30)                # small, consistent improvement
paired = stats.ttest_rel(after, before)
assert np.isclose(paired.pvalue, stats.ttest_1samp(after - before, 0).pvalue)
assert paired.pvalue < 0.001 and stats.ttest_ind(after, before).pvalue > 0.3

# a permutation test needs no distributional formula and lands close to the t-test
g1, g2 = rng.normal(0, 1, 40), rng.normal(0.5, 1, 40)
perm = stats.permutation_test((g1, g2), lambda u, v: u.mean() - v.mean(),
                              n_resamples=20_000, random_state=1)
assert abs(perm.pvalue - stats.ttest_ind(g1, g2).pvalue) < 0.01
```

## Power and sample size
Power is the chance of rejecting \(H_0\) when a real effect of a given size exists. Before a test, choose the smallest effect worth acting on, the **minimum detectable effect** \(\delta\), and collect enough data to detect it with, typically, 80% power. For two proportions \(p_1\) and \(p_2 = p_1 + \delta\), the sample per group is approximately

\[
n \approx \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \left[p_1(1-p_1) + p_2(1-p_2)\right]}{\delta^2},
\]

with \(z_{1-\alpha/2} = 1.96\) for a two-sided \(\alpha = 0.05\) and \(z_{1-\beta} = 0.84\) for 80% power. A 10% baseline and a one-point lift need about 14,750 users per group. Because \(\delta\) is squared, halving the effect you want to see roughly quadruples the sample.

To raise power for a fixed effect: more data, a less noisy metric, or variance reduction such as CUPED (Deng, Xu, Kohavi and Walker, 2013), which subtracts the part of the metric predicted by pre-experiment behaviour. Raising the minimum detectable effect adds no power; it only declares smaller effects not worth finding. [A/B testing](../ab-testing/) covers sample size, CUPED and the rest of experiment design.

```python
def n_per_group(p1, p2, alpha=0.05, power=0.80):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2))


n_needed = n_per_group(0.10, 0.11)
assert 14_700 < n_needed < 14_800
assert 3.8 < n_per_group(0.10, 0.105) / n_needed < 4.2      # half the effect, about four times the users

# simulate 4,000 experiments at that size: about 80% of them detect the lift
reps = 4000
c_a, c_b = rng.binomial(n_needed, 0.10, reps), rng.binomial(n_needed, 0.11, reps)
pooled = (c_a + c_b) / (2 * n_needed)
z = (c_b - c_a) / n_needed / np.sqrt(pooled * (1 - pooled) * 2 / n_needed)
assert 0.78 < (np.abs(z) > 1.96).mean() < 0.82
```

## Many tests, and looking too often
At \(\alpha = 0.05\), each true null has a 5% chance of a false alarm, and those chances compound. With 20 independent true nulls, the chance of at least one "significant" result is \(1 - 0.95^{20} \approx 64\%\). Two standard fixes:

- **Bonferroni:** test each of \(m\) hypotheses at \(\alpha / m\). This controls the **family-wise error rate**, the chance of any false positive, and is strict.
- **Benjamini-Hochberg** (1995): sort the p-values, find the largest \(k\) with \(p_{(k)} \le q k / m\), and reject the \(k\) smallest. For independent (or positively dependent) tests this controls the **false discovery rate**, the expected share of false positives among the rejections, at \(q\), and finds many more real effects when there are some.

**Peeking** is the same problem in time. Checking a running experiment every day and stopping the first time \(p < 0.05\) gives the noise many chances to cross the line; with 20 looks, a quarter of experiments with no effect at all end up "significant". Fix the sample size in advance, or use a sequential method built for repeated looks.

```python
assert abs((1 - 0.95 ** 20) - 0.64) < 0.01


def benjamini_hochberg(p, q=0.05):
    m = len(p)
    order = np.argsort(p)
    below = p[order] <= q * np.arange(1, m + 1) / m
    k = below.nonzero()[0].max() + 1 if below.any() else 0
    reject = np.zeros(m, bool)
    reject[order[:k]] = True
    return reject


pv = rng.uniform(size=50) ** 3
assert (benjamini_hochberg(pv) == (stats.false_discovery_control(pv) <= 0.05)).all()

# 100 tests, 10 with a real effect, repeated 2,000 times
reps, m = 2000, 100
shift = np.r_[np.full(10, 3.0), np.zeros(m - 10)]
pvals = 2 * stats.norm.sf(np.abs(rng.normal(shift, 1.0, (reps, m))))
null = shift == 0
bonf = pvals <= 0.05 / m
bh = np.array([benjamini_hochberg(row) for row in pvals])
assert (pvals[:, null] <= 0.05).any(axis=1).mean() > 0.98    # uncorrected: almost always a false alarm
assert bonf[:, null].any(axis=1).mean() < 0.06               # Bonferroni: family-wise rate under 5%
assert (bh[:, null].sum(axis=1) / np.maximum(bh.sum(axis=1), 1)).mean() < 0.055   # BH: FDR about 5%
assert bh[:, ~null].mean() > bonf[:, ~null].mean() + 0.1     # and BH finds more of the real effects

# peeking: 20 looks at an experiment with no effect
looks, per_look, reps = 20, 100, 4000
running = rng.normal(0, 1, (reps, looks, per_look)).sum(axis=2).cumsum(axis=1)
z = running / np.sqrt(per_look * np.arange(1, looks + 1))
assert abs((np.abs(z[:, -1]) > 1.96).mean() - 0.05) < 0.012  # one look at the end: 5%
assert 0.22 < (np.abs(z) > 1.96).any(axis=1).mean() < 0.28   # stop at the first "win": about a quarter
```

## Estimating parameters: MLE and MAP
**Maximum likelihood** picks the parameter under which the observed data are most probable, \(\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(\text{data} \mid \theta)\). **MAP** also weighs in a prior, \(\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\text{data} \mid \theta)\,p(\theta)\). With a flat prior the two coincide, though only in the parametrization where the prior is flat. For linear regression with Gaussian noise of standard deviation \(\sigma\) and a Gaussian prior of standard deviation \(\tau\) on each weight, MAP is exactly ridge regression with penalty \(\lambda = \sigma^2 / \tau^2\): a tighter prior means stronger shrinkage. [MLE and MAP](../statistical-estimation-mle-map/), [regression as maximum likelihood](../regression-as-maximum-likelihood/) and [L1 and L2 regularization](../regularization-l1-l2/) develop this.

```python
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

X = rng.normal(size=(60, 5))
y = X @ rng.normal(0, 1, 5) + rng.normal(0, 0.5, 60)
noise_sd, prior_sd = 0.5, 0.3
lam = noise_sd ** 2 / prior_sd ** 2

w_map = np.linalg.solve(X.T @ X + lam * np.eye(5), X.T @ y)
neg_log_posterior = lambda w: ((y - X @ w) ** 2).sum() / (2 * noise_sd ** 2) + (w ** 2).sum() / (2 * prior_sd ** 2)
assert np.allclose(minimize(neg_log_posterior, np.zeros(5)).x, w_map, atol=1e-4)
assert np.allclose(Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_, w_map, atol=1e-8)

w_mle = np.linalg.lstsq(X, y, rcond=None)[0]
assert np.linalg.norm(w_map) < np.linalg.norm(w_mle)                 # the prior shrinks
w_wide = np.linalg.solve(X.T @ X + 1e-8 * np.eye(5), X.T @ y)        # a nearly flat prior recovers the MLE
assert np.allclose(w_wide, w_mle, atol=1e-6)
```

## Correlation, confounding and Simpson's paradox
A difference between two groups that chose themselves is not an effect. Users who adopt a feature are usually the engaged ones, and a **confounder** like engagement drives both who gets the treatment and the outcome. Random assignment makes the groups alike in everything except the treatment. Without it, the options are adjusting for measured confounders, difference-in-differences, instrumental variables and regression discontinuity, each with an assumption to defend; see [causal inference](../causal-inference/).

**Simpson's paradox** is confounding at its starkest: a change can win in every segment and lose overall when the segments are mixed differently in the two groups. In a proper randomized test with large groups the mix matches up to small noise, so a reversal signals a broken split, such as a ramp that reached mobile users first. Check the observed split against the plan (a sample ratio mismatch test), within each major segment as well as overall: in the example below the totals are a perfect 50/50 and only the per-segment check catches the problem. Then compare within segments or reweight to a common mix.

```python
# a change that helps both segments by 10%, ramped unevenly: most treatment users are on mobile
rate = {"mobile": {"control": 0.020, "treatment": 0.022},
        "desktop": {"control": 0.100, "treatment": 0.110}}
users = {"control": {"mobile": 2000, "desktop": 8000},
         "treatment": {"mobile": 8000, "desktop": 2000}}


def overall(arm, mix):
    return sum(mix[s] * rate[s][arm] for s in rate) / sum(mix.values())


assert all(rate[s]["treatment"] > rate[s]["control"] for s in rate)              # wins in every segment
assert overall("treatment", users["treatment"]) < overall("control", users["control"])   # loses overall
common_mix = {s: users["control"][s] + users["treatment"][s] for s in rate}
assert overall("treatment", common_mix) > overall("control", common_mix)          # same mix: wins again

# sample ratio mismatch: the overall 10,000 / 10,000 split passes, the per-segment split does not
totals = [sum(users[arm].values()) for arm in ("control", "treatment")]
assert stats.chisquare(totals).pvalue > 0.5
for s in rate:
    assert stats.chisquare([users["control"][s], users["treatment"][s]]).pvalue < 1e-6
```

## Pitfalls
- **"Not significant" read as "no effect".** An underpowered test misses real effects most of the time. Report the interval: one spanning both zero and a large effect means the data cannot tell.
- **"Significant" read as "important".** With millions of users a negligible difference has a tiny p-value. Judge the size of the effect and its interval, not the p-value.
- **The wrong unit of analysis.** Randomize by user but count sessions as independent and the SE comes out too small, because sessions from one user are correlated. Analyse at the unit you randomized, or use a method that accounts for clustering.
- **Effects that change over time.** Novelty inflates early lifts and change aversion deepens early dips. Run full weekly cycles and compare early and late periods.
- **Interference between units.** In social or two-sided products, treated users affect control users, and the comparison is biased in a direction that depends on how they interact. Randomize by cluster or market instead.
- **Choosing the analysis after seeing the data.** Trying metrics, segments or outlier rules until something crosses 0.05 is multiple testing without the correction. Fix the primary metric and test in advance.
- **Trusting "n > 30".** It is a rough guide for mild skew; for heavy-tailed metrics, check the normal approximation by simulation.

The simulations here draw independent units from fixed distributions, so they show each formula working when those assumptions hold; correlated sessions, effects that change over time and interference, from the pitfalls above, break them. For the evaluation side of the same ideas, such as comparing models offline and running online tests, see [Evaluating ML systems](/posts/evaluating-ml-systems/) and [is this coin fair?](../is-this-coin-fair/).

This page is a rewritten summary of standard material; for fuller treatments see Casella and Berger's Statistical Inference and, on peeking, Evan Miller's essay [How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html).
