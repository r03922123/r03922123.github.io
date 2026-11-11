# Uncertainty in a rate: the Beta posterior, the standard error and the bootstrap

An item was clicked 3 times in 4 impressions, another 300 times in 400. Both have a click-through rate of 75%, and nobody would trust the two numbers equally: from a uniform prior, their 95% credible intervals run from 0.28 to 0.95 and from 0.71 to 0.79. A single estimate throws away how much evidence stands behind it; I keep that information by putting a distribution on the rate.

There are two standard ways to do it. The Bayesian way describes what you believe about the unknown rate with a whole distribution; for a probability, that is almost always a **Beta**. The frequentist way keeps the rate fixed and asks how much the estimate would move if you collected the data again, which leads to the **standard error** and, when no formula exists, the **bootstrap**. I check the formulas and numbers below in code. On 3,000 simulated new items, ranking by posterior mean instead of the raw rate puts items in the top 20 whose true rates are higher by more than a percentage point on average.

## From counts to a distribution over p
Suppose a coin with unknown heads probability \(p\) lands heads \(h\) times and tails \(t\) times. Start with no preference among values of \(p\), a uniform prior. Bayes' rule says the posterior is proportional to prior times likelihood, and the likelihood of a particular sequence is \(p^h (1-p)^t\), so

\[ f(p \mid h, t) \;\propto\; p^{h} (1 - p)^{t}, \qquad 0 \le p \le 1. \]

That shape has a name. The **Beta distribution** with parameters \(a, b > 0\) has density

\[ f(p) = \frac{p^{a-1} (1-p)^{b-1}}{B(a, b)}, \qquad B(a, b) = \frac{\Gamma(a)\,\Gamma(b)}{\Gamma(a + b)}, \]

where the Beta function \(B(a, b)\) is just the constant that makes the density integrate to 1. Matching exponents, the posterior after \(h\) heads and \(t\) tails from a uniform start is \(\text{Beta}(h + 1, t + 1)\). The "+1"s are easy to drop, and dropping them is a common slip: the exponents are \(a - 1\) and \(b - 1\), not \(a\) and \(b\). (The uniform distribution is itself \(\text{Beta}(1, 1)\).)

The code computes the posterior the slow way, by applying Bayes' rule on a grid of 10,001 values of \(p\), and compares it with the Beta density.

```python
import math
import numpy as np
from scipy import stats, integrate, special

grid = np.linspace(0, 1, 10_001)
dp = grid[1] - grid[0]


def grid_posterior(h, t, prior):
    unnorm = prior * grid ** h * (1 - grid) ** t       # Bayes' rule, point by point
    return unnorm / (unnorm.sum() * dp)


for h, t in [(3, 1), (0, 5), (30, 10)]:
    post = grid_posterior(h, t, np.ones_like(grid))
    assert np.allclose(post, stats.beta.pdf(grid, h + 1, t + 1), atol=2e-3)

# the normalizing constant is the Beta function
a, b = 4.0, 2.0
area, _ = integrate.quad(lambda p: p ** (a - 1) * (1 - p) ** (b - 1), 0, 1)
assert math.isclose(area, special.beta(a, b))
assert math.isclose(special.beta(a, b), math.gamma(a) * math.gamma(b) / math.gamma(a + b))
```

## Reading a Beta
The mean, mode and variance are

\[ \mathbb{E}[p] = \frac{a}{a + b}, \qquad \text{mode} = \frac{a - 1}{a + b - 2}\ (a, b > 1), \qquad \text{Var}(p) = \frac{\mathbb{E}[p]\,(1 - \mathbb{E}[p])}{a + b + 1}. \]

The mean sets where the belief sits and \(a + b\) sets how tight it is: read \(a\) and \(b\) as counts of successes and failures, and the variance shrinks roughly as one over the total count.

| parameters | shape | reads as |
|---|---|---|
| \(\text{Beta}(1, 1)\) | flat | no information |
| \(\text{Beta}(4, 2)\) | broad, leaning right | 3 heads, 1 tail, from a flat start |
| \(\text{Beta}(301, 101)\) | narrow spike near 0.75 | 300 heads, 100 tails |
| \(\text{Beta}(0.5, 0.5)\) | U-shaped | the Jeffreys prior; mass near 0 and 1 |

Back to the opening example. The 3-of-4 item has posterior \(\text{Beta}(4, 2)\), whose 95% **credible interval** (the central 95% of the posterior) runs from 0.28 to 0.95. The 300-of-400 item has \(\text{Beta}(301, 101)\) and an interval from 0.71 to 0.79. Same rate, an interval eight times narrower. Note also that the posterior mean of the first is \(4/6 \approx 0.67\), not 0.75: with little data, the prior still pulls.

For large counts the posterior standard deviation is close to the familiar standard error \(\sqrt{\hat p (1 - \hat p) / n}\) from the [binomial post](../bernoulli-binomial/), so the Bayesian and frequentist error bars nearly agree. They diverge exactly where it matters, with small counts or rates near 0 and 1.

```python
for a, b in [(4, 2), (301, 101), (0.5, 0.5), (12, 3)]:
    m = a / (a + b)
    assert math.isclose(stats.beta.mean(a, b), m)
    assert math.isclose(stats.beta.var(a, b), m * (1 - m) / (a + b + 1))
    if a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)
        assert abs(grid[np.argmax(stats.beta.pdf(grid, a, b))] - mode) < 1e-3

small = stats.beta.interval(0.95, 4, 2)
large = stats.beta.interval(0.95, 301, 101)
assert [round(x, 2) for x in small] == [0.28, 0.95]
assert [round(x, 2) for x in large] == [0.71, 0.79]
assert (small[1] - small[0]) > 7.5 * (large[1] - large[0])

# with plenty of data the posterior sd and the binomial standard error agree
p_hat, n = 300 / 400, 400
assert abs(stats.beta.std(301, 101) - math.sqrt(p_hat * (1 - p_hat) / n)) < 5e-4
```

## Priors are pseudo-counts
The uniform prior is a special case. Start from any \(\text{Beta}(a, b)\), observe \(h\) successes and \(t\) failures, and the posterior is

\[ \text{Beta}(a + h,\; b + t). \]

The prior and posterior are in the same family, which is what **conjugate** means, and updating is addition. It follows that updating one observation at a time or all at once gives the same answer, and that the posterior mean is a weighted average of the prior mean and the observed rate:

\[ \frac{a + h}{a + b + n} = w \cdot \frac{h}{n} + (1 - w) \cdot \frac{a}{a + b}, \qquad w = \frac{n}{n + a + b}. \]

So a \(\text{Beta}(a, b)\) prior behaves like \(a + b\) imaginary observations at rate \(a / (a + b)\). With little real data the estimate stays near the prior mean; as \(n\) grows, \(w \to 1\) and the data take over. This is **shrinkage**.

Where does a good prior come from? Often from the population itself. If established items have click-through rates with mean \(m\) and variance \(v\), choose \(a\) and \(b\) so the Beta has the same mean and variance: from the variance formula, \(a + b = m(1 - m)/v - 1\). Fitting the prior from data this way is called **empirical Bayes**.

**Worked example: ranking new items.** Take 3,000 items with true rates drawn from a population averaging about 5%, each shown between 5 and 500 times. Ranking by raw click-through rate fills the top 20 with items that got lucky on a handful of impressions. Ranking by posterior mean, with a prior fitted from 500 established items, picks items whose true rates are clearly higher.

```python
rng = np.random.default_rng(1)

# conjugate updating: one at a time equals all at once, and the posterior mean is a weighted average
a0, b0 = 2.0, 18.0
flips = rng.random(50) < 0.3
a, b = a0, b0
for x in flips:
    a, b = a + x, b + (1 - x)
h, n = flips.sum(), len(flips)
assert (a, b) == (a0 + h, b0 + n - h)
w = n / (n + a0 + b0)
assert math.isclose(a / (a + b), w * h / n + (1 - w) * a0 / (a0 + b0))

# empirical Bayes: fit the prior to established items by matching mean and variance
established = rng.beta(3, 57, 500)
m, v = established.mean(), established.var()
strength = m * (1 - m) / v - 1
a0, b0 = m * strength, (1 - m) * strength
assert math.isclose(stats.beta.mean(a0, b0), m) and math.isclose(stats.beta.var(a0, b0), v)

# 3,000 new items with very different amounts of traffic
true_rate = rng.beta(3, 57, 3000)
shown = rng.integers(5, 501, 3000)                  # 5 to 500 impressions
clicks = rng.binomial(shown, true_rate)
raw = clicks / shown
posterior_mean = (a0 + clicks) / (a0 + b0 + shown)

top_raw = np.argsort(-raw)[:20]
top_post = np.argsort(-posterior_mean)[:20]
assert shown[top_raw].mean() < 0.5 * shown[top_post].mean()          # raw favours tiny samples
assert true_rate[top_post].mean() > true_rate[top_raw].mean() + 0.01  # shrinkage picks better items
```

## Questions a posterior can answer
Once the belief about each rate is a distribution, many questions become one line of sampling.

**Is B better than A?** Variant A gets 120 conversions in 1,000 visits and B gets 145 in 1,000. With uniform priors the posteriors are \(\text{Beta}(121, 881)\) and \(\text{Beta}(146, 856)\). Draw from both and count how often B's draw is larger: about 95%. The same number comes from the integral \(\int f_B(x)\,F_A(x)\,dx\), the probability that A falls below a given value of B, averaged over B. The draws also give an interval for the lift \(p_B - p_A\), roughly \(-0.005\) to \(0.055\). "95% probability that B is better" is a direct statement about the rates, which a p-value is not, but it depends on the prior. [A/B testing](../ab-testing/) and [p-values](../p-values/) give the frequentist side.

**Which arm should I show next?** **Thompson sampling**, proposed by William R. Thompson in 1933, chooses by drawing once from each arm's posterior and playing the arm with the largest draw. An arm with little data has a wide posterior and sometimes draws high, so it gets explored; an arm that is clearly worse rarely wins a draw, so it is played less and less. In the simulation below, three arms with rates of 4%, 5% and 6% start from flat priors, and after 20,000 rounds the best arm has received most of the traffic.

```python
rng = np.random.default_rng(2)

draws_a = rng.beta(121, 881, 400_000)
draws_b = rng.beta(146, 856, 400_000)
prob_b_better = (draws_b > draws_a).mean()
exact, _ = integrate.quad(lambda x: stats.beta.pdf(x, 146, 856) * stats.beta.cdf(x, 121, 881), 0, 1,
                          points=[0.12, 0.145], limit=200)
assert abs(prob_b_better - exact) < 3e-3 and round(exact, 2) == 0.95
lo, hi = np.percentile(draws_b - draws_a, [2.5, 97.5])
assert lo < 0 < hi and round(lo, 3) == -0.005 and round(hi, 3) == 0.055

# Thompson sampling on three arms
rates = np.array([0.04, 0.05, 0.06])
alpha, beta_ = np.ones(3), np.ones(3)
pulls = np.zeros(3, dtype=int)
for _ in range(20_000):
    arm = np.argmax(rng.beta(alpha, beta_))
    reward = rng.random() < rates[arm]
    alpha[arm] += reward
    beta_[arm] += 1 - reward
    pulls[arm] += 1
assert pulls[2] > 0.75 * pulls.sum()
```

## The frequentist view: the standard error
Now drop the prior and treat the unknown quantity as fixed. Let \(X_1, \dots, X_n\) be independent draws from any distribution with mean \(\mu\) and variance \(\sigma^2\), and let \(\bar X = \frac{1}{n} \sum_i X_i\). Linearity of expectation gives \(\mathbb{E}[\bar X] = \mu\), so the sample mean is unbiased. For the variance, constants come out squared and, because the draws are independent, the variances of the terms add:

\[ \text{Var}(\bar X) = \frac{1}{n^2} \sum_{i=1}^{n} \text{Var}(X_i) = \frac{\sigma^2}{n}. \]

The standard deviation of \(\bar X\), \(\sigma / \sqrt{n}\), is the **standard error**. You do not know \(\sigma\), so you plug in the sample variance \(S^2 = \frac{1}{n-1} \sum_i (X_i - \bar X)^2\), whose \(n - 1\) makes it unbiased for \(\sigma^2\) (dividing by \(n\) underestimates by a factor \((n-1)/n\)). The independence step does real work: positively correlated draws make the true variance larger than \(\sigma^2 / n\). And \(S / \sqrt{n}\) is only an estimate, shaky for tiny samples or heavy tails.

```python
rng = np.random.default_rng(3)
n, reps = 25, 200_000
x = rng.exponential(scale=2.0, size=(reps, n))       # skewed, mean 2, variance 4
means = x.mean(axis=1)
assert abs(means.mean() - 2.0) < 0.005                # unbiased
assert abs(means.var() / (4.0 / n) - 1) < 0.02        # variance sigma^2 / n
assert abs(x.var(axis=1, ddof=1).mean() / 4.0 - 1) < 0.01           # S^2 with n - 1 is unbiased
assert abs(x.var(axis=1, ddof=0).mean() / (4.0 * (n - 1) / n) - 1) < 0.01   # with n it is low
```

## The bootstrap: error bars without a formula
The standard error formula covers the mean. For a median, a ratio of means, a percentile or an AUC there is often no simple formula. The **bootstrap**, introduced by Bradley Efron in 1979, replaces the formula with computation. The sample is the best picture of the population, so you draw new samples from it, with replacement and of the same size, recompute the statistic on each, and use the spread of those values as the spread of the statistic.

**Confidence interval.** Resample each group separately, compute the statistic each time, and take the 2.5th and 97.5th percentiles: the **percentile interval**. Its promise is about the procedure: across repeated experiments, about 95% of intervals built this way contain the true value of the quantity being estimated. That coverage is only approximate, and it degrades for small or very skewed samples.

**Test of no difference.** To ask whether two groups differ, build the world where they do not: pool them, draw both resamples from the pool, and record the difference each time. The p-value is the fraction of these null differences at least as extreme as the observed one, written \((1 + \text{count}) / (1 + B)\) so it is never exactly zero. Shuffling the pooled values without replacement instead gives the closely related permutation test.

**Worked example.** Session lengths in two groups are right-skewed, 60 and 80 users. The code builds bootstrap intervals for the difference in means and in medians, checks the mean interval against Welch's t interval, checks the bootstrap p-value against a permutation test, and finally runs the whole procedure on 300 fresh experiments to measure its real coverage.

```python
rng = np.random.default_rng(4)


def bootstrap_diff(x, y, n_boot, rng, stat=np.mean):
    xi = rng.integers(0, len(x), size=(n_boot, len(x)))   # resample each group within itself
    yi = rng.integers(0, len(y), size=(n_boot, len(y)))
    return stat(y[yi], axis=1) - stat(x[xi], axis=1)


x = rng.lognormal(1.0, 0.8, 60)
y = rng.lognormal(1.25, 0.8, 80)

diffs = bootstrap_diff(x, y, 10_000, rng)
lo, hi = np.percentile(diffs, [2.5, 97.5])
welch = stats.ttest_ind(y, x, equal_var=False).confidence_interval()
assert abs(lo - welch.low) < 0.1 and abs(hi - welch.high) < 0.1

med_lo, med_hi = np.percentile(bootstrap_diff(x, y, 10_000, rng, np.median), [2.5, 97.5])
assert med_lo < np.median(y) - np.median(x) < med_hi      # no closed form needed


def bootstrap_pvalue(x, y, n_boot, rng):
    observed = abs(y.mean() - x.mean())
    pool = np.concatenate([x, y])                          # "no difference": one shared population
    xi = rng.integers(0, len(pool), size=(n_boot, len(x)))
    yi = rng.integers(0, len(pool), size=(n_boot, len(y)))
    null = np.abs(pool[yi].mean(axis=1) - pool[xi].mean(axis=1))
    return (1 + (null >= observed).sum()) / (1 + n_boot)


p_boot = bootstrap_pvalue(x, y, 10_000, rng)
perm = stats.permutation_test((y, x), lambda a, b, axis: a.mean(axis) - b.mean(axis),
                              n_resamples=9_999, vectorized=True, random_state=0)
assert abs(p_boot - perm.pvalue) < 0.03

# does a "95%" interval cover the truth 95% of the time? lognormal mean is exp(mu + s^2 / 2)
true_diff = math.exp(1.25 + 0.32) - math.exp(1.0 + 0.32)
covered = 0
for _ in range(300):
    xs, ys = rng.lognormal(1.0, 0.8, 60), rng.lognormal(1.25, 0.8, 80)
    lo, hi = np.percentile(bootstrap_diff(xs, ys, 2_000, rng), [2.5, 97.5])
    covered += lo <= true_diff <= hi
assert 0.88 <= covered / 300 <= 0.98
```

With these sample sizes the bootstrap and Welch intervals agree to within a few hundredths, and measured coverage lands near, usually a little under, the nominal 95%. That gap is typical of the percentile interval on skewed data; refinements such as the BCa interval (available as `method="BCa"` in `scipy.stats.bootstrap`) narrow it.

## Pitfalls
- **A credible interval is not a confidence interval.** "95% probability that \(p\) is in this range" is a Bayesian statement and depends on the prior. "This procedure covers the true \(p\) 95% of the time" is the frequentist one. They agree numerically for large samples and weak priors, and nowhere else is that guaranteed.
- **A prior is data you did not observe.** \(\text{Beta}(50, 950)\) counts as a thousand observations. Choose its strength deliberately, check it against the population, and report it.
- **Independence again.** Both the Beta-binomial update and \(\sigma^2/n\) assume independent trials. Clicks from the same user, or sessions from the same day, are correlated, and the honest error bar is wider. With the bootstrap, resample whole users or clusters, not rows.
- **The bootstrap cannot invent information.** It reflects the sample it is given. With 10 observations its intervals tend to be too narrow, and for statistics that depend on extremes, such as the maximum, it can fail outright.
- **Beta is for probabilities.** For counts per unit of exposure, such as events per hour, the conjugate pair is the Gamma prior with the Poisson likelihood; the [Poisson post](../poisson-distribution/) covers the model.

## Summary
- **Beta(a, b)** is a belief about a probability, centred at \(a / (a + b)\) and tighter as \(a + b\) grows; a \(\text{Beta}(a, b)\) prior plus \(h\) successes and \(t\) failures gives \(\text{Beta}(a + h, b + t)\), so the prior acts as pseudo-counts that shrink small samples.
- **Posteriors answer direct questions** such as \(P(p_B > p_A)\), and drive Thompson sampling.
- **The standard error** of a mean is \(\sigma / \sqrt{n}\) for independent draws; **the bootstrap** gets intervals and tests for any statistic by resampling, with approximate coverage.

The ranking, Thompson sampling and coverage results each come from one simulated population with a fixed seed and independent trials, so they show how the methods behave when their assumptions hold, not how much they gain on real traffic. Two choices stay open on real data, and each is the analyst's to justify and report: how strong a prior to use, and at what level to resample when trials are correlated.

The Beta-as-belief view and the two-group bootstrap follow the way Stanford's CS109 (Probability for Computer Scientists), in Chris Piech's course notes, teaches them; the examples and code here are new. For the related estimation story, see [MLE and MAP](../statistical-estimation-mle-map/), and the [probability cheatsheet](../probability-cheatsheet/) collects the formulas.
