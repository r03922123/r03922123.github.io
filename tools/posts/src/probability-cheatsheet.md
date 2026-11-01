---
title: Probability on one page: the rules, the distributions and the traps
summary: A compact reference for the probability that comes up most in ML work: the event rules and Bayes, PMF versus PDF versus CDF, the expectation and variance identities, a table of eight distributions with their means and variances, the limit theorems, and the parameter conventions and reasoning slips that cause most mistakes. Every formula is checked in code.
group: Probability
order: 9
tags: probability · reference · distributions · Bayes' rule
---
This page collects the probability that keeps coming back when you build and evaluate models, in the order you usually need it: how to combine events, how to describe a random quantity, how means and variances behave, which distribution to reach for, and what happens when you average many draws. It is a reference, so each section is short and, where one exists, points to the post that develops the idea properly.

Nothing here is taken on faith. The formulas on the page are checked below, either exactly or against `scipy` and a simulation, so the code doubles as a set of worked examples.

## Events: the five rules
For events \(A\) and \(B\), with \(A^c\) meaning "not \(A\)":

- **Complement:** \(P(A^c) = 1 - P(A)\). Often the fastest route to "at least one": \(P(\text{at least one}) = 1 - P(\text{none})\).
- **Addition:** \(P(A \cup B) = P(A) + P(B) - P(A \cap B)\). The overlap is counted twice, so subtract it once.
- **Multiplication:** \(P(A \cap B) = P(A \mid B)\,P(B)\). This is the definition of conditional probability, rearranged, and it always holds.
- **Independence:** \(A\) and \(B\) are independent when \(P(A \cap B) = P(A)\,P(B)\), equivalently \(P(A \mid B) = P(A)\) when \(P(B) > 0\).
- **Total probability:** if \(B_1, \dots, B_k\) split the world into disjoint cases, \(P(A) = \sum_i P(A \mid B_i)\,P(B_i)\).

Bayes' rule is the multiplication rule written both ways and solved for the conditional you want, with total probability filling in the denominator:

\[
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B \mid A)\,P(A) + P(B \mid A^c)\,P(A^c)}
\]

The classic example: a disease affects 1% of people, and a test catches 99% of cases (sensitivity) while clearing 99% of healthy people (specificity). Given a positive result, the chance of disease is only 50%. The numerator, true positives, is small because the disease is rare; the denominator adds an equally large pile of false positives drawn from the 99% who are healthy. Run the same test at 0.1% prevalence and a positive is right less than one time in ten. [Conditional probability and Bayes' rule](../conditional-probability-bayes/) builds all of this from counting outcomes.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)


def p_disease_given_positive(prevalence, sensitivity, specificity):
    true_pos = sensitivity * prevalence
    false_pos = (1 - specificity) * (1 - prevalence)
    return true_pos / (true_pos + false_pos)            # total probability in the denominator


assert abs(p_disease_given_positive(0.01, 0.99, 0.99) - 0.5) < 1e-12
assert p_disease_given_positive(0.001, 0.99, 0.99) < 0.1

# the same answer by simulating a population of two million
n = 2_000_000
sick = rng.random(n) < 0.01
positive = np.where(sick, rng.random(n) < 0.99, rng.random(n) < 0.01)
assert abs(sick[positive].mean() - 0.5) < 0.01

# "at least one" via the complement: chance of at least one six in four rolls
assert abs((1 - (5 / 6) ** 4) - 0.5177) < 1e-4

# addition rule by counting all 36 rolls of two dice: A = first die is 6, B = total is 10 or more
rolls = [(i, j) for i in range(1, 7) for j in range(1, 7)]
A = {r for r in rolls if r[0] == 6}
B = {r for r in rolls if sum(r) >= 10}
assert len(A | B) / 36 == (len(A) + len(B) - len(A & B)) / 36 == 9 / 36
```

Two things to keep apart. **Mutually exclusive** events cannot both happen, so \(P(A \cap B) = 0\). **Independent** events carry no information about each other. If both have positive probability, mutually exclusive events are as dependent as events get: learning that \(A\) happened tells you \(B\) did not.

## Describing a random variable: PMF, PDF, CDF
A random variable is a rule that turns each outcome of an experiment into a number, such as the number of clicks in a session or the latency of a request.

- A **discrete** variable has a **probability mass function**, \(p(x) = P(X = x)\). Each value is a probability, between 0 and 1, and the values sum to 1.
- A **continuous** variable has a **probability density function** \(f(x)\). A density is not a probability: \(P(X = x) = 0\) for every single point, probabilities come from areas, \(P(a \le X \le b) = \int_a^b f(x)\,dx\), and a density can exceed 1. Uniform on \([0, 0.5]\) has density 2 everywhere on its range.
- Both have a **cumulative distribution function**, \(F(x) = P(X \le x)\), which rises from 0 to 1. Interval probabilities are differences, \(P(a < X \le b) = F(b) - F(a)\).

For a discrete variable the inequality matters. \(P(X \le k)\) and \(P(X < k)\) differ by exactly \(P(X = k)\), which is where most off-by-one errors in tail probabilities come from. For a continuous one the two are equal. The **survival function** \(1 - F(x) = P(X > x)\) is what you want for tails, and `scipy` computes it directly as `sf`, without the rounding loss of subtracting a number close to 1 from 1.

```python
uniform_half = stats.uniform(loc=0, scale=0.5)
assert uniform_half.pdf(0.25) == 2.0                    # a density above 1 is fine
assert abs(uniform_half.cdf(0.4) - uniform_half.cdf(0.1) - 0.6) < 1e-12

binom = stats.binom(n=10, p=0.3)
k = 4
assert abs(binom.cdf(k) - binom.cdf(k - 1) - binom.pmf(k)) < 1e-12   # P(X <= 4) - P(X < 4) = P(X = 4)
assert abs(binom.pmf(np.arange(11)).sum() - 1) < 1e-12

normal = stats.norm()
assert normal.sf(10) > 0 and 1 - normal.cdf(10) == 0.0  # the tail is ~7.6e-24; 1 - cdf rounds it away
```

## Expectation and variance: the identities
The mean is the probability-weighted average, \(\mathbb{E}[X] = \sum_x x\,p(x)\) or \(\int x f(x)\,dx\). The variance is the mean squared distance from the mean, and the second form is usually easier to compute:

\[
\text{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
\]

The rules that do the heavy lifting:

- **Linearity:** \(\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c\), whether or not \(X\) and \(Y\) are independent.
- **Shift and scale:** \(\text{Var}(aX + b) = a^2\,\text{Var}(X)\). Shifting moves the mean, not the spread.
- **Sums:** \(\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X, Y)\), where \(\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]\). The covariance term drops out only when the variables are uncorrelated.
- **Averages:** the mean of \(n\) independent draws with variance \(\sigma^2\) has variance \(\sigma^2 / n\), so its standard error is \(\sigma / \sqrt{n}\). Halving the error costs four times the data.
- **Functions:** \(\mathbb{E}[g(X)] = \sum_x g(x)\,p(x)\), which is in general **not** \(g(\mathbb{E}[X])\). For convex \(g\), Jensen's inequality says \(\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])\).
- **Total expectation:** \(\mathbb{E}[X] = \mathbb{E}\big[\mathbb{E}[X \mid Y]\big]\), the average of the per-group averages, weighted by group size.

```python
n = 1_000_000
x = rng.normal(2.0, 1.5, n)
y = 0.8 * x + rng.normal(0, 1.0, n)                     # y depends on x
cov = np.cov(x, y)[0, 1]

assert abs((3 * x - 2 * y + 1).mean() - (3 * x.mean() - 2 * y.mean() + 1)) < 1e-9   # linearity, dependent or not
assert abs((5 * x + 7).var() - 25 * x.var()) < 1e-6
assert abs((x + y).var(ddof=1) - (x.var(ddof=1) + y.var(ddof=1) + 2 * cov)) < 1e-6
assert abs((x + y).var() - (x.var() + y.var())) > 3     # forgetting the covariance is badly wrong here

# standard error of a mean shrinks like 1 / sqrt(n)
for size in (25, 100, 400):
    means = rng.exponential(2.0, (20_000, size)).mean(axis=1)   # sigma = 2
    assert abs(means.std() - 2.0 / np.sqrt(size)) < 0.03 * 2.0 / np.sqrt(size)

# Jensen: E[x^2] > (E[x])^2 by exactly the variance
assert abs((x ** 2).mean() - x.mean() ** 2 - x.var()) < 1e-6

# total expectation: average of group means weighted by group size
group = rng.integers(0, 3, n)
z = rng.normal(group * 10.0, 1.0)
weighted = sum((group == g).mean() * z[group == g].mean() for g in range(3))
assert abs(weighted - z.mean()) < 1e-9
```

## The distributions worth knowing
Eight distributions cover most of what you meet. The table gives the parameterization used on this site; the "use it for" column is the part to remember.

| Distribution | Use it for | PMF or density | Mean | Variance |
|---|---|---|---|---|
| Bernoulli\((p)\) | one yes-or-no outcome, \(k \in \{0, 1\}\) | \(p^k (1-p)^{1-k}\) | \(p\) | \(p(1-p)\) |
| Binomial\((n, p)\) | successes in \(n\) independent trials | \(\binom{n}{k} p^k (1-p)^{n-k}\) | \(np\) | \(np(1-p)\) |
| Geometric\((p)\) | trials up to and including the first success, \(k \ge 1\) | \((1-p)^{k-1} p\) | \(1/p\) | \((1-p)/p^2\) |
| Poisson\((\lambda)\) | counts of rare events in a window | \(e^{-\lambda} \lambda^k / k!\) | \(\lambda\) | \(\lambda\) |
| Uniform\((a, b)\) | "anything in a range, equally" | \(1/(b - a)\) on \([a, b]\) | \((a+b)/2\) | \((b-a)^2/12\) |
| Exponential\((\lambda)\) | waiting time to the next event at rate \(\lambda\) | \(\lambda e^{-\lambda x}\), \(x \ge 0\) | \(1/\lambda\) | \(1/\lambda^2\) |
| Normal\((\mu, \sigma^2)\) | sums and averages of many small effects | \(\frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2 / (2\sigma^2)}\) | \(\mu\) | \(\sigma^2\) |
| Beta\((\alpha, \beta)\) | uncertainty about a rate in \([0, 1]\) | \(\propto x^{\alpha-1}(1-x)^{\beta-1}\) | \(\frac{\alpha}{\alpha+\beta}\) | \(\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}\) |

Each has its own post: [Bernoulli and binomial](../bernoulli-binomial/), [Poisson](../poisson-distribution/), [exponential](../exponential-distribution/), [normal](../normal-distribution/) and [Beta](../beta-uncertainty/).

The code checks every row three ways: the mean and variance from `scipy`, the hand-written formula against `scipy`'s PMF or PDF, and the mean and variance of a large sample. Watch the constructor arguments, because they are where the conventions differ.

```python
from math import comb, exp, factorial, gamma, pi, sqrt

p, n_trials, lam, a, b, mu, sigma, al, be = 0.3, 12, 2.5, 1.0, 4.0, 1.0, 2.0, 2.0, 5.0
beta_norm = gamma(al + be) / (gamma(al) * gamma(be))
table = {
    #               scipy object                         formula for P(X = k) or f(x)              mean           variance                     test points
    "bernoulli":   (stats.bernoulli(p),                   lambda k: p ** k * (1 - p) ** (1 - k),     p,             p * (1 - p),                 [0, 1]),
    "binomial":    (stats.binom(n_trials, p),             lambda k: comb(n_trials, k) * p ** k * (1 - p) ** (n_trials - k),
                                                                                                     n_trials * p,  n_trials * p * (1 - p),      [0, 3, 7]),
    "geometric":   (stats.geom(p),                        lambda k: (1 - p) ** (k - 1) * p,          1 / p,         (1 - p) / p ** 2,            [1, 2, 6]),
    "poisson":     (stats.poisson(lam),                   lambda k: exp(-lam) * lam ** k / factorial(k),
                                                                                                     lam,           lam,                         [0, 2, 5]),
    "uniform":     (stats.uniform(loc=a, scale=b - a),    lambda x: 1 / (b - a),                     (a + b) / 2,   (b - a) ** 2 / 12,           [1.5, 3.9]),
    "exponential": (stats.expon(scale=1 / lam),           lambda x: lam * exp(-lam * x),             1 / lam,       1 / lam ** 2,                [0.1, 1.0]),
    "normal":      (stats.norm(loc=mu, scale=sigma),      lambda x: exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * sqrt(2 * pi)),
                                                                                                     mu,            sigma ** 2,                  [-1.0, 2.5]),
    "beta":        (stats.beta(al, be),                   lambda x: beta_norm * x ** (al - 1) * (1 - x) ** (be - 1),
                                                                                                     al / (al + be), al * be / ((al + be) ** 2 * (al + be + 1)),
                                                                                                                                                  [0.1, 0.6]),
}
for name, (dist, formula, mean, var, points) in table.items():
    assert np.isclose(dist.mean(), mean) and np.isclose(dist.var(), var), name
    density = dist.pmf if hasattr(dist, "pmf") else dist.pdf
    for t in points:
        assert np.isclose(density(t), formula(t)), (name, t)
    N = 400_000
    sample = dist.rvs(size=N, random_state=rng)
    assert abs(sample.mean() - mean) < 5 * sqrt(var / N), name    # within 5 standard errors
    assert abs(sample.var() / var - 1) < 0.025, name                # within 2.5% of the true variance
```

## How they connect
A handful of relationships turn one distribution into another, and they explain when an approximation is safe.

- **Bernoulli to binomial:** a sum of \(n\) independent Bernoulli\((p)\) variables is Binomial\((n, p)\).
- **Binomial to Poisson:** with \(n\) large and \(p\) small, Binomial\((n, p)\) is close to Poisson\((np)\). This is why rare-event counts are modelled as Poisson.
- **Poisson to exponential:** if events arrive as a Poisson process at rate \(\lambda\), the gaps between them are Exponential\((\lambda)\).
- **Memorylessness:** the exponential forgets: \(P(X > s + t \mid X > s) = P(X > t)\). Among discrete distributions the geometric has the same property. Neither is a good model for things that wear out or warm up.
- **Sums of normals:** independent normals add to a normal, with means and variances adding.
- **Beta and binomial:** a Beta\((\alpha, \beta)\) prior on \(p\), after \(k\) successes in \(n\) trials, becomes a Beta\((\alpha + k, \beta + n - k)\) posterior. Counting is updating.

```python
# Binomial(2000, 0.0015) against Poisson(3): total variation distance is tiny
ks = np.arange(0, 40)
tv = 0.5 * np.abs(stats.binom(2000, 0.0015).pmf(ks) - stats.poisson(3.0).pmf(ks)).sum()
assert tv < 0.002

# memorylessness of the exponential, exactly and by simulation
ex = stats.expon(scale=1 / lam)
s, t = 0.7, 0.4
assert np.isclose(ex.sf(s + t) / ex.sf(s), ex.sf(t))
waits = ex.rvs(size=1_000_000, random_state=rng)
assert abs((waits[waits > s] > s + t).mean() - ex.sf(t)) < 0.005

# exponential gaps make Poisson counts: count arrivals per unit window
arrivals = np.cumsum(rng.exponential(1 / lam, 200_000))
counts = np.bincount(arrivals[arrivals < 50_000].astype(int))
assert abs(counts.mean() - lam) < 0.03 and abs(counts.var() - lam) < 0.1   # Poisson: mean = variance

# Beta-binomial updating: prior Beta(2, 3), observe 7 successes in 20.
# Multiply prior by likelihood on a grid, normalise, and compare with the closed form.
grid = np.linspace(0.0005, 0.9995, 1000)
unnorm = stats.beta(2, 3).pdf(grid) * stats.binom(20, grid).pmf(7)
numeric = unnorm / (unnorm.sum() * (grid[1] - grid[0]))
post = stats.beta(2 + 7, 3 + 13)
assert np.allclose(numeric, post.pdf(grid), atol=1e-3 * post.pdf(grid).max())

# sums of independent normals: N(1, 2^2) + N(-3, 1.5^2) is N(-2, 2.5^2)
total = rng.normal(1, 2, 1_000_000) + rng.normal(-3, 1.5, 1_000_000)
assert stats.kstest(total, stats.norm(-2, 2.5).cdf).statistic < 0.002
```

## Averages of many draws
Two theorems explain why averages behave so well.

- The **law of large numbers:** the sample mean of independent, identically distributed draws with a finite mean converges to the true mean as \(n\) grows. This is why a test-set accuracy means anything.
- The **central limit theorem:** for independent, identically distributed draws with finite variance \(\sigma^2\), the sample mean is approximately Normal\((\mu, \sigma^2 / n)\) for large \(n\), whatever the shape of the original distribution. This is where confidence intervals of the form \(\bar{x} \pm 1.96\,\sigma / \sqrt{n}\) come from.

"Large" depends on the skew. Means of a symmetric, light-tailed variable look normal after a handful of draws; means of a heavily skewed one, like revenue per user, need many more. [Coin flips and the normal approximation](../coin-flips-normal-approximation/) works through the binomial case by hand.

```python
# means of 50 exponential draws (very skewed) are close to normal; a 95% interval covers about 95% of the time
size, reps = 50, 100_000
draws = rng.exponential(1.0, (reps, size))               # mu = sigma = 1
means = draws.mean(axis=1)
z = (means - 1.0) / (1.0 / np.sqrt(size))
coverage = (np.abs(z) < 1.96).mean()
assert 0.93 < coverage < 0.96
assert abs(stats.skew(draws[:, 0]) - 2.0) < 0.1                # one exponential draw has skewness 2
assert abs(stats.skew(means) - 2.0 / np.sqrt(size)) < 0.03     # the mean of n draws has skewness 2 / sqrt(n)
```

## Pitfalls
- **Flipping the conditional.** \(P(\text{positive} \mid \text{disease})\) is not \(P(\text{disease} \mid \text{positive})\). Precision and recall are the same pair under other names: precision is \(P(y = 1 \mid \hat{y} = 1)\), recall is \(P(\hat{y} = 1 \mid y = 1)\), and Bayes' rule links them through the base rate and the rate of positive predictions, precision = recall \(\times P(y = 1) / P(\hat{y} = 1)\).
- **Assuming independence because the story sounds unrelated.** Clicks from the same user, items on the same page and days in the same week are correlated. Multiplying their probabilities, or dropping a positive covariance from a variance, makes uncertainty look smaller than it is.
- **Reading a density as a probability.** A PDF value of 3 is not an error, and a PDF value is not the chance of that exact point.
- **Off-by-one in discrete tails.** "At least 5" is \(1 - F(4)\), not \(1 - F(5)\). In `scipy` that is `sf(4)`.
- **Mixed-up parameters.** `stats.norm` takes the standard deviation as `scale`, not the variance. `stats.expon` takes `scale = 1 / rate`. `stats.geom` counts trials including the success (starting at 1), while some texts count failures before it (starting at 0), which shifts the mean by one.
- **Pushing a function through a mean.** The average of \(\log X\) is not the log of the average of \(X\), and the loss of an averaged prediction is not the average of the losses. Jensen's inequality says which way the gap goes.
- **Trusting the normal approximation in the tails.** The CLT is about the bulk of the distribution. For rare events, small \(n p\), or heavily skewed data, use the exact distribution or simulate.

```python
# precision = recall * P(y = 1) / P(y_hat = 1), on a simulated classifier
y = rng.random(500_000) < 0.05
y_hat = np.where(y, rng.random(y.size) < 0.8, rng.random(y.size) < 0.03)
precision, recall = y[y_hat].mean(), y_hat[y].mean()
assert np.isclose(precision, recall * y.mean() / y_hat.mean())
assert precision < 0.6 < recall                                  # same classifier, very different numbers

# "at least 5" is sf(4), not sf(5)
bn = stats.binom(10, 0.3)
assert np.isclose(bn.sf(4), bn.pmf(np.arange(5, 11)).sum())
assert not np.isclose(bn.sf(5), bn.pmf(np.arange(5, 11)).sum())

# scipy's geometric starts at 1; shifting by loc=-1 counts failures and moves the mean, not the variance
g1, g0 = stats.geom(0.3), stats.geom(0.3, loc=-1)
assert g1.pmf(0) == 0 and np.isclose(g0.pmf(0), 0.3)
assert np.isclose(g1.mean() - g0.mean(), 1) and np.isclose(g1.var(), g0.var())

# Jensen with the concave log: the mean of the logs sits below the log of the mean
revenue = rng.lognormal(0.0, 1.0, 1_000_000)
assert np.log(revenue).mean() < np.log(revenue.mean()) - 0.4    # gap is sigma^2 / 2 = 0.5 for a lognormal
```

For worked problems that put these tools to use, see [is this coin fair?](../is-this-coin-fair/) and the [ML fundamentals quick reference](../ml-fundamentals-quick-reference/).
