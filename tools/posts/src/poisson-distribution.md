---
title: The Poisson distribution: counting events in a window
summary: Where the Poisson comes from as the limit of many unlikely trials, why its mean and variance are both λ, how it splits, merges and links to exponential waiting times, and how to fit it and tell when real counts break it, each claim checked in code.
group: Probability
order: 3
tags: Poisson · counts · probability distributions · overdispersion
---
Errors logged per hour, clicks on an ad per day, typos per page, goals per football match. Each is a count of events in a fixed window, where the events are individually unlikely but there are many chances for one to happen. The Poisson distribution is the default model for counts like these. It has one parameter, the average rate \(\lambda\), and a single formula:

\[ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k = 0, 1, 2, \dots \]

This post derives that formula from the [binomial](../bernoulli-binomial/), works out its mean and variance, shows the properties that make it convenient (rates scale, Poissons add, thinned Poissons stay Poisson), then covers fitting it and the ways real counts fail to be Poisson.

## The limit of many unlikely trials
Split an hour into \(n\) tiny slots. In each slot an error happens with probability \(p\), independently, and never twice in the same slot. The number of errors is \(\text{Bin}(n, p)\) with mean \(np\). Now make the slots finer while keeping the average fixed at \(\lambda = np\), so \(p = \lambda / n\). For a fixed \(k\):

\[ \binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1-\frac{\lambda}{n}\right)^{n-k} = \frac{\lambda^k}{k!} \cdot \underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to\, 1} \cdot \underbrace{\left(1-\frac{\lambda}{n}\right)^{-k}}_{\to\, 1} \cdot \underbrace{\left(1-\frac{\lambda}{n}\right)^{n}}_{\to\, e^{-\lambda}} . \]

The first bracket is a product of \(k\) factors \((n - i)/n\), each tending to 1, and \(k\) does not grow. The second has a fixed exponent and a base tending to 1. For the third, take logs and use \(\ln(1 - x) = -x - x^2/2 - \cdots\):

\[ n \ln\left(1 - \frac{\lambda}{n}\right) = -\lambda - \frac{\lambda^2}{2n} - \cdots \;\longrightarrow\; -\lambda . \]

So the binomial probability tends to \(\lambda^k e^{-\lambda} / k!\). The same limit follows from L'Hôpital's rule on \(\ln(1 - \lambda/n) / (1/n)\): the derivatives in \(n\) are \(\frac{\lambda/n^2}{1 - \lambda/n}\) and \(-1/n^2\), whose ratio is \(-\lambda / (1 - \lambda/n) \to -\lambda\).

How good is the approximation at finite \(n\)? Le Cam's inequality bounds the total variation distance between a sum of independent Bernoulli\((p_i)\) variables and a Poisson with the same mean by \(\sum_i p_i^2\), which here is \(\lambda^2 / n\). The code checks the formula against scipy and the bound at several \(n\):

```python
import math
import numpy as np
from scipy import stats


def poisson_pmf(k, lam):
    return lam ** k * math.exp(-lam) / math.factorial(k)


for lam in (0.5, 3.0, 12.0):
    for k in range(40):
        assert math.isclose(poisson_pmf(k, lam), stats.poisson.pmf(k, lam), rel_tol=1e-9)
    assert math.isclose(sum(poisson_pmf(k, lam) for k in range(100)), 1.0)    # sums to one


def tv_binomial_vs_poisson(n, lam):
    k = np.arange(0, 200)
    return 0.5 * np.abs(stats.binom.pmf(k, n, lam / n) - stats.poisson.pmf(k, lam)).sum()


lam = 3.0
gaps = [tv_binomial_vs_poisson(n, lam) for n in (10, 100, 1000, 10000)]
for n, tv in zip((10, 100, 1000, 10000), gaps):
    assert tv <= lam ** 2 / n                                    # Le Cam's bound
assert all(a > 5 * b for a, b in zip(gaps, gaps[1:]))           # roughly 10x closer per 10x n
print([round(g, 5) for g in gaps])
```

The distance falls about tenfold for every tenfold increase in \(n\): 0.086 at \(n = 10\), 0.0076 at \(n = 100\). This is why the Poisson is a good stand-in for a binomial whenever \(n\) is large and \(p\) small, without either being known separately.

## Mean and variance are both λ
The mean follows from pulling one \(\lambda\) out of the sum and re-indexing:

\[ \mathbb{E}[X] = \sum_{k \ge 1} k \frac{\lambda^k e^{-\lambda}}{k!} = \lambda \sum_{k \ge 1} \frac{\lambda^{k-1} e^{-\lambda}}{(k-1)!} = \lambda . \]

The same trick on \(k(k-1)\) gives \(\mathbb{E}[X(X-1)] = \lambda^2\), so

\[ \operatorname{Var}(X) = \mathbb{E}[X(X-1)] + \mathbb{E}[X] - \mathbb{E}[X]^2 = \lambda^2 + \lambda - \lambda^2 = \lambda . \]

A shortcut sometimes seen is to take the binomial's \(np(1 - p) = \lambda(1 - \lambda/n)\) and let \(n \to \infty\). It lands on the right answer, but swapping a limit with an expectation needs a justification that the direct sum above does not.

Mean equal to variance is the Poisson's fingerprint. The standard deviation is \(\sqrt{\lambda}\), so relative noise \(1/\sqrt{\lambda}\) shrinks as counts grow: a window that averages 4 events swings by about 50% from window to window, one that averages 400 by about 5%. For large \(\lambda\) the shape is close to a [normal](../normal-distribution/) with mean and variance \(\lambda\). In the same regime \(\sqrt{X}\) has variance close to \(1/4\), whatever the rate: by the delta method, \(\operatorname{Var}\sqrt{X} \approx \lambda \cdot \left(\frac{1}{2\sqrt{\lambda}}\right)^2 = \frac{1}{4}\). That is why square roots are the usual way to stabilise count data before plotting or averaging. The approximation needs \(\lambda\) of roughly 10 or more; at \(\lambda = 1\) the variance of \(\sqrt{X}\) is about 0.40.

```python
k = np.arange(0, 400)
for lam in (0.7, 4.0, 25.0):
    p = stats.poisson.pmf(k, lam)
    mean = (k * p).sum()
    assert math.isclose(mean, lam)
    assert math.isclose((k * (k - 1) * p).sum(), lam ** 2)                 # factorial moment
    assert math.isclose(((k - mean) ** 2 * p).sum(), lam)                 # variance = mean

rng = np.random.default_rng(0)
for lam in (20.0, 80.0, 300.0):
    x = rng.poisson(lam, 200_000)
    assert abs(np.sqrt(x).var() - 0.25) < 0.01                            # sqrt stabilises variance
assert np.sqrt(rng.poisson(1.0, 200_000)).var() > 0.35                    # but not for small lambda
```

## Rates scale, Poissons add and split
Three properties come straight from the "many tiny independent slots" picture, and they are what make the Poisson easy to work with.

1. **Rates scale with the window.** If events arrive at rate \(\lambda\) per hour, the count in \(t\) hours is \(\text{Poisson}(\lambda t)\).
2. **Independent Poissons add.** \(\text{Poisson}(a) + \text{Poisson}(b) = \text{Poisson}(a + b)\). Errors from two independent services, merged, are still Poisson.
3. **Thinning.** If each event is kept independently with probability \(q\), the kept count is \(\text{Poisson}(q\lambda)\), and the kept and dropped counts are independent. Ad impressions are Poisson, each is clicked with probability 2%, so clicks are Poisson at 2% of the impression rate.

Underneath all three is the Poisson process: events in disjoint windows are independent and the rate is constant. Its gaps between events are [exponential](../exponential-distribution/) with mean \(1/\lambda\), and that gives a second way to simulate Poisson counts: draw exponential gaps, lay them end to end, and count how many land in each unit window.

```python
# superposition: convolving two Poisson pmfs gives the Poisson with the summed rate
a, b = 2.5, 4.0
k = np.arange(0, 60)
conv = np.convolve(stats.poisson.pmf(k, a), stats.poisson.pmf(k, b))[:60]
assert np.allclose(conv, stats.poisson.pmf(k, a + b))

# thinning: keep each of N ~ Poisson(lam) events with probability q
lam, q = 6.0, 0.3
n = np.arange(0, 120)
kept = [(stats.poisson.pmf(n, lam) * stats.binom.pmf(j, n, q)).sum() for j in range(30)]
assert np.allclose(kept, stats.poisson.pmf(np.arange(30), q * lam))
# kept and dropped are independent: the joint pmf factorises
for j in range(10):
    for m in range(10):
        joint = stats.poisson.pmf(j + m, lam) * stats.binom.pmf(j, j + m, q)
        assert math.isclose(joint, stats.poisson.pmf(j, q * lam) * stats.poisson.pmf(m, (1 - q) * lam))

# exponential gaps -> Poisson counts per unit window
rate, horizon = 3.0, 50_000
arrivals = np.cumsum(rng.exponential(1 / rate, int(rate * horizon * 1.1)))
counts = np.bincount(arrivals[arrivals < horizon].astype(int), minlength=horizon)
assert abs(counts.mean() - rate) < 0.05 and abs(counts.var() - rate) < 0.1
observed = np.bincount(counts, minlength=8)[:8] / horizon
assert np.allclose(observed, stats.poisson.pmf(np.arange(8), rate), atol=0.01)
two_unit = counts.reshape(-1, 2).sum(axis=1)                             # windows twice as long
assert abs(two_unit.mean() - 2 * rate) < 0.1 and abs(two_unit.var() - 2 * rate) < 0.3
```

## Worked examples
**Errors per hour.** A service logs 3 errors an hour on average. The chance of a clean hour is \(e^{-3} \approx 0.050\), and the chance of 6 or more errors is about 0.084, so an alert at "6 per hour" would fire roughly twice a day on a perfectly healthy service. Over an 8-hour shift the count is \(\text{Poisson}(24)\).

**Rare defects.** A batch has 2000 parts, each defective with probability 0.001. The exact count is \(\text{Bin}(2000, 0.001)\); the Poisson with \(\lambda = 2\) gives the probability of more than 4 defects as 0.0527 against the exact 0.0526.

```python
p_clean = stats.poisson.pmf(0, 3)
p_alert = stats.poisson.sf(5, 3)                       # P(X >= 6) = 1 - P(X <= 5)
assert round(p_clean, 3) == 0.050 and round(p_alert, 3) == 0.084
assert 1.5 < 24 * p_alert < 2.5                         # about two false alerts a day

exact, approx = stats.binom.sf(4, 2000, 0.001), stats.poisson.sf(4, 2.0)
assert round(exact, 4) == 0.0526 and round(approx, 4) == 0.0527
```

## Fitting λ and modelling rates
Given counts \(x_1, \dots, x_n\) the log-likelihood is \(\sum_i (x_i \ln\lambda - \lambda) + \text{const}\), whose derivative \(\sum_i x_i / \lambda - n\) is zero at \(\hat\lambda = \bar x\): the [maximum-likelihood estimate](../statistical-estimation-mle-map/) is the sample mean, with standard error \(\sqrt{\hat\lambda / n}\). If windows have different lengths \(t_i\), it becomes total events over total exposure, \(\sum x_i / \sum t_i\).

When the rate depends on features, Poisson regression models \(\log \lambda_i = w^\top x_i\). The log link keeps rates positive and makes effects multiplicative, and the loss is the negative log-likelihood above, in the same spirit as treating [regression as maximum likelihood](../regression-as-maximum-likelihood/). scikit-learn's `PoissonRegressor` fits it; note that it applies an L2 penalty by default, switched off here.

```python
from scipy import optimize
from sklearn.linear_model import PoissonRegressor

x = rng.poisson(4.2, 5000)
nll = lambda lam: -(x * np.log(lam) - lam).sum()
best = optimize.minimize_scalar(nll, bounds=(0.1, 20), method="bounded")
assert abs(best.x - x.mean()) < 1e-3                                # MLE = sample mean

features = rng.uniform(0, 1, (20_000, 2))
true_w, true_b = np.array([1.2, -0.7]), 0.5
y = rng.poisson(np.exp(features @ true_w + true_b))
model = PoissonRegressor(alpha=0.0, max_iter=300).fit(features, y)
assert np.allclose(model.coef_, true_w, atol=0.05) and abs(model.intercept_ - true_b) < 0.05
```

## When counts are not Poisson
The assumptions are strong, and real counts break them in recognisable ways.

- **Overdispersion.** The variance is larger than the mean. This is the usual case, and it happens whenever the rate itself varies between units: some users are heavy clickers, some days are busy. If each unit's rate is drawn from a gamma distribution, the counts are negative binomial, with variance \(\mu + \mu^2 / r\). A Poisson model fitted to such data gets the mean right but its standard errors too small, so effects look more significant than they are. Check the ratio of variance to mean before trusting a Poisson fit.
- **Excess zeros.** Many units can never produce an event (a user who never opens the app). A zero-inflated or hurdle model separates "can it happen" from "how often".
- **A rate that changes over time.** Traffic by hour of day is not one Poisson; model the rate as a function of time, or condition on the hour.
- **Bursts.** Events that trigger more events (retries after an outage, cascading alerts) violate independence, and the counts come in clumps.
- **Bounded counts with large \(p\).** Correct answers out of 10 questions are binomial, not Poisson; the Poisson approximation needs small \(p\).

```python
mu, r = 4.0, 2.0
rates = rng.gamma(shape=r, scale=mu / r, size=200_000)            # rate varies per unit
mixed = rng.poisson(rates)
assert abs(mixed.mean() - mu) < 0.05
assert abs(mixed.var() - (mu + mu ** 2 / r)) < 0.3                # variance 12, not 4
assert mixed.var() / mixed.mean() > 2.5                           # overdispersed

nb = stats.nbinom(r, r / (r + mu))                                # the gamma-Poisson mixture
observed = np.bincount(mixed, minlength=15)[:15] / mixed.size
assert np.allclose(observed, nb.pmf(np.arange(15)), atol=0.005)
assert not np.allclose(observed, stats.poisson.pmf(np.arange(15), mu), atol=0.02)
```

## Summary
- Use the Poisson for counts of independent events in a window at a steady rate, or as an approximation to a binomial with large \(n\) and small \(p\).
- Its mean and variance are both \(\lambda\); the MLE of \(\lambda\) is the sample mean, or events over exposure.
- Rates scale with the window, independent Poissons add, and random thinning keeps it Poisson. The gaps between events are exponential.
- Before relying on it, compare variance to mean. If the variance is clearly larger, reach for the negative binomial or a model of how the rate varies.

The binomial-limit derivation is the standard textbook argument, laid out in many places including [this Mathematics Stack Exchange answer](https://math.stackexchange.com/a/836585); it is rewritten here with new examples and code. The approximation bound is L. Le Cam, "An approximation theorem for the Poisson binomial distribution", Pacific Journal of Mathematics, 1960.
