# Bernoulli and binomial: one yes-or-no, then many

An ad gets 40 clicks in 1,000 impressions: a click-through rate of 4%, with a rough 95% interval from 2.8% to 5.2%, wide enough that a 4.5% variant could be no better. I get that interval from the binomial. Much of the data an ML system sees is yes-or-no like this: did the user click, did the request fail, is the label positive. One such outcome is a **Bernoulli** random variable. Count how many came out "yes" across a fixed number of independent tries and you have a **binomial** random variable. A click-through rate, an accuracy on a test set and the binary cross-entropy loss are all statements about them.

This post defines both, derives their mean and variance without grinding through sums, checks every formula in code, then works through examples and the assumptions that fail most often in practice.

## Bernoulli: a single trial
A Bernoulli variable \(X \sim \text{Bernoulli}(p)\) takes the value 1 ("success") with probability \(p\) and 0 with probability \(1 - p\). Its probability mass function (PMF) fits on one line:

\[ P(X = k) = p^k (1 - p)^{1 - k}, \qquad k \in \{0, 1\}. \]

Plug in \(k = 1\) and you get \(p\); plug in \(k = 0\) and you get \(1 - p\). The compact form is worth remembering because its logarithm, \(k \log p + (1 - k)\log(1 - p)\), is exactly the negative of the binary cross-entropy loss. Training a logistic regression is fitting a Bernoulli model; [regression as maximum likelihood](../regression-as-maximum-likelihood/) and [cross-entropy vs MSE](../cross-entropy-vs-mse/) follow that thread.

The mean is one line too: \(\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1 - p) = p\). For the variance, use \(\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2\) and one trick: since \(X\) is 0 or 1, \(X^2 = X\), so \(\mathbb{E}[X^2] = p\) and

\[ \text{Var}(X) = p - p^2 = p(1 - p). \]

This parabola is zero at \(p = 0\) and \(p = 1\), where the outcome is certain, and peaks at \(1/4\) when \(p = 1/2\). A fair coin is the hardest binary outcome to predict, and it is also the one with the most entropy (see [entropy and information](../entropy-and-information/)).

```python
import math
import itertools
import numpy as np
from scipy import stats


def bernoulli_pmf(k, p):
    return p ** k * (1 - p) ** (1 - k)


for p in (0.0, 0.1, 0.5, 0.93, 1.0):
    mean = sum(k * bernoulli_pmf(k, p) for k in (0, 1))
    var = sum((k - mean) ** 2 * bernoulli_pmf(k, p) for k in (0, 1))
    assert math.isclose(mean, p) and math.isclose(var, p * (1 - p), abs_tol=1e-15)
    assert math.isclose(var, stats.bernoulli.var(p), abs_tol=1e-15)

ps = np.linspace(0, 1, 1001)
assert ps[np.argmax(ps * (1 - ps))] == 0.5          # the variance peaks at a fair coin
assert (ps * (1 - ps)).max() == 0.25
```

## Binomial: counting successes
Now run \(n\) Bernoulli trials that are **independent** and share the **same** \(p\), and let \(X\) be the number of successes. Then \(X \sim \text{Binomial}(n, p)\). Both conditions matter, and the second is easy to forget: a count of independent trials with different success probabilities is not binomial (it has its own name, the Poisson binomial, and its variance \(\sum_i p_i(1 - p_i)\) is never larger than the binomial's \(n\bar p(1 - \bar p)\) at the average rate \(\bar p\)).

To get the PMF, take any particular sequence with \(k\) successes and \(n - k\) failures, say \(1, 1, 0, 1, 0\). By independence its probability is the product of its parts, \(p^k (1 - p)^{n - k}\), whatever the order. There are \(\binom{n}{k}\) such sequences, one per choice of which \(k\) positions hold the successes, and they are disjoint events, so

\[ P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \qquad k = 0, 1, \dots, n. \]

By the binomial theorem these sum to \((p + (1 - p))^n = 1\), which is where the distribution gets its name.

The mean and variance need no sums at all. Write \(X = X_1 + \dots + X_n\) with each \(X_i \sim \text{Bernoulli}(p)\). Expectation is linear whether or not the \(X_i\) are independent, and variances add when they are independent:

\[ \mathbb{E}[X] = np, \qquad \text{Var}(X) = np(1 - p). \]

The code checks the PMF against the most literal definition there is: list all \(2^n\) outcome sequences, add up the probability of those with \(k\) successes, and compare.

```python
def binomial_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


n, p = 10, 0.3
brute = [0.0] * (n + 1)
for seq in itertools.product((0, 1), repeat=n):       # all 1024 outcome sequences
    brute[sum(seq)] += math.prod(bernoulli_pmf(x, p) for x in seq)

pmf = [binomial_pmf(k, n, p) for k in range(n + 1)]
assert np.allclose(pmf, brute) and np.allclose(pmf, stats.binom.pmf(range(n + 1), n, p))
assert math.isclose(sum(pmf), 1.0)

ks = np.arange(n + 1)
mean = (ks * pmf).sum()
var = ((ks - mean) ** 2 * pmf).sum()
assert math.isclose(mean, n * p) and math.isclose(var, n * p * (1 - p))

# the mode is floor((n + 1) p), with a tie one below it when (n + 1) p is a whole number
for nn in range(1, 30):
    for pp in np.linspace(0.01, 0.99, 99):
        probs = stats.binom.pmf(np.arange(nn + 1), nn, pp)
        mode = math.floor((nn + 1) * pp)
        assert math.isclose(probs[mode], probs.max())
assert math.isclose(binomial_pmf(3, 9, 0.4), binomial_pmf(4, 9, 0.4))   # (n + 1) p = 4: counts 3 and 4 tie

# different p per trial (Poisson binomial): never more spread than the binomial at the average p
p_i = np.array([0.05, 0.2, 0.5, 0.7, 0.95])
assert (p_i * (1 - p_i)).sum() <= len(p_i) * p_i.mean() * (1 - p_i.mean())

# and by simulation: the sum of n Bernoulli draws behaves the same way
rng = np.random.default_rng(0)
sums = (rng.random((200_000, n)) < p).sum(axis=1)
assert abs(sums.mean() - n * p) < 0.01 and abs(sums.var() - n * p * (1 - p)) < 0.02
```

For \(0 < p < 1\), the most likely count, the mode, is \(\lfloor (n + 1)p \rfloor\) (when \((n + 1)p\) is a whole number, that value and the one below it tie). The shape is symmetric only at \(p = 1/2\); for small \(p\) it piles up near zero with a long right tail.

## The proportion, and how many trials you need
Usually you care about the fraction of successes \(\hat p = X / n\) rather than the count. Dividing by \(n\) gives

\[ \mathbb{E}[\hat p] = p, \qquad \text{SE}(\hat p) = \sqrt{\frac{p(1 - p)}{n}}. \]

\(\hat p\) is also the maximum likelihood estimate: the log-likelihood \(k \log p + (n - k) \log(1 - p)\) has derivative \(k/p - (n - k)/(1 - p)\), which is zero at \(p = k/n\). The more important fact is the \(\sqrt{n}\) in the denominator. Halving the error bar takes four times the data.

**Worked example: click-through rate.** An ad gets 40 clicks in 1,000 impressions, so \(\hat p = 0.04\) with a standard error of about \(0.0062\). A rough 95% interval is \(0.04 \pm 1.96 \times 0.0062\), or 2.8% to 5.2%: wide enough that a 4.5% variant could easily be no better. To pin the rate to within half a percentage point at 95% confidence, solve \(1.96 \sqrt{p(1-p)/n} \le 0.005\) for \(n\), which comes to about 5,900 impressions. For proper intervals and tests, [A/B testing](../ab-testing/) and [beta uncertainty](../beta-uncertainty/) go further.

```python
k_clicks, impressions = 40, 1000
p_hat = k_clicks / impressions
se = math.sqrt(p_hat * (1 - p_hat) / impressions)
assert round(se, 4) == 0.0062
low, high = p_hat - 1.96 * se, p_hat + 1.96 * se
assert round(low, 3) == 0.028 and round(high, 3) == 0.052

needed = math.ceil(1.96 ** 2 * p_hat * (1 - p_hat) / 0.005 ** 2)
assert needed == 5901

# the MLE: maximize the log-likelihood over a fine grid of p
grid = np.linspace(0.001, 0.999, 9981)
loglik = k_clicks * np.log(grid) + (impressions - k_clicks) * np.log(1 - grid)
assert math.isclose(grid[np.argmax(loglik)], p_hat)

# four times the data halves the standard error
assert math.isclose(math.sqrt(p_hat * (1 - p_hat) / (4 * impressions)), se / 2)
```

## More worked examples
**Majority-vote ensemble.** Five classifiers each label an example correctly with probability 0.7, and their mistakes are independent. The ensemble is right when at least three of them are, which is a binomial tail: \(P(X \ge 3)\) for \(X \sim \text{Binomial}(5, 0.7)\), about 0.837. With 25 such voters it reaches 0.983. This is the arithmetic behind bagging and random forests, and the catch is in the word "independent": models trained on the same data make correlated mistakes, so the real gain is much smaller. [From decision trees to boosting](../decision-trees-to-boosting/) covers how forests try to decorrelate their trees.

**At least one failure.** A request fans out to 20 services, each succeeding independently with probability 0.999. The number of failures is \(\text{Binomial}(20, 0.001)\), and the chance that the request sees at least one is \(1 - 0.999^{20} \approx 2\%\). Per-service reliability that looks excellent compounds into a user-visible failure rate nearly twenty times larger.

**Is this coin fair?** A coin lands heads 60 times in 100 flips. If it were fair, the chance of 60 or more heads is \(P(X \ge 60) \approx 0.028\) under \(\text{Binomial}(100, 0.5)\); counting the equally extreme low side as well gives a two-sided p-value of about 0.057. The exact binomial tail answers the question directly, no approximation needed. [Is this coin fair?](../is-this-coin-fair/) and [p-values](../p-values/) take this further.

```python
def tail_at_least(k, n, p):
    return sum(binomial_pmf(j, n, p) for j in range(k, n + 1))


assert round(tail_at_least(3, 5, 0.7), 3) == 0.837
assert round(tail_at_least(13, 25, 0.7), 3) == 0.983
assert math.isclose(tail_at_least(3, 5, 0.7), stats.binom.sf(2, 5, 0.7))

# correlated voters: the same five models, but 60% of the time they all copy one shared vote
trials = 200_000
shared = rng.random(trials) < 0.7
own = rng.random((trials, 5)) < 0.7
copy = rng.random(trials) < 0.6
votes = np.where(copy[:, None], shared[:, None], own)
assert abs(votes.mean() - 0.7) < 0.005                # each model is still 70% accurate
assert (votes.sum(axis=1) >= 3).mean() < 0.78         # but the ensemble gains far less

assert round(1 - 0.999 ** 20, 4) == 0.0198
assert math.isclose(1 - 0.999 ** 20, 1 - stats.binom.pmf(0, 20, 0.001))

one_sided = tail_at_least(60, 100, 0.5)
assert round(one_sided, 3) == 0.028
assert math.isclose(2 * one_sided, stats.binomtest(60, 100, 0.5).pvalue)
```

In the simulation, letting the five voters copy one shared vote 60% of the time leaves each 70% accurate but keeps the majority below 0.78, against 0.837 for independent voters.

## Its neighbours
The binomial sits in the middle of a small family, and knowing the links saves a lot of re-deriving.

| relative | relation to the binomial |
|---|---|
| Bernoulli | the binomial with \(n = 1\) |
| Categorical, multinomial | more than two outcomes per trial; the multinomial counts each outcome over \(n\) trials |
| Geometric | the number of trials up to the first success, instead of successes in a fixed number of trials |
| Poisson | the limit as \(n \to \infty\) and \(p \to 0\) with \(np = \lambda\) fixed: rare events over many chances |
| Normal | the approximation when \(np(1 - p)\) is large, by the central limit theorem |
| Beta | the conjugate prior for \(p\): a Beta prior plus binomial data gives a Beta posterior |

The two approximations are worth seeing numerically. With \(n = 1000\) and \(p = 0.002\), the binomial and a Poisson with \(\lambda = 2\) differ by less than 0.0003 at every count. With \(n = 100\) and \(p = 0.5\), the normal with mean 50 and standard deviation 5 gets \(P(X \le 55)\) to within 0.0001 once you evaluate it at 55.5 (the **continuity correction**), and is off by more than 0.02 if you do not. See [Poisson](../poisson-distribution/), [the normal distribution](../normal-distribution/) and [coin flips and the normal approximation](../coin-flips-normal-approximation/).

```python
k = np.arange(0, 40)
assert np.abs(stats.binom.pmf(k, 1000, 0.002) - stats.poisson.pmf(k, 2)).max() < 3e-4

exact = stats.binom.cdf(55, 100, 0.5)
assert abs(stats.norm.cdf(55.5, 50, 5) - exact) < 1e-4   # with continuity correction
assert abs(stats.norm.cdf(55, 50, 5) - exact) > 0.02     # without it
```

## Pitfalls
- **Independence is an assumption, not a fact.** Clicks from one user are correlated, and so are labels from one annotator and failures during one outage. If you treat 10,000 clicks from 200 users as 10,000 independent trials, the standard error will be too small, sometimes by a lot. Count trials at the level where independence plausibly holds, or use a model with extra variance.
- **One shared success rate is an assumption too.** If each user has their own click rate, then the per-user click counts are not draws from one binomial at the average rate: each is a binomial with a different \(p\), so as a group they are a mixture, and they spread out more than \(mp(1 - p)\) for \(m\) impressions per user would suggest. This is called **overdispersion**. The beta-binomial model, where each user's \(p\) is itself drawn from a Beta distribution, is the standard fix; the code below shows the variance inflating.
- **The normal approximation fails near 0 and 1.** With \(n = 1000\) and \(p = 0.001\) you expect one success; a symmetric interval \(\hat p \pm 1.96\,\text{SE}\) can dip below zero and is badly calibrated. Use the exact binomial tail, a Wilson interval or a Beta posterior.
- **The trial count must be fixed in advance.** The binomial assumes \(n\) was chosen before looking. Flipping until the result looks significant and then computing a binomial p-value gives a wrong answer, a problem [A/B testing](../ab-testing/) runs into as "peeking".
- **Large counts break naive arithmetic.** \(\binom{2000}{1000}\) does not fit in a float and \(0.5^{2000}\) underflows to zero, so the textbook formula returns nonsense. Work in log space with the log-gamma function, or call `scipy.stats.binom.logpmf`.

```python
# overdispersion: each of 2,000 users has their own click rate drawn from Beta(2, 48), mean 0.04
m = 50                                               # impressions per user
a, b = 2.0, 48.0
user_p = rng.beta(a, b, size=2000)
clicks = rng.binomial(m, user_p)
p_bar = a / (a + b)
binom_var = m * p_bar * (1 - p_bar)
rho = 1 / (a + b + 1)                                 # beta-binomial: variance inflated by 1 + (m - 1) rho
assert math.isclose(stats.betabinom.var(m, a, b), binom_var * (1 + (m - 1) * rho))
assert clicks.var() > 1.5 * binom_var                # the plain binomial understates the spread

# large n: the naive formula fails, log space does not
with np.errstate(all="ignore"):
    assert 0.5 ** 2000 == 0.0
try:
    float(math.comb(2000, 1000))
    raise AssertionError("expected overflow")
except OverflowError:
    pass


def binomial_logpmf(k, n, p):
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_comb + k * math.log(p) + (n - k) * math.log(1 - p)


assert math.isclose(binomial_logpmf(1000, 2000, 0.5), stats.binom.logpmf(1000, 2000, 0.5))
assert round(math.exp(binomial_logpmf(1000, 2000, 0.5)), 4) == 0.0178
```

## Summary
- **Bernoulli:** one binary outcome with success probability \(p\); mean \(p\), variance \(p(1 - p)\), largest at \(p = 1/2\). Its negative log-likelihood is the binary cross-entropy loss.
- **Binomial:** the number of successes in \(n\) independent trials with the same \(p\); PMF \(\binom{n}{k} p^k (1-p)^{n-k}\), mean \(np\), variance \(np(1 - p)\).
- **The proportion** \(k/n\) estimates \(p\) with standard error \(\sqrt{p(1-p)/n}\): four times the data for half the error.
- **Check the assumptions** before trusting the error bars: independent trials, one shared \(p\), and \(n\) fixed in advance.

These simulations build in their own departures from the binomial: voters that copy one shared vote 60% of the time, and users whose click rates come from one Beta distribution with mean 0.04. So they show the direction of each effect, not its size on real data. How correlated real trials are stays open, and the binomial cannot answer it; counting at the level where independence plausibly holds, or a model with extra variance such as the beta-binomial, is where to go next.

The next posts in this group cover the Poisson, exponential and normal distributions, and the [probability cheatsheet](../probability-cheatsheet/) collects all of them on one page. The derivations here are standard textbook material; Joseph Blitzstein and Jessica Hwang's Introduction to Probability (2nd edition, 2019) treats them, and the idea of reading each distribution as a story about how the data was generated, in much more depth.
