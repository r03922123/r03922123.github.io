# MLE and MAP: estimating a parameter from data

A model has a knob: the bias of a coin, the mean of a measurement, the rate of a queue. You have data. Which setting of the knob should you believe? Two answers cover most of machine learning. **Maximum likelihood (MLE)** picks the setting under which the observed data were most probable. **Maximum a posteriori (MAP)** does the same after first weighting each setting by how plausible you thought it was before seeing anything. Almost every loss function you train with is one or the other in disguise.

This post builds both from the definitions, works the standard examples, and checks every claim numerically, including two that are often stated wrong: that the MLE is unbiased, and that MAP is "MLE plus a prior" with no other consequences.

## The likelihood is a function of the parameter
Suppose \(x_1, \dots, x_n\) are drawn independently from a distribution with density (or mass function) \(p(x \mid \theta)\). Independence turns the joint probability into a product, and the **likelihood** is that product read as a function of \(\theta\) with the data held fixed:

\[ L(\theta) = \prod_{i=1}^{n} p(x_i \mid \theta), \qquad \ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln p(x_i \mid \theta). \]

The reading matters. As a function of \(x\), \(p(x \mid \theta)\) is a probability distribution. As a function of \(\theta\), it is not: it need not integrate to 1, and "the likelihood of \(\theta = 0.3\)" is not "the probability that \(\theta = 0.3\)". Turning it into a probability over \(\theta\) needs a prior, which is exactly the step MAP takes.

The logarithm is not optional decoration. It turns a product into a sum, which is easier to differentiate, and it keeps numbers representable. The coin example below multiplies two thousand factors of roughly 0.3 and 0.7; the product is around \(10^{-530}\), far below the smallest positive double (about \(10^{-324}\)), so it underflows to zero, while the sum of the logs is an ordinary number near \(-1200\). Because \(\ln\) is increasing, both have the same maximizer.

## The recipe
1. Write down \(\ell(\theta)\).
2. Set its derivative (the **score**) to zero and solve.
3. Check that the solution is a maximum. A negative second derivative (one parameter) or a negative definite Hessian (several) at the stationary point makes it a local maximum. Negative semi-definite is necessary but not enough, since a flat direction can hide a saddle. A local maximum is the global one when \(\ell\) is concave everywhere; otherwise compare the candidates. Also check the boundary of the parameter space, where the derivative need not vanish.

For a coin with heads probability \(p\), \(h\) heads and \(t\) tails give \(\ell(p) = h \ln p + t \ln(1-p)\). The score \(h/p - t/(1-p)\) is zero at \(\hat p = h / (h + t)\), and the second derivative \(-h/p^2 - t/(1-p)^2\) is negative everywhere, so \(\ell\) is concave and this is the global maximum. The code confirms it by brute force on a grid, and shows the raw product underflowing.

```python
import math
import numpy as np
from scipy import stats, optimize

rng = np.random.default_rng(0)
flips = rng.random(2000) < 0.3
h, t = int(flips.sum()), int((~flips).sum())

grid = np.linspace(0.0005, 0.9995, 1999)
raw = grid ** h * (1 - grid) ** t                  # the likelihood as a plain product
assert raw.max() == 0.0                            # underflows everywhere: useless
loglik = h * np.log(grid) + t * np.log(1 - grid)   # the same function, in log space
assert loglik.max() / math.log(10) < -500 and loglik.max() > -1300   # about 10^-530: a fine log, an impossible double
p_hat = h / (h + t)
assert abs(grid[loglik.argmax()] - p_hat) <= 0.0005
assert all(-h / p ** 2 - t / (1 - p) ** 2 < 0 for p in grid)   # concave: one global max
```

## Three examples where the answer surprises
**The Gaussian variance is biased.** For a normal sample, the recipe gives \(\hat\mu = \bar x\) and \(\hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar x)^2\), dividing by \(n\). Its expectation is \(\frac{n-1}{n}\sigma^2\): the sample mean is fitted to the same points, so they sit closer to it than to the true mean. With \(n = 5\) the MLE underestimates the variance by 20% on average. The familiar \(n - 1\) divisor fixes the bias, and is not the MLE.

**The uniform's endpoint has no zero derivative.** For data from \(\text{Uniform}(0, \theta)\), the likelihood is \(\theta^{-n}\) when \(\theta \ge \max_i x_i\) and zero otherwise, since a smaller \(\theta\) could not have produced the largest point. That function only decreases, so the score is never zero; the maximum sits on the edge, \(\hat\theta = \max_i x_i\). It is biased low, with mean \(\frac{n}{n+1}\theta\), and its spread shrinks like \(1/n\) rather than the usual \(1/\sqrt n\), because the support of the distribution depends on \(\theta\), which breaks the regularity assumptions behind the usual theory.

**Most models have no closed form.** The Gamma distribution's shape parameter has no algebraic MLE. You minimize the negative log-likelihood numerically, which is what every training loop does. Optimizing over \(\ln\) of the parameters keeps them positive without constraints, and is safe because of the invariance property below.

```python
# Gaussian: the MLE divides by n, and is biased by (n - 1) / n
n, sigma2 = 5, 4.0
samples = rng.normal(10.0, math.sqrt(sigma2), size=(200_000, n))
mle_var = samples.var(axis=1, ddof=0)
assert abs(mle_var.mean() - (n - 1) / n * sigma2) < 0.03
assert abs(samples.var(axis=1, ddof=1).mean() - sigma2) < 0.03
x = samples[0]
mu_fit, sd_fit = stats.norm.fit(x)                  # scipy's fit is the MLE
assert math.isclose(sd_fit ** 2, x.var(ddof=0))

# Uniform(0, theta): the maximum, biased low, spread shrinking like 1/n
theta = 3.0
for n in (10, 40):
    mx = rng.uniform(0, theta, size=(100_000, n)).max(axis=1)
    assert abs(mx.mean() - n / (n + 1) * theta) < 0.005
    if n == 10:
        sd10 = mx.std()
assert 3.0 < sd10 / mx.std() < 4.5                  # 4x the data: about 3.5x tighter (4x in the limit), not 2x
x = rng.uniform(0, theta, 25)
th = np.linspace(0.01, 5, 50_000)
lik = np.where(th >= x.max(), th ** -25.0, 0.0)
assert abs(th[lik.argmax()] - x.max()) < 1e-3       # the edge, not a stationary point

# Gamma: minimize the negative log-likelihood numerically
g = rng.gamma(shape=2.5, scale=1.5, size=3000)


def nll(log_params):
    k, s = np.exp(log_params)                       # optimize in log space: always positive
    return -stats.gamma.logpdf(g, a=k, scale=s).sum()


res = optimize.minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8})
k_hat, s_hat = np.exp(res.x)
k_ref, _, s_ref = stats.gamma.fit(g, floc=0)
assert abs(k_hat - k_ref) < 1e-3 and abs(s_hat - s_ref) < 1e-3
assert abs(k_hat - 2.5) < 0.2 and abs(s_hat - 1.5) < 0.15
```

## Why the MLE is the default
Bias aside, the MLE has three properties that make it the default in large samples.

- **Consistency.** As \(n\) grows, \(\hat\theta\) converges to the true value, under mild conditions. Both biased examples above are consistent: their bias and their variance both shrink to zero.
- **Asymptotic normality.** Under smoothness conditions, \(\hat\theta\) is approximately \(\mathcal{N}(\theta, 1 / (n I(\theta)))\), where the **Fisher information** \(I(\theta) = -\mathbb{E}\big[\partial^2 \ln p(x \mid \theta) / \partial\theta^2\big]\) measures how sharply the log-likelihood curves around the truth. For a coin, \(I(p) = 1 / (p(1-p))\), which gives the familiar standard error \(\sqrt{p(1-p)/n}\). Under the same regularity conditions, the Cramér–Rao bound says no unbiased estimator can have a smaller variance than \(1/(nI)\), so in large samples the MLE is about as precise as it gets. The uniform endpoint is outside those conditions, which is how its spread shrinks faster than the bound would allow.
- **Invariance.** The MLE of \(g(\theta)\) is \(g(\hat\theta)\). The MLE of a standard deviation is the square root of the MLE of the variance; the MLE of a log-odds is the logit of \(\hat p\). Reparametrizing never changes the answer.

```python
# asymptotic normality for the coin: sd matches sqrt(p (1 - p) / n), and Wald intervals cover ~95%
p, n = 0.3, 400
p_hats = rng.binomial(n, p, size=50_000) / n
assert abs(p_hats.std() - math.sqrt(p * (1 - p) / n)) < 3e-4
se = np.sqrt(p_hats * (1 - p_hats) / n)
coverage = np.mean(np.abs(p_hats - p) <= 1.96 * se)
assert 0.93 < coverage < 0.97

# Fisher information by its definition, as an average of minus the second derivative
xs = (rng.random(200_000) < p).astype(float)
second = -xs / p ** 2 - (1 - xs) / (1 - p) ** 2
assert abs(-second.mean() - 1 / (p * (1 - p))) < 0.05
```

## MAP: put the prior back
The Bayesian view treats \(\theta\) itself as uncertain, with a prior \(\pi(\theta)\). Bayes' rule gives the posterior, and MAP takes its peak:

\[ \pi(\theta \mid x) = \frac{L(\theta)\,\pi(\theta)}{p(x)} \;\propto\; L(\theta)\,\pi(\theta), \qquad \hat\theta_{\text{MAP}} = \arg\max_\theta \big[\ell(\theta) + \ln \pi(\theta)\big]. \]

The evidence \(p(x)\) does not depend on \(\theta\), so it drops out of the argmax, which is why MAP is cheap: no integral is needed. The log-prior is an extra term in the objective, which is the general shape of a **regularizer**.

**Coins and smoothing.** With a \(\text{Beta}(a, b)\) prior on \(p\), the posterior after \(h\) heads and \(t\) tails is \(\text{Beta}(a + h, b + t)\), and its mode is

\[ \hat p_{\text{MAP}} = \frac{h + a - 1}{n + a + b - 2}. \]

A uniform prior, \(\text{Beta}(1, 1)\), gives back the MLE. \(\text{Beta}(2, 2)\) gives \((h + 1)/(n + 2)\), add-one (Laplace) smoothing. This is where MAP earns its keep: an item with 0 clicks in 20 impressions has an MLE click rate of exactly zero, which a ranker would treat as certain failure and a log-loss would punish without limit. A prior centred on the population rate gives a small positive estimate that moves toward the data as impressions accumulate. The [Beta post](../beta-uncertainty/) covers choosing that prior.

**Gaussian means and shrinkage.** For a normal mean with known noise \(\sigma^2\) and prior \(\mathcal{N}(\mu_0, \tau^2)\), the MAP estimate is a precision-weighted average,

\[ \hat\mu_{\text{MAP}} = w\,\bar x + (1 - w)\,\mu_0, \qquad w = \frac{n/\sigma^2}{n/\sigma^2 + 1/\tau^2}. \]

Taking logs, the objective is squared error plus \(\frac{\sigma^2}{\tau^2}(\mu - \mu_0)^2\) after scaling: an L2 penalty with strength \(\lambda = \sigma^2/\tau^2\). Applied to regression weights with \(\mu_0 = 0\), that is ridge regression, worked through in [regression as maximum likelihood](../regression-as-maximum-likelihood/) and [L1 and L2 regularization](../regularization-l1-l2/). (Implementations usually leave the intercept unpenalized, which corresponds to a flat prior on it.)

Shrinkage is a [bias-variance trade](../bias-variance-tradeoff/), and the code measures it. With five points per estimate, MAP beats the MLE in mean squared error when the truth is near the prior and loses when it is far away.

```python
def map_rate(h, n, a, b):
    return (h + a - 1) / (n + a + b - 2)


# the formula is the argmax of log-likelihood + log-prior
h, n, a, b = 7, 30, 3.0, 12.0
grid = np.linspace(1e-4, 1 - 1e-4, 100_000)
logpost = h * np.log(grid) + (n - h) * np.log(1 - grid) + stats.beta.logpdf(grid, a, b)
assert abs(grid[logpost.argmax()] - map_rate(h, n, a, b)) < 1e-4
assert map_rate(h, n, 1, 1) == h / n                     # flat prior: the MLE
assert map_rate(h, n, 2, 2) == (h + 1) / (n + 2)         # Beta(2, 2): add-one smoothing

# zero clicks in 20 impressions, prior centred on a 5% population rate
assert 0 / 20 == 0.0 and map_rate(0, 20, 2, 39) > 0
assert abs(map_rate(0, 20, 2, 39) - map_rate(0, 20_000, 2, 39)) > 0.01   # the data win eventually
assert map_rate(0, 20_000, 2, 39) < 1e-4

# Gaussian mean: closed form vs a numerical argmax of the log-posterior
sigma, tau, mu0, n = 2.0, 1.0, 0.0, 5
x = rng.normal(0.8, sigma, n)
w = (n / sigma ** 2) / (n / sigma ** 2 + 1 / tau ** 2)
closed = w * x.mean() + (1 - w) * mu0
neg_logpost = lambda m: ((x - m) ** 2).sum() / (2 * sigma ** 2) + (m - mu0) ** 2 / (2 * tau ** 2)
assert abs(optimize.minimize_scalar(neg_logpost).x - closed) < 1e-6
# the same thing as squared error plus an L2 penalty with lambda = sigma^2 / tau^2
ridge = lambda m: ((x - m) ** 2).sum() + sigma ** 2 / tau ** 2 * (m - mu0) ** 2
assert abs(optimize.minimize_scalar(ridge).x - closed) < 1e-6

# mean squared error, simulated and exact: variance w^2 sigma^2 / n plus squared bias
for mu_true, map_wins in [(0.5, True), (3.0, False)]:
    xbar = rng.normal(mu_true, sigma / math.sqrt(n), 200_000)   # the sample mean's distribution
    mse_mle = np.mean((xbar - mu_true) ** 2)
    mse_map = np.mean((w * xbar + (1 - w) * mu0 - mu_true) ** 2)
    exact_map = w ** 2 * sigma ** 2 / n + (1 - w) ** 2 * (mu_true - mu0) ** 2
    assert abs(mse_mle - sigma ** 2 / n) < 0.01 and abs(mse_map - exact_map) < 0.01
    assert (mse_map < mse_mle) == map_wins
```

## The trap: MAP depends on how you write the parameter
The MLE is invariant to reparametrization. MAP is not, and the reason is worth seeing once. A density changes when you change variables: if \(\phi = g(\theta)\), then \(\pi_\phi(\phi) = \pi_\theta(\theta)\,\lvert d\theta / d\phi \rvert\). The likelihood picks up no such factor, because it is not a density over the parameter. The Jacobian moves the posterior's peak.

Take a coin with a uniform prior on \(p\), 3 heads and 1 tail. In \(p\), the MAP is \(3/4\), the same as the MLE. Now describe the same belief in log-odds, \(\phi = \ln\frac{p}{1-p}\). Since \(dp/d\phi = p(1-p)\), the posterior over \(\phi\) is proportional to \(p^{4}(1-p)^{2}\), and its peak, mapped back, is \(p = 4/6\). Same prior belief, same data, two different "most probable" values. The posterior mean is not invariant either: the mean of \(\phi\), mapped back through the sigmoid, is about 0.70, while the mean of \(p\) is \(4/6\). Posterior quantiles are invariant under any increasing change of variable (a median maps to a median), so the median and quantile intervals are the safer summaries when the parametrization is arbitrary.

```python
h, t = 3, 1
phi = np.linspace(-6, 6, 200_001)
p_of_phi = 1 / (1 + np.exp(-phi))
# uniform prior on p, rewritten as a density on phi: multiply by |dp/dphi| = p (1 - p)
logpost_phi = h * np.log(p_of_phi) + t * np.log(1 - p_of_phi) + np.log(p_of_phi * (1 - p_of_phi))
p_from_phi = p_of_phi[logpost_phi.argmax()]
assert abs(p_from_phi - (h + 1) / (h + t + 2)) < 1e-4    # 4/6, not 3/4
assert abs(map_rate(h, h + t, 1, 1) - 0.75) < 1e-12      # MAP in p
# the MLE does not move: maximizing the likelihood over phi gives logit(3/4)
loglik_phi = h * np.log(p_of_phi) + t * np.log(1 - p_of_phi)
assert abs(phi[loglik_phi.argmax()] - math.log(0.75 / 0.25)) < 1e-4
# a posterior quantile is invariant: the median in p equals the median in phi, mapped back
med_p = stats.beta.median(h + 1, t + 1)
cdf_phi = np.cumsum(np.exp(logpost_phi - logpost_phi.max()))
med_phi = phi[np.searchsorted(cdf_phi / cdf_phi[-1], 0.5)]
assert abs(1 / (1 + math.exp(-med_phi)) - med_p) < 1e-3
# the posterior mean is not: E[phi] mapped back is about 0.70, E[p] is 4/6
wts = np.exp(logpost_phi - logpost_phi.max())
mean_phi = (phi * wts).sum() / wts.sum()
assert abs(mean_phi - (1 + 1 / 2 + 1 / 3 - 1)) < 1e-3   # digamma(4) - digamma(2), exact
assert abs(1 / (1 + math.exp(-mean_phi)) - stats.beta.mean(h + 1, t + 1)) > 0.02
```

## When to use which
- **MLE** when data are plentiful relative to the number of parameters. The prior's influence fades anyway, and the MLE comes with ready-made standard errors from the Fisher information.
- **MAP** when data are thin or sparse: rare events, new items, many parameters per observation. It is the cheapest way to add sensible shrinkage, and in practice it is how most regularized models are trained, whether or not anyone calls it Bayesian.
- **The full posterior** when the decision depends on uncertainty and not just a point, such as exploring new items, reporting an interval or comparing two variants. A point estimate, MAP included, throws away the spread.

## Pitfalls
- **"The MLE is unbiased" is false in general.** The Gaussian variance and the uniform's endpoint are both biased; what holds is consistency and, under regularity conditions, asymptotic unbiasedness.
- **A stationary point is not automatically the maximum.** Check concavity or compare candidates, and check the boundary: the uniform's MLE sits there, and for perfectly separable data a logistic regression's MLE does not exist at all (its weights run to infinity).
- **Multiple local maxima.** Mixtures and neural networks have many. A numerical optimizer returns some stationary point, not necessarily the best one; restarts or a good initialization matter.
- **Off by one in the Beta mode.** The mode of \(\text{Beta}(a, b)\) is \((a-1)/(a+b-2)\), while the mean is \(a/(a+b)\). Add-one smoothing is the posterior mean under a uniform prior and the mode under \(\text{Beta}(2, 2)\).
- **The mode can be unrepresentative.** A Laplace prior can make a MAP weight exactly zero (this is the lasso) while its posterior mean is not zero, and in high dimensions the peak of a distribution can sit far from where its mass is.
- **The prior is data you did not observe.** A strong prior on a small sample decides the answer. Say what it is, and check that the estimate moves when the data say so.

The material is standard; for fuller treatments, see G. Casella and R. L. Berger, Statistical Inference (2nd edition), on maximum likelihood and its properties, and K. P. Murphy, Probabilistic Machine Learning: An Introduction (2022), on MAP estimation and Bayesian inference. Stanford's CS109 course notes also cover the MLE recipe and MAP; the examples and code here are new.
