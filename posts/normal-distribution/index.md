# The normal distribution: the bell curve, checked

If one training configuration gives validation accuracy of roughly \(\mathcal{N}(0.842, 0.004^2)\) across seeds, a single run reaches 0.85 about 2.3% of the time, but the best of 20 seeds reaches it about 37% of the time. That bell is the **normal** (or Gaussian) distribution, the symmetric hump that sums of many small, independent nudges tend toward, such as measurement error or the average of a few thousand clicks. It is the default noise model behind least squares, the reference curve for z-scores, p-values and confidence intervals, and the shape that averages settle into.

I take it apart, with code that checks each section: the density and why its constant is \(\sqrt{2\pi}\), standardizing so that one function answers every question, which combinations of normals stay normal, why averages become normal and how fast, the normal approximation to the binomial, and where the bell curve is the wrong model.

## The density
A normal variable with mean \(\mu\) and standard deviation \(\sigma > 0\) is written \(X \sim \mathcal{N}(\mu, \sigma^2)\). Note that the second parameter is the **variance**. Its density is

\[ f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \qquad x \in \mathbb{R}. \]

Read it from the inside out. The exponent is a downward parabola in \(x\), peaked at \(\mu\); exponentiating a parabola gives the bell, symmetric about \(\mu\), with inflection points at \(\mu \pm \sigma\). The front factor only makes the area equal to 1. It comes from the Gaussian integral \(\int_{-\infty}^{\infty} e^{-z^2/2}\,dz = \sqrt{2\pi}\), and the extra \(\sigma\) appears because stretching the bell by \(\sigma\) multiplies its area by \(\sigma\).

For the **standard normal** \(Z \sim \mathcal{N}(0, 1)\) both parameters drop out completely: \(\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}\). It is a common slip to set \(\mu = 0\) but leave a \(\sigma\) inside the exponent while dropping it from the constant; the result no longer integrates to 1.

Like any density, \(f\) is not a probability. Its peak is \(1/(\sigma\sqrt{2\pi})\), which is about 4 when \(\sigma = 0.1\). Only areas under it are probabilities.

The log of the density is the part that matters most in ML. Dropping the terms that do not involve \(\mu\), \(\log f(x) = -(x - \mu)^2 / (2\sigma^2)\), so maximizing a normal likelihood over \(\mu\) is minimizing squared error. That is why least squares is maximum likelihood under Gaussian noise ([regression as maximum likelihood](../regression-as-maximum-likelihood/)). The maximum-likelihood estimates are the sample mean and the variance that divides by \(n\), not \(n - 1\).

```python
import math
import numpy as np
from scipy import integrate, optimize, stats

rng = np.random.default_rng(0)


def normal_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * math.sqrt(2 * math.pi))


# The Gaussian integral behind the constant.
gauss, _ = integrate.quad(lambda z: math.exp(-z * z / 2), -np.inf, np.inf)
assert math.isclose(gauss, math.sqrt(2 * math.pi), rel_tol=1e-10)

for mu, sigma in [(0.0, 1.0), (3.0, 0.1), (-2.0, 5.0)]:
    area, _ = integrate.quad(normal_pdf, mu - 12 * sigma, mu + 12 * sigma, args=(mu, sigma))
    assert abs(area - 1) < 1e-9
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 17)
    assert np.allclose(normal_pdf(xs, mu, sigma), stats.norm.pdf(xs, loc=mu, scale=sigma))  # scale is sigma

assert normal_pdf(3.0, 3.0, 0.1) > 3.98                     # a density can exceed 1

# Keeping sigma in the exponent of the "standard" density breaks it: area sigma, not 1.
bad, _ = integrate.quad(lambda z: math.exp(-0.5 * (z / 2.0) ** 2) / math.sqrt(2 * math.pi), -40, 40)
assert math.isclose(bad, 2.0, rel_tol=1e-9)

# MLE: the sample mean and the divide-by-n standard deviation, by brute-force optimization.
data = rng.normal(loc=1.5, scale=0.7, size=2_000)
nll = lambda th: -stats.norm.logpdf(data, loc=th[0], scale=math.exp(th[1])).sum()
mu_hat, log_sigma_hat = optimize.minimize(nll, x0=[0.0, 0.0]).x
assert abs(mu_hat - data.mean()) < 1e-4
assert abs(math.exp(log_sigma_hat) - data.std(ddof=0)) < 1e-4
assert np.allclose(stats.norm.fit(data), (data.mean(), data.std(ddof=0)))
```

## One table for every normal
Linear maps keep normals normal. If \(X \sim \mathcal{N}(\mu, \sigma^2)\) and \(a \ne 0\), then

\[ aX + b \sim \mathcal{N}(a\mu + b,\; a^2\sigma^2). \]

The mean shifts and scales, the variance scales by \(a^2\) (the shift \(b\) cannot change spread), and the shape stays a bell. With \(a = 0\) you get the constant \(b\), which is not normal at all, so the condition matters.

Choose \(a = 1/\sigma\) and \(b = -\mu/\sigma\) and you get the **z-score** \(Z = (X - \mu)/\sigma \sim \mathcal{N}(0, 1)\). Every normal probability is therefore a question about one function, the standard normal CDF \(\Phi\):

\[ P(X \le x) = P\!\left(Z \le \frac{x - \mu}{\sigma}\right) = \Phi\!\left(\frac{x - \mu}{\sigma}\right), \qquad P(a \le X \le b) = \Phi\!\left(\frac{b - \mu}{\sigma}\right) - \Phi\!\left(\frac{a - \mu}{\sigma}\right). \]

Symmetry gives \(\Phi(-z) = 1 - \Phi(z)\), so tables only list positive \(z\). \(\Phi\) has no elementary closed form, but it is a rescaled error function, \(\Phi(z) = \tfrac{1}{2}\left(1 + \operatorname{erf}(z/\sqrt{2})\right)\), which is how libraries compute it. The same standardizing step, applied per feature or per activation, is what feature scaling and [batch and layer normalization](../batchnorm-vs-layernorm/) do.

**Worked example.** One training configuration, rerun with different seeds, gives validation accuracy that is roughly \(\mathcal{N}(0.842, 0.004^2)\). How often does a single run reach 0.85? The z-score is \((0.85 - 0.842)/0.004 = 2\), so \(P = 1 - \Phi(2) \approx 2.3\%\). And \(P(0.836 \le \text{acc} \le 0.85) = \Phi(2) - \Phi(-1.5) \approx 0.910\). If someone reports their best of 20 seeds, a 0.85 is no longer surprising: \(1 - (1 - 0.023)^{20} \approx 37\%\).

| Within | Probability | Two-sided \(z\) for | \(z\) |
|---|---|---|---|
| \(\mu \pm 1\sigma\) | 0.683 | 90% | 1.645 |
| \(\mu \pm 2\sigma\) | 0.954 | 95% | 1.960 |
| \(\mu \pm 3\sigma\) | 0.997 | 99% | 2.576 |

The left columns give the chance of landing within 1, 2 or 3 standard deviations of the mean; the right columns, the cutoff for a two-sided interval at each confidence level.

```python
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
for z in [-3.0, -1.5, 0.0, 0.7, 2.0]:
    assert math.isclose(Phi(z), stats.norm.cdf(z), abs_tol=1e-15)
    assert math.isclose(Phi(-z), 1 - Phi(z), abs_tol=1e-15)

# aX + b by simulation: mean a*mu + b, sd |a|*sigma, still normal.
X = rng.normal(5.0, 2.0, size=400_000)
Y = -3.0 * X + 1.0
assert abs(Y.mean() - (-14.0)) < 0.05 and abs(Y.std() - 6.0) < 0.03
assert stats.kstest(Y, "norm", args=(-14.0, 6.0)).pvalue > 0.01

# The seed example, three ways: z-score, scipy with loc/scale, simulation.
mu, sigma = 0.842, 0.004
p_085 = 1 - Phi((0.85 - mu) / sigma)
assert math.isclose(p_085, stats.norm.sf(0.85, loc=mu, scale=sigma), rel_tol=1e-9)
assert abs(p_085 - 0.02275) < 1e-5
runs = rng.normal(mu, sigma, size=1_000_000)
assert abs((runs >= 0.85).mean() - p_085) < 0.001
assert abs(Phi(2) - Phi(-1.5) - 0.9104) < 1e-4
assert abs((1 - (1 - p_085) ** 20) - 0.37) < 0.01           # best of 20 seeds

# The rule of thumb and the usual critical values.
for k, p in [(1, 0.683), (2, 0.954), (3, 0.997)]:
    assert abs(Phi(k) - Phi(-k) - p) < 5e-4
for conf, z in [(0.90, 1.645), (0.95, 1.960), (0.99, 2.576)]:
    assert abs(stats.norm.ppf(0.5 + conf / 2) - z) < 5e-4
```

## Sums and averages
Independent normals add to a normal, with means adding and **variances** (not standard deviations) adding:

\[ X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2,\; \sigma_1^2 + \sigma_2^2). \]

So the average of \(n\) independent \(\mathcal{N}(\mu, \sigma^2)\) draws is \(\mathcal{N}(\mu, \sigma^2/n)\): the standard error shrinks like \(1/\sqrt{n}\). Four times the data halves the error bar, which is the arithmetic behind sample sizes in [A/B testing](../ab-testing/).

The word "independent" is doing real work. Two variables can each be normal and be uncorrelated, yet their sum is not normal. Take \(X \sim \mathcal{N}(0, 1)\) and a fair random sign \(S\), and set \(Y = SX\). Then \(Y\) is standard normal and \(\operatorname{Cov}(X, Y) = 0\), but \(X + Y\) is exactly 0 half the time. For normal variables, zero correlation implies independence only when the pair is **jointly** normal, and here it is not.

```python
X1 = rng.normal(1.0, 2.0, size=400_000)
X2 = rng.normal(-3.0, 1.5, size=400_000)
total = X1 + X2
assert stats.kstest(total, "norm", args=(-2.0, 2.5)).pvalue > 0.01   # sd sqrt(4 + 2.25) = 2.5, not 3.5

n = 25
means = rng.normal(10.0, 3.0, size=(100_000, n)).mean(axis=1)
assert abs(means.std() - 3.0 / math.sqrt(n)) < 0.01                 # standard error sigma / sqrt(n)

# Normal marginals and zero correlation, but the sum is not normal.
Xn = rng.normal(size=400_000)
S = rng.choice([-1.0, 1.0], size=400_000)
Yn = S * Xn
assert stats.kstest(Yn, "norm").pvalue > 0.01                       # Y alone is standard normal
assert abs(np.corrcoef(Xn, Yn)[0, 1]) < 0.01                        # and uncorrelated with X
assert abs(((Xn + Yn) == 0).mean() - 0.5) < 0.005                    # yet X + Y has a point mass at 0
```

## Why it is everywhere: the central limit theorem
The normal is not common because nature prefers it. It is common because of the **central limit theorem**: if \(X_1, \dots, X_n\) are independent and identically distributed with mean \(\mu\) and finite variance \(\sigma^2\), then the standardized average \(\sqrt{n}(\bar X - \mu)/\sigma\) approaches \(\mathcal{N}(0, 1)\) in distribution as \(n\) grows, whatever the shape of the original distribution.

Two caveats decide whether it helps you. First, the speed depends on skew. For exponential data (skewness 2), the largest gap between the CDF of the standardized average and \(\Phi\) falls like \(1/\sqrt{n}\), the rate the Berry-Esseen theorem guarantees: about 0.06 at \(n = 5\) and still 0.013 at \(n = 100\). The largest absolute gap sits near the centre, but the tails are where p-values live, and there the relative error is what counts: at \(n = 100\) the true chance that the standardized average exceeds 3 is about twice the normal's 0.00135. Symmetric data converge much faster: the sum of 12 uniforms, minus 6, is an old and serviceable normal generator. Second, the variance must be finite. The average of Cauchy draws is Cauchy again, no matter how many you average, so it never settles.

```python
# Sum of n Exp(1) draws is exactly Gamma(n), so the gap to the normal can be computed without sampling.
z = np.linspace(-8, 8, 20_001)
gap = {n: np.abs(stats.gamma.cdf(n + math.sqrt(n) * z, n) - stats.norm.cdf(z)).max() for n in [1, 5, 30, 100]}
assert gap[1] > gap[5] > gap[30] > gap[100]
assert 0.055 < gap[5] < 0.065 and 0.012 < gap[100] < 0.014
for n in [5, 30, 100]:
    assert abs(gap[n] * math.sqrt(n) - 0.133) < 0.003                # the 1/sqrt(n) rate
    zi = np.abs(stats.gamma.cdf(n + math.sqrt(n) * z, n) - stats.norm.cdf(z)).argmax()
    assert abs(z[zi]) < 0.1                                          # largest absolute gap near the centre
# ...but the relative error is in the tail: P(standardized mean > 3) at n = 100 is about 2x the normal's.
ratio = stats.gamma.sf(100 + 10 * 3, 100) / stats.norm.sf(3)
assert 1.9 < ratio < 2.2 and abs(stats.norm.sf(3) - 0.00135) < 1e-5

# Symmetric parent: 12 uniforms minus 6 has mean 0, variance 1 and is already close.
U12 = rng.random((200_000, 12)).sum(axis=1) - 6
assert stats.kstest(U12, "norm").statistic < 0.006

# Infinite variance: the IQR of a Cauchy average does not shrink (standard Cauchy IQR is 2).
cauchy_means = rng.standard_cauchy(size=(5_000, 1_000)).mean(axis=1)
q25, q75 = np.percentile(cauchy_means, [25, 75])
assert 1.8 < q75 - q25 < 2.2
```

## The normal approximation to the binomial
A binomial count is a sum of \(n\) independent Bernoulli trials ([Bernoulli and binomial](../bernoulli-binomial/)), so the central limit theorem applies:

\[ \text{Binomial}(n, p) \approx \mathcal{N}\big(np,\; np(1 - p)\big). \]

Two refinements make it usable. The binomial lives on integers, so \(P(X \ge 40)\) should be read as the area from 39.5, the **continuity correction**. And the approximation needs the bell to fit inside \([0, n]\); a common rule of thumb asks for \(np\) and \(n(1 - p)\) of at least about 10.

**Worked example.** 1,000 impressions at a 3% click-through rate: mean 30, standard deviation \(\sqrt{29.1} \approx 5.39\). The exact \(P(X \ge 40)\) is 0.0437. The normal without correction gives 0.0319; with it, 0.0391. The correction recovers most of the error, and what is left is the binomial's right skew at a mean of only 30.

At \(p = 0.002\) the mean is 2 and the rule fails. The normal then gives 0.038 for \(P(X \ge 5)\) against an exact 0.052, and with the same continuity correction it puts 3.8% of its mass below \(-0.5\), on negative click counts that cannot happen (7.8% lies below zero). The [Poisson](../poisson-distribution/) approximation is the right tool there. The coin-flip version of this story is in [coin flips and the normal approximation](../coin-flips-normal-approximation/).

```python
n, p = 1_000, 0.03
m, s = n * p, math.sqrt(n * p * (1 - p))
exact = stats.binom.sf(39, n, p)                             # P(X >= 40)
plain = stats.norm.sf(40, m, s)
corrected = stats.norm.sf(39.5, m, s)
assert abs(exact - 0.0437) < 1e-4 and abs(plain - 0.0319) < 1e-4 and abs(corrected - 0.0391) < 1e-4
assert abs(corrected - exact) < abs(plain - exact) / 2
sim = rng.binomial(n, p, size=1_000_000)
assert abs((sim >= 40).mean() - exact) < 0.001

n, p = 1_000, 0.002                                          # np = 2: rule of thumb fails
m, s = n * p, math.sqrt(n * p * (1 - p))
exact = stats.binom.sf(4, n, p)                              # P(X >= 5)
normal_approx = stats.norm.sf(4.5, m, s)
poisson_approx = stats.poisson.sf(4, m)
assert abs(poisson_approx - exact) < 0.001 < 0.01 < abs(normal_approx - exact)
assert abs(stats.norm.cdf(-0.5, m, s) - 0.038) < 0.001         # mass on impossible negative counts
assert abs(stats.norm.cdf(0, m, s) - 0.078) < 0.001
```

## Pitfalls
Reach for the normal when a quantity is an average or a sum of many independent pieces, for measurement noise, and as the large-sample shape of estimators (means, proportions, regression coefficients). It is also the maximum-entropy distribution for a given mean and variance: if those two numbers are all you are willing to assume, the normal assumes the least else ([entropy and information](../entropy-and-information/)).

- **Tails.** The normal's tails fall off like \(e^{-z^2/2}\), faster than almost any real data. A 6\(\sigma\) move has probability about \(10^{-9}\) under a normal; under a Student-t with 3 degrees of freedom rescaled to the same variance it is about \(10^{-3}\), a million times likelier. Returns, latencies and gradient norms have heavy tails; check before quoting "six sigma".
- **Skewed, positive quantities.** Latency, revenue, session length and file sizes are bounded below and skewed right. Model their logarithm, or use a log-normal, gamma or [exponential](../exponential-distribution/), rather than a bell that puts mass below zero.
- **Variance, not standard deviation.** \(\mathcal{N}(\mu, \sigma^2)\) names the variance, but `scipy.stats.norm(loc, scale)` and `numpy`'s `normal(loc, scale)` take \(\sigma\). Variances add for independent sums; standard deviations do not.
- **Normal marginals are not a normal pair.** Two normal variables with zero correlation are independent only if they are jointly normal, as the sign-flip example showed.
- **The CLT is slow for skewed data.** Small-sample z-tests on skewed metrics give wrong tail probabilities; bootstrap or use more data ([p-values](../p-values/)).
- **Far tails in floating point.** \(1 - \Phi(z)\) rounds to 0 past about \(z = 8.3\). Use `norm.sf` and `norm.logsf`, which compute the upper tail directly.
- **Least squares assumes it.** Squared-error loss is the Gaussian negative log-likelihood, so a few outliers pull the fit hard. A heavier-tailed noise model leads to absolute or Huber loss ([cross-entropy vs MSE](../cross-entropy-vs-mse/)).

```python
# Maximum entropy at fixed variance 1: normal beats uniform and Laplace.
h_norm = stats.norm(scale=1).entropy()
h_unif = stats.uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)).entropy()
h_lap = stats.laplace(scale=1 / math.sqrt(2)).entropy()
assert math.isclose(h_norm, 0.5 * math.log(2 * math.pi * math.e))
assert h_norm > h_lap > h_unif

# Six sigma under a normal versus a unit-variance Student-t with 3 degrees of freedom (variance 3).
p_norm = stats.norm.sf(6)
p_t3 = stats.t.sf(6 * math.sqrt(3), df=3)
assert 9e-10 < p_norm < 1.1e-9 and 9e-4 < p_t3 < 1e-3 and p_t3 / p_norm > 5e5

# The far tail: 1 - cdf underflows to zero, sf and logsf do not.
assert 1 - stats.norm.cdf(8.0) > 0 and 1 - stats.norm.cdf(8.3) == 0.0 and stats.norm.sf(8.3) > 5e-17
assert 1 - stats.norm.cdf(10) == 0.0
assert math.isclose(stats.norm.sf(10), 7.62e-24, rel_tol=1e-3)
assert stats.norm.sf(40) == 0.0                                 # underflows even with sf...
assert abs(stats.norm.logsf(40) - (-800 - math.log(40 * math.sqrt(2 * math.pi)))) < 1e-3   # ...but logsf is fine
```

The code checks these claims on chosen cases: exponential data for the central limit theorem, a Student-t with 3 degrees of freedom for the tails, two click-through rates for the binomial. It shows how far the normal can be off, not how far it is off for a given metric. Check a real metric's skew and tails before trusting a normal tail probability, or bootstrap.

For the binomial side see [is this coin fair?](../is-this-coin-fair/), and for the distributions side by side, the [probability cheatsheet](../probability-cheatsheet/). This is standard textbook material; chapter 5 of Blitzstein and Hwang's Introduction to Probability is a good fuller treatment.
