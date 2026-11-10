# Regression as maximum likelihood: where squared error and cross-entropy come from

Squared error for regression and cross-entropy for classification are usually presented as conventions. They are not. Each is the negative log-likelihood of a specific assumption about how the targets were generated: Gaussian noise gives squared error, a biased coin gives cross-entropy. This post derives both from one recipe and checks every step numerically, then uses the same recipe to get the right loss for medians and counts, and to read L2 and L1 regularization as priors.

## One recipe
Maximum likelihood picks the parameters under which the observed data were most probable. In supervised learning that takes three steps.

1. Choose a distribution for the target given the input, \(p(y \mid x, w)\), whose parameter (a mean, a probability, a rate) is a function of \(x\) with weights \(w\).
2. Assume the examples are independent given their inputs, so the likelihood of the whole data set is a product over examples.
3. Take the negative log. The product becomes a sum, maximizing becomes minimizing, and the per-example term is the loss.

\[ \hat w \;=\; \arg\max_w \prod_{i=1}^{n} p(y_i \mid x_i, w) \;=\; \arg\min_w \sum_{i=1}^{n} -\log p(y_i \mid x_i, w) . \]

The log does not move the optimum, because it is strictly increasing. What it does is make the arithmetic possible: a product of a few thousand probabilities is smaller than the smallest positive float.

```python
import numpy as np

rng = np.random.default_rng(0)
probs = rng.uniform(0.3, 0.7, size=2000)    # the probability a model gave each observed label
assert np.prod(probs) == 0.0                # the likelihood underflows float64...
nll = -np.log(probs).sum()                  # ...the negative log-likelihood is an ordinary number
assert 1000 < nll < 2000
```

## Linear regression: Gaussian noise gives squared error
Assume each target is a linear function of its input plus its own independent Gaussian noise:

\[ y_i = w^\top x_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2) \ \text{independently, so} \ y_i \sim \mathcal{N}(w^\top x_i, \sigma^2) . \]

Summing the negative log of the Gaussian density over the examples gives

\[ -\log L(w, \sigma) \;=\; \frac{n}{2}\log(2\pi\sigma^2) \;+\; \frac{1}{2\sigma^2}\sum_{i=1}^{n}\left(y_i - w^\top x_i\right)^2 . \]

For any fixed \(\sigma\), the first term is a constant and the second is the residual sum of squares (RSS) times a positive number. Neither changes which \(w\) is best, so the maximum likelihood weights are exactly the least squares weights, the solution of the normal equations \(X^\top X w = X^\top y\). Strictly, the two objectives are equal up to an additive constant and a positive scale, not proportional, which matters once likelihoods of different models are compared.

The code checks the full Gaussian negative log-likelihood against `scipy.stats.norm`, then minimizes it at three very different noise levels, landing on the least squares solution each time:

```python
from scipy.optimize import minimize
from scipy.stats import norm

n, w_true, noise_sd = 500, np.array([1.5, -2.0, 0.5]), 0.8
X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])     # column 0 is the intercept
y = X @ w_true + rng.normal(0, noise_sd, n)


def gaussian_nll(w, sd):
    r = y - X @ w
    return 0.5 * len(y) * np.log(2 * np.pi * sd**2) + (r @ r) / (2 * sd**2)


w_ols = np.linalg.solve(X.T @ X, X.T @ y)                          # normal equations
assert np.isclose(gaussian_nll(w_ols, 0.8), -norm.logpdf(y, X @ w_ols, 0.8).sum())

for sd in (0.1, 1.0, 10.0):                                        # sd rescales the loss but never moves its minimum
    w_hat = minimize(gaussian_nll, np.zeros(3), args=(sd,), method="BFGS").x
    assert np.allclose(w_hat, w_ols, atol=1e-4)
```

### The noise variance has an MLE too
Squared error throws \(\sigma\) away, but the likelihood keeps it. Setting the derivative with respect to \(\sigma^2\) to zero gives \(\hat\sigma^2 = \mathrm{RSS}/n\), the mean squared residual. That estimate is biased low: the fitted line was chosen to sit close to these particular points, so its residuals are smaller than the true noise. With \(p\) weights, \(\mathbb{E}[\mathrm{RSS}] = (n - p)\,\sigma^2\), which is why the usual unbiased estimate divides by \(n - p\) instead. A simulation of 20,000 small data sets with \(n = 10\) and \(p = 3\) puts the shortfall at 30%:

```python
rss = np.sum((y - X @ w_ols) ** 2)
fit = minimize(lambda t: gaussian_nll(t[:3], np.exp(t[3])), np.zeros(4), method="BFGS")
assert np.isclose(np.exp(fit.x[3]) ** 2, rss / n, rtol=1e-4)      # joint MLE of the variance = RSS / n

n_small, p_small, reps = 10, 3, 20_000
X_small = np.column_stack([np.ones(n_small), rng.normal(size=(n_small, p_small - 1))])
H = X_small @ np.linalg.solve(X_small.T @ X_small, X_small.T)      # hat matrix: fitted values = H @ y
Y = X_small @ rng.normal(size=p_small) + rng.normal(0, 1.0, (reps, n_small))   # true variance 1
rss_many = np.sum((Y - Y @ H.T) ** 2, axis=1)
assert abs((rss_many / n_small).mean() - 0.7) < 0.02               # low by a factor (n - p) / n
assert abs((rss_many / (n_small - p_small)).mean() - 1.0) < 0.02   # dividing by n - p fixes it
```

The variance estimate is what turns a point prediction into a range: \(\hat y \pm 1.96\,\hat\sigma\) is a rough 95% prediction interval. It ignores the uncertainty in \(\hat w\), which is small when \(n\) is much larger than \(p\). See [the normal distribution](../normal-distribution/) for where 1.96 comes from.

## Change the noise, change the loss
Squared error encodes three assumptions about the noise: it is symmetric, it has the same variance everywhere, and it has light tails. The Gaussian density falls off like \(e^{-r^2 / (2\sigma^2)}\) in the residual \(r\), so a residual ten times larger costs a hundred times more, and a few gross errors can drag the whole fit.

Use the Laplace distribution instead, whose density falls off like \(e^{-\lvert r \rvert / b}\), and the negative log-likelihood becomes the sum of absolute residuals. The two losses also estimate different things. With no inputs, squared error is minimized by the sample mean and absolute error by the sample median; likewise, a flexible model trained on each estimates the conditional mean or the conditional median of \(y\). With ten corrupted targets out of 200, the absolute-error fit (least absolute deviations, here scikit-learn's `QuantileRegressor` at the 0.5 quantile) stays near the true slope of 0.5, while least squares nearly doubles it:

```python
from scipy.optimize import minimize_scalar
from sklearn.linear_model import QuantileRegressor

sample = rng.exponential(1.0, 301)                                 # skewed, so mean and median differ
best_sq = minimize_scalar(lambda c: np.sum((sample - c) ** 2)).x
best_abs = minimize_scalar(lambda c: np.sum(np.abs(sample - c)), bounds=(0, 5),
                           method="bounded", options={"xatol": 1e-10}).x
assert np.isclose(best_sq, sample.mean()) and np.isclose(best_abs, np.median(sample), atol=1e-6)

x = rng.uniform(0, 10, 200)
y_line = 2.0 + 0.5 * x + rng.normal(0, 0.5, x.size)
y_line[np.argsort(x)[-10:]] += 15                                  # ten gross errors, all at large x

slope_sq = np.polyfit(x, y_line, 1)[0]                             # least squares
slope_abs = QuantileRegressor(quantile=0.5, alpha=0).fit(x[:, None], y_line).coef_[0]
assert abs(slope_abs - 0.5) < 0.05 and abs(slope_sq - 0.5) > 0.4
```

Counts follow the same recipe. If \(y_i \sim \operatorname{Poisson}(\lambda_i)\) with \(\lambda_i = e^{w^\top x_i}\), which keeps the rate positive, the negative log-likelihood is

\[ -\log L(w) \;=\; \sum_{i=1}^{n} \left( e^{w^\top x_i} - y_i\, w^\top x_i + \log y_i! \right) . \]

The \(\log y_i!\) term has no \(w\) in it, so it can be dropped for fitting. What remains is Poisson regression. Minimizing it directly gives the same answer as scikit-learn's `PoissonRegressor`:

```python
from scipy.special import gammaln
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

x_count = rng.uniform(0, 2, 2000)
X_count = np.column_stack([np.ones_like(x_count), x_count])
counts = rng.poisson(np.exp(0.3 + 0.8 * x_count))


def poisson_nll(w):
    eta = X_count @ w
    return np.sum(np.exp(eta) - counts * eta)                      # log(y!) dropped: it has no w in it


w_pois = minimize(poisson_nll, np.zeros(2), jac=lambda w: X_count.T @ (np.exp(X_count @ w) - counts),
                  method="BFGS").x
full_nll = -poisson.logpmf(counts, np.exp(X_count @ w_pois)).sum()
assert np.isclose(full_nll, poisson_nll(w_pois) + gammaln(counts + 1).sum())
ref = PoissonRegressor(alpha=0, tol=1e-10, max_iter=1000).fit(x_count[:, None], counts)
assert np.allclose(w_pois, [ref.intercept_, *ref.coef_], atol=1e-4)
```

The [Poisson distribution](../poisson-distribution/) post covers when counts really are Poisson, and what to do when their variance exceeds their mean.

## Logistic regression: a biased coin gives cross-entropy
For a binary label, the natural distribution is a Bernoulli whose probability depends on the input:

\[ y_i \sim \operatorname{Bernoulli}(p_i), \qquad p_i = \operatorname{sigmoid}(w^\top x_i) = \frac{1}{1 + e^{-w^\top x_i}} . \]

The probability of the observed label is \(p_i^{y_i}(1 - p_i)^{1 - y_i}\), which is \(p_i\) when the label is 1 and \(1 - p_i\) when it is 0. The negative log-likelihood is

\[ -\log L(w) \;=\; -\sum_{i=1}^{n} \Big( y_i \log p_i + (1 - y_i)\log(1 - p_i) \Big) , \]

which is binary cross-entropy summed over the examples (the "log loss" averages it instead). So training a classifier with cross-entropy is fitting a Bernoulli model by maximum likelihood. For why squared error is the wrong loss here, see [cross-entropy vs MSE](../cross-entropy-vs-mse/).

Written in terms of the logit \(z_i = w^\top x_i\), each term is \(\log(1 + e^{z_i}) - y_i z_i\). That is the form to compute: evaluated as a softplus, for example `np.logaddexp(0, z)`, it never overflows, whereas \(\log(1 - p_i)\) taken from an already-rounded sigmoid becomes \(\log 0\) once \(z_i\) passes about 37 ([stable softmax and cross-entropy](../stable-softmax-cross-entropy/) covers the multi-class case). Its gradient is

\[ \nabla_w \big( -\log L(w) \big) \;=\; \sum_{i=1}^{n} (p_i - y_i)\, x_i : \]

prediction minus label, times input. That is the same shape as the squared-error gradient \(\frac{1}{\sigma^2}\sum_i (\hat y_i - y_i)\,x_i\). Unlike least squares there is no closed form, but the Hessian \(X^\top \operatorname{diag}\big(p_i(1 - p_i)\big)\,X\) is positive semidefinite, so the loss is convex, and on data that are not separable, Newton's method usually converges in a handful of steps (six to machine precision here). The code checks the gradient by finite differences, fits by Newton's method, and matches both `log_loss` and `LogisticRegression` (with a huge `C`, so it is effectively unpenalized):

```python
from scipy.optimize import check_grad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

Xb = np.column_stack([np.ones(1000), rng.normal(size=(1000, 2))])
yb = (rng.random(1000) < 1 / (1 + np.exp(-Xb @ np.array([-0.5, 2.0, -1.0])))).astype(float)


def bernoulli_nll(w):
    z = Xb @ w
    return np.sum(np.logaddexp(0, z) - yb * z)                     # log(1 + e^z) - y z, no overflow


def bernoulli_grad(w):
    return Xb.T @ (1 / (1 + np.exp(-Xb @ w)) - yb)


assert check_grad(bernoulli_nll, bernoulli_grad, rng.normal(size=3)) < 1e-3
assert 1 - 1 / (1 + np.exp(-37.0)) == 0                          # naive 1 - p rounds to 0: log(1 - p) = -inf
assert np.isclose(np.logaddexp(0, 1000.0), 1000.0)                # the softplus form stays finite


def fit_logistic(X, y, steps=20):                                  # Newton's method
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ w))
        hessian = X.T @ (X * (p * (1 - p))[:, None])
        w -= np.linalg.solve(hessian, X.T @ (p - y))
    return w


w_log = fit_logistic(Xb, yb)
p_hat = 1 / (1 + np.exp(-Xb @ w_log))
assert np.isclose(bernoulli_nll(w_log), len(yb) * log_loss(yb, p_hat))   # NLL = summed cross-entropy
ref = LogisticRegression(C=1e12, tol=1e-12, max_iter=10_000).fit(Xb[:, 1:], yb)
assert np.allclose(w_log, [ref.intercept_[0], *ref.coef_[0]], atol=1e-4)
assert np.allclose(fit_logistic(Xb, yb, steps=6), w_log, atol=1e-10)    # Newton: six steps suffice
```

### One pattern, three models
Linear, logistic and Poisson regression are all generalized linear models with their canonical link function, and for each of them the gradient of the negative log-likelihood is proportional to \(X^\top(\hat y - y)\). The component of that gradient for the intercept, whose input is always 1, is \(\sum_i (\hat y_i - y_i)\), so at the optimum the predictions add up to the targets. Residuals sum to zero, predicted probabilities average to the base rate, and predicted counts add up to the observed total:

```python
assert np.isclose((y - X @ w_ols).sum(), 0, atol=1e-8)                      # linear
assert np.isclose(p_hat.sum(), yb.sum())                                    # logistic
assert np.isclose(np.exp(X_count @ w_pois).sum(), counts.sum(), rtol=1e-6)  # Poisson
```

## When the MLE does not exist
If a hyperplane separates the two classes perfectly, there is no best \(w\). Scaling up a separating \(w\) pushes every \(p_i\) toward its label and the likelihood toward 1, but never reaches it. Gradient descent never converges: the weights keep growing, roughly with the log of the number of steps (Soudry et al., 2018, show the direction converges to the maximum-margin separator), and the predicted probabilities drift toward 0 and 1. Add an L2 penalty and a minimum exists again:

```python
x_sep = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
X_sep, y_sep = np.column_stack([np.ones(6), x_sep]), (x_sep > 0).astype(float)


def gradient_descent(lam, steps, lr=0.5):
    w = np.zeros(2)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X_sep @ w))
        w -= lr * (X_sep.T @ (p - y_sep) + lam * w)
    return w


w_1k, w_10k = gradient_descent(0.0, 1_000), gradient_descent(0.0, 10_000)
growth = np.linalg.norm(w_10k) - np.linalg.norm(w_1k)
assert abs(growth - np.log(10)) < 0.1                            # no MLE: |w| grows by about log(10) per tenfold steps
p_1k, p_10k = (1 / (1 + np.exp(-X_sep @ w)) for w in (w_1k, w_10k))
assert np.all(np.abs(p_10k - y_sep) < np.abs(p_1k - y_sep))      # every probability drifts toward its label
w_1k, w_10k = gradient_descent(0.1, 1_000), gradient_descent(0.1, 10_000)
assert np.allclose(w_1k, w_10k)                                   # with a penalty it settles
```

This is common: with at least as many features as examples, points in general position are separable under any labeling, so unpenalized logistic regression on wide or sparse features often has no MLE at all. Scikit-learn's `LogisticRegression` applies an L2 penalty by default.

## Priors are regularizers
Maximum likelihood treats every \(w\) as equally plausible before seeing any data. Maximum a posteriori (MAP) estimation adds a prior \(p(w)\) and maximizes the posterior, which by Bayes' rule is proportional to likelihood times prior. In negative logs, the prior is just one more term:

\[ \hat w_{\text{MAP}} \;=\; \arg\min_w \Big( -\log L(w) - \log p(w) \Big) . \]

With independent Gaussian priors \(w_j \sim \mathcal{N}(0, \tau^2)\), the extra term is \(\lVert w \rVert_2^2 / (2\tau^2)\) plus a constant. Multiply the linear regression objective by \(2\sigma^2\) and it becomes

\[ \sum_{i=1}^{n} \left(y_i - w^\top x_i\right)^2 + \frac{\sigma^2}{\tau^2}\, \lVert w \rVert_2^2 , \]

which is ridge regression with \(\lambda = \sigma^2/\tau^2\). A tight prior (small \(\tau\)) or noisy data (large \(\sigma\)) both mean stronger shrinkage. To keep the check short, the prior below covers every weight, the intercept included; in practice the intercept is usually left unpenalized.

```python
from sklearn.linear_model import Lasso, Ridge

tau = 0.5                                                          # prior: each weight ~ N(0, tau^2)
lam = noise_sd**2 / tau**2


def neg_log_posterior(w):
    return gaussian_nll(w, noise_sd) + (w @ w) / (2 * tau**2)


w_map = minimize(neg_log_posterior, np.zeros(3), method="BFGS").x
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)
assert np.allclose(w_map, w_ridge, atol=1e-4)
assert np.allclose(w_ridge, Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_)
```

With Laplace priors of scale \(b\), the extra term is \(\lVert w \rVert_1 / b\) and the MAP estimate is the lasso; in scikit-learn's scaling, its penalty is \(\alpha = \sigma^2 / (n b)\). The absolute value has a kink at zero, so a weight sits at exactly zero whenever the pull of the data on it, the size of its gradient there, is at most \(1/b\). The code checks those optimality conditions at the lasso solution, which zeroes out exactly the five irrelevant features:

```python
n_l1, b, sd_l1 = 300, 0.02, 1.0                                    # b is the Laplace prior scale
X_l1 = rng.normal(size=(n_l1, 8))
y_l1 = X_l1 @ np.array([2.0, -1.5, 1.0, 0, 0, 0, 0, 0]) + rng.normal(0, sd_l1, n_l1)
w_l1 = Lasso(alpha=sd_l1**2 / (n_l1 * b), fit_intercept=False, tol=1e-12, max_iter=100_000).fit(X_l1, y_l1).coef_

pull = X_l1.T @ (y_l1 - X_l1 @ w_l1) / sd_l1**2                   # minus the gradient of the Gaussian NLL
on = w_l1 != 0
assert np.allclose(pull[on], np.sign(w_l1[on]) / b, atol=1e-3)    # nonzero weights: pull balances the prior
assert np.all(np.abs(pull[~on]) <= 1 / b)                         # zero weights: the pull is too weak to move them
assert on.tolist() == [True] * 3 + [False] * 5
```

Two cautions. The Laplace prior gives zero probability to any weight being exactly zero; sparsity is a property of the MAP point, and the posterior mean under the same prior is not sparse. And the correspondence is an interpretation, not a tuning method: \(\lambda\) is chosen by validation, not derived from a believed \(\tau\). There is more in [L1 and L2 regularization](../regularization-l1-l2/) and [MLE and MAP](../statistical-estimation-mle-map/).

## Choosing a loss from the target
| Target | Noise model | Loss (negative log-likelihood) | The prediction estimates |
|---|---|---|---|
| Real-valued | Gaussian, constant variance | squared error | conditional mean |
| Real-valued, with outliers | Laplace | absolute error | conditional median |
| Real-valued, known per-example variance | Gaussian, variance \(\sigma_i^2\) | squared error weighted by \(1/\sigma_i^2\) | conditional mean |
| Binary | Bernoulli | binary cross-entropy | \(P(y = 1 \mid x)\) |
| One of K classes | categorical | softmax cross-entropy | class probabilities |
| Counts | Poisson | \(e^{\eta} - y\,\eta\) with \(\eta = w^\top x\) | conditional mean count |

## Pitfalls
- **Independence is an assumption too.** The product over examples assumes independence given the inputs. With duplicated rows or many rows per user, the point estimate may be fine, but standard errors and likelihood-based tests come out too small.
- **Constants matter when comparing models.** Dropping \(\frac{n}{2}\log(2\pi\sigma^2)\) or \(\log y_i!\) is fine for fitting one model. To compare noise models on the same targets, say Gaussian against Laplace, compare full negative log-likelihoods with each scale fitted. An MSE and a cross-entropy are never comparable numbers.
- **Check the residuals.** If their spread grows with the prediction, the constant-variance assumption is wrong. Weighted least squares, a log transform or a different distribution will fit better.
- **Reweighting changes the model.** Class weights and resampling change the likelihood being maximized, so the fitted probabilities are no longer calibrated to the original base rate.
- **The MLE of the variance understates the noise.** \(\mathrm{RSS}/n\) is badly biased when \(p\) is not small compared with \(n\). Divide by \(n - p\).

The probabilistic reading of regression is standard textbook material; for fuller derivations, see C. M. Bishop, Pattern Recognition and Machine Learning (2006), chapters 3 and 4, and K. P. Murphy, Probabilistic Machine Learning: An Introduction (2022), its chapters on logistic, linear and generalized linear models.
