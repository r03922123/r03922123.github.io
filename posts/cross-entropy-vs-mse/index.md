# Why classifiers use cross-entropy, not squared error

Start a one-feature logistic regression confidently wrong, at \(w_0 = -8\), on 400 points from two slightly overlapping classes. With gradient descent at learning rate 1, cross-entropy reaches 95% accuracy in five steps and squared error takes 2,482. A binary classifier usually ends in a sigmoid: a score \(z\) becomes a probability \(p = \sigma(z)\). Two losses look reasonable for comparing \(p\) with a label \(y \in \{0, 1\}\): squared error, which serves regression well, and cross-entropy, also called log loss. Nearly everyone uses cross-entropy, and the reason is not taste. Squared error behind a sigmoid stops pushing exactly when the model is confidently wrong, while cross-entropy is the negative log-likelihood of the coin-flip model that binary labels actually follow.

I derive the first claim and check it against autograd, watch it stall that training run, set out the second, and then cover the cases where squared error is still the right choice.

## Two losses on one sigmoid
Write \(z\) for the logit (for logistic regression, \(z = w^\top x + b\)) and \(p = \sigma(z) = 1/(1 + e^{-z})\) for the predicted probability that \(y = 1\). For one example the two candidates are

\[ \ell_{\text{CE}} = -\big[\,y \log p + (1 - y)\log(1 - p)\,\big], \qquad \ell_{\text{MSE}} = \tfrac{1}{2}(p - y)^2 . \]

The \(\tfrac12\) only tidies the algebra. Without it every squared-error gradient below doubles, which changes nothing in the argument.

## The gradient with respect to the logit
Gradient descent moves the weights in proportion to \(\partial \ell / \partial z\) (times \(x\), for a linear model), so that is the number to look at. The sigmoid's derivative is \(\sigma'(z) = p(1 - p)\), and the chain rule gives

\[ \frac{\partial \ell_{\text{CE}}}{\partial z} = \frac{p - y}{p(1-p)} \cdot p(1-p) = p - y, \qquad \frac{\partial \ell_{\text{MSE}}}{\partial z} = (p - y)\,p(1-p) . \]

For cross-entropy, the \(p(1-p)\) contributed by the sigmoid cancels the \(1/(p(1-p))\) contributed by the log, and what is left is the plain error. Squared error keeps the factor \(p(1-p)\), which goes to zero at both ends. That is harmless when the model is confidently right and disastrous when it is confidently wrong. Here is a positive example (\(y = 1\)) at a range of logits:

| Logit \(z\) | \(p\) | CE gradient | MSE gradient |
|---|---|---|---|
| −10 | 0.000045 | −1.000 | −0.000045 |
| −5 | 0.0067 | −0.993 | −0.0066 |
| −2 | 0.119 | −0.881 | −0.092 |
| −0.69 | 0.333 | −0.667 | −0.148 |
| 0 | 0.5 | −0.500 | −0.125 |
| 2 | 0.881 | −0.119 | −0.0125 |
| 5 | 0.9933 | −0.0067 | −0.000044 |

At \(z = -10\) the error is nearly as large as it can be, yet squared error pushes twenty thousand times more weakly than cross-entropy. The MSE column peaks at \(p = 1/3\) (\(z = -\ln 2\)), where its push is \(4/27 \approx 0.148\); for logits below that it gets **weaker the more wrong the prediction is**. The CE gradient only grows in size as the prediction gets worse, up to its cap of 1.

```python
import numpy as np
import torch
import torch.nn.functional as F


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def grad_ce(z, y):                  # d/dz of -[y log p + (1 - y) log(1 - p)]
    return sigmoid(z) - y


def grad_mse(z, y):                 # d/dz of (p - y)^2 / 2
    p = sigmoid(z)
    return (p - y) * p * (1 - p)


# the hand-derived gradients agree with autograd, for both labels
z = torch.linspace(-12, 12, 97, dtype=torch.float64, requires_grad=True)
for label in (0.0, 1.0):
    y = torch.full_like(z, label)
    ce = F.binary_cross_entropy_with_logits(z, y, reduction="sum")
    mse = (0.5 * (torch.sigmoid(z) - y) ** 2).sum()
    g_ce, = torch.autograd.grad(ce, z)
    g_mse, = torch.autograd.grad(mse, z)
    assert np.allclose(g_ce.numpy(), grad_ce(z.detach().numpy(), label))
    assert np.allclose(g_mse.numpy(), grad_mse(z.detach().numpy(), label))

# the table, for a positive example
table = [(-10, 0.000045, -1.000, -0.000045), (-5, 0.0067, -0.993, -0.0066), (-2, 0.119, -0.881, -0.092),
         (-np.log(2), 0.333, -0.667, -0.148), (0, 0.5, -0.5, -0.125), (2, 0.881, -0.119, -0.0125),
         (5, 0.9933, -0.0067, -0.000044)]
for zv, p, ce, mse in table:
    assert np.allclose([sigmoid(zv), grad_ce(zv, 1), grad_mse(zv, 1)], [p, ce, mse], rtol=0.02, atol=0)
assert grad_ce(-10, 1) / grad_mse(-10, 1) > 20_000

# MSE pushes hardest at p = 1/3, and less and less as the model gets more wrong
grid = np.linspace(-15, 15, 300_001)
push = np.abs(grad_mse(grid, 1))
assert abs(grid[push.argmax()] + np.log(2)) < 1e-3 and abs(push.max() - 4 / 27) < 1e-9
assert np.all(np.diff(push[grid < -np.log(2)]) > 0)
```

## A training run that stalls
To see what this does to training, fit a one-feature logistic regression \(p = \sigma(wx + b)\) by full-batch gradient descent on 400 points: 200 positives centred at \(x = 2\) and 200 negatives at \(x = -2\), with unit spread, so the classes overlap only slightly. The right weight is positive. Start it at a negative \(w_0\), so that nearly every point is confidently misclassified, and count the steps until accuracy reaches 95%.

```python
from scipy.stats import norm

# evenly spaced normal quantiles instead of random draws, so the counts are exact
q = norm.ppf((np.arange(200) + 0.5) / 200)
x = np.concatenate([2 + q, -2 + q])
y = np.concatenate([np.ones(200), np.zeros(200)])


def steps_to_accuracy(grad_fn, lr, w, b=0.0, target=0.95, max_steps=10_000):
    for step in range(max_steps):
        z = w * x + b
        if np.mean((z > 0) == (y == 1)) >= target:
            return step
        g = grad_fn(z, y)                                 # dLoss/dz for every example
        w, b = w - lr * np.mean(g * x), b - lr * np.mean(g)
    return None


steps = {w0: (steps_to_accuracy(grad_ce, 1, w0),
              steps_to_accuracy(grad_mse, 1, w0),
              steps_to_accuracy(grad_mse, 4, w0)) for w0 in (-1, -2, -4, -8)}
assert np.mean((-4 * x > 0) == (y == 1)) < 0.03           # the start really is confidently wrong
assert steps == {-1: (1, 5, 2), -2: (2, 21, 6), -4: (3, 189, 49), -8: (5, 2482, 623)}
```

| Start \(w_0\) | CE, learning rate 1 | MSE, learning rate 1 | MSE, learning rate 4 |
|---|---|---|---|
| −1 | 1 | 5 | 2 |
| −2 | 2 | 21 | 6 |
| −4 | 3 | 189 | 49 |
| −8 | 5 | 2,482 | 623 |

Cross-entropy needs at most five steps from any of these starts, because its gradient is the error itself, however confident the mistake. Squared error needs hundreds, then thousands. The last column gives it a learning rate four times larger, which makes up for the largest value \(p(1-p)\) can take, 1/4, and from \(w_0 = -8\) it is still 125 times slower. No constant learning rate can make up for a factor that shrinks like \(e^{-|z|}\). What little MSE learns early on comes from the few points near \(x = 0\), whose logits stay small whatever \(w\) is.

Deep networks hit the same wall whenever an output unit saturates on the wrong side, say after a bad initialization or a loss spike.

## Plateaus and convexity
The factor \(p(1-p)\) also shapes the loss surface. Differentiate once more in \(z\), for a positive example:

\[ \frac{\partial^2 \ell_{\text{CE}}}{\partial z^2} = p(1-p) > 0, \qquad \frac{\partial^2 \ell_{\text{MSE}}}{\partial z^2} = -p\,(1-p)^2\,(1 - 3p) . \]

Cross-entropy curves upward everywhere. Because \(z\) is linear in the weights, the logistic regression loss is convex in \(w\) and \(b\), so gradient descent has no bad local minimum to get stuck in. Squared error curves downward wherever \(p < 1/3\), which is exactly the confidently wrong region, and the result is a plateau before a cliff. On the data above, as a function of \(w\) with \(b = 0\), the MSE loss is 0.484 at \(w = -8\), still 0.474 at \(w = -4\), and only then falls to 0.125 at \(w = 0\). Cross-entropy drops at a steady slope through the same points: 16.1, 8.1, 0.69. With more features, squared error on a single sigmoid unit can have exponentially many local minima in the input dimension (Auer, Herbster and Warmuth, NeurIPS 1995).

```python
# second derivative in the logit, for a positive example, by autograd
z = torch.linspace(-10, 10, 2001, dtype=torch.float64, requires_grad=True)
p = torch.sigmoid(z)
for name, loss in (("ce", F.binary_cross_entropy_with_logits(z, torch.ones_like(z), reduction="sum")),
                   ("mse", (0.5 * (p - 1) ** 2).sum())):
    g, = torch.autograd.grad(loss, z, create_graph=True)
    h, = torch.autograd.grad(g.sum(), z)
    if name == "ce":
        assert torch.all(h > 0) and torch.allclose(h, p * (1 - p))
    else:
        assert torch.allclose(h, -p * (1 - p) ** 2 * (1 - 3 * p))
        assert torch.all(h[z < -np.log(2) - 1e-3] < 0) and torch.all(h[z > -np.log(2) + 1e-3] > 0)

# the training loss as a function of w on the data above, with b = 0
ws = np.linspace(-8, 8, 801)
zs = np.outer(ws, x)
ce_curve = np.mean(np.logaddexp(0, zs) - y * zs, axis=1)    # CE computed from the logit: see below
mse_curve = np.mean(0.5 * (sigmoid(zs) - y) ** 2, axis=1)
assert np.all(np.diff(ce_curve, 2) > 0)                     # convex
assert np.any(np.diff(mse_curve, 2) < 0)                    # not convex
at = {w: int(round((w + 8) * 50)) for w in (-8, -4, 0)}     # grid index of each w
assert [round(mse_curve[at[w]], 3) for w in (-8, -4, 0)] == [0.484, 0.474, 0.125]
assert [round(ce_curve[at[w]], 2) for w in (-8, -4, 0)] == [16.08, 8.06, 0.69]
```

## The likelihood view
The second argument is about what the loss means. A binary label is a coin flip that lands 1 with probability \(p\), the [Bernoulli distribution](../bernoulli-binomial/): \(P(y \mid p) = p^{y}(1-p)^{1-y}\). Its negative log is exactly \(\ell_{\text{CE}}\), so minimizing cross-entropy over a dataset is [maximum likelihood](../statistical-estimation-mle-map/) for that model.

Squared error is maximum likelihood too, for a different model: \(y = p + \varepsilon\) with Gaussian noise of fixed variance, since \(-\log \mathcal{N}(y;\, p, \sigma^2) = (y - p)^2/(2\sigma^2) + \text{const}\) (see [regression as maximum likelihood](../regression-as-maximum-likelihood/)). A label that can only be 0 or 1 has no Gaussian noise around it, and its variance is not fixed: a coin with \(p = 0.5\) varies most, a coin with \(p = 0.99\) hardly at all. The gradients in \(p\) show the difference exactly:

\[ \frac{\partial \ell_{\text{CE}}}{\partial p} = \frac{p - y}{p(1-p)} = \frac{\partial \ell_{\text{MSE}} / \partial p}{\operatorname{Var}(y)} . \]

Cross-entropy is squared error divided by the Bernoulli variance, the inverse-variance weighting that weighted least squares would choose for this noise. A miss at \(p = 0.99\) is strong evidence, since a coin that biased rarely lands the other way, so cross-entropy weights it heavily. Squared error treats it like any other residual.

### Both are proper, only one is the likelihood
None of this makes squared error wrong in the limit of infinite data. Both losses are **strictly proper scoring rules**: if the true probability is \(p\), the prediction that minimizes the expected loss is \(p\) itself, for either one. Squared error on probabilities, without the \(\tfrac12\), is the Brier score, a standard measure of probability forecasts and of [calibration](/posts/evaluating-ml-systems/). The losses differ in how fast gradient descent gets there and in how they price a confident mistake: predicting 0.001 for a positive costs 6.91 in log loss but only 0.998 in Brier score, whose maximum is 1.

```python
from scipy.optimize import minimize_scalar


def log_loss(q, y):
    return -np.log(q if y else 1 - q)


def brier(q, y):
    return (q - y) ** 2


for p_true in (0.05, 0.3, 0.8):
    for loss in (log_loss, brier):
        expected = lambda q: p_true * loss(q, 1) + (1 - p_true) * loss(q, 0)
        best = minimize_scalar(expected, bounds=(1e-9, 1 - 1e-9), method="bounded",
                               options={"xatol": 1e-9}).x
        assert abs(best - p_true) < 1e-5                  # both recover the true probability

assert round(log_loss(0.001, 1), 2) == 6.91 and round(brier(0.001, 1), 3) == 0.998

# dCE/dp is dMSE/dp divided by the Bernoulli variance p(1 - p)
q = torch.linspace(0.01, 0.99, 99, dtype=torch.float64, requires_grad=True)
for label in (0.0, 1.0):
    target = torch.full_like(q, label)
    d_ce, = torch.autograd.grad(F.binary_cross_entropy(q, target, reduction="sum"), q)
    d_mse, = torch.autograd.grad((0.5 * (q - target) ** 2).sum(), q)
    assert torch.allclose(d_ce, d_mse / (q * (1 - q)))
```

## More than two classes
With \(K\) classes the sigmoid becomes a softmax and the same cancellation happens: the gradient of cross-entropy with respect to the logits is \(\operatorname{softmax}(z) - \operatorname{onehot}(y)\), the predicted distribution minus the true one. A two-class softmax over the logits \((0, z)\) gives class 1 probability \(\sigma(z)\), so the binary case is a special case. [Stable softmax cross-entropy](../stable-softmax-cross-entropy/) builds the multiclass version from scratch, and [entropy and information](../entropy-and-information/) explains where the name comes from.

```python
torch.manual_seed(0)
logits = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
labels = torch.tensor([0, 3, 1, 1, 2, 0])
grad, = torch.autograd.grad(F.cross_entropy(logits, labels, reduction="sum"), logits)
assert torch.allclose(grad, torch.softmax(logits, dim=1) - F.one_hot(labels, 4))

zb = torch.linspace(-8, 8, 33, dtype=torch.float64)
two_class = torch.stack([torch.zeros_like(zb), zb], dim=1)
assert torch.allclose(torch.softmax(two_class, dim=1)[:, 1], torch.sigmoid(zb))
```

## Computing it safely
The clean gradient survives only if the code never takes the log of a rounded probability. In float32, \(\sigma(-120) \approx e^{-120} \approx 10^{-52}\) is below the smallest representable number, so it rounds to 0 and its log is \(-\infty\). Compute the loss from the logit instead:

\[ \ell_{\text{CE}}(z, y) = \log\!\left(1 + e^{z}\right) - yz = \max(z, 0) - yz + \log\!\left(1 + e^{-|z|}\right) . \]

The right-hand form never exponentiates a positive number. PyTorch's `binary_cross_entropy_with_logits` (the functional form of `BCEWithLogitsLoss`) evaluates an equivalent stable form, which is why the documentation recommends it over a sigmoid followed by `BCELoss`.

A common workaround is worse than the problem: clamping \(p\) to \([10^{-7}, 1 - 10^{-7}]\) before the log. That avoids the infinity, but a clamp has zero gradient outside its range, so every example wrong by more than the clamp allows (for a positive, any logit below \(\ln 10^{-7} \approx -16\)), the very examples the clean gradient was meant to fix, contributes nothing at all.

```python
z = torch.tensor([-120.0, -20.0, 0.0, 20.0, 120.0])        # float32, the usual training dtype
ones = torch.ones(5)
naive = -torch.log(torch.sigmoid(z))
stable = F.binary_cross_entropy_with_logits(z, ones, reduction="none")
by_hand = torch.clamp(z, min=0) - ones * z + torch.log1p(torch.exp(-z.abs()))
assert torch.isinf(naive[0]) and torch.allclose(stable, by_hand)
assert abs(stable[0].item() - 120) < 1e-3                   # the loss for z = -120 is just 120

# clamping the probability silences exactly the examples that most need a gradient
z = torch.tensor([-20.0], requires_grad=True)
clamped = torch.sigmoid(z).clamp(1e-7, 1 - 1e-7)
(-torch.log(clamped)).sum().backward()
assert z.grad.item() == 0.0
z.grad = None
F.binary_cross_entropy_with_logits(z, torch.ones(1)).backward()
assert abs(z.grad.item() + 1) < 1e-6                        # from the logit: full strength, -1
```

## When squared error is the right call
- **Regression on a continuous target** with roughly constant noise: prices, durations, ratings. That is the Gaussian likelihood, and there is no sigmoid to saturate.
- **Scoring probabilities.** The Brier score is bounded, splits into calibration, resolution and uncertainty terms, and is not dominated by a single confident miss, which makes it a steady metric to report next to log loss.
- **Noisy labels.** Cross-entropy's unbounded penalty is the flip side of its strength: a mislabeled example that the model rightly finds implausible gets the largest gradient in the batch. Bounded losses tolerate label noise better, and generalized cross-entropy (Zhang and Sabuncu, NeurIPS 2018) moves between cross-entropy and mean absolute error with a single parameter.
- **Squared error on the raw outputs.** Regressing the logits themselves onto one-hot targets has no saturating factor at all, and Hui and Belkin (ICLR 2021) found it competitive with cross-entropy on a range of classification benchmarks. The case against squared error is specifically squared error behind a sigmoid.

## Pitfalls
- **A sigmoid applied twice.** A model that ends in `torch.sigmoid` and is trained with `BCEWithLogitsLoss` squashes its outputs into \((0.5, 0.731)\): it can never predict a negative, and the loss on a positive cannot fall below 0.313. The loss still goes down a little, so the bug is easy to miss.
- **Clamped probabilities**, as above: no infinities, and no gradient for the worst mistakes.
- **Class imbalance.** The gradient \(p - y\) is summed over examples, so a 99:1 split lets the majority class set the direction. Weight the rare class (`pos_weight` in PyTorch) or resample; either keeps the clean gradient, just rescaled.

```python
z = torch.linspace(-50, 50, 1001)
twice = torch.sigmoid(torch.sigmoid(z))                     # what the loss sees if the model already applied it
assert twice.min() >= 0.5 and twice.max() < 0.732
floor = F.binary_cross_entropy_with_logits(torch.sigmoid(z), torch.ones_like(z), reduction="none").min()
assert round(floor.item(), 3) == 0.313
```

The step counts come from one synthetic dataset with a single feature and evenly spaced normal quantiles, so they show how the factor \(p(1-p)\) slows training, not how many steps a real model loses. I train no deep network, so the claim that a saturated output unit stalls one rests on the gradient algebra alone. Two questions stay untested here: how much label noise it takes before a bounded loss wins, and whether squared error on the raw logits matches cross-entropy beyond the benchmarks Hui and Belkin used.

The derivations follow the treatment of sigmoid output units in Goodfellow, Bengio and Courville, Deep Learning (MIT Press, 2016), section 6.2.2, and of logistic regression in Bishop, Pattern Recognition and Machine Learning (Springer, 2006), section 4.3.2.
