# Numerically stable softmax and cross-entropy

Softmax and cross-entropy are one line of math each, and the one-line code for each breaks on inputs a real network can produce. In float32, a logit of 100 overflows, and a gap of 120 between two logits makes a probability exactly zero, and the log of that zero is infinite. This post derives the two fixes every deep learning framework uses: subtract the maximum before exponentiating, and compute log-probabilities directly with log-sum-exp instead of taking the log of a probability. Each fix is checked against a brute force, and the post ends with the common bugs that quietly undo them.

Throughout, \(z \in \mathbb{R}^K\) is a vector of logits (the raw scores a network outputs for \(K\) classes) and \(y\) is the index of the correct class.

## Where floating point runs out
Softmax turns logits into probabilities:

\[ \operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \]

The trouble is \(e^{x}\), which leaves the range of representable numbers at quite modest arguments:

| Type | \(e^{x}\) overflows to inf above | \(e^{x}\) underflows to 0 below |
|---|---|---|
| float16 | about 11.1 | about −17.3 |
| float32 | about 88.7 | about −104.0 |
| float64 | about 709.8 | about −745.1 |

Overflow makes the numerator and the denominator both infinite, and inf / inf is nan. Underflow is just as bad: if every logit is very negative, every exponential is 0 and the result is 0 / 0, also nan.

```python
import math
from decimal import Decimal, getcontext

import numpy as np
import scipy.special as sp
import torch
import torch.nn.functional as F


def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


with np.errstate(all="ignore"):                                       # the warnings are the point here
    assert np.isnan(softmax_naive(np.array([1000.0, 999.0]))).all()    # inf / inf
    assert np.isnan(softmax_naive(np.array([-1000.0, -999.0]))).all()  # 0 / 0
    assert np.isinf(np.exp(np.float16(12.0)))                          # half precision gives up at 11.1
```

## Softmax ignores a constant shift
Multiply the numerator and the denominator by \(e^{-c}\) for any constant \(c\):

\[ \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{-c}\, e^{z_i}}{\sum_j e^{-c}\, e^{z_j}} = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} \]

Subtracting the same number from every logit leaves softmax unchanged. The useful choice is \(c = m = \max_j z_j\), which buys two guarantees:

- Every exponent \(z_j - m\) is at most 0, so every exponential is at most 1 and nothing overflows.
- The largest logit contributes exactly \(e^0 = 1\), so the denominator is at least 1. It is never zero, and the output is never all zeros.

No other shift gives both. Any \(c\) below the max can still overflow: the mean of the logits \([0, 1000]\) is 500, and \(e^{500}\) still overflows float32. Any \(c\) above the max can underflow every term to zero.

```python
def softmax(z, axis=-1):
    z = np.asarray(z)
    e = np.exp(z - z.max(axis=axis, keepdims=True))   # exponents <= 0, the largest is exactly 0
    return e / e.sum(axis=axis, keepdims=True)        # so the denominator is >= 1


rng = np.random.default_rng(0)
batch = rng.normal(size=(5, 7))
assert np.allclose(softmax(batch), softmax_naive(batch))      # same answer where the naive one works
assert np.allclose(softmax(batch + 1234.5), softmax(batch))    # shift invariance
assert np.allclose(softmax(50 * batch), sp.softmax(50 * batch, axis=-1))

r = 1 / (1 + math.exp(-1))                                     # softmax([1, 0]) = [r, 1 - r]
assert np.allclose(softmax([1000.0, 999.0]), [r, 1 - r])
assert np.allclose(softmax([-1000.0, -999.0]), [1 - r, r])
assert np.isfinite(softmax(np.array([12.0, 10.0, 3.0], dtype=np.float16))).all()

pair = np.array([0.0, 1000.0], dtype=np.float32)
with np.errstate(all="ignore"):
    assert np.isnan(softmax_naive(pair - pair.mean())).any()   # shift by the mean: e^500 overflows
    assert np.isnan(softmax_naive(pair - 1200.0)).all()        # shift past the max: 0 / 0
```

## Never take the log of a probability
Cross-entropy between a target distribution \(p\) and a predicted distribution \(q\) is

\[ H(p, q) = -\sum_i p_i \log q_i = H(p) + D_{\mathrm{KL}}(p \,\|\, q) \]

It is not a similarity score, and it is not symmetric in \(p\) and \(q\). It is the entropy of the targets plus the KL divergence \(D_{\mathrm{KL}}(p \,\|\, q)\), so for fixed targets, minimizing it is the same as minimizing the divergence, and it reaches its smallest value, \(H(p)\), at \(q = p\) ([entropy and information](../entropy-and-information/) has the details). With a hard label, \(p\) is one-hot on class \(y\) and the sum collapses to a single term, \(-\log q_y\): the negative log-likelihood of the correct class. [Cross-entropy vs MSE](../cross-entropy-vs-mse/) covers why this, and not squared error, is the classification loss.

The stable softmax does not make \(-\log q_y\) safe. When the model is confidently wrong, \(q_y\) is a legitimately tiny number such as \(e^{-120}\), which float32 rounds to exactly 0, so the loss comes out infinite. Those are exactly the examples the loss most needs to learn from.

```python
z = np.array([0.0, 120.0], dtype=np.float32)   # confidently wrong: the correct class is 0
p = softmax(z)
assert p[0] == 0.0                              # e^-120 is below float32's smallest positive number
with np.errstate(divide="ignore"):
    assert np.isinf(-np.log(p[0]))              # the loss is inf; the true value is about 120
```

The fix is never to form \(q_y\) at all. Take the log of the shifted softmax and simplify:

\[ \log \operatorname{softmax}(z)_i = z_i - m - \log \sum_j e^{z_j - m} = z_i - \operatorname{LSE}(z) \]

Here \(\operatorname{LSE}\) is log-sum-exp, evaluated in the shifted form on the right:

\[ \operatorname{LSE}(z) = \log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m} \]

The sum inside the log lies between 1 and \(K\), so the log lies between 0 and \(\log K\). It is never the log of zero, and it gives the bounds \(\max_j z_j \le \operatorname{LSE}(z) \le \max_j z_j + \log K\), which is why log-sum-exp is often called a smooth maximum. The loss for one example is now the difference of two finite numbers:

\[ \ell(z, y) = -\log \operatorname{softmax}(z)_y = \operatorname{LSE}(z) - z_y \]

To check the code beyond the range where naive floats work, the brute force below uses Python's `decimal` module at 50 significant digits. Its exponent range is so wide that \(e^{900}\) is just a large number.

```python
def logsumexp(z, axis=-1):
    z = np.asarray(z)
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def log_softmax(z, axis=-1):
    z = np.asarray(z)
    return z - np.expand_dims(logsumexp(z, axis), axis)


def cross_entropy(logits, targets):
    """Mean of -log softmax(logits)[target]: logits (n, K), integer targets (n,)."""
    logits = np.asarray(logits)
    return np.mean(logsumexp(logits) - logits[np.arange(len(logits)), targets])


assert np.isclose(cross_entropy(z[None], [0]), 120.0)          # finite, and right

lse = logsumexp(50 * batch)
assert np.allclose(lse, sp.logsumexp(50 * batch, axis=-1))
assert (batch.max(-1) * 50 <= lse).all() and (lse <= 50 * batch.max(-1) + np.log(7)).all()
assert np.allclose(log_softmax(50 * batch), sp.log_softmax(50 * batch, axis=-1))

getcontext().prec = 50


def cross_entropy_decimal(row, target):          # brute force with no overflow or underflow
    total = sum(Decimal(float(v)).exp() for v in row)
    return float(total.ln() - Decimal(float(row[target])))


big = rng.normal(scale=300, size=(20, 6))        # logits in the hundreds
y_big = rng.integers(0, 6, size=20)
with np.errstate(all="ignore"):
    assert not np.isfinite(-np.log(softmax_naive(big)[np.arange(20), y_big])).all()   # naive fails here
exact = np.mean([cross_entropy_decimal(row, t) for row, t in zip(big, y_big)])
assert math.isclose(cross_entropy(big, y_big), exact, rel_tol=1e-12)

logits = rng.normal(scale=30, size=(64, 10))
y = rng.integers(0, 10, size=64)
assert math.isclose(cross_entropy(logits, y),
                    F.cross_entropy(torch.tensor(logits), torch.tensor(y)).item(), rel_tol=1e-12)
```

## The gradient is softmax minus one-hot
The derivative of log-sum-exp is softmax, because \(\partial \operatorname{LSE} / \partial z_i = e^{z_i} / \sum_j e^{z_j}\). So

\[ \frac{\partial \ell}{\partial z_i} = \operatorname{softmax}(z)_i - \mathbb{1}[i = y] \]

Every component lies in \([-1, 1]\) and the components sum to zero, however large the loss. That is the second reason frameworks fuse softmax and cross-entropy into one operation: the backward pass never divides by a probability. Going through \(q\) instead, the chain rule multiplies \(-1/q_y\) by the softmax Jacobian, which contains \(q_y\). Once \(q_y\) has underflowed, that product is \(\infty \cdot 0\), which is nan.

```python
def cross_entropy_grad(logits, targets):
    """Gradient of the mean cross-entropy with respect to the logits."""
    g = softmax(logits)
    g[np.arange(len(logits)), targets] -= 1
    return g / len(logits)


t = torch.tensor(logits, requires_grad=True)
F.cross_entropy(t, torch.tensor(y)).backward()
g = cross_entropy_grad(logits, y)
assert np.allclose(g, t.grad.numpy())
assert np.allclose(g.sum(axis=1), 0) and (np.abs(g) <= 1 / len(logits)).all()

small, y_small, h = rng.normal(size=(3, 4)), np.array([0, 3, 1]), 1e-6   # central finite differences
numeric = np.zeros_like(small)
for idx in np.ndindex(small.shape):
    d = np.zeros_like(small)
    d[idx] = h
    numeric[idx] = (cross_entropy(small + d, y_small) - cross_entropy(small - d, y_small)) / (2 * h)
assert np.allclose(numeric, cross_entropy_grad(small, y_small), atol=1e-8)

zt = torch.tensor([[0.0, 120.0]], requires_grad=True)       # the log-of-softmax route
(-torch.log(torch.softmax(zt, dim=1)[0, 0])).backward()
assert torch.isnan(zt.grad).all()                           # inf loss, nan gradient

zt = torch.tensor([[0.0, 120.0]], requires_grad=True)       # the fused route
F.cross_entropy(zt, torch.tensor([0])).backward()
assert torch.allclose(zt.grad, torch.tensor([[-1.0, 1.0]]))  # softmax minus one-hot
```

## The binary case: softplus
With one logit per example, a label \(y \in \{0, 1\}\) and \(\sigma(z) = 1 / (1 + e^{-z})\), binary cross-entropy is \(-[y \log \sigma(z) + (1 - y) \log(1 - \sigma(z))]\). Since \(\log \sigma(z) = -\log(1 + e^{-z})\) and \(\log(1 - \sigma(z)) = -\log(1 + e^{z})\), it simplifies to

\[ \ell(z, y) = \operatorname{softplus}(z) - y z, \qquad \operatorname{softplus}(z) = \log(1 + e^{z}) \]

Softplus is log-sum-exp over the pair \([0, z]\), because a sigmoid is a two-class softmax with one logit pinned at 0. The same max shift therefore applies:

\[ \operatorname{softplus}(z) = \max(z, 0) + \log\left(1 + e^{-|z|}\right) \]

Compute the last term with `log1p`, which stays accurate when \(e^{-|z|}\) is tiny. The naive route fails even sooner than in the multi-class case: in float32, \(\sigma(20)\) rounds to exactly 1, so a negative example with logit 20 gets an infinite loss.

```python
def bce_with_logits(z, y):
    """Mean binary cross-entropy from logits: softplus(z) - y * z, with a stable softplus."""
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    softplus = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    return np.mean(softplus - y * z)


zb = rng.normal(scale=50, size=1000)
yb = rng.integers(0, 2, size=1000)
assert math.isclose(bce_with_logits(zb, yb),
                    F.binary_cross_entropy_with_logits(torch.tensor(zb), torch.tensor(yb, dtype=torch.float64)).item(),
                    rel_tol=1e-12)
assert np.allclose(np.maximum(zb, 0) + np.log1p(np.exp(-np.abs(zb))),
                   sp.logsumexp(np.stack([np.zeros_like(zb), zb]), axis=0))      # softplus = LSE([0, z])

s = torch.sigmoid(torch.tensor(20.0))
assert s == 1.0 and torch.isinf(-torch.log(1 - s))          # naive: inf for a negative example
assert bce_with_logits([20.0], [0]) == 20.0 + math.log1p(math.exp(-20.0))
```

## Pitfalls
- **Probabilities fed to a loss that expects logits.** PyTorch's `F.cross_entropy` and `nn.CrossEntropyLoss` apply log-softmax themselves. A softmax layer in front of them raises no error, but inputs confined to [0, 1] can never be confident, so the loss can never fall below \(\log(1 + (K - 1)/e)\), about 1.46 nats for ten classes, however good the model. Keras losses take a `from_logits` flag, and it has to match what the model actually outputs.
- **The epsilon hack.** `log(p + 1e-12)` turns inf into a finite number, but not the right one. It caps the loss at \(-\log 10^{-12} \approx 27.6\), and once \(p\) has underflowed to 0 the gradient is exactly zero, so the worst-classified examples stop contributing to training.
- **Fully masked rows.** Attention sets disallowed scores to \(-\infty\) before the softmax (see [attention scaling](../attention-scaling/)). If every entry in a row is masked, as for a padding position, the max is \(-\infty\) and \(-\infty - (-\infty)\) is nan, which then propagates through later layers. Fill with the dtype's most negative finite value instead, or skip such rows explicitly.
- **Half precision.** In float16, \(e^{x}\) overflows just above 11. The max shift protects the forward pass, but long sums of small terms and the final log are still better done in float32. PyTorch's autocast runs softmax and log-softmax in float32 for this reason.
- **Soft targets and distillation.** With label smoothing or a teacher's distribution as the target, use \(-\sum_i p_i \log \operatorname{softmax}(z)_i\) with a real log-softmax. `F.cross_entropy` accepts class probabilities as the target, and `F.kl_div` expects log-probabilities as its first argument, not probabilities.

```python
K = 10
floor = cross_entropy(np.eye(K)[:1], [0])        # the most confident "probability" input: [1, 0, ..., 0]
assert math.isclose(floor, math.log(1 + (K - 1) / math.e)) and 1.46 < floor < 1.47
assert cross_entropy(softmax(logits), y) >= floor   # double softmax: stuck above the floor, whatever the model

zt = torch.tensor([0.0, 120.0], requires_grad=True)
loss = -torch.log(torch.softmax(zt, dim=0)[0] + 1e-12)
loss.backward()
assert math.isclose(loss.item(), 12 * math.log(10), rel_tol=1e-6)   # 27.6 instead of 120
assert (zt.grad == 0).all()                                         # the true gradient is [-1, 1]

scores = torch.tensor([[2.0, 1.0, 0.5], [0.3, 0.2, 0.1]])
mask = torch.tensor([[True, True, False], [False, False, False]])   # row 2 may attend to nothing
assert torch.isnan(torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)[1]).all()
safe = torch.softmax(scores.masked_fill(~mask, torch.finfo(scores.dtype).min), dim=-1)
assert torch.isfinite(safe).all() and safe[0, 2] == 0


def soft_cross_entropy(logits, target_probs):
    return np.mean(-(target_probs * log_softmax(logits)).sum(axis=-1))


teacher = softmax(rng.normal(scale=3, size=(64, 10)))
assert math.isclose(soft_cross_entropy(logits, teacher),
                    F.cross_entropy(torch.tensor(logits), torch.tensor(teacher)).item(), rel_tol=1e-12)
assert math.isclose(soft_cross_entropy(logits, np.eye(10)[y]), cross_entropy(logits, y), rel_tol=1e-12)

kl = F.kl_div(F.log_softmax(torch.tensor(logits), dim=-1), torch.tensor(teacher), reduction="batchmean").item()
teacher_entropy = np.mean(-(teacher * np.log(teacher)).sum(axis=-1))
assert math.isclose(kl, soft_cross_entropy(logits, teacher) - teacher_entropy, rel_tol=1e-10)   # H(p, q) = H(p) + KL
```

## Which function to call
| You have | Compute | In PyTorch |
|---|---|---|
| Logits and class indices | \(\operatorname{LSE}(z) - z_y\) | `F.cross_entropy(logits, y)` |
| Logits and soft targets | \(-\sum_i p_i \log \operatorname{softmax}(z)_i\) | `F.cross_entropy(logits, probs)` |
| One logit and a 0/1 label | \(\operatorname{softplus}(z) - y z\) | `F.binary_cross_entropy_with_logits` |
| Log-probabilities for scoring or sampling | \(z - \operatorname{LSE}(z)\) | `F.log_softmax(logits, dim=-1)` |
| Probabilities to show or threshold | the max-shifted softmax | `torch.softmax(logits, dim=-1)` |

The rule behind the table: keep everything as logits or log-probabilities for as long as possible, and turn them into probabilities only at the end, when a person needs to read them or a threshold needs to be applied.

The outline of this post, the max shift followed by the log-softmax rearrangement, follows Jay Mody's post [Numerically Stable Softmax and Cross Entropy](https://jaykmody.com/blog/stable-softmax/).
