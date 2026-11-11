---
title: Dropout: random masks, ensembles and a hidden L2 penalty
summary: What dropout does in a forward pass, why the survivors are scaled by 1/(1-p), and what it buys on a small network. Then the two exact results behind the "ensemble" story, the L2 penalty it becomes on a linear model, why it should sit after the last batch-norm layer rather than before one, and where it is still used.
group: Machine learning fundamentals
order: 8
tags: regularization · dropout · ensembles · neural networks
---
Fit a network with about 17,000 weights to 20 noisy points and it memorizes the noise: training MSE 0.03 on noisy points, test MSE 0.17 on the true curve. Dropout moves those to 0.13 and 0.12. It is the simplest regularizer in deep learning to write down: on every training step, set each unit's output to zero with probability \(p\), independently, and use a fresh random pattern on the next step. At inference nothing is dropped. A unit can no longer count on any particular neighbour being present, so the network has to spread what it knows across many units instead of building fragile combinations that only work together.

I write dropout from scratch, check it against PyTorch and show what it does to both errors, then make three common one-liners precise. "It trains an ensemble" is exact only in special cases. "It is like weight decay" is exact for a linear model, with a twist. And "never use it with batch norm" is really a rule about placement: the trouble comes from dropout that feeds a batch-norm layer.

## The forward pass
For an activation vector \(h\), draw a mask \(m\) with independent entries \(m_i \sim \text{Bernoulli}(1 - p)\) and output

\[ \tilde h = \frac{m \odot h}{1 - p} \]

Each entry is either zero or \(h_i / (1 - p)\). Dividing by the keep probability is **inverted dropout**: it makes the expected output equal to the input,

\[ \mathbb{E}[\tilde h_i] = (1 - p) \cdot \frac{h_i}{1 - p} + p \cdot 0 = h_i \]

so the layer can simply be switched off at inference. Without that division, training would see activations that are on average a factor \(1 - p\) smaller than the ones at test time, and every later layer would be tuned to the wrong scale. The original papers ([Hinton et al., 2012](https://arxiv.org/abs/1207.0580); [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)) did the equivalent thing the other way round: no scaling during training, and the outgoing weights multiplied by the keep probability at test time. Both give the same function; frameworks implement the inverted form, which leaves inference untouched.

The backward pass uses the same mask: a dropped unit gets no gradient on that step, and a kept one gets its gradient scaled by \(1 / (1 - p)\).

```python
import numpy as np
import torch

torch.set_num_threads(1)


def dropout(h, p, training, rng):
    if not training or p == 0:
        return h
    mask = rng.random(h.shape) >= p                   # keep with probability 1 - p
    return h * mask / (1 - p)


rng = np.random.default_rng(0)
h = rng.normal(size=(2000, 500))
out = dropout(h, 0.3, True, rng)
assert abs((out == 0).mean() - 0.3) < 0.005           # about p of the units are zeroed
kept = out != 0
assert np.allclose(out[kept], h[kept] / 0.7)          # the rest are scaled up by 1 / (1 - p)
draws = np.stack([dropout(h[:5, :5], 0.3, True, rng) for _ in range(40_000)])
assert np.allclose(draws.mean(0), h[:5, :5], atol=0.03)    # unbiased: the expectation is h itself
assert dropout(h, 0.3, False, rng) is h               # identity at inference

# PyTorch's nn.Dropout does the same, and its gradient goes through the same mask
torch.manual_seed(0)
x = torch.randn(1000, 100, requires_grad=True)
layer = torch.nn.Dropout(p=0.3)
y = layer(x)
y.sum().backward()
on = y.detach() != 0
assert torch.allclose(y.detach()[on], x.detach()[on] / 0.7)
assert torch.allclose(x.grad, on / 0.7)               # dropped units receive no gradient
assert torch.equal(layer.eval()(x), x)

# Classic dropout (no training scale, weights times the keep probability at test) is the same function
W = rng.normal(size=(500, 10))
mask = rng.random(h.shape) >= 0.3
assert np.allclose((h * mask / 0.7) @ W, (h * mask) @ (W / 0.7))   # identical training passes
assert np.allclose(h @ W, h @ (0.7 * (W / 0.7)))                   # identical test passes
```

## What it buys
A network with about 17,000 weights fitted to 20 noisy points is the standard way to overfit on purpose. The targets are \(\sin(x)\) plus noise with standard deviation 0.5, so a model that had learned only the signal would have a training MSE near 0.25. The block below trains the same two-hidden-layer network with and without dropout (\(p = 0.5\) after each hidden layer) on four datasets, and measures the test error against the noiseless curve.

| | Train MSE | Test MSE (against \(\sin x\)) |
|---|---|---|
| No dropout | 0.03 | 0.17 |
| Dropout, \(p = 0.5\) | 0.13 | 0.12 |

Without dropout the training error falls far below the noise level: the network memorized the noise, and test error pays for it. Dropout makes memorizing hard: a wiggle through one training point has to survive whichever half of the units is missing. Training error goes up and test error goes down, as a regularizer should (the [bias–variance trade-off](../bias-variance-tradeoff/) again). These numbers come from one sine-wave task averaged over four seeds, so they show the mechanism, not how much dropout will help on other data.

```python
def fit(p, seed, n=20, steps=1500, width=128):
    torch.manual_seed(seed)
    x = torch.rand(n, 1) * 6 - 3
    y = torch.sin(x) + 0.5 * torch.randn_like(x)
    net = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.ReLU(), torch.nn.Dropout(p),
                              torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Dropout(p),
                              torch.nn.Linear(width, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        ((net(x) - y) ** 2).mean().backward()
        opt.step()
    net.eval()                                         # dropout off for evaluation
    x_test = torch.linspace(-3, 3, 500).unsqueeze(1)
    with torch.no_grad():
        return net, ((net(x) - y) ** 2).mean().item(), ((net(x_test) - torch.sin(x_test)) ** 2).mean().item()


runs = {p: [fit(p, seed) for seed in range(4)] for p in (0.0, 0.5)}
train = {p: np.mean([r[1] for r in rs]) for p, rs in runs.items()}
test = {p: np.mean([r[2] for r in rs]) for p, rs in runs.items()}
assert train[0.0] < 0.06 < 0.25                       # far below the noise variance: memorized
assert train[0.5] > 2 * train[0.0]                    # dropout fits the training set less closely
assert test[0.5] < 0.8 * test[0.0]                    # and the noiseless curve better
```

## The ensemble story, made precise
A layer with \(n\) droppable units has \(2^n\) possible masks, and each mask picks out a thinned sub-network. Training with dropout trains a random one of them for each example on each step, and all of them share the same weights. Switching dropout off at test time is then described as averaging all \(2^n\) sub-networks. That is true in two cases and only approximately true otherwise.

**Linear after the dropout.** If everything between the dropout and the output is linear, the deterministic output is exactly the average of the sub-networks' outputs, by the unbiasedness shown above.

**One softmax layer.** For a softmax classifier with dropout on its inputs, the test-time output is exactly the **normalized geometric mean** of all \(2^n\) sub-networks' predicted distributions, weighted by mask probability. The log of each sub-network's prediction is its logits minus a normalizer that is the same for every class, so averaging log-probabilities averages the logits, and the average of the logits is the deterministic logits. Note that this is the geometric mean, not the arithmetic mean of the probabilities that an ordinary ensemble would use.

**A nonlinearity after the dropout.** Then the expected output is not the output at the expected input, and the deterministic network is only an approximation of the ensemble average. Srivastava et al. found it a good one on trained networks; with the random weights below it is visibly off. The block checks all three claims by enumerating every mask of a 10-unit layer.

```python
from itertools import product
from scipy.special import log_softmax, softmax

rng = np.random.default_rng(1)
n, p = 10, 0.3
x = rng.normal(size=n)
W, W1, W2 = rng.normal(size=(4, n)), rng.normal(size=(16, n)), rng.normal(size=(4, 16))
masks = np.array(list(product([0, 1], repeat=n)))     # all 2^10 = 1024 sub-networks
prob = np.prod(np.where(masks == 1, 1 - p, p), axis=1)
assert len(masks) == 1024 and np.isclose(prob.sum(), 1)
x_drop = masks * x / (1 - p)                          # each row is one sub-network's input

logits = x_drop @ W.T
assert np.allclose(prob @ logits, W @ x)              # linear: the average is exact

geo = softmax(prob @ log_softmax(logits, axis=1))     # normalized, weighted geometric mean
assert np.allclose(geo, softmax(W @ x))               # softmax: exact, as the geometric mean
assert not np.allclose(prob @ softmax(logits, axis=1), softmax(W @ x), atol=1e-3)   # not the arithmetic mean

relu = lambda z: np.maximum(z, 0)
avg_out = prob @ (relu(x_drop @ W1.T) @ W2.T)         # ReLU after the dropout
assert not np.allclose(avg_out, W2 @ relu(W1 @ x), atol=0.1)    # the deterministic net is an approximation
```

Since dropout is a distribution over sub-networks, it can also be left on at test time on purpose. Averaging \(T\) stochastic passes approximates the ensemble average more faithfully, and their spread is a cheap uncertainty estimate: Gal and Ghahramani ([2016](https://arxiv.org/abs/1506.02142)) call this **Monte Carlo dropout** and relate it to approximate Bayesian inference. On the regression network above, the spread is several times larger far outside the training range than inside it.

```python
net = runs[0.5][0][0].train()                         # dropout on again, deliberately
grid = torch.tensor([[0.0], [1.5], [8.0], [-8.0]])
with torch.no_grad():
    samples = torch.stack([net(grid) for _ in range(500)]).squeeze(-1)
std = samples.std(0)
assert std[2:].min() > 3 * std[:2].max()              # uncertain where there was no data
net.eval()
```

## On a linear model, an adaptive L2 penalty
Apply inverted dropout to the inputs of linear regression and average the squared loss over masks. The noisy prediction \(\tilde x^\top w\) has mean \(x^\top w\) and variance \(\frac{p}{1 - p} \sum_j x_j^2 w_j^2\), so

\[ \mathbb{E}_m \lVert y - \tilde X w \rVert^2 = \lVert y - X w \rVert^2 + \frac{p}{1 - p} \sum_j \Big( \sum_i x_{ij}^2 \Big) w_j^2 \]

That is ridge regression ([L1 and L2 regularization](../regularization-l1-l2/)) with a separate penalty on each weight, proportional to how large its feature is in the data. The minimizer is \(\big(X^\top X + \tfrac{p}{1-p} \operatorname{diag}(X^\top X)\big)^{-1} X^\top y\). Unlike weight decay, the penalty is unchanged when a feature is rescaled, because the weight rescales the other way. Put differently, dropout on a linear regression is plain ridge with a single strength \(\tfrac{p}{1-p}\) applied after dividing each feature by its root sum of squares, so the penalty adapts to the data instead of to arbitrary units. Wager, Wang and Liang ([2013](https://arxiv.org/abs/1307.1493)) develop this view of dropout as adaptive regularization; for logistic regression they show the penalty also weights each example by the model's predicted variance, so it differs from ridge in more than scale. In a deep network the penalty also depends on the activations, so there is no closed form, but the intuition carries over: dropout regularizes in activation space, weight decay in parameter space, and the two are routinely used together.

```python
rng = np.random.default_rng(2)
N, d, p = 60, 8, 0.4
X = rng.normal(size=(N, d)) * rng.uniform(0.2, 3, size=d)     # features on different scales
y = X @ rng.normal(size=d) + rng.normal(size=N)
w = rng.normal(size=d)

masks = np.array(list(product([0, 1], repeat=d)))
prob = np.prod(np.where(masks == 1, 1 - p, p), axis=1)
X_tilde = X[None] * masks[:, None, :] / (1 - p)               # every masked copy of the data
expected_loss = prob @ ((y - X_tilde @ w) ** 2).sum(axis=1)
col_sq = (X**2).sum(axis=0)
assert np.isclose(expected_loss, ((y - X @ w) ** 2).sum() + p / (1 - p) * col_sq @ w**2)

w_dropout = np.linalg.solve(np.einsum("k,kni,knj->ij", prob, X_tilde, X_tilde),
                            np.einsum("k,kni,n->i", prob, X_tilde, y))   # minimizer of the exact average
w_ridge = np.linalg.solve(X.T @ X + p / (1 - p) * np.diag(col_sq), X.T @ y)
assert np.allclose(w_dropout, w_ridge)

X_scaled = X * np.array([1000, 1, 1, 1, 1, 1, 1, 1])          # change one feature's units
w_scaled = np.linalg.solve(X_scaled.T @ X_scaled + p / (1 - p) * np.diag((X_scaled**2).sum(0)), X_scaled.T @ y)
assert np.allclose(w_scaled * [1000, 1, 1, 1, 1, 1, 1, 1], w_ridge)   # same fit, weight rescaled

Z = X / np.sqrt(col_sq)                                       # each column scaled to unit norm
w_z = np.linalg.solve(Z.T @ Z + p / (1 - p) * np.eye(d), Z.T @ y)
assert np.allclose(w_z / np.sqrt(col_sq), w_ridge)            # ordinary ridge on rescaled features
```

## Dropout and batch norm
Dropout preserves the mean of its input, but not the variance. In training, the variance of an entry after inverted dropout is \(\mathbb{E}[h^2] / (1 - p) - \mathbb{E}[h]^2\), larger than the variance of \(h\); at test time dropout is off and the variance falls back. A [batch norm](../batchnorm-vs-layernorm/) layer placed directly after the dropout learns its running variance from the inflated training signal and then, at inference, divides a smaller signal by it. Its outputs come out with the wrong scale, and every later layer sees inputs unlike those it was trained on. Li et al. ([2019](https://arxiv.org/abs/1801.05134)) call this the **variance shift** and trace the poor results of dropout-then-BN networks to it.

The fix is placement, not abstinence. Swapping the order inside one block (batch norm, then dropout) removes the mismatch at that batch-norm layer, but not at the next one: the extra variance passes through the following linear or convolutional layer and shifts the statistics there instead. The safe arrangement is dropout only after the last batch-norm layer, as in the classifier heads of many CNNs, so that every set of normalization statistics is computed on clean activations. In the block, a unit with mean 1 and variance 1 goes through dropout with \(p = 0.5\): its training variance is \(2 / 0.5 - 1 = 3\), so dropout-then-BN produces test outputs with standard deviation \(1/\sqrt{3} \approx 0.58\) instead of 1. Batch norm first is fine on its own, but a linear layer and a second batch norm after the dropout bring the problem back: that layer's training variance is doubled, and its test outputs have standard deviation about \(1/\sqrt{2} \approx 0.71\).

```python
torch.manual_seed(0)
drop_then_bn = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(64, momentum=None))
bn_then_drop = torch.nn.Sequential(torch.nn.BatchNorm1d(64, momentum=None), torch.nn.Dropout(0.5))
for _ in range(200):                                  # momentum=None: running stats are plain averages
    batch = 1 + torch.randn(256, 64)                  # mean 1, variance 1
    drop_then_bn(batch), bn_then_drop(batch)

assert abs(drop_then_bn[1].running_var.mean() - 3) < 0.05   # learned from the inflated variance
test_batch = 1 + torch.randn(20_000, 64)
with torch.no_grad():
    std_bad = drop_then_bn.eval()(test_batch).std(0).mean()
    std_good = bn_then_drop.eval()(test_batch).std(0).mean()
assert abs(std_bad - 3**-0.5) < 0.02                  # wrong scale at inference
assert abs(std_good - 1) < 0.02                       # BN first: no mismatch at this layer

bn_drop_lin_bn = torch.nn.Sequential(torch.nn.BatchNorm1d(64, momentum=None), torch.nn.Dropout(0.5),
                                     torch.nn.Linear(64, 64), torch.nn.BatchNorm1d(64, momentum=None))
for _ in range(200):
    bn_drop_lin_bn(1 + torch.randn(256, 64))
with torch.no_grad():
    std_next = bn_drop_lin_bn.eval()(test_batch).std(0).mean()
assert abs(std_next - 2**-0.5) < 0.03                 # the shift reappears at the next batch norm
```

## Where it is used today
Dropout was central to the large fully connected networks of the early 2010s. Today it is used more selectively, sometimes in a changed shape:

| Place | Typical setting |
|---|---|
| Hidden layers of an MLP, classifier heads | \(p\) between 0.1 and 0.5 |
| Inputs | small, around 0.1 to 0.2, if at all |
| Transformers: attention weights and each sublayer's output, before the residual add | 0.1 in BERT-style and GPT-2-style models, often 0 for pretraining on huge corpora |
| Fine-tuning and adapters (for example the dropout option in [LoRA](../lora-explained/)) | 0.05 to 0.1 |
| Convolutional feature maps | whole channels (`nn.Dropout2d`), since neighbouring pixels are too correlated for per-pixel masks to remove much |
| Very deep residual networks | whole residual blocks, as **stochastic depth** ([Huang et al., 2016](https://arxiv.org/abs/1603.09382)) |

Dropout earns its place when the data cannot keep up with the model. A language model pretrained for about one pass over trillions of tokens rarely sees an example twice, so there is little to memorize, and many recent pretraining recipes set dropout to zero. Convolutional networks lean on batch norm, augmentation and weight decay in their bodies and keep dropout for the head. Rates above 0.5 rarely help. Tune \(p\) on validation data, alongside weight decay (more levers in the [overfitting playbook](../overfitting-playbook/)).

```python
fmap = torch.ones(8, 16, 5, 5)
dropped = torch.nn.Dropout2d(p=0.5).train()(fmap)
per_channel = dropped.flatten(2)
assert ((per_channel == 0).all(-1) | (per_channel == 2).all(-1)).all()   # each channel kept or zeroed whole
```

## Pitfalls
- **Evaluating in training mode.** `nn.Dropout` follows `model.train()` and `model.eval()`. Forgetting `eval()` makes predictions random and, on average, slightly off.
- **The functional form ignores the mode.** `torch.nn.functional.dropout` has `training=True` as its default, so calling it inside `forward` without `training=self.training` keeps dropping units after `model.eval()`.
- **Which \(p\)?** PyTorch's `p` and Keras's `rate` are the probability of dropping; the original paper's \(p\) and the old TensorFlow `keep_prob` are the probability of keeping. A "\(p = 0.8\)" copied across conventions drops four times too much.
- **Dropout anywhere before a batch norm.** It shifts the variance between training and inference, as shown above, even with other layers in between. Keep dropout after the last batch-norm layer.
- **Comparing losses across modes.** With dropout on, training loss is measured on a handicapped network, so it can sit above validation loss without anything being wrong. Compare the two in evaluation mode before concluding there is no overfitting.

```python
import torch.nn.functional as F


class Leaky(torch.nn.Module):
    def forward(self, x):
        return F.dropout(x, p=0.5)                    # bug: training defaults to True


class Fixed(torch.nn.Module):
    def forward(self, x):
        return F.dropout(x, p=0.5, training=self.training)


ones = torch.ones(4, 100)
assert (Leaky().eval()(ones) == 0).any()              # still dropping in eval mode
assert torch.equal(Fixed().eval()(ones), ones)
```

The exact results above hold at any size; the measured ones may not carry to trained, deep networks. The error table comes from one sine-wave task, and the nonlinear ensemble check uses random weights, so it shows the deterministic network can miss the ensemble average, not how far a trained one does. A deep network's penalty has no closed form, so "dropout regularizes in activation space" is carried over from the linear case, not proven here. How much dropout buys on a given model is a question for its validation data.

The masking scheme, the weight-scaling rule and the ensemble interpretation are from Hinton et al. (2012) and Srivastava et al. (2014). The linear-model penalty follows the analysis in Srivastava et al. and Wager, Wang and Liang (2013), and the batch-norm interaction is from Li et al. (2019).
