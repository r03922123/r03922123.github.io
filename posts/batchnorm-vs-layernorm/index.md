# BatchNorm vs LayerNorm: one formula, two axes

A normalization layer takes a set of activations, subtracts their mean, divides by their standard deviation, and applies a learned scale and shift. BatchNorm and LayerNorm both do exactly this. They differ only in which set: BatchNorm averages one feature over the examples in a batch, LayerNorm averages all the features of one example. That choice of axis decides nearly everything else: whether training and inference differ, whether a batch of one works, whether the layer can be folded away at serving time, and whether it suits CNNs or Transformers.

Both are built from scratch below and checked against PyTorch, each consequence gets a small experiment, and the post ends with RMSNorm, the stripped-down LayerNorm that most recent large language models use.

## One formula, two axes
Take a set \(S\) of activations with mean \(\mu_S\) and variance \(\sigma_S^2\). Every layer here computes

\[ \hat{x} = \frac{x - \mu_S}{\sqrt{\sigma_S^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta \]

where \(\epsilon\) (typically \(10^{-5}\)) keeps the division safe, \(\gamma\) and \(\beta\) are learned per feature, and the variance is the biased one, dividing by the size of \(S\). Only \(S\) changes:

| Layer | Input shape | Statistics over | One mean and variance per |
|---|---|---|---|
| BatchNorm | \([B, C, H, W]\) | batch and spatial axes \((B, H, W)\) | channel |
| BatchNorm | \([B, D]\) | batch axis \(B\) | feature |
| LayerNorm | \([B, T, D]\) | feature axis \(D\) | token |
| GroupNorm | \([B, C, H, W]\) | a group of channels, and \((H, W)\) | example and group |
| InstanceNorm | \([B, C, H, W]\) | spatial axes \((H, W)\) | example and channel |

GroupNorm and InstanceNorm sit in between: like LayerNorm, they never look across the batch. GroupNorm with one group is LayerNorm over \((C, H, W)\), the form Ba, Kiros and Hinton described for convolutional layers.

Without \(\gamma\) and \(\beta\), BatchNorm would pin every feature to mean 0 and variance 1, and LayerNorm every token. For BatchNorm the pair can undo the normalization: with \(\gamma = \sqrt{\sigma^2 + \epsilon}\) and \(\beta = \mu\), set from the statistics it divides by, the layer is the identity, so it costs no expressive power. It changes how the same functions are parameterized, and so how easy they are to optimize. LayerNorm has no such escape, because its statistics belong to each token while \(\gamma\) and \(\beta\) belong to each feature: a token's own mean and scale are discarded for good, and two tokens that differ only by those map to the same output.

```python
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0)
torch.manual_seed(0)
EPS = 1e-5


def normalize(x, axes, gamma=1.0, beta=0.0, eps=EPS):
    """(x - mean) / sqrt(var + eps) over `axes`, then the learned scale and shift."""
    mu = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)                 # biased: divides by the count
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


images = rng.normal(2.0, 3.0, size=(8, 4, 5, 5))          # [B, C, H, W]
tokens = rng.normal(2.0, 3.0, size=(8, 6, 16))            # [B, T, D]
bn, ln = normalize(images, (0, 2, 3)), normalize(tokens, -1)
assert np.allclose(bn.mean(axis=(0, 2, 3)), 0) and np.allclose(bn.var(axis=(0, 2, 3)), 1, atol=1e-5)
assert np.allclose(ln.mean(axis=-1), 0) and np.allclose(ln.var(axis=-1), 1, atol=1e-5)

ti, tt = torch.tensor(images), torch.tensor(tokens)
assert np.allclose(bn, F.batch_norm(ti, None, None, training=True, eps=EPS).numpy())
assert np.allclose(ln, F.layer_norm(tt, (16,), eps=EPS).numpy())
assert np.allclose(normalize(images, (1, 2, 3)), F.group_norm(ti, 1, eps=EPS).numpy())   # one group
assert np.allclose(normalize(images, (2, 3)), F.instance_norm(ti, eps=EPS).numpy())
grouped = normalize(images.reshape(8, 2, 2, 5, 5), (2, 3, 4)).reshape(images.shape)
assert np.allclose(grouped, F.group_norm(ti, 2, eps=EPS).numpy())                        # two groups of two

mu, var = images.mean(axis=(0, 2, 3), keepdims=True), images.var(axis=(0, 2, 3), keepdims=True)
assert np.allclose(normalize(images, (0, 2, 3), gamma=np.sqrt(var + EPS), beta=mu), images)   # BatchNorm can undo itself
assert np.allclose(normalize(3 * tokens + 5, -1), normalize(tokens, -1), atol=1e-5)          # LayerNorm forgets shift and scale
```

## BatchNorm: statistics from the batch
In training, BatchNorm uses the current batch's statistics. At inference that is wrong: there may be a single example, and a prediction should not depend on what else is processed with it. So BatchNorm also keeps a running average of the batch statistics and uses it in evaluation mode. PyTorch's update is an exponential moving average,

\[ \hat{\mu} \leftarrow (1 - m)\,\hat{\mu} + m\,\mu_{\text{batch}}, \qquad m = 0.1 \text{ by default,} \]

and likewise for the variance, except that the stored variance is the unbiased estimate (dividing by \(n - 1\)) while the batch is normalized with the biased one. The class below follows these conventions and matches `nn.BatchNorm1d` in both modes.

```python
class BatchNorm1d:
    """BatchNorm over [B, C] inputs, with PyTorch's conventions for the running statistics."""

    def __init__(self, c, momentum=0.1, eps=EPS):
        self.gamma, self.beta = np.ones(c), np.zeros(c)
        self.running_mean, self.running_var = np.zeros(c), np.ones(c)
        self.momentum, self.eps, self.training = momentum, eps, True

    def __call__(self, x):
        if self.training:
            mu, var, n, m = x.mean(axis=0), x.var(axis=0), len(x), self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mu
            self.running_var = (1 - m) * self.running_var + m * var * n / (n - 1)   # unbiased copy
        else:
            mu, var = self.running_mean, self.running_var
        return self.gamma * (x - mu) / np.sqrt(var + self.eps) + self.beta


ours, ref = BatchNorm1d(3), nn.BatchNorm1d(3).double()
for _ in range(50):
    batch = rng.normal([1.0, -2.0, 5.0], [0.5, 2.0, 3.0], size=(32, 3))
    assert np.allclose(ours(batch), ref(torch.tensor(batch)).detach().numpy())
assert np.allclose(ours.running_mean, ref.running_mean.numpy())
assert np.allclose(ours.running_var, ref.running_var.numpy())
assert np.allclose(ours.running_mean, [1.0, -2.0, 5.0], atol=0.5)       # the average has found the data

ours.training = False
ref.eval()
batch = rng.normal(size=(5, 3))
assert np.allclose(ours(batch), ref(torch.tensor(batch)).detach().numpy())
```

Two properties of the training-mode layer show up directly. An example's output depends on the rest of its batch: a noise often credited with a mild regularizing effect, a little like [dropout](../dropout/), and the reason evaluation mode exists. And the output ignores a positive rescaling of the previous layer's weights, while a bias added before the layer is subtracted straight back out with the mean. That is why a convolution followed by BatchNorm is created with `bias=False`.

```python
row = rng.normal(size=(1, 3))
crowd_a = np.vstack([row, rng.normal(0.0, 1.0, size=(15, 3))])
crowd_b = np.vstack([row, rng.normal(4.0, 1.0, size=(15, 3))])
fresh = BatchNorm1d(3)
assert not np.allclose(fresh(crowd_a)[0], fresh(crowd_b)[0])      # training: depends on the neighbours
fresh.training = False
assert np.allclose(fresh(crowd_a)[0], fresh(crowd_b)[0])          # evaluation: does not

h, W, b = rng.normal(size=(32, 6)), rng.normal(size=(6, 3)), rng.normal(size=3)
assert np.allclose(normalize(h @ (10 * W), 0), normalize(h @ W, 0), atol=1e-4)   # weight scale cancels
assert np.allclose(normalize(h @ W + b, 0), normalize(h @ W, 0))                 # and so does a bias
```

Why BatchNorm helps is less settled than the recipe. Ioffe and Szegedy motivated it as reducing internal covariate shift, the drift in each layer's input distribution as earlier layers train. Santurkar et al. injected noise after BatchNorm to bring that drift back, found that training barely suffered, and argued that the real effect is a smoother loss surface that tolerates larger learning rates.

## Where BatchNorm breaks
All of this assumes each batch is a fair sample of the data. When it is not, BatchNorm degrades, usually without an error.

**A batch of one.** With \([1, D]\) inputs each feature's set has one element, so \(x - \mu_S = 0\) and every input maps to \(\beta\). \(\epsilon\) prevents a division by zero, so nothing crashes; the layer just erases its input. PyTorch raises an error for this case in training mode. A \([1, C, H, W]\) batch is allowed, since each channel still has \(H \times W\) values, and in evaluation mode a batch of one is always fine. "BatchNorm fails at batch size one" is a statement about training, not inference.

**Small batches.** The error in batch statistics shrinks like \(1/\sqrt{B}\), so the network trains on inputs normalized by noisy estimates and is tested on inputs normalized by averaged ones. Simulated for one feature, in units of its standard deviation:

| Batch size | Mean gap between training-mode and evaluation-mode output |
|---|---|
| 2 | 0.75 |
| 8 | 0.35 |
| 32 | 0.17 |
| 128 | 0.085 |

At batch size 2 every training-mode output is \(+1\) or \(-1\) (up to \(\epsilon\)), whatever the input. Detection and segmentation models, which often fit one or two images per GPU, therefore freeze BatchNorm at its pretrained statistics, swap it for GroupNorm, whose accuracy Wu and He showed stays flat as the batch shrinks, or synchronize the statistics across devices (`nn.SyncBatchNorm`).

**Batches that are not a random sample.** A batch drawn from one class or one user has the statistics of that slice only. In contrastive learning, BatchNorm also lets the examples in a batch exchange information, which the model can exploit instead of learning useful features; MoCo shuffles examples across devices before BatchNorm to stop this (see [contrastive loss](../contrastive-loss/)).

```python
one = rng.normal(size=(1, 3))
assert np.allclose(normalize(one, 0, beta=0.7), 0.7)              # every input becomes beta
try:
    nn.BatchNorm1d(3)(torch.randn(1, 3))
    raise AssertionError("PyTorch accepted a training batch of one")
except ValueError:
    pass
nn.BatchNorm2d(3)(torch.randn(1, 3, 8, 8))                        # allowed: 64 values per channel
nn.BatchNorm1d(3).eval()(torch.randn(1, 3))                       # allowed: uses the running statistics


def train_eval_gap(batch_size, trials=20_000, mu=3.0, sigma=2.0):
    """Mean |training-mode output - evaluation-mode output|, evaluation using the true statistics."""
    x = rng.normal(mu, sigma, size=(trials, batch_size))          # each row is one batch of one feature
    return np.abs(normalize(x, 1) - (x - mu) / math.sqrt(sigma**2 + EPS)).mean()


expected = {2: 0.75, 8: 0.35, 32: 0.17, 128: 0.085}
for size, gap in expected.items():
    measured = train_eval_gap(size)
    assert abs(measured - gap) < 0.01
    assert size < 8 or 0.9 < measured * math.sqrt(size) < 1.0     # about 1 / sqrt(B)

pairs = rng.normal(3.0, 2.0, size=(1000, 2))
assert np.allclose(np.abs(normalize(pairs, 1, eps=0)), 1)         # batch of two: always +1 or -1
```

## Folding BatchNorm away at inference
In evaluation mode BatchNorm is a fixed per-channel affine map, \(y = s \odot (x - \hat\mu) + \beta\) with \(s = \gamma / \sqrt{\hat\sigma^2 + \epsilon}\). After a linear or convolutional layer \(x = Wh + b\), the two compose into one:

\[ W' = \operatorname{diag}(s)\,W, \qquad b' = s \odot (b - \hat{\mu}) + \beta \]

So at serving time BatchNorm is free, and inference toolchains merge it into the preceding convolution. LayerNorm cannot be folded like this: its statistics depend on the input it is applied to.

```python
lin, norm = nn.Linear(6, 4).double(), nn.BatchNorm1d(4).double()
with torch.no_grad():
    norm.weight.uniform_(0.5, 2.0)                                # stand-ins for learned gamma and beta
    norm.bias.normal_()
    for _ in range(100):                                          # training-mode passes fill the statistics
        norm(lin(3 * torch.randn(32, 6, dtype=torch.float64) + 1))
norm.eval()
assert not torch.allclose(norm.running_var, torch.ones(4, dtype=torch.float64))

with torch.no_grad():
    s = norm.weight / torch.sqrt(norm.running_var + norm.eps)
    fused = nn.Linear(6, 4).double()
    fused.weight.copy_(s[:, None] * lin.weight)                   # W' = diag(s) W
    fused.bias.copy_(s * (lin.bias - norm.running_mean) + norm.bias)
    probe_x = torch.randn(10, 6, dtype=torch.float64)
    assert torch.allclose(fused(probe_x), norm(lin(probe_x)))     # one layer instead of two
```

## LayerNorm: statistics from the example itself
LayerNorm takes its mean and variance over the features of one example, or of one token. Nothing depends on the batch: there are no running statistics, training and evaluation compute the same function, and a batch of one is like any other. Ba, Kiros and Hinton introduced it for recurrent networks, where BatchNorm would need statistics for every time step, and it became the Transformer's normalization. Batch independence matters for sequences in three ways:

- **Padding.** Sequences are padded to the longest one in the batch. Unless every call is masked, BatchNorm counts the padding, so a sentence's representation depends on the lengths of its batch-mates. LayerNorm never looks at another token.
- **Causality.** A language model trains on all positions at once, with attention masked so that position \(t\) sees only positions up to \(t\). BatchNorm over batch and time undoes that mask: position \(t\) is normalized with statistics that include later tokens, a leak of the future that is gone at inference.
- **Noisy statistics.** Even with masking, Shen et al. found that batch statistics in NLP fluctuate far more than in vision, one reason plain BatchNorm trails LayerNorm in Transformers.

The first two are easy to measure. In a causal model the gradient of position 0's output with respect to later inputs should be zero; under BatchNorm it is not:

```python
D = 16
assert len(list(nn.LayerNorm(D).buffers())) == 0                  # nothing to track
assert len(list(nn.BatchNorm1d(D).buffers())) == 3                # running mean, variance and a counter


def bn_tokens(x):
    """BatchNorm on a [B, T, D] batch of sequences: each feature over all B * T tokens."""
    return F.batch_norm(x.reshape(-1, D), None, None, training=True).reshape(x.shape)


def ln_tokens(x):
    return F.layer_norm(x, (D,))


seq = torch.randn(2, 5, D, dtype=torch.float64)
padded = torch.cat([seq, torch.zeros(2, 20, D, dtype=torch.float64)], dim=1)   # same tokens, padded to 25
assert not torch.allclose(bn_tokens(padded)[:, :5], bn_tokens(seq))   # padding moves the real tokens
assert torch.allclose(ln_tokens(padded)[:, :5], ln_tokens(seq))       # LayerNorm never sees it

x = torch.randn(4, 6, D, dtype=torch.float64, requires_grad=True)
w = torch.randn(4, D, dtype=torch.float64)
for norm_fn, leaks in ((bn_tokens, True), (ln_tokens, False)):
    grad, = torch.autograd.grad((norm_fn(x)[:, 0] * w).sum(), x)      # what position 0's output depends on
    assert (grad[:, 1:].abs().max() > 1e-8) == leaks                  # positions 1 to 5 are its future
```

Placement matters too. The original Transformer normalized after each residual addition (post-LN); GPT-2 and most later models normalize each sub-layer's input instead and add a final LayerNorm before the output (pre-LN):

```plaintext
post-LN:   x = LayerNorm(x + Sublayer(x))
pre-LN:    x = x + Sublayer(LayerNorm(x))
```

Pre-LN keeps an unnormalized identity path through the residual stream. Xiong et al. showed that post-LN starts with large gradients near the output, hence its need for a learning-rate warm-up, and that pre-LN trains without one. [Transformers and LLMs: key ideas](../transformers-and-llms-key-ideas/) assembles the full block.

Nor is LayerNorm only for sequences. Ba et al. found it helped CNNs less than BatchNorm, but ConvNeXt later replaced BatchNorm with a per-position LayerNorm over channels and gained slightly in accuracy. "CNNs use BatchNorm" is a strong default, not a law.

## RMSNorm: drop the mean
LayerNorm re-centers (subtracts the mean) and re-scales (divides by the standard deviation). Zhang and Sennrich hypothesized that re-scaling is what stabilizes training, and proposed RMSNorm, which keeps only that step:

\[ y = \gamma \odot \frac{x}{\operatorname{RMS}(x)}, \qquad \operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon} \]

There is usually no \(\beta\). On features that already average to zero it equals LayerNorm. It computes one statistic per token instead of two, skips the centering step, and has half the parameters; the paper reports quality comparable to LayerNorm at a lower running time, with savings that vary by model and implementation. T5 uses this form, as do LLaMA, Mistral and Gemma.

A justification sometimes given for dropping the mean, that \(\beta\) can shift the output anyway, does not hold: \(\beta\) is the same constant for every input, while the mean differs from token to token. LayerNorm ignores a common shift and a common scale of a token's features; RMSNorm ignores only the scale. The case for RMSNorm is empirical: losing the shift invariance does not seem to hurt, and the layer is cheaper.

```python
def rms_norm(x, gamma=1.0, eps=1e-6):
    return gamma * x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)


v, g = rng.normal(1.0, 2.0, size=(4, 64)), rng.normal(size=64)
assert np.allclose(rms_norm(v, g), F.rms_norm(torch.tensor(v), (64,), torch.tensor(g), eps=1e-6).numpy())
assert np.allclose(np.sqrt((rms_norm(v) ** 2).mean(axis=-1)), 1)             # unit RMS per token

centered = v - v.mean(axis=-1, keepdims=True)
assert np.allclose(rms_norm(centered), normalize(v, -1, eps=1e-6))           # LayerNorm = center, then RMSNorm

assert np.allclose(rms_norm(7 * v), rms_norm(v))                             # both ignore a common scale
assert np.allclose(normalize(7 * v, -1), normalize(v, -1))
assert np.allclose(normalize(v + 3, -1), normalize(v, -1))                   # LayerNorm ignores a shift
assert not np.allclose(rms_norm(v + 3), rms_norm(v))                         # RMSNorm does not
```

## Which one to use
| Situation | Use | Why |
|---|---|---|
| CNN with tens of images per device | BatchNorm | Well tested, and folds into the convolution at inference |
| CNN with one or two images per device | GroupNorm, or synchronized BatchNorm | Per-device batch statistics are too noisy |
| Transformer, RNN or other sequence model | LayerNorm, pre-LN | Independent of the batch, the padding and future tokens |
| Large language model | RMSNorm, pre-norm | The same independence, at lower cost |
| Training on one example at a time (online learning, some RL) | LayerNorm or RMSNorm | No batch to take statistics from |

## Pitfalls
- **Evaluating in training mode.** Without `model.eval()`, predictions depend on the batch and the running statistics absorb test data. `torch.no_grad()` does not help: it turns off gradients, not batch statistics.
- **Fine-tuning with small batches.** Training BatchNorm on a few examples per step overwrites good running statistics with noisy ones. Keep the BatchNorm layers in evaluation mode while the rest of the model trains, remembering that `model.train()` switches them back.
- **Momentum means opposite things.** PyTorch's `momentum=0.1` weights the new batch; Keras's default `momentum=0.99` weights the old average. Copying the number across, in either direction, gives the newest batch a weight of 0.9 or 0.99, so the running statistics track little more than the last batch.
- **The wrong axis.** `nn.LayerNorm(d)` normalizes the trailing dimensions. On a channels-first \([B, C, H, W]\) tensor it raises an error, unless \(W = C\), when it silently normalizes across the width. Permute to channels-last first, as ConvNeXt does.
- **Half precision.** In float16 the square of anything above about 256 overflows (the largest float16 is 65504), so a float16 sum of squares becomes infinite and RMSNorm outputs zeros. Compute the statistics in float32: PyTorch's autocast does this for LayerNorm, and LLaMA's reference RMSNorm upcasts its input first.

```python
probe = nn.BatchNorm1d(3)
with torch.no_grad():
    probe(torch.randn(16, 3) + 10)                                # "inference" without .eval()
assert probe.running_mean.min() > 0.5                             # the statistics moved anyway

copied = nn.BatchNorm1d(1, momentum=0.99)                         # Keras's number, PyTorch's meaning
copied(torch.full((4, 1), 5.0) + torch.randn(4, 1))
assert copied.running_mean.item() > 4                             # one batch nearly overwrote the average

img = torch.randn(2, 8, 4, 4)                                     # channels-first [B, C, H, W]
try:
    nn.LayerNorm(8)(img)
    raise AssertionError("expected a shape error")
except RuntimeError:
    pass
square = torch.randn(2, 4, 4, 4)                                  # C == W: no error, wrong axis
wrong = nn.LayerNorm(4)(square)
assert torch.allclose(wrong.mean(dim=-1), torch.zeros(2, 4, 4), atol=1e-6)      # normalized across width
assert not torch.allclose(wrong.mean(dim=1), torch.zeros(2, 4, 4), atol=1e-2)   # not across channels
right = nn.LayerNorm(4)(square.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        # channels-last, per pixel
assert torch.allclose(right.mean(dim=1), torch.zeros(2, 4, 4), atol=1e-6)

h16 = torch.full((1, 8), 300.0, dtype=torch.float16)
assert torch.isinf((h16**2).mean(dim=-1)).all()                   # 300 ** 2 = 90000 > 65504
assert (h16 / torch.sqrt((h16**2).mean(dim=-1, keepdim=True) + 1e-6) == 0).all()   # the token is erased
up = h16.float()
assert torch.allclose((up / torch.sqrt((up**2).mean(dim=-1, keepdim=True) + 1e-6)).half(), torch.ones_like(h16))
```

## References
- S. Ioffe, C. Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015.
- J. L. Ba, J. R. Kiros, G. E. Hinton. Layer Normalization. arXiv 2016.
- S. Santurkar, D. Tsipras, A. Ilyas, A. Madry. How Does Batch Normalization Help Optimization? NeurIPS 2018.
- Y. Wu, K. He. Group Normalization. ECCV 2018.
- B. Zhang, R. Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019.
- K. He, H. Fan, Y. Wu, S. Xie, R. Girshick. Momentum Contrast for Unsupervised Visual Representation Learning. CVPR 2020.
- R. Xiong et al. On Layer Normalization in the Transformer Architecture. ICML 2020.
- S. Shen, Z. Yao, A. Gholami, M. W. Mahoney, K. Keutzer. PowerNorm: Rethinking Batch Normalization in Transformers. ICML 2020.
- Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie. A ConvNet for the 2020s. CVPR 2022.
