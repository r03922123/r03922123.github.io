# Why attention divides by the square root of d

Score one random query against 32 random keys, all with 64 unit-variance components, and an unscaled softmax puts on average 0.82 of its weight on a single key; at 1024 components it puts 0.95. Scaled dot-product attention prevents this with one constant that looks arbitrary: the scores \(QK^\top\) are divided by \(\sqrt{d}\), where \(d\) is the size of one attention head. Without that division, a transformer with 64-dimensional heads and unit-variance queries and keys starts training with attention weights that are already close to one-hot, and for 15% of queries the gradient that should teach them and the keys where to look is under a tenth of what scaled attention typically gives. I derive the constant from a two-line variance calculation, measure what happens to softmax and its gradient with and without it, and then look at where the assumptions behind it stop holding.

Throughout, \(d\) is the per-head dimension of queries and keys (often written \(d_k\)), \(L\) is the number of keys a query attends over, and

\[ \operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V . \]

## A dot product's spread grows with d
One attention score is a sum of \(d\) products, \(s = q \cdot k = \sum_{i=1}^{d} q_i k_i\). Suppose, as at initialization, that the components of \(q\) and \(k\) are independent with mean 0 and variances \(\sigma_q^2\) and \(\sigma_k^2\). For one term, independence gives \(\mathbb{E}[q_i k_i] = 0\) and

\[ \operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] - \big(\mathbb{E}[q_i]\,\mathbb{E}[k_i]\big)^2 = \sigma_q^2\,\sigma_k^2 . \]

The \(d\) terms are independent of each other, so their variances add:

\[ \operatorname{Var}(q \cdot k) = d\,\sigma_q^2\,\sigma_k^2, \qquad \operatorname{std}(q \cdot k) = \sqrt{d}\,\sigma_q \sigma_k . \]

With unit-variance components a 64-dimensional head produces scores with a standard deviation of 8, and a 1024-dimensional one a standard deviation of 32. Dividing by \(\sqrt{d}\) brings the variance back to \(\sigma_q^2 \sigma_k^2\) whatever the head size, which is the whole point: the softmax sees inputs on the same scale in a small model and a large one.

The product rule above needs the means to be zero. In general \(\operatorname{Var}(XY) = \sigma_X^2\sigma_Y^2 + \mu_X^2\sigma_Y^2 + \mu_Y^2\sigma_X^2\) for independent \(X, Y\), so "the variance of a product is the product of the variances" is a special case, not a rule.

Where do unit-variance components come from? Not from LayerNorm alone. LayerNorm rescales each token's input vector \(x\) so that its components have mean 0 and mean square 1, before a learned gain that starts at 1 (see [BatchNorm vs LayerNorm](../batchnorm-vs-layernorm/)). The query is \(q = W_q x\), and with zero-mean random weights its components have variance \(d_{\text{model}} \operatorname{Var}(W_q)\). An initialization with weight variance \(1/d_{\text{model}}\) makes that 1. Common defaults are smaller: PyTorch's `nn.Linear` draws weights with variance \(1/(3\,d_{\text{model}})\), which gives \(\sigma_q^2 = 1/3\). That changes the constant \(\sigma_q \sigma_k\) in front, not the \(\sqrt{d}\) growth, and the growth is what the scaling removes.

```python
import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)

for d in [16, 64, 256, 1024]:
    q, k = rng.standard_normal((2, 20_000, d))
    s = (q * k).sum(axis=1)
    assert abs(s.var() / d - 1) < 0.05                   # Var(q . k) = d
    assert abs((s / np.sqrt(d)).var() - 1) < 0.05        # scaled: variance 1 for every d

sq, sk = 2.0, 0.5                                        # unequal variances multiply
q, k = rng.normal(0, sq, (20_000, 64)), rng.normal(0, sk, (20_000, 64))
assert abs((q * k).sum(1).var() / (64 * sq**2 * sk**2) - 1) < 0.05

mx, sx, my, sy = 0.5, 2.0, -1.0, 0.5                     # non-zero means: the extra terms matter
x, y = rng.normal(mx, sx, 1_000_000), rng.normal(my, sy, 1_000_000)
full = sx**2 * sy**2 + mx**2 * sy**2 + my**2 * sx**2
assert abs((x * y).var() / full - 1) < 0.02
assert abs((x * y).var() / (sx**2 * sy**2) - 1) > 0.5   # product of variances is badly off

d_model = 512                                            # unit-variance x, weights with variance 1/d_model
x = rng.standard_normal((5_000, d_model))
W_q = rng.normal(0, d_model ** -0.5, (d_model, 64))
assert abs((x @ W_q).var() - 1) < 0.05

torch.manual_seed(0)
lin = torch.nn.Linear(d_model, 64)                      # PyTorch's default init
assert abs(lin.weight.var().item() * 3 * d_model - 1) < 0.05   # variance 1 / (3 d_model)
```

## What large scores do to softmax
Softmax only cares about differences between scores, and it exaggerates them exponentially. A gap of 8 between the top score and the next one already puts \(e^{8} \approx 3000\) times more weight on the top key. When the scores' spread grows like \(\sqrt{d}\), so does the typical gap between the largest score and the rest, and the softmax output drifts toward a one-hot vector.

The experiment below draws one query and \(L = 32\) random keys with unit-variance components, two thousand times, and reports the average largest attention weight and the average entropy of the weights (uniform attention over 32 keys has entropy \(\ln 32 \approx 3.47\); see [entropy and information](../entropy-and-information/)):

| Head size \(d\) | Unscaled: top weight | Unscaled: entropy | Scaled: top weight | Scaled: entropy |
|---|---|---|---|---|
| 16 | 0.63 | 1.15 | 0.16 | 3.01 |
| 64 | 0.82 | 0.50 | 0.16 | 3.02 |
| 256 | 0.91 | 0.23 | 0.16 | 3.01 |
| 1024 | 0.95 | 0.12 | 0.16 | 3.01 |

Without the scaling, attention gets sharper with every increase in head size, for no reason other than the arithmetic of sums. With it, the distribution looks the same at every size: noticeably peaked, far from one-hot.

```python
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))        # max-shifted, see the stable softmax post
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p):
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)


L = 32
stats = {}
for d in [16, 64, 256, 1024]:
    q = rng.standard_normal((2_000, 1, d))
    k = rng.standard_normal((2_000, L, d))
    s = (q * k).sum(axis=-1)                             # (2000, L) scores
    for name, z in [("raw", s), ("scaled", s / np.sqrt(d))]:
        p = softmax(z)
        stats[d, name] = (p.max(axis=-1).mean(), entropy(p).mean())

top = {key: v[0] for key, v in stats.items()}
assert top[16, "raw"] < top[64, "raw"] < top[256, "raw"] < top[1024, "raw"]
assert top[1024, "raw"] > 0.9 and stats[1024, "raw"][1] < 0.2      # nearly one-hot
for d in [16, 64, 256, 1024]:
    assert 0.14 < top[d, "scaled"] < 0.19                          # same shape at every size
    assert abs(stats[d, "scaled"][1] - 3.0) < 0.1
assert stats[16, "scaled"][1] < np.log(L)                          # but not uniform either
```

## Saturation kills the gradient, for every key
The derivative of softmax \(p = \operatorname{softmax}(z)\) is

\[ \frac{\partial p_i}{\partial z_j} = p_i\,(\delta_{ij} - p_j), \qquad J = \operatorname{diag}(p) - p\,p^\top . \]

It is tempting to say that a one-hot softmax starves the losing keys of gradient while the winner keeps learning. In fact the winner gets nothing either. If \(p_m\) is the largest weight, every diagonal entry \(p_i(1 - p_i)\) and every off-diagonal entry \(p_i p_j\) has a factor that is at most \(1 - p_m\) (either \(1 - p_m\) itself, or the weight of a key that is not the winner, which is at most \(1 - p_m\)). So

\[ |J_{ij}| \le 1 - p_m \quad \text{for all } i, j , \]

and as attention becomes one-hot the entire Jacobian goes to zero. Nothing upstream of the softmax, neither the query nor any key, receives a signal about whether a different key would have been better. The block below checks the formula against autograd, checks the bound, and then measures the gradient that reaches the query through a full attention step:

```python
z = torch.randn(10, dtype=torch.float64)
p = torch.softmax(z, dim=0)
J = torch.diag(p) - torch.outer(p, p)
assert torch.allclose(torch.autograd.functional.jacobian(lambda t: torch.softmax(t, 0), z), J)

for scale in [1.0, 5.0, 20.0, 80.0]:                     # sharper and sharper softmax
    p = torch.softmax(scale * z, dim=0)
    J = torch.diag(p) - torch.outer(p, p)
    assert J.abs().max() <= 1 - p.max() + 1e-12


def query_grad_norms(d, scale, n=500, L=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 1, d, generator=g, requires_grad=True)
    k, v = torch.randn(n, L, d, generator=g), torch.randn(n, L, d, generator=g)
    upstream = torch.randn(n, 1, d, generator=g)         # a random downstream loss
    out = torch.softmax(q @ k.transpose(1, 2) * scale, dim=-1) @ v
    (out * upstream).sum().backward()
    return q.grad.norm(dim=-1).flatten()                 # one gradient norm per query


starved = {}
for d in [64, 256, 1024]:
    raw, scaled = query_grad_norms(d, 1.0), query_grad_norms(d, d ** -0.5)
    typical = scaled.median()
    starved[d] = (raw < 0.1 * typical).float().mean().item()
    print(f"d={d:5d}  starved queries: unscaled {starved[d]:.0%}, "
          f"scaled {(scaled < 0.1 * typical).float().mean().item():.0%}   "
          f"90th percentile / typical: unscaled {(raw.quantile(0.9) / typical).item():.0f}x, "
          f"scaled {(scaled.quantile(0.9) / typical).item():.1f}x")
    assert (scaled < 0.1 * typical).sum() == 0           # every scaled query learns
    assert raw.quantile(0.9) > 10 * typical              # while a few unscaled ones get huge steps
assert starved[64] < starved[256] < starved[1024] and starved[1024] > 0.5
```

The printout (500 queries, 32 keys each) shows two failures at once. Call a query "starved" if its gradient is below a tenth of the typical scaled-attention gradient. Without the scaling, 15% of queries are starved at \(d = 64\), 36% at \(d = 256\) and 60% at \(d = 1024\); with it, none are. Meanwhile the unscaled queries that do get a signal (the ones whose top scores happen to be close together, so the softmax is not yet saturated) get much larger steps: the 90th percentile is 17 to 23 times the typical size, because the chain rule multiplies by the raw keys rather than by \(k / \sqrt{d}\). Most queries cannot learn where to look while the rest take huge, erratic steps: early training of the attention pattern under those conditions is slow at best and unstable at worst. (The values and the rest of the network still receive gradient; it is the query and key projections that are starved.)

## Why the square root, and not d
Dividing by a constant \(c\) is the same as running softmax at temperature \(c\). Two other choices show why \(\sqrt{d}\) is the natural one.

- **No scaling** lets the temperature effectively fall as \(d\) grows, so the model's behaviour at initialization depends on its width. The table above is that dependence.
- **Dividing by \(d\)** overcorrects for random vectors: the scores now have variance \(1/d\), so at \(d = 1024\) a typical score is about 0.03 and attention is a near-uniform average of the values. The query's gradient also carries the factor \(1/d\), so learning to focus is slow.

\(\sqrt{d}\) is the one choice that keeps the score variance constant under the independence assumption, which is the same reasoning behind initializing a linear layer with weight variance proportional to \(1/\text{fan-in}\).

## Where the assumption stops holding
The argument is about initialization, when queries and keys are close to independent. Training breaks it in two ways.

**Correlated queries and keys.** If \(q\) and \(k\) point in similar directions, the dot product is no longer a sum of zero-mean terms. In the extreme \(q = k\), each term is \(q_i^2\) with mean \(\sigma^2\), the sum has mean \(d\sigma^2\), and dividing by \(\sqrt{d}\) still leaves scores of order \(\sqrt{d}\). This is one argument behind the maximal update parametrization (µP) of Yang et al. (2022), which scales attention scores by \(1/d\) rather than \(1/\sqrt{d}\) so that hyperparameters tuned on a small model transfer to a wide one.

**Growing weights.** Nothing stops \(W_q\) and \(W_k\) from growing during training, and the scale factor is fixed. Large models have been observed to diverge when attention logits grow until softmax collapses to one-hot; Dehghani et al. (2023) fixed this in a 22-billion-parameter vision transformer by applying LayerNorm to queries and keys before the dot product ("QK-norm"), which bounds the scores directly.

```python
for d in [64, 256, 1024]:
    q = rng.standard_normal((5_000, d))
    s_indep = (q * rng.standard_normal((5_000, d))).sum(1) / np.sqrt(d)
    s_same = (q * q).sum(1) / np.sqrt(d)                 # q = k: mean d / sqrt(d) = sqrt(d)
    assert abs(s_indep.mean()) < 0.1 and abs(s_indep.std() - 1) < 0.05
    assert abs(s_same.mean() / np.sqrt(d) - 1) < 0.02   # the scaling no longer tames it
    assert abs(s_same.std() - np.sqrt(2)) < 0.1          # Var(q_i^2) = 2, so std = sqrt(2 d / d)
```

## In code
The scale goes on the per-head dimension. In PyTorch, `F.scaled_dot_product_attention` applies \(1/\sqrt{d}\) with \(d\) taken from the last dimension of the query, and fused kernels such as [FlashAttention](../flashattention/) apply it inside the kernel. Writing it by hand, the scale can multiply \(q\) before the matrix product instead of the \(L \times L\) score matrix afterwards; the result is the same up to floating-point rounding, and it can even be folded into \(W_q\) (and its bias, if it has one) once at load time.

```python
def attention(q, k, v, mask=None):
    d = q.size(-1)                                       # per-head size, not d_model
    scores = q @ k.transpose(-2, -1) / d ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


torch.manual_seed(0)
batch, heads, L, d_model = 2, 8, 16, 512
d_head = d_model // heads
q, k, v = torch.randn(3, batch, heads, L, d_head, dtype=torch.float64)
causal = torch.ones(L, L, dtype=torch.bool).tril()

ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
assert torch.allclose(attention(q, k, v, causal), ref)
twice = F.scaled_dot_product_attention(q * d_head ** -0.5, k, v, is_causal=True)
assert not torch.allclose(twice, ref)                    # pre-scaled q, scaled again inside
assert torch.allclose(F.scaled_dot_product_attention(q * d_head ** -0.5, k, v, is_causal=True, scale=1.0), ref)
pre = torch.softmax((q * d_head ** -0.5) @ k.transpose(-2, -1), dim=-1) @ v
assert torch.allclose(pre, attention(q, k, v))           # scale q first: same output

x = torch.randn(L, d_model, dtype=torch.float64)         # folding the scale into W_q
W_q, W_k = torch.randn(2, d_model, d_head, dtype=torch.float64) * d_model ** -0.5
folded = (x @ (W_q * d_head ** -0.5)) @ (x @ W_k).T
assert torch.allclose(folded, (x @ W_q) @ (x @ W_k).T / d_head ** 0.5)

wrong = torch.softmax(q @ k.transpose(-2, -1) / d_model ** 0.5, dim=-1)   # the d_model bug
right = torch.softmax(q @ k.transpose(-2, -1) / d_head ** 0.5, dim=-1)
assert entropy(wrong.numpy()).mean() > entropy(right.numpy()).mean() + 0.3   # blurrier attention
```

## Pitfalls
- **Dividing by \(\sqrt{d_{\text{model}}}\).** With 8 heads of size 64 in a 512-wide model, that divides by about 22.6 instead of 8, so the score standard deviation at initialization is \(1/\sqrt{8} \approx 0.35\) and every head starts out close to uniform. The last assertion above shows the effect.
- **Scaling twice.** Passing already-scaled queries to `F.scaled_dot_product_attention` applies the factor again. Either scale by hand or let the function do it (it also takes a `scale=` argument), not both.
- **Assuming the argument holds after training.** It fixes the scale at initialization only. If training is unstable and the attention entropy of some heads is collapsing toward zero, look at the logit magnitudes; QK-norm or a learned temperature addresses that, the \(\sqrt{d}\) does not.
- **Thinking the scale prevents overflow.** It keeps scores moderate, which helps in float16, but the numerical safety of softmax comes from subtracting the max, covered in [numerically stable softmax and cross-entropy](../stable-softmax-cross-entropy/).
- **Confusing the variance rule.** \(\operatorname{Var}(XY) = \operatorname{Var}(X)\operatorname{Var}(Y)\) needs independence and zero means. Non-zero-mean queries and keys add terms the simple argument leaves out.

Every measurement above draws queries and keys as independent random vectors, the situation at initialization, so it shows why the constant is \(\sqrt{d}\) when training starts, not how large scores get later. The one departure from independence I check is the extreme case \(q = k\), and the divergence of a 22-billion-parameter model is Dehghani et al.'s report, not a measurement here. How fast logits grow while a given model trains stays open, and that model's logit magnitudes are the place to look. For how this piece fits into the rest of the architecture, see [transformers and LLMs: key ideas](../transformers-and-llms-key-ideas/).

## References
1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017. Section 3.2.1 introduces scaled dot-product attention, and its footnote gives the variance argument for unit-variance, independent components.
2. Yang et al., [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466), 2022. The µP parametrization, including \(1/d\) attention scaling.
3. Dehghani et al., [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442), ICML 2023. Query-key normalization against attention logit growth.

The measured demonstration of score variance growing with head size follows the one in Andrej Karpathy's lecture "Let's build GPT: from scratch, in code, spelled out".
