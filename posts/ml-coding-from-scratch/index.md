# ML building blocks from scratch, checked against the library

Most of deep learning is built from about a dozen small pieces: a linear map, a nonlinearity, a softmax, a normalization, attention, a recurrence, a convolution. Each one is a few lines of tensor code, and writing them yourself is the fastest way to learn what the shapes are, which axis everything reduces over, and where the numerical traps are.

Writing them is half the exercise; proving them right is the other half. Every piece below is asserted equal to the PyTorch, scikit-learn or brute-force version, usually to nine decimal places, with the library's weights copied across where needed, which is also the best way to learn its weight layout.

## The harness
Everything runs in float64, so a wrong implementation cannot hide inside a loose tolerance.

```python
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
rng = np.random.default_rng(0)


def close(a, b, tol=1e-9):
    torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b), rtol=tol, atol=tol, check_dtype=False)
```

## Activations, softmax and cross-entropy
GELU is \(x \, \Phi(x)\), the input times the standard normal CDF; the tanh formula is an approximation PyTorch exposes as `approximate="tanh"`. Softmax subtracts the row maximum before exponentiating. The shift cancels mathematically but keeps \(e^{z}\) from overflowing. Cross-entropy is computed from the log-softmax directly, never as the log of a softmax, for the reasons in [stable softmax and cross-entropy](../stable-softmax-cross-entropy/).

```python
def gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))


def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, dim=-1):
    e = (x - x.amax(dim, keepdim=True)).exp()
    return e / e.sum(dim, keepdim=True)


def log_softmax(x, dim=-1):
    z = x - x.amax(dim, keepdim=True)
    return z - z.exp().sum(dim, keepdim=True).log()


def cross_entropy(logits, y):                     # logits [B, C], y [B] integer classes
    return -log_softmax(logits)[torch.arange(len(y)), y].mean()


x = torch.randn(8, 10) * 30                       # large logits on purpose
y = torch.randint(0, 10, (8,))
close(x.clamp_min(0), F.relu(x))
close(gelu(x), F.gelu(x))
close(gelu_tanh(x), F.gelu(x, approximate="tanh"))
close(softmax(x), F.softmax(x, -1))
close(log_softmax(x), F.log_softmax(x, -1))
close(cross_entropy(x, y), F.cross_entropy(x, y))

big = torch.tensor([1000.0, 0.0])
assert torch.isnan(big.exp() / big.exp().sum()).any()      # the naive softmax overflows
close(softmax(big), torch.tensor([1.0, 0.0]))
```

## Linear layers, embeddings and an MLP by hand
`nn.Linear(d_in, d_out)` stores its weight as `[d_out, d_in]` and computes \(xW^\top + b\). An embedding table is a linear layer applied to a one-hot vector, implemented as a row lookup because the matrix product would be mostly zeros.

```python
lin = nn.Linear(5, 3)
x = torch.randn(4, 5)
assert lin.weight.shape == (3, 5)
close(x @ lin.weight.T + lin.bias, lin(x))

emb = nn.Embedding(10, 4)
ids = torch.tensor([3, 7, 3])
close(emb.weight[ids], emb(ids))
close(F.one_hot(ids, 10).double() @ emb.weight, emb(ids))
```

A two-layer MLP classifier is where it pays to know every shape. The backward pass follows from three facts: the gradient of mean cross-entropy with respect to the logits is \((p - \text{onehot}(y))/B\); a matrix product \(Y = XW\) sends gradient \(X^\top \, dY\) to \(W\) and \(dY \, W^\top\) to \(X\); and ReLU passes gradient only where its input was positive. The hand-written gradients must match autograd exactly.

```python
def mlp_loss_and_grads(X, y, W1, b1, W2, b2):
    B = len(X)
    h_pre = X @ W1 + b1                    # [B, H]
    h = h_pre.clamp_min(0)                 # [B, H]
    logits = h @ W2 + b2                   # [B, C]
    loss = cross_entropy(logits, y)
    d_logits = softmax(logits)             # [B, C]
    d_logits[torch.arange(B), y] -= 1
    d_logits /= B
    dW2, db2 = h.T @ d_logits, d_logits.sum(0)          # [H, C], [C]
    d_h_pre = (d_logits @ W2.T) * (h_pre > 0)           # [B, H]
    dW1, db1 = X.T @ d_h_pre, d_h_pre.sum(0)            # [D, H], [H]
    return loss, [dW1, db1, dW2, db2]


B, D, H, C = 32, 5, 16, 3
X, y = torch.randn(B, D), torch.randint(0, C, (B,))
params = [torch.randn(D, H) / D ** 0.5, torch.zeros(H), torch.randn(H, C) / H ** 0.5, torch.zeros(C)]
loss, grads = mlp_loss_and_grads(X, y, *params)

auto = [p.clone().requires_grad_() for p in params]
ref_loss = F.cross_entropy(F.relu(X @ auto[0] + auto[1]) @ auto[2] + auto[3], y)
ref_loss.backward()
close(loss, ref_loss.detach())
for g, p in zip(grads, auto):
    close(g, p.grad)

# the same gradients train it: three Gaussian blobs, full-batch gradient descent
centers = torch.tensor([[2.0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0], [0, 0, 2.0, 0, 0]])
y = torch.randint(0, C, (300,))
X = centers[y] + 0.7 * torch.randn(300, D)
for _ in range(300):
    loss, grads = mlp_loss_and_grads(X, y, *params)
    params = [p - 0.5 * g for p, g in zip(params, grads)]
pred = (F.relu(X @ params[0] + params[1]) @ params[2] + params[3]).argmax(1)
assert (pred == y).double().mean() > 0.95
```

## LayerNorm, BatchNorm and dropout
Both normalizations subtract a mean and divide by a standard deviation; LayerNorm averages over each example's features, BatchNorm over the batch for each feature (the [BatchNorm vs LayerNorm](../batchnorm-vs-layernorm/) post covers the consequences). Two details catch almost everyone. The variance used to normalize is the biased one, dividing by \(n\), while `torch.var` divides by \(n-1\) by default. And BatchNorm's running variance, used at inference, is updated with the unbiased variance.

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta


class BatchNorm1d:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = torch.ones(d), torch.zeros(d)
        self.mean, self.var = torch.zeros(d), torch.ones(d)
        self.momentum, self.eps, self.training = momentum, eps, True

    def __call__(self, x):                                  # x [B, d]
        if self.training:
            mu, var = x.mean(0), x.var(0, unbiased=False)
            m = self.momentum
            self.mean = (1 - m) * self.mean + m * mu
            self.var = (1 - m) * self.var + m * x.var(0, unbiased=True)
        else:
            mu, var = self.mean, self.var
        return (x - mu) / torch.sqrt(var + self.eps) * self.gamma + self.beta


ln = nn.LayerNorm(6)
nn.init.normal_(ln.weight), nn.init.normal_(ln.bias)
x = torch.randn(2, 4, 6) * 3 + 1
close(layer_norm(x, ln.weight, ln.bias), ln(x))

mine, ref = BatchNorm1d(6), nn.BatchNorm1d(6)
for _ in range(5):
    x = torch.randn(16, 6) * 2 + 3
    close(mine(x), ref(x))
close(mine.mean, ref.running_mean)
close(mine.var, ref.running_var)
close(x.var(0), x.var(0, unbiased=False) * 16 / 15)       # torch.var defaults to n - 1
assert np.isclose(np.var(x.numpy(), axis=0), x.var(0, unbiased=False).numpy()).all()   # np.var defaults to n
mine.training = False
ref.eval()
x = torch.randn(4, 6)
close(mine(x), ref(x))
```

Dropout zeros each activation with probability \(p\) during training and scales the survivors by \(1/(1-p)\), so the expected activation is unchanged and inference is a no-op. That is "inverted" dropout; the [dropout](../dropout/) post explains why the scaling sits on the training side. Random masks cannot be compared element by element, so the checks are structural: every output is either zero or the input scaled by \(1/(1-p)\), PyTorch's output has the same structure, and the mean is preserved.

```python
def dropout(x, p, training=True):
    if not training or p == 0:
        return x
    keep = (torch.rand_like(x) >= p).to(x.dtype)
    return x * keep / (1 - p)


x, p = torch.rand(200_000) + 1, 0.3
for out in (dropout(x, p), F.dropout(x, p)):
    zero = out == 0
    close(out[~zero], x[~zero] / (1 - p))
    assert abs(zero.double().mean() - p) < 0.01
    assert abs(out.mean() - x.mean()) < 0.01
assert dropout(x, p, training=False) is x
```

## Attention, causal and bidirectional
Scaled dot-product attention compares every query with every key, turns the scores into weights with a softmax over the keys, and averages the values:

\[ \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \]

The \(\sqrt{d_k}\) keeps the scores' variance near one as the head dimension grows (see [why attention is scaled](../attention-scaling/)). Bidirectional attention, as in an encoder, uses no mask. Causal attention, as in a decoder, sets the score of every future key to \(-\infty\) so its weight is exactly zero.

```python
def attention(q, k, v, mask=None):           # q [..., Tq, d], k [..., Tk, d], v [..., Tk, dv]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])     # [..., Tq, Tk]
    if mask is not None:                      # mask True = may attend
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores) @ v                # [..., Tq, dv]


T = 6
q, k, v = torch.randn(3, 2, T, 8).unbind(0)
causal = torch.ones(T, T, dtype=torch.bool).tril()
close(attention(q, k, v), F.scaled_dot_product_attention(q, k, v))
close(attention(q, k, v, causal), F.scaled_dot_product_attention(q, k, v, is_causal=True))

# causality: changing token 4 cannot change outputs at positions 0..3
k2, v2 = k.clone(), v.clone()
k2[:, 4] += 1
v2[:, 4] += 1
a, b = attention(q, k, v, causal), attention(q, k2, v2, causal)
close(a[:, :4], b[:, :4])
assert not torch.allclose(a[:, 4:], b[:, 4:])

nothing = torch.zeros(T, T, dtype=torch.bool)               # every key masked
assert torch.isnan(attention(q, k, v, nothing)).all()       # a plain softmax gives NaN here
```

Multi-head attention runs \(h\) of these in parallel on \(d_\text{model}/h\)-dimensional slices, then mixes the heads with an output projection. The only tricky lines are the reshapes: split the model dimension into heads, move the head axis in front of time, and undo both afterwards. Taking keys and values from a second sequence turns the same module into the cross-attention of an encoder-decoder model.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(d_model, d_model) for _ in range(4))

    def split(self, t):                                    # [B, T, D] -> [B, h, T, dh]
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.dh).transpose(1, 2)

    def merge(self, t):                                    # [B, h, T, dh] -> [B, T, D]
        B, _, T, _ = t.shape
        return t.transpose(1, 2).reshape(B, T, self.h * self.dh)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context        # cross-attention reads keys/values from context
        out = attention(self.split(self.q(x)), self.split(self.k(context)), self.split(self.v(context)), mask)
        return self.o(self.merge(out))


def copy_into_torch_mha(mine, ref):
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mine.q.weight, mine.k.weight, mine.v.weight]))
        ref.in_proj_bias.copy_(torch.cat([mine.q.bias, mine.k.bias, mine.v.bias]))
        ref.out_proj.weight.copy_(mine.o.weight)
        ref.out_proj.bias.copy_(mine.o.bias)


D, h = 16, 4
mha, ref = MultiHeadAttention(D, h), nn.MultiheadAttention(D, h, batch_first=True)
copy_into_torch_mha(mha, ref)
x, mem = torch.randn(2, T, D), torch.randn(2, 9, D)
close(mha(x), ref(x, x, x)[0])
close(mha(x, mask=causal), ref(x, x, x, attn_mask=~causal)[0])   # torch MHA: True = blocked
close(mha(x, context=mem), ref(x, mem, mem)[0])
```

## A Transformer block
An encoder block is attention and a two-layer feed-forward network, each wrapped in a residual connection with a LayerNorm. The pre-norm arrangement below normalizes the input to each sublayer and leaves the residual path untouched, which is what most current models use; the original Transformer of Vaswani et al. normalized after the addition instead. A decoder-only (GPT-style) block is the same code called with the causal mask. PyTorch's `nn.TransformerEncoderLayer` with `norm_first=True` and dropout off must agree to rounding error once the weights are copied across.

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff1, self.ff2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        return x + self.ff2(gelu(self.ff1(self.ln2(x))))


block = EncoderBlock(D, h, 4 * D)
ref = nn.TransformerEncoderLayer(D, h, 4 * D, dropout=0.0, activation="gelu",
                                 batch_first=True, norm_first=True).eval()
copy_into_torch_mha(block.attn, ref.self_attn)
for a, b in [(block.ff1, ref.linear1), (block.ff2, ref.linear2), (block.ln1, ref.norm1), (block.ln2, ref.norm2)]:
    b.load_state_dict(a.state_dict())
with torch.no_grad():
    close(block(x), ref(x))
    close(block(x, mask=causal), ref(x, src_mask=~causal))
```

## The KV cache
A decoder generates one token at a time, and with a causal mask the keys and values of earlier tokens never change. Recomputing them at every step makes generation quadratic in projections for no reason. A KV cache stores them and appends one new key and value per step, so each step projects one token and attends over the cache. No mask is needed, because the cache only ever holds the past. It must reproduce the full causal forward pass; [LLM inference](../llm-inference/) covers what the cache costs in memory.

```python
def decode_with_cache(mha, x):
    keys, values, outs = [], [], []
    for t in range(x.shape[1]):
        xt = x[:, t:t + 1]                                  # [B, 1, D]: only the new token
        keys.append(mha.split(mha.k(xt)))
        values.append(mha.split(mha.v(xt)))
        out = attention(mha.split(mha.q(xt)), torch.cat(keys, 2), torch.cat(values, 2))
        outs.append(mha.merge(out))
    return mha.o(torch.cat(outs, 1))


with torch.no_grad():
    close(decode_with_cache(mha, x), mha(x, mask=causal))
```

## Attention without the full score matrix
Standard attention materializes the \(T \times T\) score matrix. FlashAttention (Dao et al., 2022) avoids that by streaming over blocks of keys and keeping, for each query, a running maximum \(m\), a running softmax denominator \(\ell\) and a running weighted sum of values. When a new block raises the maximum from \(m\) to \(m'\), everything accumulated so far is rescaled by \(e^{m - m'}\). This "online softmax" trick is exact, not an approximation, as the assert shows. The speedup comes from doing this in fast on-chip GPU memory; the [FlashAttention](../flashattention/) post covers that part.

```python
def tiled_attention(q, k, v, block=4):
    m = torch.full(q.shape[:-1] + (1,), float("-inf"))     # running max of scores
    l = torch.zeros_like(m)                                # running sum of exp(score - m)
    acc = torch.zeros(q.shape[:-1] + (v.shape[-1],))       # running sum of exp(score - m) * v
    for s in range(0, k.shape[-2], block):
        scores = q @ k[..., s:s + block, :].transpose(-2, -1) / math.sqrt(q.shape[-1])
        m_new = torch.maximum(m, scores.amax(-1, keepdim=True))
        p = (scores - m_new).exp()
        scale = (m - m_new).exp()
        l = l * scale + p.sum(-1, keepdim=True)
        acc = acc * scale + p @ v[..., s:s + block, :]
        m = m_new
    return acc / l


q, k, v = torch.randn(3, 2, 4, 37, 8).unbind(0)
close(tiled_attention(q, k, v, block=5), attention(q, k, v))
```

## LSTM and GRU cells
An LSTM keeps a cell state \(c\) and updates it through input, forget and output gates. The input and output gates come from Hochreiter and Schmidhuber (1997); the forget gate, now standard, was added by Gers, Schmidhuber and Cummins (2000). A GRU merges the state and the cell and uses reset and update gates (Cho et al., 2014). Both cells are a handful of lines. The details that break a comparison with PyTorch are the gate order inside the stacked weight matrices, `i, f, g, o` for the LSTM and `r, z, n` for the GRU, and the fact that PyTorch's GRU applies the reset gate to the hidden-state term after the matrix product and its bias. Cho et al. apply it to \(h\) before the product, a slightly different cell, so a GRU written from the paper will not match `nn.GRU`.

```python
def lstm(x, W_ih, W_hh, b_ih, b_hh):                      # x [B, T, I], W_ih [4H, I]
    B, T, _ = x.shape
    h = c = torch.zeros(B, W_hh.shape[1])
    outs = []
    for t in range(T):
        i, f, g, o = (x[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs, 1)                            # [B, T, H]


def gru(x, W_ih, W_hh, b_ih, b_hh):                       # W_ih [3H, I]
    B, T, _ = x.shape
    h = torch.zeros(B, W_hh.shape[1])
    outs = []
    for t in range(T):
        xr, xz, xn = (x[:, t] @ W_ih.T + b_ih).chunk(3, 1)
        hr, hz, hn = (h @ W_hh.T + b_hh).chunk(3, 1)
        r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        h = (1 - z) * n + z * h
        outs.append(h)
    return torch.stack(outs, 1)


x = torch.randn(3, 7, 5)
with torch.no_grad():
    for mine, ref in [(lstm, nn.LSTM(5, 4, batch_first=True)), (gru, nn.GRU(5, 4, batch_first=True))]:
        weights = ref.weight_ih_l0, ref.weight_hh_l0, ref.bias_ih_l0, ref.bias_hh_l0
        close(mine(x, *weights), ref(x)[0])
```

## Convolution
A 2-D convolution slides a \(k \times k\) filter over the image and takes a dot product at each position. Strictly, deep learning libraries compute a cross-correlation: the filter is not flipped, which makes no difference to a learned filter. One clear way to write it: pad the input, take the \(k^2\) shifted and strided slices of it, one per filter offset, and contract them with the filter weights in a single `einsum`. Stacking those slices is im2col. The stacked tensor holds every patch, \(k^2\) times the input's size at stride 1, and is the same matrix `F.unfold` builds, only kept as six axes instead of two.

```python
def im2col(x, kh, kw, stride=1, pad=0):                  # x [B, C, H, W] -> [B, C, kh, kw, Ho, Wo]
    x = F.pad(x, (pad, pad, pad, pad))
    Ho = (x.shape[2] - kh) // stride + 1
    Wo = (x.shape[3] - kw) // stride + 1
    return torch.stack([torch.stack([x[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride]
                                     for j in range(kw)], 2) for i in range(kh)], 2)


def conv2d(x, w, b, stride=1, pad=0):                     # w [O, C, kh, kw]
    cols = im2col(x, w.shape[2], w.shape[3], stride, pad)
    return torch.einsum("bcijhw,ocij->bohw", cols, w) + b[:, None, None]


x, w, b = torch.randn(2, 3, 11, 9), torch.randn(4, 3, 3, 3), torch.randn(4)
for stride, pad in [(1, 0), (2, 1), (3, 2)]:
    close(conv2d(x, w, b, stride, pad), F.conv2d(x, w, b, stride=stride, padding=pad))
    cols = im2col(x, 3, 3, stride, pad)
    close(cols.reshape(2, 3 * 9, -1), F.unfold(x, 3, padding=pad, stride=stride))   # the same patch matrix
close(conv2d(x, w.flip(2, 3), b), F.conv2d(x, w.flip(2, 3), b))                     # no flip anywhere: cross-correlation
```

## LoRA
LoRA (Hu et al., 2021) freezes a pretrained weight \(W\) and learns a low-rank update \(\Delta W = \frac{\alpha}{r} BA\), with \(A\) of shape \([r, d_\text{in}]\) and \(B\) of shape \([d_\text{out}, r]\). \(B\) starts at zero, so the adapted layer starts out identical to the original, and after training the update can be merged into \(W\) so inference costs nothing extra. Three asserts cover those claims; [LoRA explained](../lora-explained/) and the long-form [LoRA from scratch](/posts/lora-from-scratch/) go further.

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base, self.scale = base.requires_grad_(False), alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def merged(self):
        return self.base.weight + self.scale * self.B @ self.A


base = nn.Linear(64, 32)
lora = LoRALinear(base)
x = torch.randn(5, 64)
with torch.no_grad():
    close(lora(x), base(x))                                # B = 0: no change at initialization
    lora.B.normal_()
    close(lora(x), F.linear(x, lora.merged(), base.bias))  # merging is exact
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
assert trainable == 4 * (64 + 32)                          # vs 64 * 32 + 32 in the frozen layer
```

## Beam search
Greedy decoding takes the most likely token at each step and can walk into a sequence whose total probability is low. Beam search keeps the \(k\) best partial sequences by total log-probability at each step. On a toy model small enough to enumerate every sequence, the checks are clean: a beam of width 1 is greedy decoding, a beam wide enough to keep everything finds the true best sequence, and here greedy is strictly worse than the best.

```python
from itertools import product

V, L = 4, 4
gen = torch.Generator().manual_seed(0)
logp = torch.log_softmax(torch.randn(V + 1, V, generator=gen) * 2, 1)   # row V is the start; next token depends on the last


def score(seq):
    prev, s = V, 0.0
    for tok in seq:
        s, prev = s + logp[prev, tok].item(), tok
    return s


def beam_search(width):
    beams = [((), 0.0)]
    for _ in range(L):
        grown = [(seq + (t,), s + logp[seq[-1] if seq else V, t].item()) for seq, s in beams for t in range(V)]
        beams = sorted(grown, key=lambda b: -b[1])[:width]
    return beams[0]


greedy, prev = [], V
for _ in range(L):
    prev = int(logp[prev].argmax())
    greedy.append(prev)

best = max(product(range(V), repeat=L), key=score)
assert beam_search(1)[0] == tuple(greedy)
assert beam_search(V ** L)[0] == best
assert score(greedy) < score(best)                         # -3.48 vs -1.97: greedy walked into a bad path
```

Real decoders add a length penalty, since summed log-probabilities favour short outputs, and stop each beam at an end-of-sequence token.

## Classical models in NumPy
Linear regression has a closed form, the least-squares solution, best computed with `lstsq` rather than by inverting \(X^\top X\) (see [regression as maximum likelihood](../regression-as-maximum-likelihood/)). Logistic regression has none, but its loss is convex, so plain gradient descent on the mean log-loss reaches the same optimum as scikit-learn's solver. The gradient is \(X^\top(\sigma(Xw) - y)/n\), the same "prediction minus label" shape as the softmax gradient above.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

n = 400
Xn = rng.normal(size=(n, 3))
Xb = np.c_[np.ones(n), Xn]                                 # intercept column
y_reg = Xn @ [1.5, -2.0, 0.5] + 3 + rng.normal(size=n)
w_ls = np.linalg.lstsq(Xb, y_reg, rcond=None)[0]
sk = LinearRegression().fit(Xn, y_reg)
close(w_ls, np.r_[sk.intercept_, sk.coef_])

y_clf = (rng.random(n) < 1 / (1 + np.exp(-(Xn @ [2.0, -1.0, 0.0] + 0.5)))).astype(float)
w = np.zeros(4)
for _ in range(3000):
    p = 1 / (1 + np.exp(-Xb @ w))
    w -= 1.0 * Xb.T @ (p - y_clf) / n
sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(Xn, y_clf)
close(w, np.r_[sk.intercept_, sk.coef_[0]], tol=1e-5)
```

K-nearest neighbours is a distance matrix and a vote. K-means (Lloyd's algorithm) alternates two steps until the assignments stop changing: assign each point to its nearest center, then move each center to the mean of its points. Given the same starting centers, scikit-learn's `KMeans` runs the same iterations, so the results agree exactly.

```python
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def sq_dists(A, B):                                        # [n, m] squared Euclidean distances
    return (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None, :]


def knn_predict(X_train, y_train, X_test, k=5):
    nearest = np.argsort(sq_dists(X_test, X_train), axis=1)[:, :k]
    return (y_train[nearest].mean(1) > 0.5).astype(int)   # two classes, odd k: no ties


def kmeans(X, centers):
    labels = None
    while True:
        new = sq_dists(X, centers).argmin(1)
        if labels is not None and (new == labels).all():
            return centers, labels
        labels = new
        centers = np.array([X[labels == j].mean(0) if (labels == j).any() else centers[j]
                            for j in range(len(centers))])


Xk = np.r_[rng.normal(0, 1, (100, 2)), rng.normal(4, 1, (100, 2)), rng.normal([0, 5], 1, (100, 2))]
yk = (Xk[:, 0] + rng.normal(0, 1, 300) > 2).astype(int)
X_test = rng.normal(2, 3, (50, 2))
assert (knn_predict(Xk, yk, X_test) == KNeighborsClassifier(5).fit(Xk, yk).predict(X_test)).all()

init = Xk[rng.choice(300, 3, replace=False)]
centers, labels = kmeans(Xk, init)
sk = KMeans(3, init=init, n_init=1, algorithm="lloyd", tol=0).fit(Xk)
close(centers, sk.cluster_centers_)
assert (labels == sk.labels_).all()
```

A decision tree tries every feature and every threshold, keeps the split that most reduces Gini impurity, and recurses. scikit-learn places each threshold halfway between consecutive sorted values, so doing the same gives the same splits, which the predictions confirm. Two candidate splits with exactly equal impurity could still be broken differently, since scikit-learn visits features in a random order; [from decision trees to boosting](../decision-trees-to-boosting/) builds on this.

```python
from sklearn.tree import DecisionTreeClassifier


def gini(y):
    p = np.bincount(y, minlength=2) / len(y)
    return 1 - (p ** 2).sum()


def build_tree(X, y, depth):
    if depth == 0 or gini(y) == 0:
        return np.bincount(y, minlength=2).argmax()
    best = None
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for t in (values[:-1] + values[1:]) / 2:
            left = X[:, f] <= t
            cost = left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])
            if best is None or cost < best[0]:
                best = (cost, f, t)
    _, f, t = best
    left = X[:, f] <= t
    return (f, t, build_tree(X[left], y[left], depth - 1), build_tree(X[~left], y[~left], depth - 1))


def tree_predict(node, x):
    while isinstance(node, tuple):
        f, t, l, r = node
        node = l if x[f] <= t else r
    return node


Xt = rng.normal(size=(200, 3))
yt = ((Xt[:, 0] > 0.3) ^ (Xt[:, 1] * Xt[:, 2] > 0)).astype(int)
tree = build_tree(Xt, yt, depth=3)
sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xt, yt)
X_test = rng.normal(size=(500, 3))
assert (np.array([tree_predict(tree, x) for x in X_test]) == sk.predict(X_test)).all()
```

## Pitfalls
The bugs these asserts catch, roughly from most to least common:

- **Wrong reduction axis.** Softmax over the queries instead of the keys, LayerNorm over the batch. The output has the right shape and is quietly wrong.
- **`view` after `transpose`.** A transposed tensor is not contiguous, so merging heads needs `reshape` (or `.contiguous().view`). Reshaping straight from `[B, T, D]` to `[B, h, T, dh]` without the transpose runs and mixes tokens across heads.
- **Biased versus unbiased variance.** Normalization uses the biased variance; `torch.var` and `np.var` disagree on the default.
- **Mask conventions.** In `F.scaled_dot_product_attention` a boolean `True` means "may attend"; in `nn.MultiheadAttention` it means "blocked". A row whose keys are all masked has nothing to normalize: a plain softmax returns NaN there (\(-\infty\) minus \(-\infty\) after the max shift, zero over zero without it), while recent PyTorch versions of `F.scaled_dot_product_attention` return zeros. Padding masks need care.
- **Forgetting the \(\sqrt{d_k}\).** Attention still trains, but the softmax saturates as the head size grows.
- **Train and eval mode.** BatchNorm and dropout behave differently in each; evaluating in training mode changes both the outputs and the running statistics.
- **Weight layout.** `nn.Linear` stores `[out, in]`; LSTM and GRU gates are stacked in a fixed order.
- **Loose tolerances.** In float32 a 1e-4 tolerance hides real bugs, such as a missing \(\epsilon\) in a normalization. Test in float64 and tighten.

The checklist of building blocks follows a post on [Yuan Meng's blog](https://www.yuan-meng.com/); the code and explanations here are original. For building these pieces up from nothing, Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and Sebastian Raschka's [PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) are excellent.
