---
title: FlashAttention: exact attention without the N-by-N matrix
summary: Standard attention writes an N-by-N score matrix to GPU memory and reads it back, so it is limited by memory traffic, not arithmetic. FlashAttention computes the same output in tiles that stay on chip, using an online softmax, and recomputes the scores in the backward pass. Both passes rebuilt in NumPy and checked against PyTorch, with the traffic counted.
group: LLMs and generative AI
order: 3
tags: attention · GPU · transformers · inference · training
---
FlashAttention, by Tri Dao, Daniel Fu, Stefano Ermon, Atri Rudra and Christopher Ré (2022), is a way of computing ordinary softmax attention. It is not an approximation and it does not change the model: the output is the same up to floating-point rounding. What changes is the order of the work. Standard attention computes the whole \(N \times N\) matrix of scores, writes it to GPU memory, and reads it back for the softmax and again for the product with the values. FlashAttention never builds that matrix. It processes queries and keys in small tiles that fit in fast on-chip memory, and the full matrix exists only one tile at a time.

Two ideas make that possible: a softmax computed one block of keys at a time, and a backward pass that recomputes the scores instead of storing them. Below, both are built in NumPy, checked against PyTorch, and the memory traffic is counted.

Throughout, \(N\) is the sequence length, \(d\) the head dimension, and for one head

\[ S = \frac{QK^\top}{\sqrt{d}}, \qquad P = \operatorname{softmax}(S) \text{ row by row}, \qquad O = PV . \]

## Where the time goes
A GPU has two kinds of memory that matter here. High-bandwidth memory (HBM) is the large one, tens of gigabytes. On-chip SRAM is small, a couple of hundred kilobytes per streaming multiprocessor, and about an order of magnitude faster. The FlashAttention paper's figures for an A100 are 1.5 to 2.0 TB/s for HBM and an estimated 19 TB/s for SRAM. A kernel loads its inputs from HBM into SRAM, computes, and writes its outputs back.

Matrix products do many operations per byte loaded. The softmax does a handful per element of \(S\), so a kernel that reads \(S\) from HBM and writes \(P\) back is **memory-bound**: it waits on HBM, not on arithmetic. Standard attention is a chain of such kernels, each moving an \(N \times N\) matrix in and out of HBM, and training adds passes for the mask and dropout and keeps \(P\) for the backward pass.

The memory alone becomes the problem first. In half precision the score matrix for one head at \(N = 8192\) is 128 MiB, and a layer with 32 heads needs 4 GiB of it per sequence.

```python
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp

rng = np.random.default_rng(0)


def standard_attention(Q, K, V, mask=None):
    S = Q @ K.T / Q.shape[1] ** 0.5                # N x N, written to memory
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    P = np.exp(S - S.max(axis=1, keepdims=True))   # read back, another N x N written
    P /= P.sum(axis=1, keepdims=True)
    return P @ V                                   # and read back once more


N, d = 256, 64
Q, K, V = rng.standard_normal((3, N, d))
causal = np.tri(N, dtype=bool)
t = torch.from_numpy
ref = standard_attention(Q, K, V)
ref_causal = standard_attention(Q, K, V, causal)
assert np.allclose(ref, F.scaled_dot_product_attention(t(Q), t(K), t(V)).numpy())
assert np.allclose(ref_causal, F.scaled_dot_product_attention(t(Q), t(K), t(V), is_causal=True).numpy())

MiB = 2 ** 20
assert 8192 * 8192 * 2 == 128 * MiB                # one head, fp16
assert 32 * 8192 * 8192 * 2 == 4 * 1024 * MiB      # 32 heads: 4 GiB per layer, per sequence
```

## Softmax one block at a time
The obstacle to tiling is the softmax. To normalize one row of scores you need the sum of the exponentials over every key, and for numerical safety you subtract the row maximum before exponentiating (see [numerically stable softmax](../stable-softmax-cross-entropy/)). Both look like they need the whole row first.

They do not. Keep a running maximum \(m\) and a running sum \(\ell\) of \(e^{s - m}\). When a new score \(s\) arrives, the new maximum is \(m' = \max(m, s)\), and the old sum was computed relative to the old maximum, so it is rescaled:

\[ \ell' = \ell\, e^{m - m'} + e^{s - m'} . \]

At the end \(m\) is the true maximum and \(\ell\) the true sum, so \(m + \log \ell\) is the log-sum-exp of the row. This **online softmax** was described by Milakov and Gimelshein (2018). The same rescaling works for whole blocks of keys, and it extends to the weighted sum of values. If a block of keys has maximum \(m_B\), sum \(\ell_B\) and unnormalized output \(a_B = \sum_k e^{s_k - m_B} v_k\), two blocks merge as

\[ m = \max(m_1, m_2), \quad \ell = e^{m_1 - m}\ell_1 + e^{m_2 - m}\ell_2, \quad a = e^{m_1 - m} a_1 + e^{m_2 - m} a_2 , \]

and the answer is \(o = a / \ell\). The merge is associative and commutative, so the keys can be split anywhere, visited in any order and combined in any grouping. The check below splits one query's 256 keys at random points, shuffles the pieces, and gets the reference output both by folding them left to right and by merging them pairwise as a tree:

```python
def online_stats(x):
    m, l = -np.inf, 0.0
    for xi in x:
        m_new = max(m, xi)
        l = l * np.exp(m - m_new) + np.exp(xi - m_new)
        m = m_new
    return m, l


x = rng.standard_normal(1000) * 300                # scores large enough to overflow exp
m, l = online_stats(x)
assert m == x.max() and np.isclose(l, np.exp(x - x.max()).sum())
assert np.isclose(m + np.log(l), logsumexp(x))
with np.errstate(over="ignore"):
    assert not np.isfinite(np.exp(x).sum())        # the unshifted sum overflows


def partial(s, v):                                 # (max, sum, unnormalized output) of one block
    m = s.max()
    p = np.exp(s - m)
    return m, p.sum(), p @ v


def merge(a, b):
    (ma, la, acc_a), (mb, lb, acc_b) = a, b
    m = max(ma, mb)
    ca, cb = np.exp(ma - m), np.exp(mb - m)
    return m, ca * la + cb * lb, ca * acc_a + cb * acc_b


s = Q[0] @ K.T / np.sqrt(d)                        # one query's scores
for trial in range(5):
    cuts = np.sort(rng.choice(np.arange(1, N), size=rng.integers(1, 10), replace=False))
    chunks = np.split(np.arange(N), cuts)
    rng.shuffle(chunks)                            # any split, any order
    state = partial(s[chunks[0]], V[chunks[0]])
    for c in chunks[1:]:
        state = merge(state, partial(s[c], V[c]))
    m, l, acc = state
    assert np.allclose(acc / l, ref[0])
    parts = [partial(s[c], V[c]) for c in chunks]
    while len(parts) > 1:                          # any grouping: pairwise, like a reduction tree
        parts = [merge(*parts[k:k + 2]) if k + 1 < len(parts) else parts[k] for k in range(0, len(parts), 2)]
    assert np.allclose(parts[0][2] / parts[0][1], ref[0])
```

## The tiled forward pass
With the merge in hand the forward pass is two loops. Take a block of \(B_r\) queries and keep its running \(m\), \(\ell\) and accumulator on chip. Stream the keys and values past it in blocks of \(B_c\): compute the \(B_r \times B_c\) tile of scores, fold it into the running state, discard it. After the last key block, divide by \(\ell\) and write the output block to HBM, together with one number per query, the log-sum-exp \(L = m + \log \ell\), which the backward pass needs.

This is the loop order of FlashAttention-2 (Dao, 2023): queries outside, keys inside. The original paper put the key blocks in the outer loop, which meant reading and writing the partial outputs and statistics to HBM on every inner step. Swapping the loops keeps them on chip, and since query blocks are independent, each one can go to a different thread block, which parallelizes long sequences even at batch size 1.

A causal mask adds a free saving: a key block that lies entirely after a query block contributes nothing, so the inner loop stops at the diagonal and roughly half the tiles are never computed.

```python
def flash_forward(Q, K, V, Br=32, Bc=32, mask=None, causal=False, safe=True, io=None):
    N, d = Q.shape
    O, L = np.empty_like(Q), np.empty(N)
    for i in range(0, N, Br):
        Qi = Q[i:i + Br] / d ** 0.5
        br = len(Qi)
        m, l, acc = np.full(br, -np.inf, Q.dtype), np.zeros(br, Q.dtype), np.zeros((br, d), Q.dtype)   # on chip
        if io is not None:
            io["read"] += Qi.size
        stop = min(i + br, N) if causal else N    # causal: skip blocks past the diagonal
        for j in range(0, stop, Bc):
            Kj, Vj = K[j:j + Bc], V[j:j + Bc]
            if io is not None:
                io["read"] += Kj.size + Vj.size
                io["tiles"] += 1
            S = Qi @ Kj.T                          # Br x Bc, never leaves the chip
            if causal:
                S = np.where(np.arange(j, j + len(Kj)) <= np.arange(i, i + br)[:, None], S, -np.inf)
            if mask is not None:
                S = np.where(mask[i:i + br, j:j + len(Kj)], S, -np.inf)
            m_new = np.maximum(m, S.max(axis=1))
            shift = np.where(np.isneginf(m_new), 0.0, m_new) if safe else m_new
            P = np.exp(S - shift[:, None])
            c = np.exp(m - shift)                  # rescale what was accumulated so far
            l = c * l + P.sum(axis=1)
            acc = c[:, None] * acc + P @ Vj
            m = m_new
        O[i:i + br] = acc / l[:, None]
        L[i:i + br] = m + np.log(l)                # one number per query is all that is kept
        if io is not None:
            io["write"] += O[i:i + br].size + br
    return O, L


O, L = flash_forward(Q, K, V)
assert np.allclose(O, ref)
assert np.allclose(L, logsumexp(Q @ K.T / np.sqrt(d), axis=1))
for Br, Bc in [(16, 64), (48, 40), (256, 256), (1, 7)]:     # block sizes need not divide N
    assert np.allclose(flash_forward(Q, K, V, Br, Bc)[0], ref)
    assert np.allclose(flash_forward(Q, K, V, Br, Bc, causal=True)[0], ref_causal)
    assert np.allclose(flash_forward(Q, K, V, Br, Bc, mask=causal)[0], ref_causal)

io_full, io_causal = ({"read": 0, "write": 0, "tiles": 0} for _ in range(2))
flash_forward(Q, K, V, io=io_full)
flash_forward(Q, K, V, causal=True, io=io_causal)
T = N // 32
assert io_full["tiles"] == T * T and io_causal["tiles"] == T * (T + 1) // 2   # 64 vs 36 tiles
```

## Counting the traffic
The counters make the saving concrete. Each query block is read once, the outputs and the log-sum-exp are written once, and the keys and values are read once per query block, \(N / B_r\) times in total:

\[ \text{FlashAttention: } 2Nd + N + \frac{2N^2 d}{B_r} \qquad \text{vs.} \qquad \text{standard: } 4Nd + 4N^2 \]

elements moved, where the standard count is the plain three-step version above: write \(S\), read it, write \(P\), read it, plus the inputs and output. The tile must fit in SRAM of size \(M\), so \(B_r\) can grow to about \(M/d\), and \(2N^2d/B_r\) becomes \(O(N^2 d^2 / M)\), the paper's bound. With \(d\) in the range 64 to 128 and \(M\) around 100K elements, \(d^2/M\) is well below 1. The paper also proves that no exact attention algorithm can do asymptotically better for every SRAM size between \(d\) and \(Nd\).

```python
assert io_full["read"] == N * d + T * 2 * N * d
assert io_full["write"] == N * d + N


def standard_traffic(N, d):
    return 4 * N * d + 4 * N * N


def flash_traffic(N, d, Br):
    return 2 * N * d + N + 2 * N * N * d // Br


assert io_full["read"] + io_full["write"] == flash_traffic(N, d, 32)
for n in [1024, 4096, 16384, 65536]:
    std, fl = 2 * standard_traffic(n, 64), 2 * flash_traffic(n, 64, 128)   # bytes in fp16
    print(f"N={n:6d}  standard {std / 1e6:9.1f} MB  flash {fl / 1e6:8.1f} MB  ratio {std / fl:.2f}  "
          f"extra memory {2 * n * n / MiB:6.0f} MiB vs {4 * n / 1024:3.0f} KiB")
    assert 3.7 < std / fl < 4.0                    # tends to 2 Br / d = 4
```

For one head with \(d = 64\), \(B_r = 128\) and half precision:

| \(N\) | Standard traffic | FlashAttention traffic | Extra memory, standard | Extra memory, FlashAttention |
|---|---|---|---|---|
| 1,024 | 8.9 MB | 2.4 MB | 2 MiB | 4 KiB |
| 4,096 | 136.3 MB | 34.6 MB | 32 MiB | 16 KiB |
| 16,384 | 2,155.9 MB | 541.1 MB | 512 MiB | 64 KiB |
| 65,536 | 34,393.3 MB | 8,606.8 MB | 8 GiB | 256 KiB |

The traffic still grows like \(N^2\), only with a constant about four times smaller in this simple model; real unfused implementations make more passes (masking, dropout, the backward pass), and the paper measured about a ninth of the HBM traffic (4.4 GB against 40.3 GB) for a GPT-2-medium-sized attention, forward and backward. The memory column is the bigger change: the quadratic buffer becomes one float per query, which is what allows training at lengths where standard attention runs out of memory. The paper reports about 3x faster GPT-2 training at length 1K and the first transformers to beat chance on the Path-X long-range benchmark (length 16K).

## The backward pass: recompute instead of store
The gradients of attention need \(P\). Standard attention keeps it from the forward pass, which is exactly the \(N^2\) memory FlashAttention avoids. Instead it keeps \(Q, K, V, O\) and \(L\), and rebuilds any tile of \(P\) on chip as \(P_{ij} = \exp(S_{ij} - L_i)\), with no second pass to normalize. Given the upstream gradient \(dO\):

\[ dV = P^\top dO, \qquad dP = dO\, V^\top, \qquad dS_{ij} = P_{ij}\,(dP_{ij} - D_i), \qquad dQ = \frac{dS\,K}{\sqrt{d}}, \qquad dK = \frac{dS^\top Q}{\sqrt{d}} . \]

The \(dS\) line is the softmax Jacobian \(\operatorname{diag}(p) - pp^\top\) applied row by row, and the term it subtracts is \(D_i = \sum_k P_{ik}\, dP_{ik} = \sum_k P_{ik}\,(dO_i \cdot v_k) = dO_i \cdot O_i\). That identity is why a full row of \(P\) is never needed: \(D\) comes from two \(N \times d\) matrices that are already stored.

The backward loop runs the other way round from the forward, as in both papers: key blocks outside, so \(dK_j\) and \(dV_j\) accumulate on chip, and query blocks inside, adding into \(dQ\) (on a GPU, with atomic adds, since several key blocks write to the same rows).

```python
def flash_backward(Q, K, V, O, L, dO, Br=32, Bc=32):
    N, d = Q.shape
    scale = 1 / np.sqrt(d)
    D = (dO * O).sum(axis=1)                       # rowsum(dO * O) = rowsum(P * dP)
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for j in range(0, N, Bc):
        Kj, Vj = K[j:j + Bc], V[j:j + Bc]
        dKj, dVj = np.zeros_like(Kj), np.zeros_like(Vj)
        for i in range(0, N, Br):
            Qi, dOi = Q[i:i + Br], dO[i:i + Br]
            P = np.exp(Qi @ Kj.T * scale - L[i:i + Br, None])   # recomputed tile
            dVj += P.T @ dOi
            dP = dOi @ Vj.T
            dS = P * (dP - D[i:i + Br, None])
            dQ[i:i + Br] += dS @ Kj * scale
            dKj += dS.T @ Qi * scale
        dK[j:j + Bc], dV[j:j + Bc] = dKj, dVj
    return dQ, dK, dV


dO = rng.standard_normal((N, d))
q, k, v = (t(a.copy()).requires_grad_() for a in (Q, K, V))
F.scaled_dot_product_attention(q, k, v).backward(t(dO))
for mine, autograd in zip(flash_backward(Q, K, V, O, L, dO, Br=32, Bc=64), (q.grad, k.grad, v.grad)):
    assert np.allclose(mine, autograd.numpy())
P_full = np.exp(Q @ K.T / np.sqrt(d) - L[:, None])
assert np.allclose(P_full.sum(axis=1), 1)          # L alone normalizes any recomputed tile
assert np.allclose((dO * O).sum(1), (P_full * (dO @ V.T)).sum(1))   # the D identity
```

Recomputation costs arithmetic. The forward pass does two \(N \times N \times d\) matrix products and the standard backward four; FlashAttention's backward adds a fifth to rebuild \(S\), so attention does about 7/6 of the matrix-product work. It is still faster, because the time it saves was spent waiting on HBM. Dropout follows the same logic: the random mask is not stored but regenerated in the backward pass from the saved random-number-generator state.

## FlashAttention-2 and -3
FlashAttention-2 (Dao, 2023) kept the algorithm and reorganized the work: the loop swap above, fewer non-matrix-multiply operations (dividing by \(\ell\) once at the end, not every step), and a better split of each tile among warps. It roughly doubled the speed of the first version. FlashAttention-3 (Shah, Bikshandi, Zhang, Thakkar, Ramani and Dao, 2024) targets the H100: it overlaps data movement, matrix products and the softmax using the Hopper GPU's asynchronous hardware, and adds an FP8 path.

In practice you rarely call any of these directly. PyTorch's `F.scaled_dot_product_attention` dispatches to a FlashAttention kernel when the device, data type, head size and mask allow it, and falls back to other kernels otherwise. The `flash-attn` package and libraries such as xFormers expose the kernels directly.

## When it helps, and pitfalls
- **Long sequences in training and prefill gain the most.** The saving scales with \(N^2\). For short sequences the attention matrix is small and the rest of the layer dominates.
- **Decoding is a different bottleneck.** Generating one token means one query against a long KV cache, so there is no \(N \times N\) matrix to avoid; the cost is reading the cache. Splitting the keys across thread blocks and merging the partial results with the same \((m, \ell, a)\) rule is the idea behind Flash-Decoding. See [LLM inference](../llm-inference/).
- **Exact is not bitwise identical.** The sums happen in a different order, so results differ from the naive kernel in the last bits (below \(10^{-6}\) in float32 in the check below, more in float16). Tolerance-based tests are fine; tests that compare hashes of outputs are not.
- **Fully masked tiles produce NaN if the update is written naively.** With left padding, the first key tile a query sees can be entirely masked. Then the running maximum is still \(-\infty\), and \(e^{-\infty - (-\infty)}\) is NaN, which poisons every later step. The `safe` shift in `flash_forward` substitutes 0 in that case. Real kernels guard this; a hand-written or custom kernel may not.
- **Unusual attention variants need kernel support.** A mask the kernel does not understand, a bias added to the scores, or an unusual head size can silently send PyTorch to a slower path. Check which backend runs (`torch.nn.attention.sdpa_kernel` can restrict the choice so a fallback fails loudly) or use PyTorch's FlexAttention, which compiles fused kernels for custom score modifications.
- **It does not reduce FLOPs.** Compute still scales as \(N^2\); FlashAttention changes how efficiently that compute runs, so it complements approximate attention, pruning or distillation rather than replacing them.

```python
pad = np.ones((N, N), dtype=bool)
pad[:, :40] = False                                # the first 40 keys are left padding
want = standard_attention(Q, K, V, pad)
with np.errstate(invalid="ignore"):
    naive, _ = flash_forward(Q, K, V, mask=pad, safe=False)
assert np.isnan(naive).all()                       # first tile fully masked: NaN everywhere
assert np.allclose(flash_forward(Q, K, V, mask=pad)[0], want)

Q32, K32, V32 = (a.astype(np.float32) for a in (Q, K, V))
a, b = standard_attention(Q32, K32, V32), flash_forward(Q32, K32, V32)[0]
assert not np.array_equal(a, b) and np.abs(a - b).max() < 1e-6   # same math, different bits
```

For where the \(1/\sqrt{d}\) in every block above comes from, see [why attention divides by the square root of d](../attention-scaling/), and for the surrounding architecture, [transformers and LLMs: key ideas](../transformers-and-llms-key-ideas/).

## References
1. Dao, Fu, Ermon, Rudra and Ré, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
2. Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023.
3. Shah, Bikshandi, Zhang, Thakkar, Ramani and Dao, [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608), 2024.
4. Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018.

The explanation here follows the FlashAttention papers and Tri Dao's talk on FlashAttention in the Stanford MedAI seminar series; the code is a NumPy reconstruction for reading, not the CUDA kernels.
