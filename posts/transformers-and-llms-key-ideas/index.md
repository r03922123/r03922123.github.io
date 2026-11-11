# Transformers and LLMs: the key engineering ideas

At batch size one, a 7-billion-parameter model in 16-bit floats reads about 14 GB of weights for every token it generates, so a data-centre GPU manages about 140 steps per second at best. A transformer language model is a short stack of ideas: attention, an MLP, residual connections, normalization, and a next-token loss. What makes such a model affordable to train and serve is a second set of ideas, mostly about memory. This post covers the ones that matter most, each with the reason it exists, the mechanism, and a small experiment whose assertions check the claim. I build three of them in small models and check with asserts that each one matches the slower computation it replaces: the KV cache, speculative decoding (in distribution) and blockwise attention.

Throughout, \(L\) is the number of layers, \(H\) the number of query heads, \(d_h\) the size of one head, \(N\) the number of tokens in the sequence, and \(p\) and \(q\) are probability distributions over the vocabulary.

## Why generation is slow
Text is produced one token at a time. Processing the prompt (the prefill) is one big parallel pass that keeps the GPU busy. Each generation step (the decode) does little arithmetic per weight but must read every weight from GPU memory.

A 7-billion-parameter model in 16-bit floats is about 14 GB. At batch size one, a decode step reads those 14 GB to do roughly \(2 \times 7 \times 10^9\) operations, about one per byte. A data-centre GPU reads roughly 2 TB/s but computes over a hundred times more operations per second than that, so bandwidth sets the limit: about \(2 \times 10^{12} / 1.4 \times 10^{10} \approx 140\) steps per second at best. Most ideas below either move fewer bytes per token or produce more tokens per pass over the weights.

## The KV cache
In causal attention, the key and value vectors of token \(i\) depend only on tokens \(1, \dots, i\), so they never change once computed. Keep each layer's keys and values in a cache; at step \(t\), compute the query, key and value of the new token only, append the key and value, and attend from the one new query over all \(t\) cached keys.

Per layer, a step then costs \(O(d^2)\) for the new token's projections and MLP plus \(O(t\,d)\) for attention over the cache. Without the cache, step \(t\) would redo the projections and MLP for all \(t\) tokens and the full \(t \times t\) score matrix. With it, attention is still linear in \(t\) per step and quadratic over a whole generation: the cache removes recomputation over the prefix, but it does not make attention any cheaper than one new row of scores per step. The code builds a one-layer model (whose attention allows fewer key/value heads than query heads, for later), decodes with and without a cache, and checks they agree.

```python
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
d_model, n_heads, vocab = 64, 8, 50
d_head = d_model // n_heads


class TinyLM(torch.nn.Module):
    """Embedding, one causal attention layer with n_kv key/value heads, unembedding."""

    def __init__(self, n_kv):
        super().__init__()
        self.n_kv = n_kv
        self.emb = torch.nn.Embedding(vocab, d_model)
        self.wq = torch.nn.Linear(d_model, n_heads * d_head, bias=False)
        self.wkv = torch.nn.Linear(d_model, 2 * n_kv * d_head, bias=False)
        self.wo = torch.nn.Linear(n_heads * d_head, d_model, bias=False)
        self.out = torch.nn.Linear(d_model, vocab, bias=False)
        self.rows_projected = 0                                    # token rows pushed through wq/wkv

    def forward(self, tokens, cache=None):
        x = self.emb(tokens)                                       # (T, d_model), only the new tokens
        T = x.shape[0]
        self.rows_projected += T
        q = self.wq(x).view(T, n_heads, d_head).transpose(0, 1)    # (H, T, d_head)
        k, v = self.wkv(x).view(T, 2, self.n_kv, d_head).permute(1, 2, 0, 3)
        if cache is not None:                                      # prepend cached keys and values
            k, v = torch.cat([cache[0], k], dim=1), torch.cat([cache[1], v], dim=1)
        S = k.shape[1]                                             # total tokens so far
        group = n_heads // self.n_kv                               # query heads per key/value head
        kk, vv = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
        scores = q @ kk.transpose(1, 2) / d_head ** 0.5            # (H, T, S)
        pos_q = torch.arange(S - T, S)[:, None]                    # absolute positions of the queries
        scores = scores.masked_fill(torch.arange(S)[None, :] > pos_q, float("-inf"))
        ctx = (torch.softmax(scores, -1) @ vv).transpose(0, 1).reshape(T, -1)
        return self.out(x + self.wo(ctx)), (k, v)


def generate(model, prompt, steps, use_cache):
    tokens, cache, logits_seen = list(prompt), None, []
    logits, cache = model(torch.tensor(tokens))
    for _ in range(steps):
        logits_seen.append(logits[-1])
        tokens.append(int(logits[-1].argmax()))                    # greedy decoding
        if use_cache:
            logits, cache = model(torch.tensor(tokens[-1:]), cache)
        else:
            logits, _ = model(torch.tensor(tokens))                # recompute the whole prefix
    return tokens, torch.stack(logits_seen)


model = TinyLM(n_kv=n_heads)
prompt, steps = [3, 14, 15, 9, 2], 40
slow, slow_logits = generate(model, prompt, steps, use_cache=False)
rows_slow, model.rows_projected = model.rows_projected, 0
fast, fast_logits = generate(model, prompt, steps, use_cache=True)
assert slow == fast and torch.allclose(slow_logits, fast_logits, atol=1e-12)
assert rows_slow == sum(len(prompt) + i for i in range(steps + 1))   # quadratic in length
assert model.rows_projected == len(prompt) + steps                   # linear in length
print(f"token rows projected: {rows_slow} without a cache, {model.rows_projected} with one")
```

## How big the cache gets
Each layer stores one key and one value vector per key/value head per token, so

\[ \text{KV bytes} = 2 \times L \times H_{kv} \times d_h \times N \times \text{bytes per number}, \]

per sequence, where \(H_{kv}\) is the number of key/value heads (\(H\) in standard multi-head attention). It grows with context length and batch size and competes with the weights for GPU memory:

| Configuration | Per token | 4,096 tokens | 8 sequences of 4,096 |
|---|---|---|---|
| 32 layers, 32 KV heads of 128 (Llama 2 7B) | 512 KiB | 2 GiB | 16 GiB |
| 32 layers, 8 KV heads of 128 (Llama 3 8B) | 128 KiB | 512 MiB | 4 GiB |

At eight concurrent 4k-token requests, the first model's cache outweighs its 13 GB of weights. Batch size and context length trade against each other through this number.

```python
def kv_bytes(layers, kv_heads, d_h, tokens, bytes_per=2):
    return 2 * layers * kv_heads * d_h * tokens * bytes_per


KiB, GiB = 2 ** 10, 2 ** 30
assert kv_bytes(32, 32, 128, 1) == 512 * KiB
assert kv_bytes(32, 32, 128, 4096) == 2 * GiB
assert kv_bytes(32, 32, 128, 8 * 4096) == 16 * GiB > 6.7e9 * 2      # cache outweighs fp16 weights
assert kv_bytes(32, 8, 128, 1) == 128 * KiB
assert kv_bytes(32, 8, 128, 8 * 4096) == 4 * GiB
```

The cache also fragments: requests end at unpredictable lengths, so reserving contiguous space for the maximum wastes most of it. PagedAttention (Kwon et al., the paper behind vLLM) stores the cache in fixed-size blocks addressed through a table, like virtual memory pages. Waste drops to the last partly filled block, and requests sharing a prefix can share its blocks. The paper reports 2 to 4 times the throughput of earlier serving systems at the same latency. More in [LLM inference](../llm-inference/).

## Sharing keys and values: MQA and GQA
Multi-head attention (MHA) gives each of the \(H\) query heads its own key and value head. Multi-query attention (MQA, Shazeer, 2019) has all query heads share one key/value head, cutting the cache by a factor of \(H\), sometimes at a cost in quality. Grouped-query attention (GQA, Ainslie et al., 2023) splits the query heads into \(G\) groups that each share one key/value head: MHA is \(G = H\), MQA is \(G = 1\). With \(H = 32\) and \(G = 8\), the cache is 4 times smaller than MHA's and 8 times larger than MQA's. The GQA paper converts an MHA checkpoint by averaging the key and value heads within each group, then continuing pretraining with about 5% of the original compute.

Only keys and values shrink; every query head keeps its own attention pattern. The check compares the layer above with PyTorch's built-in grouped attention for MHA, GQA and MQA.

```python
x_tokens = torch.tensor(prompt + fast[len(prompt):])
for n_kv in [8, 2, 1]:                                             # MHA, GQA with 4 heads per group, MQA
    m = TinyLM(n_kv)
    logits, (k, v) = m(x_tokens)
    assert k.shape == (n_kv, len(x_tokens), d_head)                # the cache shrinks with n_kv
    x = m.emb(x_tokens)
    q = m.wq(x).view(-1, n_heads, d_head).transpose(0, 1)
    ref = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True, enable_gqa=True)[0]
    ref_logits = m.out(x + m.wo(ref.transpose(0, 1).reshape(len(x_tokens), -1)))
    assert torch.allclose(logits, ref_logits, atol=1e-10)
```

## Speculative decoding
Since a decode step is limited by reading the weights, checking several tokens in one pass costs about as much as producing one. Speculative decoding (Leviathan et al., 2023, and independently Chen et al., 2023) exploits this. A small draft model proposes \(k\) tokens cheaply; the target model scores all \(k\) positions in one forward pass, giving its distribution \(p\) at each position where the draft had \(q\). Left to right, each draft token \(x\) is accepted with probability \(\min(1, p(x)/q(x))\); at the first rejection, a replacement is drawn from the leftover distribution

\[ p'(x) = \frac{\max\big(0,\; p(x) - q(x)\big)}{\sum_y \max\big(0,\; p(y) - q(y)\big)}, \]

and the round ends. If all \(k\) are accepted, the target's distribution at position \(k + 1\) gives a bonus token.

The output is distributed exactly as if the target had sampled alone. Let \(\beta = \sum_x \min(p(x), q(x))\) be the probability that a draft token is accepted. Token \(x\) comes out by being drafted and accepted, with probability \(q(x)\min(1, p(x)/q(x)) = \min(p(x), q(x))\), or by a rejection and a draw from \(p'\), with probability \((1 - \beta)\,p'(x)\). The normalizer of \(p'\) is \(\sum_y (p(y) - \min(p(y), q(y))) = 1 - \beta\), so the second term is \(p(x) - \min(p(x), q(x))\) and the two add to \(p(x)\). The draft's quality affects only speed. If each token is accepted independently with probability \(\alpha\), a round produces \((1 - \alpha^{k+1})/(1 - \alpha)\) tokens per target pass on average: 2.5 tokens at \(\alpha = 0.7, k = 3\).

The experiment checks one step against \(p\), the tokens-per-pass formula, and the full algorithm on toy bigram models against the target's exact distribution of three-token continuations. Sampling noise alone puts the total variation distance near 0.03 at this sample size, so the check also runs a broken variant, one that redraws a rejected token from \(p\) instead of the leftover \(p'\), to show the test can tell the difference.

```python
rng = np.random.default_rng(0)
V = 6
p = rng.dirichlet(np.ones(V))                                      # target
q = 0.6 * p + 0.4 * rng.dirichlet(np.ones(V))                      # a decent but imperfect draft


def sample(dist, u):                                               # inverse-CDF sampling, vectorized
    return np.minimum(np.searchsorted(np.cumsum(dist), u), len(dist) - 1)


n = 400_000
drafted = sample(q, rng.random(n))
accepted = rng.random(n) < np.minimum(1, p[drafted] / q[drafted])
residual = np.maximum(p - q, 0)
out = np.where(accepted, drafted, sample(residual / residual.sum(), rng.random(n)))
assert np.abs(np.bincount(out, minlength=V) / n - p).max() < 0.003           # exactly p
assert abs(accepted.mean() - np.minimum(p, q).sum()) < 0.003                 # acceptance = beta

alpha, k = 0.7, 3                                                  # tokens per round, i.i.d. acceptance
acc = rng.random((n, k)) < alpha
run = np.where(acc.all(1), k, acc.argmin(1))                       # accepted before first rejection
assert abs((run + 1).mean() - (1 - alpha ** (k + 1)) / (1 - alpha)) < 0.01   # +1: fix-up or bonus

P = rng.dirichlet(np.ones(V), size=V)                              # target bigram: P[prev] -> next
Q = 0.5 * P + 0.5 * rng.dirichlet(np.ones(V), size=V)              # draft bigram


def speculative(start, n_new, k, rng, fix_up="leftover"):
    seq, passes = [start], 0
    while len(seq) < n_new + 1:
        draft = []
        for _ in range(k):                                         # cheap: k draft steps
            draft.append(sample(Q[(seq + draft)[-1]], rng.random()))
        passes += 1                                                # one target pass scores all k + 1
        for x in draft:
            p_i, q_i = P[seq[-1]], Q[seq[-1]]                      # both conditioned on the same prefix
            if rng.random() < min(1, p_i[x] / q_i[x]):
                seq.append(x)
                continue
            r = np.maximum(p_i - q_i, 0) if fix_up == "leftover" else p_i
            seq.append(sample(r / r.sum(), rng.random()))
            break
        else:
            seq.append(sample(P[seq[-1]], rng.random()))           # all accepted: bonus token
    return seq[1:n_new + 1], passes


def continuation_tv(fix_up, runs=20_000):
    counts, passes = np.zeros((V, V, V)), 0
    for _ in range(runs):
        (a, b, c), used = speculative(0, 3, 3, rng, fix_up)
        counts[a, b, c] += 1
        passes += used
    exact = np.einsum("a,ab,bc->abc", P[0], P, P)                 # P(a|0) P(b|a) P(c|b)
    return 0.5 * np.abs(counts / runs - exact).sum(), passes / runs  # total variation distance


tv, per_run = continuation_tv("leftover")
tv_bad, _ = continuation_tv("target")                              # wrong fix-up: redraw from p
assert tv < 0.06 < 0.1 < tv_bad                                    # noise ~0.03; the bug ~0.15
assert per_run < 2                                                 # 3 tokens in under 2 passes
print(f"TV {tv:.3f} (broken fix-up: {tv_bad:.3f}); 3 tokens in {per_run:.2f} target passes")
```

With the leftover fix-up the distance stays under 0.06, a bound set at twice the sampling noise; the broken variant lands above 0.1, and three tokens cost fewer than two target passes.

## Attention without the N-by-N matrix
In training and prefill, standard attention writes each head's \(N \times N\) score matrix to GPU memory and reads it back, more than once. FlashAttention (Dao et al., 2022) computes the same output without storing it. It streams blocks of keys and values through fast on-chip memory and keeps, per query, a running maximum, softmax denominator and weighted sum of values, rescaling the old sums by \(e^{m_{\text{old}} - m_{\text{new}}}\) when the maximum grows. That online softmax predates it (Milakov and Gimelshein, 2018); FlashAttention's contribution was organizing the computation around memory traffic, and FlashAttention-2 (Dao, 2023) split the work better across the GPU. The result is exact, with extra memory linear in \(N\). Details in [FlashAttention](../flashattention/).

```python
def blockwise_attention(q, k, v, block=16):
    m = torch.full((q.shape[0], 1), float("-inf"))                 # running max per query
    l = torch.zeros(q.shape[0], 1)                                 # running softmax denominator
    acc = torch.zeros(q.shape[0], v.shape[1])                      # running weighted sum of values
    for j in range(0, k.shape[0], block):
        s = q @ k[j:j + block].T / q.shape[1] ** 0.5
        m_new = torch.maximum(m, s.max(1, keepdim=True).values)
        scale = torch.exp(m - m_new)                               # rescale what was summed so far
        e = torch.exp(s - m_new)
        l, acc, m = l * scale + e.sum(1, keepdim=True), acc * scale + e @ v[j:j + block], m_new
    return acc / l


qa, ka, va = torch.randn(100, 32), torch.randn(237, 32) * 4, torch.randn(237, 32)
full = torch.softmax(qa @ ka.T / 32 ** 0.5, dim=1) @ va             # materializes 100 x 237
assert torch.allclose(blockwise_attention(qa, ka, va), full, atol=1e-12)
```

## Mixed precision training
Training in 32-bit floats everywhere wastes memory and leaves fast 16-bit matrix units idle. Mixed precision training (Micikevicius et al., 2018) runs forward and backward matrix multiplications in 16 bits and keeps a 32-bit master copy of the weights for the optimizer. The two 16-bit formats fail differently:

| | Exponent bits | Mantissa bits | Largest value | Gap above 1.0 |
|---|---|---|---|---|
| float16 | 5 | 10 | 65,504 | \(2^{-10} \approx 10^{-3}\) |
| bfloat16 | 8 | 7 | \(\approx 3.4 \times 10^{38}\) | \(2^{-7} \approx 8 \times 10^{-3}\) |

float16 has little range: small gradients round to zero and large activations overflow. Loss scaling multiplies the loss by a large factor before the backward pass and divides the gradients by it before the update (PyTorch's `torch.amp.GradScaler` picks the factor and skips steps that overflow). bfloat16 has float32's range and needs no scaling, at the cost of precision. Neither format is a safe place to accumulate the weights. An update is often a thousandth of the weight or less, and any update smaller than half the gap between neighbouring values (about \(2^{-11}\) of the weight in float16, \(2^{-8}\) in bfloat16) rounds away, so adding it changes nothing. That is swamping, not underflow, and it is why master weights stay in 32 bits.

```python
f16, bf16 = torch.finfo(torch.float16), torch.finfo(torch.bfloat16)
assert f16.max == 65504 and bf16.max > 3.3e38
assert f16.eps == 2 ** -10 and bf16.eps == 2 ** -7
assert torch.tensor(70_000.0).half().isinf()                       # fp16 overflow
g = torch.tensor(1e-8)                                             # a small gradient
assert g.half() == 0 and g.bfloat16() != 0                         # fp16 underflow, bf16 fine
scale = 2.0 ** 16                                                  # loss scaling rescues it
assert abs((g * scale).half().double() / scale - 1e-8) < 1e-11

w16, w32 = torch.tensor(1.0, dtype=torch.bfloat16), torch.tensor(1.0, dtype=torch.float32)
for _ in range(1000):                                              # 1000 updates of 1e-3
    w16, w32 = w16 + 1e-3, w32 + 1e-3
assert w16 == 1.0 and abs(w32.item() - 2.0) < 1e-3                 # bf16 weight never moved

torch.manual_seed(0)                                               # bf16 autocast, fp32 master weights
X = torch.randn(512, 20, dtype=torch.float32)
y = X @ torch.randn(20, 1, dtype=torch.float32)
lin = torch.nn.Linear(20, 1, dtype=torch.float32)
opt = torch.optim.SGD(lin.parameters(), lr=0.1)
for step in range(200):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred = lin(X)                                              # matmul runs in bf16
    loss = ((pred.float() - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
assert pred.dtype == torch.bfloat16 and lin.weight.dtype == torch.float32
assert loss.item() < 1e-3
```

## Instruction tuning and cheap fine-tuning
A pretrained model continues text; instruction tuning fine-tunes it on prompt and response pairs. FLAN (Wei et al., 2022) showed that training on many tasks phrased as instructions improves zero-shot performance on unseen tasks; InstructGPT (Ouyang et al., 2022) added a reward model and reinforcement learning from human preferences (see [RLHF and alignment](../rlhf-alignment/)). One detail matters: in the usual setup the loss covers response tokens only. Prompt positions get the label \(-100\), which PyTorch's cross-entropy ignores.

```python
T_prompt, T_resp = 7, 5
logits = torch.randn(T_prompt + T_resp, vocab)
targets = torch.randint(vocab, (T_prompt + T_resp,))
labels = targets.clone()
labels[:T_prompt] = -100                                           # no loss on the prompt
masked = F.cross_entropy(logits, labels, ignore_index=-100)
assert torch.allclose(masked, F.cross_entropy(logits[T_prompt:], targets[T_prompt:]))
assert not torch.allclose(masked, F.cross_entropy(logits, targets))
```

(In a real model the labels are also shifted by one position, so that position \(i\) predicts token \(i + 1\).)

Full fine-tuning leaves one full checkpoint per task. LoRA (Hu et al., 2021) freezes the pretrained weight \(W_0\) and learns a low-rank update, \(h = W_0 x + \frac{\alpha}{r} B A x\), with \(B\) initialized to zero so training starts at the pretrained model; afterwards \(\frac{\alpha}{r} BA\) is added into \(W_0\) at no inference cost. Rank 8 on a 7B model's query and value projections trains about 4 million parameters. A full build, with asserts, is in [LoRA from scratch](/posts/lora-from-scratch/), and the wider family in [parameter-efficient fine-tuning](../parameter-efficient-fine-tuning/).

## Long contexts are not read evenly
A longer window does not mean the model uses all of it equally. Liu et al. (2023) moved one relevant document among distractors and found a U-shaped curve: accuracy is highest when it sits near the start or end of the context and drops in the middle, and longer-window models were no better at this. For retrieval-augmented generation, put the strongest passages first or last and prefer fewer, better chunks (see [RAG architecture](../rag-architecture/)).

## Pitfalls
- **Treating the KV cache as free.** Its memory grows with every token of every request and often limits batch size before compute does.
- **Thinking speculative decoding changes outputs.** With the rule above it does not; a poor draft only slows it down. Looser rules, such as accepting any token the target ranks highly, do change the distribution.
- **Expecting speculative decoding to help a busy server.** It spends idle compute. At large batch sizes decoding is no longer memory-bound and the gain shrinks.
- **Using float16 without loss scaling.** Gradients below about \(3 \times 10^{-8}\) become zero. Prefer bfloat16 where supported.
- **Forgetting the prompt mask.** The model then also learns to imitate users, diluting the signal on responses.

The experiments run on toy models: one attention layer with 8 heads, bigram models over a six-token vocabulary, and a linear fit for mixed precision. They show that a method is exact or how its memory scales, and say nothing about speed on a GPU; the throughput figures are arithmetic or come from the cited papers. Whether the U-shaped curve of Liu et al. (2023) still holds for newer long-window models is not tested here. For the basic building blocks see [why attention divides by the square root of d](../attention-scaling/) and [BatchNorm vs LayerNorm](../batchnorm-vs-layernorm/).

## References
1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
2. Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019. Multi-query attention.
3. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), EMNLP 2023.
4. Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
5. Leviathan, Kalman and Matias, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
6. Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023.
7. Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018.
8. Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022, and Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023.
9. Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740), ICLR 2018.
10. Wei et al., [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652), ICLR 2022.
11. Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155), NeurIPS 2022.
12. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
13. Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), TACL 2024.

These topics follow Stanford's CME295 course, Transformers and Large Language Models; the explanations and experiments here are my own.
