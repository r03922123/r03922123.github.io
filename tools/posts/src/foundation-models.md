---
title: Foundation models: what a pretrained LLM actually is
summary: A large language model is a next-token predictor trained once on broad text and then adapted to many tasks. The objective, the causal mask, where the parameters live, how compute is split between model size and data, the KV cache and sampling, each checked in code, then how a base model becomes an assistant.
group: LLMs and generative AI
order: 1
tags: LLM · transformers · pretraining · scaling laws
---
A foundation model is a large network trained once, at great expense, on a broad pile of data with a self-supervised objective, and then reused for many tasks it was never trained on directly. The term comes from a 2021 Stanford report by Bommasani et al. It covers image and multimodal models too, but the best-known examples are the GPT, Llama, Gemini and Qwen families of large language models. Retrieval, fine-tuning and agents are all ways of using one of these, so it pays to know what the thing underneath is.

The short answer: a decoder-only transformer that, given some text, outputs a probability for every possible next token. This post builds that up piece by piece, from the objective to a chat assistant.

## The objective: predict the next token
Text is first cut into tokens (sub-word pieces from a fixed vocabulary of tens of thousands to a few hundred thousand entries). The chain rule of probability then factorizes any sequence exactly, with no approximation:

\[ p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}) . \]

So a model that predicts the next token well is a model of the whole distribution of text. Training minimizes the average negative log-probability of each actual next token, which is ordinary cross-entropy over the vocabulary (see [cross-entropy vs MSE](../cross-entropy-vs-mse/) and [entropy and information](../entropy-and-information/)). Its exponential is **perplexity**: a model that guesses uniformly over \(V\) tokens has perplexity exactly \(V\), and a good model behaves as if it were choosing among far fewer options.

The smallest possible language model makes the idea concrete: a character bigram model, where the "network" is a table of counts. It is scored here on its own training text, only to show the bookkeeping.

```python
import math
import numpy as np
import torch
import torch.nn.functional as F

text = "the cat sat on the mat. the dog sat on the log. the cat saw the dog."
vocab = sorted(set(text))
stoi = {c: i for i, c in enumerate(vocab)}
ids = [stoi[c] for c in text]
V = len(vocab)

counts = np.ones((V, V))                      # add-one smoothing: no zero probabilities
for a, b in zip(ids, ids[1:]):
    counts[a, b] += 1
P = counts / counts.sum(1, keepdims=True)    # P[a, b] = p(next = b | current = a)


def nll(next_probs, seq):
    """Sum of -log p(x_t | x_<t) over the sequence: the chain rule, in log space."""
    return -sum(math.log(next_probs(seq[:t])[seq[t]]) for t in range(1, len(seq)))


bigram = lambda prefix: P[prefix[-1]]
uniform = lambda prefix: np.full(V, 1 / V)

n = len(ids) - 1
ppl_uniform = math.exp(nll(uniform, ids) / n)
ppl_bigram = math.exp(nll(bigram, ids) / n)
assert abs(ppl_uniform - V) < 1e-9            # uniform guessing: perplexity = vocabulary size
assert ppl_bigram < ppl_uniform / 2

logits = torch.log(torch.tensor(P[ids[:-1]]))   # the same loss as a framework computes it
ce = F.cross_entropy(logits, torch.tensor(ids[1:]))
assert abs(ce.item() - nll(bigram, ids) / n) < 1e-9
print(f"vocabulary {V}, perplexity: uniform {ppl_uniform:.1f}, bigram {ppl_bigram:.1f}")
```

An LLM replaces the count table with a transformer that conditions on the whole prefix, not just the last character. The loss is the same line of code.

## The causal mask: one pass, many predictions
A transformer layer lets every position look at every other position through attention. For next-token training that would be cheating: position \(t\) could read token \(t+1\), the answer it is meant to predict. The fix is a **causal mask** that sets the attention scores to future positions to \(-\infty\) before the softmax, so their weights are exactly zero.

The mask also buys efficiency: one forward pass over \(T\) tokens yields \(T\) valid next-token predictions, all trained in parallel, each conditioned on the true prefix (teacher forcing).

```python
torch.manual_seed(0)
T, d = 6, 16
Wq, Wk, Wv = (torch.randn(d, d, dtype=torch.float64) / d**0.5 for _ in range(3))


def self_attention(x, causal=True):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.T / d**0.5                   # why sqrt(d): see the attention-scaling post
    if causal:
        future = torch.triu(torch.ones(len(x), len(x), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


x = torch.randn(T, d, dtype=torch.float64)
x2 = x.clone()
x2[4:] = torch.randn(2, d, dtype=torch.float64)      # rewrite the last two tokens

assert torch.allclose(self_attention(x)[:4], self_attention(x2)[:4])       # earlier outputs unchanged
assert not torch.allclose(self_attention(x, causal=False)[:4],
                          self_attention(x2, causal=False)[:4])            # without the mask, the future leaks
```

The \(1/\sqrt{d}\) factor has its own post: [why attention divides by the square root of d](../attention-scaling/).

## Three shapes of transformer
The original transformer (Vaswani et al., 2017) had an encoder and a decoder. Language models since then have used one half, the other, or both:

| Shape | Attention | Pretraining objective | Examples | Typical use |
|---|---|---|---|---|
| Encoder-only | bidirectional | fill in masked tokens | BERT, RoBERTa | embeddings, classification, retrieval |
| Decoder-only | causal | next token | GPT, Llama, Qwen | generation, chat, general-purpose |
| Encoder-decoder | bidirectional encoder, causal decoder with cross-attention | reconstruct corrupted spans | T5, BART | translation, summarization |

Encoder-decoders are not needed just because input and output lengths differ (a decoder-only model handles that fine); their advantage is that the input is read bidirectionally, once. Decoder-only models won the general-purpose role because one objective uses every token of raw text and every task can be phrased as "continue this text". Encoder-only models remain common for embeddings, as in [two-tower retrieval](../two-tower-retrieval/).

Inside the block, most current LLMs normalize (LayerNorm or RMSNorm) at the start of each residual branch ("pre-LN") rather than after the residual addition as in 2017; Xiong et al. (2020) showed this keeps gradients well-behaved at initialization. More in [BatchNorm vs LayerNorm](../batchnorm-vs-layernorm/).

## Where the parameters live
A GPT-style block has an attention part (query, key, value and output projections: \(4d^2\) weights) and an MLP that widens to \(4d\) and back (\(8d^2\)). So each layer holds about \(12d^2\) parameters, and a model with \(L\) layers about \(12Ld^2\) plus the embedding table. The count below reproduces the true size of GPT-2 small (124M; the GPT-2 paper's table listed it as 117M), whose output layer reuses the input embedding matrix (weight tying), and checks the formula against modules PyTorch actually builds.

```python
def gpt2_params(vocab, ctx, d, L):
    emb = vocab * d + ctx * d                             # token and position embeddings
    attn = (d * 3 * d + 3 * d) + (d * d + d)              # fused q,k,v projection + output projection
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)           # up to 4d, back down to d
    norms = 2 * 2 * d                                     # two LayerNorms, gain and bias each
    return emb + L * (attn + mlp + norms) + 2 * d         # + final LayerNorm; output head is tied


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.qkv, self.proj = torch.nn.Linear(d, 3 * d), torch.nn.Linear(d, d)
        self.up, self.down = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)


class GPTShape(torch.nn.Module):
    def __init__(self, vocab, ctx, d, L):
        super().__init__()
        self.tok, self.pos = torch.nn.Embedding(vocab, d), torch.nn.Embedding(ctx, d)
        self.blocks = torch.nn.ModuleList(Block(d) for _ in range(L))
        self.ln_f = torch.nn.LayerNorm(d)


for cfg in [(100, 32, 16, 2), (500, 64, 48, 3)]:
    assert sum(p.numel() for p in GPTShape(*cfg).parameters()) == gpt2_params(*cfg)

total = gpt2_params(50257, 1024, 768, 12)                 # GPT-2 small
assert total == 124_439_808
body = total - (50257 + 1024) * 768
assert abs(body / (12 * 12 * 768**2) - 1) < 0.01          # non-embedding part ~ 12 L d^2
print(f"GPT-2 small: {total:,} parameters, {1 - body / total:.0%} of them in embeddings")
```

At this size almost a third of the model is the embedding table. At 7B parameters, with \(d = 4096\) and 32 layers, the \(12Ld^2\) term is about 6.4B and the embeddings are a small fraction, which is why scaling studies count non-embedding parameters.

## How big a model, on how much data
Training a dense transformer with \(N\) parameters on \(D\) tokens costs about \(C \approx 6ND\) floating-point operations: roughly 2 per parameter per token for the forward pass and 4 for the backward pass. Given a compute budget, how should it be split between a bigger model and more data?

Kaplan et al. (2020) found loss falls as a smooth power law in \(N\), \(D\) and \(C\), and recommended spending most extra compute on model size. Hoffmann et al. (2022), the Chinchilla paper, redid the experiments with learning-rate schedules matched to each run's length and reached a different answer: the compute-optimal \(N\) and \(D\) both grow as about \(C^{0.5}\), which works out to roughly 20 training tokens per parameter. Their 70B Chinchilla, trained on 1.4T tokens, beat the 280B Gopher, which had used the same compute on 300B tokens.

```python
def compute_optimal(C, tokens_per_param=20):
    """Solve C = 6 N D with D = r N."""
    N = math.sqrt(C / (6 * tokens_per_param))
    return N, tokens_per_param * N


C = 6 * 70e9 * 1.4e12                                     # Chinchilla's budget, ~5.9e23 FLOPs
N, D = compute_optimal(C)
assert abs(N / 70e9 - 1) < 0.01 and abs(D / 1.4e12 - 1) < 0.01

N4, D4 = compute_optimal(4 * C)
assert abs(N4 / N - 2) < 1e-9 and abs(D4 / D - 2) < 1e-9  # 4x compute: 2x params and 2x tokens

assert 300e9 / 175e9 < 2           # GPT-3: 175B params, 300B tokens, far below 20 tokens/param
assert 15e12 / 8e9 > 90 * 20       # an 8B model on 15T tokens: ~94x past the compute-optimal ratio
```

The last line is not a mistake. "Compute-optimal" minimizes training cost, but a model is trained once and served millions of times, and serving cost scales with \(N\). Meta trained the 8B Llama 3 on over 15 trillion tokens deliberately: a smaller model trained longer is cheaper to run at the same quality.

## Generating text and the KV cache
Generation is a loop: run the model on the prefix, pick a token, append it, repeat. Done naively, step \(t\) recomputes the keys and values of all \(t\) earlier tokens, even though, thanks to the causal mask, none of them have changed. The **KV cache** stores each token's keys and values once; each new step computes only the new token's query, key and value and attends over the cache.

```python
class KVCache:
    def __init__(self):
        self.k, self.v = [], []

    def step(self, x_t):                                   # x_t: one new token's hidden state
        self.k.append(x_t @ Wk)
        self.v.append(x_t @ Wv)
        K, Vals = torch.stack(self.k), torch.stack(self.v)
        w = torch.softmax(K @ (x_t @ Wq) / d**0.5, dim=0)
        return w @ Vals


seq, cache = torch.randn(10, d, dtype=torch.float64), KVCache()
for t in range(10):
    assert torch.allclose(cache.step(seq[t]), self_attention(seq[: t + 1])[-1])


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2):    # fp16
    return 2 * layers * kv_heads * head_dim * bytes_per_value             # 2: keys and values


mha = kv_bytes_per_token(32, 32, 128)     # shaped like Llama 2 7B
gqa = kv_bytes_per_token(32, 8, 128)      # same, with 8 shared key/value heads (as in Llama 3 8B)
assert mha == 2**19                       # 0.5 MiB per token
assert mha * 4096 == 2 * 2**30            # 2 GiB for a single 4096-token sequence
assert mha // gqa == 4
```

The cache trades compute for memory, and at long contexts that memory, not the weights, limits how many requests fit on a GPU. Grouped-query attention (Ainslie et al., 2023) lets several query heads share one key/value head, shrinking the cache by that factor. Context windows have grown from 2,048 tokens for GPT-3 to 128K for Llama 3.1, but attention compute still grows as \(n^2\). FlashAttention (Dao et al., 2022) does not change that count: it computes exact attention in tiles without storing the \(n \times n\) score matrix, so extra memory is linear in \(n\) and far less data moves through GPU memory. See [FlashAttention](../flashattention/) and [LLM inference](../llm-inference/).

## Picking the next token
The model outputs a distribution; a decoding rule turns it into one token. Greedy decoding takes the most likely token and tends to loop and repeat. Sampling is usually shaped by two knobs. **Temperature** divides the logits by \(\tau\) before the softmax: \(\tau < 1\) sharpens towards greedy, \(\tau > 1\) flattens towards uniform. **Top-p** or nucleus sampling (Holtzman et al., 2019) keeps the smallest set of most-likely tokens whose probabilities add up to at least \(p\), renormalizes, and samples from that, which cuts off the long tail of unlikely tokens.

```python
def decode_dist(logits, temperature=1.0, top_p=1.0):
    z = np.asarray(logits, dtype=float) / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p)
    keep = order[: np.searchsorted(np.cumsum(p[order]), top_p) + 1]   # smallest prefix with mass >= top_p
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()


logits = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
base = decode_dist(logits)
assert np.allclose(base, np.exp(logits) / np.exp(logits).sum())
assert decode_dist(logits, temperature=0.05)[0] > 0.999            # cold: effectively greedy
flat = decode_dist(logits, temperature=100)
assert flat.max() - flat.min() < 0.05                              # hot: nearly uniform

nucleus = decode_dist(logits, top_p=0.9)
k = int((nucleus > 0).sum())
ranked = np.sort(base)[::-1]
assert ranked[:k].sum() >= 0.9 > ranked[: k - 1].sum()             # smallest set reaching 0.9
draws = np.random.default_rng(0).choice(len(logits), size=200_000, p=nucleus)
assert np.allclose(np.bincount(draws, minlength=len(logits)) / 200_000, nucleus, atol=0.005)
```

## From base model to assistant
A pretrained base model continues text; it does not follow instructions. Ask it a question and it may answer with three more questions, because that is what lists of questions look like on the web. Two further stages make it an assistant:

1. **Supervised fine-tuning (SFT):** train on curated examples of instructions and good responses, with the same next-token loss, usually applied only to the response tokens.
2. **Preference tuning:** collect human (or model) judgements of which of two responses is better, then push the model towards preferred ones, with a reward model and reinforcement learning as in InstructGPT (Ouyang et al., 2022), or directly with DPO (Rafailov et al., 2023). See [RLHF and alignment](../rlhf-alignment/).

Much of what makes LLMs useful needs no weight updates. GPT-3 (Brown et al., 2020) showed **in-context learning**: put a few input-output examples in the prompt and the model continues the pattern. Chain-of-thought prompting adds worked reasoning to those examples and markedly helps multi-step problems at scale (Wei et al., 2022); appending "Let's think step by step" with no examples also helps (Kojima et al., 2022). Why in-context learning works is still studied; the safe statement is that the prompt conditions the next-token distribution, and a large model has learned to infer the task from it.

## Which way to adapt one
| Need | Reach for | Why |
|---|---|---|
| A task the model can already do | prompting, a few examples, structured output | no training, change it in minutes |
| Facts that change, or private documents | [retrieval-augmented generation](../rag-architecture/) | knowledge stays in an index you can update and cite |
| A consistent format, style or narrow skill | fine-tuning, usually [LoRA](../lora-explained/) | cheap, small adapter files, base weights untouched |
| Lower latency or cost at fixed quality | a smaller model, quantization, distillation | serving cost scales with parameters |

Pitfalls worth keeping in mind:

- **Perplexity is not quality.** Lower loss on held-out text predicts a better base model, but says little about helpfulness or factuality after tuning; evaluate on the task (see [evaluating ML systems](/posts/evaluating-ml-systems/)).
- **Tokens are not words.** Context limits, pricing and per-token latency are all in tokens, and the same text can take very different numbers of tokens in different languages or tokenizers.
- **Fine-tuning is a poor way to add facts.** It changes behaviour readily but stores new knowledge unreliably; retrieval is usually the better tool.
- **Plan memory for the cache.** A model whose weights fit on a GPU can still run out of memory serving long contexts or large batches.

To see these pieces trained end to end, [LoRA from scratch](/posts/lora-from-scratch/) builds a tiny GPT and fine-tunes it. This post is a rewrite of study notes drawn from Andrej Karpathy's Neural Networks: Zero to Hero course, Lilian Weng's [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) and Stanford's CME 295, Transformers and Large Language Models, together with the papers named in the text.
