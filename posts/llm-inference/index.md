# LLM inference: where the time and memory go

In a simple cost model of a 7B model on a GPU with roughly an A100's figures, decoding one sequence takes 7 ms per token and leaves more than 99% of the arithmetic idle. A batch of 64 raises throughput from 143 to 9,143 tokens per second, until each sequence's 2,000-token cache must be read as well, which cuts it to 1,578 with 32 key/value heads. Training a language model is a one-off bill; serving it is a running one, measured in how long a user waits for the first word, how fast the rest arrives, and how many users one GPU carries at once. I build that cost model of a decoding step and use it to explain the standard serving techniques. Most of them answer one question: how to get more tokens out of each trip through GPU memory.

The mechanics of the KV cache, grouped-query attention and the proof that speculative decoding is exact are in [Transformers and LLMs: the key engineering ideas](../transformers-and-llms-key-ideas/). Here the focus is on what happens when many requests share a machine.

## Two phases with different bottlenecks
A request goes through two phases. **Prefill** runs the whole prompt through the model in one parallel pass and fills the KV cache. **Decode** then produces one token per forward pass, each pass reading the cache and appending to it.

The useful lens is arithmetic intensity: floating-point operations per byte moved from memory. A dense model with \(P\) parameters does about \(2P\) operations per token (one multiply and one add per weight), and every forward pass must read all \(P\) weights. In 16-bit precision that is \(2P\) bytes. A pass over \(n\) tokens therefore has intensity of roughly

\[ \frac{2Pn}{2P + \text{KV bytes read}} \;\approx\; n \ \text{FLOP per byte (ignoring the cache)} . \]

A GPU has a ridge point, the ratio of its peak compute to its memory bandwidth (the roofline model of Williams, Waterman and Patterson). Below that intensity a kernel waits on memory; above it, on arithmetic. For a GPU with 312 TFLOP/s of 16-bit tensor throughput and 2 TB/s of bandwidth, roughly an A100, the ridge is about 156 FLOP per byte.

So a 2,000-token prefill (intensity about 2,000) is compute-bound, while decoding one sequence (intensity about 1) leaves more than 99% of the arithmetic idle. The cost model below takes the step time as the larger of compute time and memory time, which is optimistic but captures the shape:

```python
import numpy as np

PEAK_FLOPS = 312e12                     # 16-bit dense tensor throughput, roughly an A100
BANDWIDTH = 2.0e12                      # bytes per second from HBM
RIDGE = PEAK_FLOPS / BANDWIDTH          # FLOP per byte where the bottleneck switches
GB = 1e9


def kv_bytes_per_token(layers, kv_heads, d_head, bytes_per=2):
    return 2 * layers * kv_heads * d_head * bytes_per          # keys and values, every layer


def step(params, new_tokens, cached_tokens=0, kv_per_token=0, bytes_per_param=2):
    """Time of one forward pass and whether compute or memory limits it."""
    flops = 2 * params * new_tokens
    moved = params * bytes_per_param + cached_tokens * kv_per_token
    t = max(flops / PEAK_FLOPS, moved / BANDWIDTH)
    return t, ("compute" if flops / moved > RIDGE else "memory")


P = 7e9                                                         # a 7B model, 32 layers, 128-dim heads
mha = kv_bytes_per_token(32, 32, 128)                           # 32 key/value heads
gqa = kv_bytes_per_token(32, 8, 128)                            # 8 key/value heads
assert 155 < RIDGE < 157 and mha == 524_288 and gqa == mha // 4

ttft, bound = step(P, 2000)                                     # prefill a 2,000-token prompt
assert bound == "compute" and abs(ttft - 0.090) < 0.001         # about 90 ms
tpot, bound = step(P, 1)                                        # decode one sequence
assert bound == "memory" and abs(tpot - 0.007) < 1e-6           # 7 ms: ~143 tokens/s, whatever the FLOPs


def tokens_per_s(batch, ctx, kv):
    t, bound = step(P, batch, batch * ctx, kv)
    return batch / t, bound, (2 * P + batch * ctx * kv) / GB


assert round(tokens_per_s(64, 0, mha)[0]) == 9143               # weights only: 64x the single stream
assert tokens_per_s(256, 0, mha)[1] == "compute"                # past the ridge at last
assert round(tokens_per_s(64, 2000, mha)[0]) == 1578            # but each sequence reads its own cache
assert round(tokens_per_s(64, 2000, gqa)[0]) == 4159
assert tokens_per_s(64, 2000, mha)[2] > 80 > tokens_per_s(64, 2000, gqa)[2]   # MHA does not even fit

table = {1: (143, 133, 140), 8: (1143, 715, 994), 32: (4571, 1346, 2859), 64: (9143, 1578, 4159)}
for b, row in table.items():                                    # every number in the table below
    assert tuple(round(tokens_per_s(b, c, kv)[0]) for c, kv in ((0, mha), (2000, mha), (2000, gqa))) == row
```

The model gives a time to first token of about 90 ms and a decode step of 7 ms at batch size one. The table shows decode throughput as the batch grows, with each sequence holding 2,000 tokens of context:

| Batch | Weights only | With 2k-token cache, 32 KV heads | With 2k-token cache, 8 KV heads |
|---|---|---|---|
| 1 | 143 tokens/s | 133 tokens/s | 140 tokens/s |
| 8 | 1,143 | 715 | 994 |
| 32 | 4,571 | 1,346 | 2,859 |
| 64 | 9,143 | 1,578 (needs 81 GB) | 4,159 (31 GB) |

Two lessons follow. **Batching is nearly free** while decode is memory-bound: one read of the weights serves every sequence in the batch. But **the cache does not amortize**: every sequence reads its own keys and values on every step, so at long contexts cache traffic, not weights, sets the step time. Shrinking the cache (grouped-query attention, cache quantization) therefore pays twice: more sequences fit, and each step moves fewer bytes.

The first lesson is visible even on a laptop CPU. Multiplying a 64 MB weight matrix by 64 vectors at once costs far less than 64 times one vector, because the matrix is read once either way:

```python
import time
import torch

torch.manual_seed(0)
W = torch.randn(4096, 4096)                                     # 64 MB of float32 weights


def best_time(batch, repeats=10):
    x = torch.randn(4096, batch)
    W @ x                                                       # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        W @ x
        times.append(time.perf_counter() - t0)
    return min(times)


assert best_time(64) < 16 * best_time(1)                        # 64x the work, well under 64x the time
```

## Latency, throughput, and the memory budget
A serving system is judged by **time to first token** (TTFT: queueing plus prefill), **time per output token** (TPOT, or inter-token latency: the decode step time) and **throughput** across all users. A response of \(n\) tokens takes about \(\text{TTFT} + (n - 1)\,\text{TPOT}\). Raising the batch size buys throughput with TPOT, and the latency target decides how far to go.

What caps the batch is memory. GPU memory holds the weights, the KV cache of every running sequence, and some workspace for activations. Take a 70B model with 80 layers and 8 key/value heads of size 128 (the shape of Llama 2 70B) on one 80 GB GPU. In 16-bit precision the weights alone are 140 GB. In 8-bit they are 70 GB, which fits but leaves room for only four 4,096-token conversations after a few GB of workspace. In 4-bit, with one 16-bit scale per group of 128 weights, they are about 36 GB, and around 40 GB is left for the cache:

```python
def fits(params, bits_per_weight, kv_per_token, ctx, gpu_gb=80, workspace_gb=4):
    weights = params * bits_per_weight / 8 / GB
    free = gpu_gb - workspace_gb - weights
    return weights, max(0, int(free * GB // (kv_per_token * ctx)))  # concurrent sequences of length ctx


kv70 = kv_bytes_per_token(80, 8, 128)
assert kv70 == 327_680                                          # 0.33 MB per token
assert fits(70e9, 16, kv70, 4096)[1] == 0                       # 140 GB of weights: does not fit
w8, n8 = fits(70e9, 8, kv70, 4096)
w4, n4 = fits(70e9, 4 + 16 / 128, kv70, 4096)                   # 4-bit weights plus group scales
assert (w8, n8) == (70.0, 4) and round(w4, 1) == 36.1 and n4 == 29

kv13 = kv_bytes_per_token(40, 40, 128)                          # a 13B model without grouped KV heads
assert 6.7e9 < kv13 * 8192 < 6.8e9                              # one 8k-token conversation: 6.7 GB
```

## Paged KV memory
Early serving systems reserved one contiguous slab of cache per request, sized for the maximum length, because a request's final length is unknown when it starts. Most requests stop far short of the maximum, so most of the slab is never written. The vLLM paper (Kwon et al., 2023) measured existing systems storing actual token states in only about 20 to 40 percent of their KV memory.

PagedAttention borrows the operating system's answer to the same problem. The cache is a pool of fixed-size blocks, say 16 tokens each. Each sequence owns a block table of block indices and gets a new block only when its last one fills. Blocks need not be adjacent; the attention kernel follows the table. Waste is limited to each sequence's last, partly filled block, and sequences that share a prefix, such as a system prompt, can point at the same blocks.

The sketch below stores keys and values in a shared pool, interleaves three sequences so their blocks are scattered, and checks that attention through the block table matches attention over contiguous arrays. Then it compares how many requests fit in the same memory under each scheme:

```python
class PagedKVCache:
    def __init__(self, n_blocks, block_size, d):
        self.k = np.zeros((n_blocks, block_size, d))
        self.v = np.zeros((n_blocks, block_size, d))
        self.bs = block_size
        self.free = list(range(n_blocks - 1, -1, -1))
        self.table, self.length = {}, {}

    def append(self, seq, k, v):
        n = self.length.get(seq, 0)
        if n % self.bs == 0:                                    # last block full: take any free one
            self.table.setdefault(seq, []).append(self.free.pop())
        block = self.table[seq][n // self.bs]
        self.k[block, n % self.bs], self.v[block, n % self.bs] = k, v
        self.length[seq] = n + 1

    def gather(self, seq):
        n, blocks = self.length[seq], self.table[seq]
        return self.k[blocks].reshape(-1, self.k.shape[-1])[:n], self.v[blocks].reshape(-1, self.v.shape[-1])[:n]

    def release(self, seq):
        self.free += self.table.pop(seq)
        del self.length[seq]


def attend(q, K, V):
    s = K @ q / np.sqrt(len(q))
    w = np.exp(s - s.max())
    return (w / w.sum()) @ V


rng = np.random.default_rng(0)
d, cache = 8, PagedKVCache(n_blocks=64, block_size=16, d=8)
truth = {s: ([], []) for s in "abc"}
for _ in range(300):                                            # three sequences decoding in turns
    s = "abc"[rng.integers(3)]
    k, v = rng.standard_normal((2, d))
    cache.append(s, k, v)
    truth[s][0].append(k)
    truth[s][1].append(v)
for s in "abc":
    K, V = cache.gather(s)
    assert np.array_equal(K, np.array(truth[s][0])) and np.array_equal(V, np.array(truth[s][1]))
    q = rng.standard_normal(d)
    assert np.allclose(attend(q, K, V), attend(q, *map(np.array, truth[s])))
    assert len(cache.table[s]) * 16 - cache.length[s] < 16     # waste: under one block per sequence
assert all(np.any(np.diff(cache.table[s]) != 1) for s in "abc")  # each sequence's blocks are scattered
cache.release("a")
assert len(cache.free) == 64 - len(cache.table["b"]) - len(cache.table["c"])

max_len, block = 2048, 16
lengths = np.minimum(rng.integers(50, 500, 10_000) + rng.geometric(1 / 200, 10_000), max_len)
budget = 122_000                                                # tokens of cache, from the 70B example
contiguous = budget // max_len                                  # every request reserves max_len
paged = int(budget // (np.ceil(lengths / block) * block).mean())
used = lengths.mean() / max_len
assert contiguous == 59 and 0.2 < used < 0.25                   # under a quarter of reserved memory used
assert paged > 4 * contiguous and lengths.mean() / (np.ceil(lengths / block) * block).mean() > 0.97
```

With these request lengths a contiguous reservation uses under a quarter of what it holds, and paging fits more than four times as many sequences in the same memory. More concurrent sequences means a larger decode batch, which, by the table above, is where the throughput comes from. The vLLM paper reports 2 to 4 times the throughput of earlier systems at similar latency.

## Continuous batching
The classic way to batch is static: collect \(B\) requests, run them together until the longest finishes, then take the next \(B\). Output lengths vary wildly, so most slots sit empty while one long answer finishes. Orca (Yu et al., 2022) introduced iteration-level scheduling, now usually called continuous batching: the scheduler acts between decode steps, and a finished sequence's slot is handed to a waiting request at the very next step.

The simulation below queues 2,000 requests with output lengths from a long-tailed distribution and charges one time unit per decode step for up to 16 sequences, which is what the memory-bound regime above implies. The continuous scheduler is simulated step by step and then cross-checked with a shortcut: each new request starts when the earliest-finishing slot frees up, which a [heap](/leetcode/heap/) gives directly.

```python
import heapq

lens = np.minimum(rng.geometric(1 / 200, 2000), 1000)            # output tokens per request
B = 16


def static_batching(lens, B):
    t, done = 0, np.zeros(len(lens))
    for s in range(0, len(lens), B):
        t += lens[s:s + B].max()                                # the batch waits for its longest member
        done[s:s + B] = t
    return t, done


def continuous_batching(lens, B):
    queue, slots, t, done, busy = list(range(len(lens)))[::-1], {}, 0, np.zeros(len(lens)), 0
    while queue or slots:
        while queue and len(slots) < B:                         # admit at every step
            slots[queue.pop()] = 0
        t += 1
        busy += len(slots)
        for r in list(slots):
            slots[r] += 1
            if slots[r] == lens[r]:
                done[r] = t
                del slots[r]
    return t, done, busy / (t * B)


def continuous_by_heap(lens, B):
    free_at, done = [0] * B, np.zeros(len(lens))
    for i, n in enumerate(lens):
        start = heapq.heappop(free_at)
        done[i] = start + n
        heapq.heappush(free_at, start + n)
    return done


t_static, d_static = static_batching(lens, B)
t_cont, d_cont, util = continuous_batching(lens, B)
assert np.array_equal(d_cont, continuous_by_heap(lens, B))      # step simulation = heap shortcut
assert lens.sum() / (t_static * B) < 0.35 and util > 0.98       # slot utilization
assert t_static > 3 * t_cont                                    # about 3.3x the throughput here
assert d_static.mean() > 3 * d_cont.mean()                      # and requests finish sooner
```

Static batching keeps the slots about 30% busy here; continuous batching keeps them about 99% busy and finishes the same work in under a third of the time. The simulation leaves out prefill. In a real server a newly admitted request's prefill must run too, and a long prompt processed in one piece stalls every running sequence's next token. Sarathi-Serve (Agrawal et al., 2024) splits prefills into chunks that ride along with decode steps; DistServe (Zhong et al., 2024) instead puts prefill and decode on separate GPUs and moves the cache between them.

## Quantization: fewer bytes per weight
If decode time is bytes read divided by bandwidth, fewer bits per weight make it faster directly and free memory for the cache. Weight-only schemes keep activations in 16 bits and dequantize weights inside the matrix multiply, so the speed-up is largest in the memory-bound regime and fades once a large batch makes the step compute-bound.

The basic recipe is round-to-nearest with a scale: divide a group of weights by \(s = \max|w| / (2^{b-1} - 1)\), round to integers, store the integers and \(s\). One large weight inflates \(s\) for everything that shares it, so smaller groups (often 128 weights) track the local range better at the cost of more scales.

Weights are not equally important. Dettmers et al. (2022) found that transformers beyond a few billion parameters develop a handful of hidden-state dimensions with very large magnitudes, and that these outlier features matter a great deal for quality. AWQ (Lin et al., 2024) observed that the weights multiplying the largest activations are the salient ones, and protects them without mixed precision: scale input channel \(j\) of the weight up by \(s_j\) before quantizing and divide the activation by \(s_j\) (folded into the previous layer), which leaves the exact product unchanged but gives salient weights a finer effective grid. It sets \(s_j = \overline{|x_j|}^{\,\alpha}\) from calibration activations and searches \(\alpha\) per layer. GPTQ (Frantar et al., 2023) takes a different route: it quantizes the columns of a weight matrix one at a time and adjusts the not-yet-quantized columns to compensate for the error, using second-order information from calibration inputs.

```python
def quantize(w, bits, group):
    """Symmetric round-to-nearest, one scale per `group` consecutive input weights in a row."""
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(w.shape[0], -1, group)
    scale = np.abs(g).max(axis=-1, keepdims=True) / qmax
    return (np.clip(np.round(g / scale), -qmax - 1, qmax) * scale).reshape(w.shape)


def rel_err(X, W, W_hat):
    return np.linalg.norm(X @ W_hat.T - X @ W.T) / np.linalg.norm(X @ W.T)


d_in = d_out = 1024
W = rng.normal(0, 0.02, (d_out, d_in))
W.flat[rng.choice(W.size, 20, replace=False)] *= 25              # a few large weights
X = rng.standard_normal((256, d_in))
err = {(b, g): rel_err(X, W, quantize(W, b, g)) for b in (8, 4) for g in (d_in, 128)}
assert err[8, 128] < 0.01 < err[8, d_in] < 0.02                  # 8-bit: about 1% output error
assert 0.1 < err[4, 128] < err[4, d_in] < 0.2                     # 4-bit: groups help

act = np.ones(d_in)
act[rng.choice(d_in, 10, replace=False)] = 30                    # ten outlier activation channels
X2 = rng.standard_normal((512, d_in)) * act
W2 = rng.normal(0, 0.02, (d_out, d_in))


def awq_err(alpha, bits=4, group=128):
    s = np.abs(X2).mean(axis=0) ** alpha
    s = s / np.sqrt(s.max() * s.min())                           # keep the scales centred around 1
    return rel_err(X2, W2, quantize(W2 * s, bits, group) / s)    # (x / s) @ (W s)^T is unchanged


alphas = np.linspace(0, 1, 11)
errs = [awq_err(a) for a in alphas]
best = alphas[int(np.argmin(errs))]
assert 0.1 < best < 0.6 and min(errs) < 0.6 * errs[0]            # about half the error of plain rounding
assert errs[-1] > errs[0]                                        # scaling too hard hurts the rest


def speedup(batch, bits):                                       # weight-only quantization, cost model above
    return step(P, batch)[0] / step(P, batch, bytes_per_param=bits / 8)[0]


assert 3.9 < speedup(1, 4) <= 4                                  # memory-bound: 4x fewer bytes, ~4x faster
assert speedup(128, 4) < 1.25                                    # near compute-bound: little left to gain
```

On these random weights (Gaussian, plus a few large ones), 8-bit rounding with groups of 128 costs under 1% of the layer's output and 4-bit about 12%; real checkpoints are evaluated on task quality, where good 4-bit methods usually lose little. In the outlier experiment, activation-aware scaling with \(\alpha\) around 0.3 cuts the 4-bit error almost in half, while \(\alpha = 1\) over-protects the salient channels and is worse than doing nothing.

## Speculative decoding, seen from the server
Speculative decoding (Leviathan et al., 2023, and independently Chen et al., 2023) is another way to spend idle arithmetic. A small draft model proposes a few tokens, and the large model checks all of them in one forward pass, which costs about the same as producing one token when decode is memory-bound. An acceptance rule based on rejection sampling makes the output distribution exactly the large model's, so only speed changes; the derivation and a simulation are in [Transformers and LLMs](../transformers-and-llms-key-ideas/). On a server two things limit the gain. The draft model's own weights and cache take memory that could have held more sequences, and at a large batch the verification pass is no longer free, so speculative decoding helps most for latency-sensitive traffic at small batch sizes.

## Splitting a model across GPUs
When weights do not fit on one device, the model is split. **Tensor parallelism** (as in Megatron-LM, Shoeybi et al., 2019) splits each layer. In an MLP, the first matrix is split by output features, so each GPU computes a slice of the hidden layer and applies the nonlinearity locally; the second is split by input features, so each GPU produces a partial output and one all-reduce sums them. Attention splits the same way by heads. Two all-reduces per layer per step keep tensor parallelism inside a node with fast GPU-to-GPU links; in return each GPU reads only its share of the weights, so decode steps get faster.

**Pipeline parallelism** puts different layers on different GPUs and passes activations from one stage to the next. Communication is small, so it works across nodes, but a single token still visits every stage in order: it adds capacity and, with several batches in flight, throughput, not lower latency per token.

```python
x = torch.randn(4, 64, dtype=torch.float64)
W1, W2 = torch.randn(256, 64, dtype=torch.float64), torch.randn(64, 256, dtype=torch.float64)
full = torch.nn.functional.gelu(x @ W1.T) @ W2.T

shards = 4
partials = [torch.nn.functional.gelu(x @ a.T) @ b.T                # each GPU: its rows of W1, columns of W2
            for a, b in zip(W1.chunk(shards, dim=0), W2.chunk(shards, dim=1))]
assert torch.allclose(sum(partials), full)                         # one all-reduce restores the output
assert all(p.shape == full.shape for p in partials)
```

## Pitfalls
- **Reading speedups at the wrong batch size.** Quantization and speculative decoding exploit idle arithmetic. At a large batch the arithmetic is no longer idle, so in the cost model above, 4-bit weights make a batch-one decode step 4 times faster for a 7B model but a batch-128 step only about 1.2 times faster.
- **Sizing for weights only.** The KV cache grows with batch size times context and can exceed the weights. Plan memory as weights plus cache plus workspace, at the longest context the service allows.
- **Averages hide tails.** One long prompt can stall everyone queued behind it; report percentiles of TTFT and TPOT, not means.
- **Trusting reconstruction error.** Layer error is a proxy. Check task metrics on the real quantized checkpoint, with calibration data close to production traffic.

Every number here comes from a cost model built on one GPU's peak figures or from small simulations: random weights, synthetic request lengths, one time unit per decode step. They show why each technique helps and where its gain fades, not what a given server will measure. The simulation also leaves out prefill, so how to schedule it alongside decode stays open here; Sarathi-Serve and DistServe answer that in different ways.

The overview of inference optimization on Lilian Weng's blog, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/), was a starting point for these notes; the explanations and simulations here are my own. For the model these techniques serve, see [Foundation models](../foundation-models/) and [FlashAttention](../flashattention/), which removes the memory cost of the attention matrix during prefill.

## References
1. Williams, Waterman and Patterson, [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785), Communications of the ACM, 2009.
2. Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
3. Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
4. Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024.
5. Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670), OSDI 2024.
6. Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339), NeurIPS 2022.
7. Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323), ICLR 2023.
8. Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978), MLSys 2024.
9. Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
10. Leviathan, Kalman and Matias, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
11. Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), 2023.
