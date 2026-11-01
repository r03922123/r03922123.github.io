---
title: LoRA explained: why low rank, how it starts, and where it goes
summary: The ideas behind low-rank adaptation rather than the code: why a fine-tuning update can be squeezed into two thin matrices, what that costs when it cannot, why B starts at zero, what the alpha over r scale really does, which layers to adapt, and the variants that change each of these choices.
group: LLMs and generative AI
order: 5
tags: LoRA · fine-tuning · PEFT · LLM
---
Fine-tuning a 7-billion-parameter model the ordinary way means a gradient and two optimizer moments for every parameter, and a full copy of the model per task. LoRA (Hu et al., 2021) freezes the pretrained weights and learns each layer's change as the product of two thin matrices. The trainable part shrinks by a factor of hundreds to thousands, and afterwards the change can be folded back into the weights, so the merged model runs exactly as fast as the original.

The implementation, step by step with asserts, is in [LoRA from scratch](/posts/lora-from-scratch/). This post is the conceptual companion: why the trick works, what it gives up, and what each design choice (initialization, the \(\alpha / r\) scale, which layers to adapt) does to training.

## The bet: the update is low rank
Take one linear layer with pretrained weight \(W_0 \in \mathbb{R}^{d \times k}\). Fine-tuning replaces it with \(W_0 + \Delta W\). LoRA restricts the change to rank \(r\):

\[ h = W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k). \]

\(W_0\) is frozen; only \(A\) and \(B\) train. The product \(BA\) is a \(d \times k\) matrix, but its rank is at most \(r\): every column of \(BA\) is a combination of the \(r\) columns of \(B\). So the layer can only move its outputs within an \(r\)-dimensional subspace, which the training itself chooses.

Why should that be enough? Work on **intrinsic dimension** suggested it would be. Li et al. (2018) found that many networks reach 90% of their usual performance while optimizing only a random low-dimensional slice of their parameters, and Aghajanyan et al. (2021) found that fine-tuning pretrained language models needs very few dimensions, fewer the larger the model. The pretrained model already holds the features, and a task mostly reweights a few directions. In the LoRA paper's GPT-3 experiments, adapters of rank 1 to 8 were competitive with full fine-tuning.

It is a bet, not a theorem. A rank-\(r\) update can only capture as much of the ideal update as its top \(r\) singular values. By the Eckart–Young theorem, the best rank-\(r\) approximation keeps the \(r\) largest singular values and drops the rest, so the relative error is

\[ \frac{\lVert \Delta W - \Delta W_r \rVert_F}{\lVert \Delta W \rVert_F} = \sqrt{\frac{\sum_{i > r} \sigma_i^2}{\sum_i \sigma_i^2}} . \]

With fast-decaying singular values a small \(r\) loses almost nothing; with a flat spectrum, rank 8 of 64 keeps an eighth of the energy. The block builds both kinds of "ideal update" and trains a rank-4 LoRA on each. Training reaches the Eckart–Young optimum and no further: the rank, not the optimizer, sets the limit.

```python
import math
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(0)
d = k = 64


def matrix_with_spectrum(sv):
    U, _ = torch.linalg.qr(torch.randn(d, d))
    V, _ = torch.linalg.qr(torch.randn(k, k))
    return U @ torch.diag(sv / sv.norm()) @ V.T          # unit Frobenius norm


def best_rank_r_error(M, r):                              # Eckart-Young, relative
    s = torch.linalg.svdvals(M)
    return (s[r:].norm() / s.norm()).item()


i = torch.arange(d, dtype=torch.float32)
decaying = matrix_with_spectrum(0.7 ** i)
flat = matrix_with_spectrum(torch.ones(d))

assert best_rank_r_error(decaying, 8) < 0.06               # rank 8: under 6% error
assert best_rank_r_error(flat, 8) > 0.93                   # rank 8: over 93% error
assert abs(best_rank_r_error(flat, 8) - math.sqrt(1 - 8 / 64)) < 1e-4


def fit_lora(target, r, steps=1500):
    A = nn.Parameter(torch.randn(r, k) / math.sqrt(k))
    B = nn.Parameter(torch.zeros(d, r))
    opt = torch.optim.Adam([A, B], lr=1e-2)
    for _ in range(steps):
        loss = ((B @ A - target) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert torch.linalg.matrix_rank(B @ A) <= r
        return ((B @ A - target).norm() / target.norm()).item()


for target in (decaying, flat):
    trained, optimum = fit_lora(target, r=4), best_rank_r_error(target, 4)
    assert optimum - 1e-4 <= trained < optimum + 1e-3      # reaches the bound, cannot beat it
```

In a real model the ideal update is unknown, and nothing guarantees a decaying spectrum. Tasks close to what the model already does (a format, a style, a label set) tend to suit LoRA; absorbing a lot of new knowledge, such as a new language, likely needs a higher-rank change.

## What it saves
The savings are easy to count. One \(4096 \times 4096\) projection has 16,777,216 entries. At \(r = 8\), \(A\) and \(B\) together have \(8 \times (4096 + 4096) = 65{,}536\), exactly 256 times fewer. Across a model shaped like Llama 2 7B (32 blocks, hidden size 4096, MLP width 11008, a 32,000-token vocabulary):

| Adapted matrices | Rank | Trainable parameters | Share of the model | Adapter file (bf16) |
|---|---|---|---|---|
| Query and value | 8 | 4.2M | 0.06% | 8 MB |
| All seven linear layers per block | 8 | 20.0M | 0.30% | 40 MB |
| All seven linear layers per block | 64 | 159.9M | 2.4% | 320 MB |

Memory is where this pays. Mixed-precision Adam costs about 16 bytes per trained parameter: bf16 weights and gradients (2 + 2), an fp32 master copy (4) and two fp32 moments (8), over 100 GB for 6.7B parameters before activations. With LoRA the frozen base costs 2 bytes per parameter and only the adapters pay 16. On GPT-3 175B the LoRA paper reports 10,000 times fewer trainable parameters and a third of the GPU memory.

LoRA does **not** make the forward pass cheaper, nor the half of the backward pass that propagates gradients to the activations: those still flow through every frozen layer, and most activations are still stored. The compute saved is the frozen matrices' weight gradients, at most about a third of a training step; the LoRA paper measured a 25% training speedup on GPT-3 175B.

```python
hidden, mlp_width, n_layers, vocab = 4096, 11008, 32, 32_000
attn_shapes = {name: (hidden, hidden) for name in ["q", "k", "v", "o"]}
mlp_shapes = {"gate": (mlp_width, hidden), "up": (mlp_width, hidden), "down": (hidden, mlp_width)}
all_shapes = {**attn_shapes, **mlp_shapes}

linear = n_layers * sum(dd * kk for dd, kk in all_shapes.values())
total = linear + 2 * vocab * hidden + (2 * n_layers + 1) * hidden   # untied embeddings, norms
assert abs(total / 1e9 - 6.74) < 0.01


def lora_count(names, r):
    return n_layers * sum(r * sum(all_shapes[n]) for n in names)


assert 4096 * 4096 == 256 * (lora_count(["q"], 8) // n_layers)    # one matrix, 256x fewer
qv8, all8, all64 = lora_count(["q", "v"], 8), lora_count(all_shapes, 8), lora_count(all_shapes, 64)
assert qv8 == 4_194_304 and all8 == 19_988_480 and all64 == 8 * all8
assert round(100 * qv8 / total, 2) == 0.06 and round(100 * all8 / total, 2) == 0.30
assert round(2 * all8 / 1e6) == 40                                  # MB at 2 bytes per value

full_ft_gb = 16 * total / 1e9                                       # weights, grads, master, moments
lora_gb = (2 * total + 16 * all8) / 1e9
assert full_ft_gb > 100 and lora_gb < 14
```

## Initialization: why B starts at zero and A does not
LoRA initializes \(A\) randomly (a Gaussian in the paper, Kaiming-uniform in common implementations) and \(B\) to zero. First, that makes \(BA = 0\), so training starts from exactly the pretrained model rather than a randomly perturbed one. Second, the gradients are asymmetric. With \(G = \partial L / \partial h\) for one input,

\[ \frac{\partial L}{\partial B} = \frac{\alpha}{r}\, G\, (Ax)^\top, \qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top G\, x^\top . \]

At \(B = 0\), \(A\) gets no gradient but \(B\) does, because \(Ax \neq 0\); after one step both train. If both started at zero, both gradients would stay zero forever: the origin is a stationary point of the factorization. Zero \(A\) with random \(B\) also starts from the pretrained model and trains, with the roles reversed, but the two are not equivalent in practice: Hayou, Ghosh and Yu (2024) showed that the standard choice (zero \(B\)) tolerates larger learning rates.

```python
def lora_layer(d_out, d_in, r, zero_A=False):
    W0 = torch.randn(d_out, d_in) / math.sqrt(d_in)          # frozen
    A = nn.Parameter(torch.zeros(r, d_in) if zero_A else torch.empty(r, d_in))
    if not zero_A:
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    B = nn.Parameter(torch.zeros(d_out, r))
    return W0, A, B


x, y = torch.randn(32, 48), torch.randn(32, 40)
W0, A, B = lora_layer(40, 48, r=4)
scale = 8 / 4
h = x @ W0.T + scale * (x @ A.T) @ B.T
assert torch.equal(h, x @ W0.T)                                 # the adapter adds exactly zero
((h - y) ** 2).mean().backward()
assert A.grad.abs().max() == 0 and B.grad.abs().max() > 0       # only B learns on step one

W0, A, B = lora_layer(40, 48, r=4, zero_A=True)                  # both zero: stuck for good
opt = torch.optim.Adam([A, B], lr=1e-2)
for _ in range(20):
    loss = ((x @ W0.T + scale * (x @ A.T) @ B.T - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
assert A.abs().max() == 0 and B.abs().max() == 0
```

## The alpha over r scale is a learning-rate knob
The adapter's output is multiplied by \(s = \alpha / r\). It is easiest to understand through the optimizer. Adam's step size is set by the learning rate \(\eta\), not by the gradient's size. On the first step \(A\) has zero gradient and every entry of \(B\) moves by about \(\pm \eta\), so the layer's change \(sBA\) is proportional to \(s\eta\): on that step, doubling \(\alpha\) is exactly the same as doubling the learning rate. Later the match is only approximate, because \(A\)'s own steps scale with \(\eta\) but not with \(\alpha\). That is why the paper calls tuning \(\alpha\) roughly the same as tuning the learning rate, sets \(\alpha\) once, and tunes the learning rate.

The \(1/r\) is meant to save retuning when \(r\) changes, but it overcorrects. Each entry of \(BA\) sums \(r\) terms of roughly random sign, so it grows like \(\sqrt{r}\), not \(r\). Dividing by \(r\) shrinks the effective step like \(1/\sqrt{r}\), so higher ranks learn more slowly and their extra capacity can go unused. Kalajdzievski (2023) made this argument formally and proposed **rsLoRA**, scaling by \(\alpha / \sqrt{r}\). The block measures one Adam step, then the loss after 60, on a full-rank target:

```python
def run(r, s, lr=1e-3, steps=1, d_=128, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(256, d_)
    Y = X @ (torch.randn(d_, d_) / math.sqrt(d_)).T            # a full-rank change to learn
    A = nn.Parameter(torch.empty(r, d_))
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    B = nn.Parameter(torch.zeros(d_, r))
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(steps):
        loss = ((X @ (s * B @ A).T - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return s * B @ A, ((X @ (s * B @ A).T - Y) ** 2).mean().item()


dW_a, _ = run(16, s=2.0, lr=1e-3)                                # first step: alpha and lr trade off
dW_b, _ = run(16, s=1.0, lr=2e-3)
assert (dW_a - dW_b).norm() < 1e-2 * dW_a.norm()                 # equal up to Adam's epsilon

alpha, ranks = 16, [4, 16, 64]
step_r = [run(r, alpha / r)[0].norm().item() for r in ranks]
step_rs = [run(r, alpha / math.sqrt(r))[0].norm().item() for r in ranks]
assert step_r[0] / step_r[2] > 3                                 # alpha / r: steps shrink with r
assert max(step_rs) / min(step_rs) < 1.3                         # alpha / sqrt(r): rank-stable

loss_r = [run(r, alpha / r, steps=60)[1] for r in ranks]
loss_rs = [run(r, alpha / math.sqrt(r), steps=60)[1] for r in ranks]
assert loss_rs[2] < 0.6 * loss_r[2]                              # at this lr, rank 64 does far better with rsLoRA
assert loss_r[0] - loss_r[2] < 0.5 * (loss_rs[0] - loss_rs[2])  # alpha / r gains much less from rank 4 to 64
assert loss_rs[0] > loss_rs[1] > loss_rs[2]
```

A common habit is \(\alpha = r\) or \(\alpha = 2r\), which fixes \(s\) at 1 or 2 and quietly removes the \(1/r\), so the step grows like \(\sqrt{r}\). Whatever the convention, \(\alpha\) and the learning rate are coupled: change one, revisit the other, and store \(s\) with the adapter.

## Where to put the adapters
The original paper adapted only attention projections. On GPT-3, with a fixed parameter budget, adapting \(W_q\) and \(W_v\) at low rank beat one matrix at higher rank: spreading the budget helped more than concentrating it. The QLoRA paper (Dettmers et al., 2023) went further and found adapters on **all** linear layers, MLP included, were needed to match full fine-tuning, with the number of adapted layers mattering more than the rank. That is now the usual default, at about five times the parameters of query-and-value alone.

A **new output head** (say, a classifier) has no pretrained weights to adapt and must be trained in full, as must embedding rows for newly added tokens. Norm gains are tiny and are sometimes trained alongside at almost no cost.

## Merge, or keep the adapter separate
Computing \(W = W_0 + s BA\) once gives a model with exactly the base's shape and cost: the "no extra latency" claim. Keeping the adapter separate costs \(r(d + k)\) extra multiply-adds per token for each adapted matrix, under 1% of that matrix's \(dk\) at \(r = 8\), \(d = k = 4096\) (in FLOPs; the extra small matrix multiplies add some latency at small batch sizes), and allows **many adapters on one base**: one copy of the frozen weights, a small adapter per task, swapped per request or mixed within a batch.

Unmerging by subtraction is only safe in high precision. In bf16 the merge rounds and subtracting does not undo it:

```python
torch.manual_seed(1)
W0_bf16 = torch.randn(256, 256).to(torch.bfloat16)
delta = (0.01 * torch.randn(256, 8) @ torch.randn(8, 256)).to(torch.bfloat16)
restored = (W0_bf16 + delta) - delta
changed = (restored != W0_bf16).float().mean().item()
assert changed > 0.01                                           # a few percent of weights drifted
```

Keep an untouched copy of the base, or merge into an fp32 copy and cast afterwards.

## Variants
Most variants change one of the choices above.

| Variant | What it changes |
|---|---|
| QLoRA (Dettmers et al., 2023) | Stores the frozen base in 4-bit NormalFloat and trains bf16 adapters on top, so very large models fit on one GPU. Each forward pass dequantizes, so steps are slower than plain LoRA. |
| rsLoRA (Kalajdzievski, 2023) | Scales by \(\alpha / \sqrt{r}\) so that higher ranks keep learning at the same speed. |
| LoRA+ (Hayou, Ghosh and Yu, 2024) | Gives \(B\) a larger learning rate than \(A\), since the two factors play different roles. |
| AdaLoRA (Zhang et al., 2023) | Parameterizes the update in an SVD-like form and prunes singular values during training, so rank is spent where it helps. |
| DoRA (Liu et al., 2024) | Splits each weight into a magnitude and a direction, trains the magnitude directly and applies LoRA to the direction. |
| PiSSA (Meng et al., 2024) | Initializes \(A\) and \(B\) from the top singular vectors of \(W_0\) and freezes the residual, instead of starting the adapter at zero. |

Other parameter-efficient methods, such as prefix and prompt tuning, train input vectors instead of weight updates. They often train fewer parameters, but they use up part of the context in every sequence and usually trail LoRA in quality. The [parameter-efficient fine-tuning](../parameter-efficient-fine-tuning/) post compares the families.

## When to use it, and pitfalls
LoRA is the default when the task is close to what the base model can already do (instruction following, a domain's tone, classification, extraction), when memory is tight, or when one model needs many task variants. Full fine-tuning earns its cost when the model must absorb a lot of new knowledge: Biderman et al. (2024) found that in continued pretraining on code and math LoRA learned noticeably less than full fine-tuning, and also forgot less of what the base model knew.

Common ways it goes wrong:

- **Rank as the first knob.** Which layers are adapted and the learning rate usually matter more. Try all linear layers at a modest rank before raising \(r\).
- **A full fine-tuning learning rate.** Adapters typically use one about an order of magnitude higher, and \(\alpha\) multiplies it.
- **Something else left trainable.** Count trainable parameters before training; the [from-scratch post](/posts/lora-from-scratch/) has an assert for this.
- **Misreading weight decay.** Decay on \(A\) and \(B\) pulls the model toward the base, not toward small weights (compare [L1 and L2 regularization](../regularization-l1-l2/)).
- **Lossy merges.** Merge in higher precision and keep the original base.
- **Mismatched base.** An adapter is a delta against one exact checkpoint; on another, even a requantized one, it silently gives a different model.

For the surrounding architecture, see [transformers and LLMs: key ideas](../transformers-and-llms-key-ideas/) and [foundation models](../foundation-models/).

## References
- E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- C. Li, H. Farkhoor, R. Liu, J. Yosinski. [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838). ICLR 2018.
- A. Aghajanyan, L. Zettlemoyer, S. Gupta. [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255). ACL 2021.
- T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). NeurIPS 2023.
- D. Kalajdzievski. [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732). 2023.
- D. Biderman et al. [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673). TMLR 2024.
