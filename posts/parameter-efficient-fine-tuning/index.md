# Parameter-efficient fine-tuning: LoRA, QLoRA, adapters and prompts compared

Updating every weight of a 7-billion-parameter model takes a hundred-odd gigabytes of training state and a full new checkpoint for every task. Parameter-efficient fine-tuning (PEFT) is the family of methods that freeze the pretrained weights and train something small next to them: a few extra layers, a few vectors, a low-rank update. The small thing is the whole result of training, often a few megabytes, and the base model is shared.

The methods differ in **where** the trainable piece sits and **what it does to inference**. I put them side by side: what each trains, how many parameters and how much memory that costs for a Llama-2-7B-sized model, how they compare on a small transformer where everything else is held fixed, and which to reach for. On that transformer, after 200 steps with one seed, LoRA on all linear layers reaches 0.90 accuracy on the new task with 3,848 trainable parameters, against 0.94 for full fine-tuning with 18,312. The mechanics of LoRA itself are covered separately, in [LoRA explained](../lora-explained/) and with a full from-scratch build in [LoRA from scratch](/posts/lora-from-scratch/), so here LoRA is one entry on the menu.

## The menu
Every method below starts from the same frozen network and adds or unfreezes a different piece.

| Method | What is trained | Where it acts | Extra inference cost |
|---|---|---|---|
| Full fine-tuning | every weight | everywhere | none, but one full copy per task |
| Adapters (Houlsby et al., 2019) | small bottleneck MLPs inserted in each block | after attention and after the FFN | extra layers in series |
| Prefix tuning (Li and Liang, 2021) | learned key and value vectors at every layer | inside attention | longer attention span |
| Prompt tuning (Lester et al., 2021) | learned embeddings prepended to the input | input layer only | longer sequence |
| IA³ (Liu et al., 2022) | one scaling vector each for keys, values and FFN activations | elementwise, inside the block | none once folded in |
| LoRA (Hu et al., 2021) | a low-rank update \(BA\) to chosen weight matrices | in parallel with each linear layer | none once merged |
| QLoRA (Dettmers et al., 2023) | LoRA, on top of a base model stored in 4 bits | same as LoRA | dequantization in each matmul |

A few details matter for what follows.

- **Adapters** are a down-projection to a small width \(m\), a nonlinearity and an up-projection, added back through a residual connection. The up-projection starts at or near zero, so the adapted model starts out (almost) identical to the pretrained one.
- **Prefix tuning** prepends \(p\) learned vectors to the keys and values of every attention layer, so every real token can attend to them. Li and Liang trained them through a small MLP for stability and kept only the resulting vectors.
- **Prompt tuning** is the input-layer-only version: \(p\) learned "soft token" embeddings in front of the text. It is the cheapest method here and also the weakest on small models. Lester et al. found the gap to full fine-tuning closes only at around ten billion parameters.
- **IA³** multiplies keys, values and the hidden FFN activations by learned vectors that start at one. It trains even fewer parameters than LoRA, and the scales can be folded into the weights afterwards.
- **LoRA** keeps \(W_0\) frozen and learns \(\Delta W = \frac{\alpha}{r} BA\) with rank \(r\). The original paper adapted only the query and value projections. QLoRA found that matching full fine-tuning needed LoRA on every linear layer.

## Counting parameters
For a decoder-only model with \(L\) blocks, width \(d\), feed-forward width \(d_{ff}\), LoRA rank \(r\), adapter width \(m\) and \(p\) prompt or prefix vectors, the trainable counts are simple formulas. Llama-2-7B has \(L = 32\), \(d = 4096\), \(d_{ff} = 11008\), a 32,000-token vocabulary, no biases and three FFN matrices per block (a gated FFN). The first check is that the formula for the whole model reproduces its published size, 6,738,415,616 parameters.

```python
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

torch.set_num_threads(1)


def full_params(L, d, ff, vocab, ff_mats=3):
    per_block = 4 * d * d + ff_mats * d * ff + 2 * d       # attention, FFN, two norms
    return L * per_block + 2 * vocab * d + d                # + embeddings, output head, final norm


def peft_params(L, d, ff, ff_mats=3, r=8, m=64, p=20):
    return {
        "LoRA r=8, q and v": L * 2 * r * (d + d),
        "LoRA r=8, all linear": L * (4 * r * (d + d) + ff_mats * r * (d + ff)),
        "Adapters m=64": L * 2 * (d * m + m + m * d + d),  # two bottlenecks per block, with biases
        "Prefix p=20": L * 2 * p * d,                       # a key and a value vector per slot per layer
        "Prompt p=20": p * d,
        "IA3": L * (d + d + ff),                            # keys, values, FFN hidden units
    }


llama7b = dict(L=32, d=4096, ff=11008)
total = full_params(**llama7b, vocab=32_000)
assert total == 6_738_415_616                               # the published Llama-2-7B count
counts = peft_params(**llama7b)
assert counts["LoRA r=8, q and v"] == 4_194_304
for name, n in counts.items():
    print(f"{name:22s} {n:>12,d}  {100 * n / total:.4f}%")
```

| Method (Llama-2-7B) | Trainable parameters | Share of the model |
|---|---|---|
| Full fine-tuning | 6,738,415,616 | 100% |
| Adapters, \(m = 64\) | 33,820,672 | 0.50% |
| LoRA \(r = 8\), all linear layers | 19,988,480 | 0.30% |
| Prefix tuning, \(p = 20\) | 5,242,880 | 0.078% |
| LoRA \(r = 8\), query and value only | 4,194,304 | 0.062% |
| IA³ | 614,400 | 0.0091% |
| Prompt tuning, \(p = 20\) | 81,920 | 0.0012% |

Every PEFT method trains well under one percent of the model, and the spread among them is about 400 times. That spread matters less than it looks. Once you are below about 1%, the memory bill is dominated by the frozen weights, not the trainable ones, which is the next section.

## Where the gigabytes go
Training with Adam in mixed precision keeps about 16 bytes per **trainable** parameter: a 16-bit weight (2 bytes), a 16-bit gradient (2), an fp32 master copy (4) and Adam's two moment estimates (8). A **frozen** parameter needs only its 16-bit weight, 2 bytes. That asymmetry is the whole memory argument for PEFT, and it gives the first two rows below. Activations come on top in every case and grow with batch size and sequence length.

QLoRA attacks the remaining cost, the frozen weights themselves. Dettmers et al. store them in 4 bits using **NormalFloat4 (NF4)**: since trained weights are roughly normal, the 16 levels are placed at quantiles of a standard normal, with an exact zero, and scaled to \([-1, 1]\). Each block of 64 weights is divided by its largest absolute value before rounding. Storing that scale in fp32 would add \(32/64 = 0.5\) bits per weight. **Double quantization** stores the scales themselves in 8 bits, with one fp32 scale per 256 of them, which brings the overhead down to about 0.127 bits. The base model is dequantized to 16 bits on the fly for each matrix multiply, and gradients flow through it into the LoRA weights, which stay in 16 bits.

```python
def nf4_levels():
    pos = norm.ppf(np.linspace(0.9677083, 0.5, 9)[:-1])    # 8 positive quantiles
    neg = -norm.ppf(np.linspace(0.9677083, 0.5, 8)[:-1])   # 7 negative quantiles
    levels = np.sort(np.concatenate([neg, [0.0], pos]))     # plus an exact zero
    return levels / np.abs(levels).max()


def quantize(w, levels, block=64):
    flat = w.reshape(-1, block)
    absmax = np.abs(flat).max(1, keepdims=True)             # one scale per block of 64
    idx = np.abs(flat[:, :, None] / absmax[:, :, None] - levels).argmin(-1)
    return (levels[idx] * absmax).reshape(w.shape)


nf4 = nf4_levels()
assert len(nf4) == 16 and nf4[7] == 0
assert np.allclose(nf4[[0, 6, 8, 14, 15]], [-1, -0.0911, 0.0796, 0.7230, 1], atol=1e-4)  # the NF4 table

w = np.random.default_rng(0).normal(0, 0.02, (1024, 256))   # weights that look like a trained layer's
int4 = np.arange(-7, 8) / 7                                  # symmetric int4: 15 evenly spaced levels, same block scaling
err = {name: ((quantize(w, lv) - w) ** 2).mean() / w.var() for name, lv in [("NF4", nf4), ("int4", int4)]}
assert 0.25 < 1 - err["NF4"] / err["int4"] < 0.31            # quantile levels fit normal weights better

scale_bits = 32 / 64                                         # one fp32 scale per 64 weights
double_quant_bits = 8 / 64 + 32 / (64 * 256)                 # 8-bit scales, one fp32 scale per 256 of those
assert scale_bits == 0.5 and round(double_quant_bits, 3) == 0.127


def train_memory_gb(n_total, n_trainable, frozen_bits):
    frozen = (n_total - n_trainable) * frozen_bits / 8
    return (frozen + 16 * n_trainable) / 1e9


lora_all = counts["LoRA r=8, all linear"]
mem = {
    "full": train_memory_gb(total, total, 16),
    "LoRA": train_memory_gb(total, lora_all, 16),
    "QLoRA": train_memory_gb(total, lora_all, 4 + double_quant_bits),
}
assert 105 < mem["full"] < 110 and 13 < mem["LoRA"] < 15 and 3.5 < mem["QLoRA"] < 4.5
assert 65e9 * (4 + double_quant_bits) / 8 / 1e9 < 48           # a 65B base in NF4 fits on a 48 GB card

emb_extra = 2 * 32_000 * 4096 * (16 - 4 - double_quant_bits) / 8 / 1e9  # embeddings and head left in 16 bits
assert 0.38 < emb_extra < 0.40
```

| Llama-2-7B, before activations | Frozen weights | Trainable state | Total |
|---|---|---|---|
| Full fine-tuning | none | 16 bytes × 6.74B | about 108 GB |
| LoRA, all linear layers, \(r = 8\) | 16-bit, 13.4 GB | 16 bytes × 20M | about 13.8 GB |
| QLoRA, same adapters | NF4 with double quantization, 3.5 GB | 16 bytes × 20M | about 3.8 GB |

The QLoRA row assumes every frozen weight is stored in NF4. Common implementations quantize only the linear layers inside the blocks and keep the embeddings and output head in 16 bits, which adds about 0.4 GB for this model.

On normally distributed weights NF4's rounding error is about 28% lower than evenly spaced 4-bit levels with the same block scaling. The last assert is the headline result of the QLoRA paper: a 65B model, which needs about 33.5 GB in NF4, can be fine-tuned on a single 48 GB GPU. They also used paged optimizers, which move optimizer state to CPU memory during memory spikes from long sequences.

## A head-to-head on a small transformer
Parameter counts say nothing about quality, so the next block runs every method on the same small problem. A two-layer transformer is pretrained on sequences of 12 tokens drawn from 16 to answer one question: which of the tokens 0 to 7 occurs most often? Then it is adapted to a related question, which of tokens 8 to 15 occurs most often, with a fresh classification head. Every method trains that head. The "head only" row trains nothing else and shows how far the frozen features get on their own. All PEFT methods get the same budget, 200 steps at learning rate \(10^{-2}\). Full fine-tuning gets the more conventional \(10^{-3}\). On this toy it would also tolerate \(10^{-2}\), but on large pretrained models a high rate on every weight tends to destabilise training and erase what pretraining learned. The reverse matters more: at full fine-tuning's \(10^{-3}\), the PEFT methods learn far too slowly, which the last check in the block shows for LoRA. QLoRA here is simulated: the base weights are rounded to the NF4 grid with the code above, then stored as ordinary floats.

```python
V, T, D, H, FF, L = 16, 12, 32, 2, 64, 2


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.up, self.down = nn.Linear(D, FF), nn.Linear(FF, D)
        self.prefix = self.ia3 = self.adapters = None        # the hooks each method fills in

    def forward(self, x):
        n, t, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        if self.ia3 is not None:
            k, v = k * self.ia3["k"], v * self.ia3["v"]
        if self.prefix is not None:                          # learned keys and values, attended to by every token
            k = torch.cat([self.prefix[0].expand(n, -1, -1), k], 1)
            v = torch.cat([self.prefix[1].expand(n, -1, -1), v], 1)
        heads = lambda z: z.view(n, z.shape[1], H, D // H).transpose(1, 2)
        a = self.o(F.scaled_dot_product_attention(heads(q), heads(k), heads(v)).transpose(1, 2).reshape(n, t, D))
        if self.adapters is not None:
            a = a + self.adapters[0](a)
        x = x + a
        f = F.gelu(self.up(self.ln2(x)))
        if self.ia3 is not None:
            f = f * self.ia3["ff"]
        f = self.down(f)
        if self.adapters is not None:
            f = f + self.adapters[1](f)
        return x + f


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Parameter(torch.randn(T, D) * 0.02)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, 8)
        self.soft_prompt = None

    def forward(self, ids):
        x = self.emb(ids) + self.pos
        if self.soft_prompt is not None:                     # learned embeddings in front of the input
            x = torch.cat([self.soft_prompt.expand(len(ids), -1, -1), x], 1)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, -T:]).mean(1))         # pool over the real tokens only


class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        return out + self.scale * (x @ self.A.T @ self.B.T) if self.enabled else out


def data(task, n, g):
    ids = torch.randint(0, V, (n, T), generator=g)
    counts = F.one_hot(ids, V).sum(1)
    return ids, (counts[:, :8] if task == "A" else counts[:, 8:]).argmax(1)


def train(model, params, task, steps, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    for _ in range(steps):
        x, y = data(task, 64, g)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def accuracy(model, task):
    x, y = data(task, 4000, torch.Generator().manual_seed(123))
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


torch.manual_seed(0)
base = Net()
train(base, base.parameters(), "A", 300, 3e-3, seed=0)
acc_A = accuracy(base, "A")
assert acc_A > 0.95


def adapter(m=4):
    a = nn.Sequential(nn.Linear(D, m), nn.GELU(), nn.Linear(m, D))
    nn.init.zeros_(a[2].weight)                              # starts as the identity
    nn.init.zeros_(a[2].bias)
    return a


def adapt(method):
    model = copy.deepcopy(base)
    model.requires_grad_(method == "full")
    torch.manual_seed(1)
    model.head = nn.Linear(D, 8)                             # a fresh head for the new task, always trained
    for b in model.blocks:
        if method == "prefix":
            b.prefix = nn.ParameterList(nn.Parameter(torch.randn(1, 8, D) * 0.5) for _ in range(2))
        if method == "ia3":
            b.ia3 = nn.ParameterDict({"k": nn.Parameter(torch.ones(D)), "v": nn.Parameter(torch.ones(D)),
                                      "ff": nn.Parameter(torch.ones(FF))})
        if method == "adapter":
            b.adapters = nn.ModuleList([adapter(), adapter()])
        if method in ("lora", "qlora"):
            for name in ("q", "k", "v", "o", "up", "down"):
                lin = getattr(b, name)
                if method == "qlora":
                    lin.weight.data = torch.tensor(quantize(lin.weight.data.numpy(), nf4), dtype=torch.float32)
                setattr(b, name, LoRALinear(lin))
    if method == "prompt":
        model.soft_prompt = nn.Parameter(torch.randn(1, 8, D) * 0.5)
    return model


results, models = {}, {}
for method in ["head", "prompt", "prefix", "ia3", "adapter", "lora", "qlora", "full"]:
    model = adapt(method)
    params = [p for p in model.parameters() if p.requires_grad]
    train(model, params, "B", 200, 1e-3 if method == "full" else 1e-2, seed=1)
    results[method] = (sum(p.numel() for p in params), accuracy(model, "B"))
    models[method] = model
    print(f"{method:8s} {results[method][0]:6d} params  accuracy on B {results[method][1]:.3f}")

# the formulas above, at this model's size (2 FFN matrices, r=4, m=4, p=8), plus the 264-parameter head
tiny = peft_params(L, D, FF, ff_mats=2, r=4, m=4, p=8)
assert results["lora"][0] == tiny["LoRA r=8, all linear"] + 264
assert results["adapter"][0] == tiny["Adapters m=64"] + 264
assert results["prefix"][0] == tiny["Prefix p=20"] + 264
assert results["prompt"][0] == tiny["Prompt p=20"] + 264 and results["ia3"][0] == tiny["IA3"] + 264

acc = {k: v[1] for k, v in results.items()}
assert acc["head"] < 0.35                                    # frozen features alone do not solve B
assert all(acc[k] > acc["head"] + 0.1 for k in acc if k != "head")
assert acc["prefix"] > acc["prompt"] + 0.2                   # every layer beats the input layer alone
assert acc["lora"] > 0.85 and abs(acc["qlora"] - acc["lora"]) < 0.05   # 4-bit base costs little
assert acc["full"] >= max(acc.values()) - 1e-9 and results["full"][0] > 4 * results["lora"][0]

slow = adapt("lora")                                         # LoRA at full fine-tuning's learning rate
train(slow, [p for p in slow.parameters() if p.requires_grad], "B", 200, 1e-3, seed=1)
acc_lora_slow = accuracy(slow, "B")
print(f"LoRA at lr 1e-3: accuracy on B {acc_lora_slow:.3f}")
assert acc_lora_slow < acc["lora"] - 0.25
```

| Method | Trainable parameters | Accuracy on the new task |
|---|---|---|
| Head only (frozen body) | 264 | 0.25 |
| Prompt tuning | 520 | 0.39 |
| Adapters | 1,432 | 0.56 |
| IA³ | 520 | 0.67 |
| Prefix tuning | 1,288 | 0.73 |
| QLoRA (simulated) | 3,848 | 0.89 |
| LoRA, all linear layers | 3,848 | 0.90 |
| Full fine-tuning | 18,312 | 0.94 |

Read this as a picture of the trade-offs, not a benchmark. One small task, one seed and one learning rate per method decide the exact order. Run for 1,000 steps instead of 200 and every method except the frozen head gets above 0.85, with prefix tuning and IA³ passing LoRA. LoRA run at full fine-tuning's \(10^{-3}\) reaches only about 0.43 in the 200 steps. Three things here are likely to hold on real models too, though one toy cannot show it. Prompt tuning trails prefix tuning, because a few vectors at the input of a small model have little leverage, while prefixes act directly inside every layer's attention. IA³ gets a lot from very few parameters. And rounding the frozen base to 4 bits barely moves LoRA, which is the bet QLoRA makes.

## Serving: merging, swapping and forgetting
Training cost is half the story. The other half is what the adapted model costs to run and how many tasks it can serve.

LoRA and IA³ can be **merged**: \(\frac{\alpha}{r}BA\) is added into \(W_0\), and an IA³ scale on the keys or values becomes a rescaling of the rows of \(W_k\) or \(W_v\), and the FFN scale, which acts after the nonlinearity, a rescaling of the columns of the down-projection. The served model then has exactly the base architecture and latency. Adapters cannot be merged, because their nonlinearity sits between two projections, so they add layers that run in series. That shows up most at batch size 1, which is common in online serving. Prefix and prompt tuning cannot be merged either, and they use up context: every request carries \(p\) extra positions.

Not merging is also an option. One frozen base in GPU memory can serve many LoRA adapters at once, picking the adapter per request, and switching an adapter off gives back the pretrained model exactly. Full fine-tuning has no such switch. Below, the fully fine-tuned network, given back its old task-A head, scores 0.19 on task A, down from 0.99: it has largely forgotten it. The LoRA model with its adapters disabled scores exactly what the base model did.

```python
x_test, _ = data("B", 256, torch.Generator().manual_seed(7))

# IA3 folds into the weights: the key and value scales into the rows of W_k and W_v (and their biases),
# the FFN scale, which acts after the nonlinearity, into the columns of W_down
ia3 = models["ia3"]
merged = copy.deepcopy(ia3)
for b in merged.blocks:
    for name in ("k", "v"):
        lin, s = getattr(b, name), b.ia3[name].detach()
        lin.weight.data *= s[:, None]
        lin.bias.data *= s
    b.down.weight.data *= b.ia3["ff"].detach()[None, :]
    b.ia3 = None
with torch.no_grad():
    assert torch.allclose(merged(x_test), ia3(x_test), atol=1e-5)

# LoRA folds into the weights too (the long version is in "LoRA from scratch")
lora = models["lora"]
lin = lora.blocks[0].q
x = torch.randn(5, D)
with torch.no_grad():
    W = lin.base.weight + lin.scale * lin.B @ lin.A
    assert torch.allclose(lin(x), x @ W.T + lin.base.bias, atol=1e-5)

# Switching adapters off recovers the base model exactly; full fine-tuning has overwritten it
for m in lora.modules():
    if isinstance(m, LoRALinear):
        m.enabled = False
lora.head = base.head
assert accuracy(lora, "A") == acc_A
full = models["full"]
full.head = base.head
assert accuracy(full, "A") < acc_A - 0.3
```

Merging a QLoRA adapter needs one more step. \(BA\) is added to a dequantized 16-bit copy of the base, and that merged model is then served in 16 bits or quantized again, which adds a little error of its own.

## Choosing a method
- **LoRA** is the default. It is well supported, cheap, and mergeable, and on every linear layer it comes close to full fine-tuning on most adaptation tasks. Start with rank 8 to 16 and \(\alpha\) at one to two times the rank.
- **QLoRA** is for when the base model in 16 bits does not fit on your GPU. It costs some speed, since every matmul dequantizes, and a merge-and-requantize step before serving.
- **Full fine-tuning** is still the strongest when the new task is far from pretraining, for example a new language, a new domain at scale or continued pretraining, and when you have the data and memory for it. Biderman et al. (2024) compared the two on code and math, in both continued pretraining and instruction fine-tuning, and found that LoRA learns less than full fine-tuning but also forgets less of what the base model knew.
- **IA³** fits many tiny per-user or per-task variants where storage is the constraint, as in the few-shot setting it was introduced for.
- **Prompt and prefix tuning** keep the weights completely untouched and are easy to batch across tasks. They are much weaker on small models, and they spend context length.
- **Adapters** came first and still work. LoRA has largely replaced them for LLMs, in good part because adapters cannot be merged.

Fine-tuning changes behaviour: format, style, a narrow skill, a label set. It is a poor way to add facts that change, because updating them means retraining, and the model cannot point to a source. Use [retrieval](../rag-architecture/) for that. Many systems do both: fine-tune for format, retrieve for facts. Preference tuning, the step after supervised fine-tuning, is covered in [RLHF and alignment](../rlhf-alignment/).

## Pitfalls
- **Using full fine-tuning's learning rate.** PEFT methods usually want a larger one, around \(10^{-4}\) to \(3 \times 10^{-4}\) for LoRA on an LLM against around \(10^{-5}\) for full fine-tuning, because they train few, freshly initialized parameters. In the toy above, LoRA at \(10^{-3}\) learned far less in 200 steps than at \(10^{-2}\).
- **Small datasets overfit quickly anyway.** A few hundred examples can be memorized by a LoRA adapter in a few epochs. Watch validation loss, and see the [overfitting playbook](../overfitting-playbook/).
- **Adapting too little of the network.** LoRA on query and value only is the original recipe and the cheapest, but adding the other attention and FFN matrices usually helps more than raising the rank.
- **Forgetting the chat template.** Fine-tune an instruction model with the same prompt format it will see when served, or the adapter learns a different format from the one you use in production.
- **Comparing by parameter count.** Below about 1% trainable, memory is set by the frozen base and the activations. Quality is set by where the trainable parameters sit, not how many there are.
- **Assuming merged equals unmerged under quantization.** For QLoRA, test the model you will actually serve, after any requantization.

The head-to-head is one toy task with one seed and 200 steps, so its order is not settled: at 1,000 steps prefix tuning and IA³ pass LoRA. The parameter and memory counts follow from formulas checked against the published Llama-2-7B size, before activations. How the methods rank in quality at that size is not measured here; the papers below are where to look.

The method details follow the original papers: Houlsby et al. (2019) for adapters, Li and Liang (2021) for prefix tuning, Lester et al. (2021) for prompt tuning, Hu et al. (2021) for LoRA, Liu et al. (2022) for IA³, Dettmers et al. (2023) for QLoRA and NF4, and Biderman et al. (2024) for the learns-less, forgets-less comparison. Some of the practical advice is adapted from Sebastian Raschka's [practical tips for fine-tuning LLMs with LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms).
