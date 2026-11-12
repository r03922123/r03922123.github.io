# Distributed training and GPU efficiency: where the memory goes and what sharding costs

A 7.5-billion-parameter model trained with Adam in mixed precision needs 120 GB for its weights, gradients and optimizer states before it stores a single activation, more than one 80 GB GPU holds. Sharded over 64 GPUs with ZeRO stage 3, the same states take 1.875 GB per GPU, and each GPU then sends 1.5 times as many bytes per step as plain data parallelism. Training on one GPU, or copying the model onto a few with PyTorch's `DistributedDataParallel` (DDP), hides these numbers until the model grows. Then the questions become which bytes live where, how many cross the network each step, and whether the GPU spends its time on arithmetic or on waiting.

This post answers them with small calculations and CPU checks. I write a memory calculator and check it against the ZeRO paper's own table, count the bytes each sharding scheme sends, confirm on a real PyTorch model that activation checkpointing trades a second forward pass for about an eighth of the stored activations without changing a gradient bit, show fp16 losing about a tenth of a real backward pass's gradients that bf16 keeps, and use a roofline to tell compute-bound from memory-bound work. Ring all-reduce and checkpoint/resume are in [ML infrastructure](../ml-infrastructure/); tensor and pipeline parallelism for serving are in [LLM inference](../llm-inference/).

## The words in the job ad
| Term | What it is |
|---|---|
| DDP | One full copy of the model per GPU; gradients averaged with an all-reduce after each backward pass |
| ZeRO stages 1, 2, 3 | Data parallelism that shards the optimizer states (1), also the gradients (2), also the weights (3) across the GPUs (Rajbhandari et al., 2020) |
| DeepSpeed | Microsoft's training library that implements ZeRO, plus offloading states to CPU memory |
| FSDP | PyTorch's fully sharded data parallelism: ZeRO-3-style sharding built into PyTorch (`fully_shard` in current versions) |
| Mixed precision | Matrix products in 16 bits (fp16 or bf16), with an fp32 master copy of the weights for the update |
| Activation checkpointing | Keep only some activations in the forward pass and recompute the rest during the backward pass |
| MFU | Model FLOPs utilization: the model's useful FLOPs per second divided by the hardware's peak |

## Where training memory goes
Training memory has two parts. **Model states** (weights, gradients, optimizer states) scale with the parameter count \(\Psi\). **Activations**, the intermediate tensors the backward pass needs, scale with batch size and sequence length. The ZeRO paper counts model states for mixed-precision Adam as 2 bytes of 16-bit weights, 2 of 16-bit gradients, and \(K = 12\) for the optimizer: an fp32 master copy of the weights and Adam's two moments, 4 bytes each. That is \(16\Psi\) bytes. Stage 1 divides the 12 by the number of GPUs \(N\), stage 2 also the gradients' 2, stage 3 everything.

```python
GB = 1e9


def model_states(psi, n_gpus, stage, k=12):
    """Bytes per GPU for weights, gradients and optimizer states, as in the ZeRO paper (section 5)."""
    weights, grads, optim = 2 * psi, 2 * psi, k * psi   # 16-bit weights and grads; fp32 master copy, Adam m and v
    if stage >= 1:
        optim /= n_gpus
    if stage >= 2:
        grads /= n_gpus
    if stage >= 3:
        weights /= n_gpus
    return weights + grads + optim


paper_table_1 = {1: (120, 120, 120), 4: (52.5, 41.3, 30), 16: (35.6, 21.6, 7.5), 64: (31.4, 16.6, 1.88),
                 256: (30.4, 15.4, 0.47), 1024: (30.1, 15.1, 0.12)}   # 7.5B parameters, GB per GPU, stages 1-3
for n, row in paper_table_1.items():
    mine = [model_states(7.5e9, n, s) / GB for s in (1, 2, 3)]
    tol = [0.005 if p < 2 else 0.05 for p in row]             # the table prints 2 decimals below 2 GB, else 1
    assert all(abs(m - p) <= e + 1e-9 for m, p, e in zip(mine, row, tol)), (n, mine)
assert model_states(7.5e9, 64, 0) / GB == 120 and model_states(7.5e9, 64, 3) / GB == 1.875

per_gpu = {s: model_states(7e9, 8, s) / GB for s in range(4)}       # 7B parameters on 8 GPUs; stage 0 = DDP
assert per_gpu == {0: 112, 1: 38.5, 2: 26.25, 3: 14}

import torch

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256))
opt = torch.optim.Adam(net.parameters())
net(torch.randn(8, 256)).sum().backward()
opt.step()
n_params = sum(p.numel() for p in net.parameters())
nbytes = lambda t: t.numel() * t.element_size()
state = sum(nbytes(p) + nbytes(p.grad) + nbytes(opt.state[p]["exp_avg"]) + nbytes(opt.state[p]["exp_avg_sq"])
            for p in net.parameters())
assert state == 16 * n_params                             # plain fp32 Adam: 4 + 4 + 4 + 4 bytes
```

The calculator matches every 7.5B entry of the paper's Table 1 to its printed precision. For a 7B model on eight 80 GB GPUs, DDP needs 112 GB per GPU and does not fit; stage 1 needs 38.5 GB, stage 2 26.25 GB, stage 3 14 GB. The last assert holds a surprise: plain fp32 Adam also costs 16 bytes per parameter. Mixed precision does not shrink model states; it buys tensor-core speed and halves the activations.

Activations grow with the micro-batch and the sequence length, and once model states are sharded they usually dominate per GPU. Korthikanti et al. (2022) count, for one transformer layer in 16 bits with sequence length \(s\), micro-batch \(b\), hidden size \(h\) and \(a\) heads, \(sbh(34 + 5as/h)\) bytes; the second term is the \(s \times s\) attention scores. Keeping only each layer's input (full recomputation) leaves \(2sbh\).

```python
def activation_bytes(s, b, h, a, layers, mode="all"):
    """Activation memory of a transformer stack, 16-bit, Korthikanti et al. (2022), Table 2, no parallelism."""
    per_layer = {"all": s * b * h * (34 + 5 * a * s / h),    # every intermediate, including the s x s scores
                 "no_scores": s * b * h * 34,                # what selective recomputation keeps
                 "full_recompute": 2 * s * b * h}[mode]      # only each layer's input
    return per_layer * layers


L, h, a, s = 32, 4096, 32, 2048
psi = 12 * L * h * h                                      # weights in the transformer blocks, embeddings left out
acts = {m: activation_bytes(s, 1, h, a, L, m) / GB for m in ("all", "no_scores", "full_recompute")}
assert psi == 6_442_450_944 and round(16 * psi / GB) == 103
assert {m: round(v, 2) for m, v in acts.items()} == {"all": 30.6, "no_scores": 9.13, "full_recompute": 0.54}
assert round(model_states(psi, 8, 3) / GB, 1) == 12.9 < acts["all"]   # ZeRO-3 on 8 GPUs: states shrink, activations do not
```

For a 6.4B-parameter shape (32 layers, hidden size 4,096) at 2,048 tokens, one sequence stores 30.6 GB of activations, 9.13 GB without the attention scores (roughly what a fused attention such as [FlashAttention](../flashattention/) also avoids storing) and 0.54 GB with full recomputation, against 103 GB of model states. ZeRO-3 on 8 GPUs cuts the model states to about 12.9 GB per GPU, and then the activations of one sequence are more than twice that. ZeRO-DP shards model states, not activations: each GPU keeps the activations of its own micro-batch.

## Activation checkpointing
`torch.utils.checkpoint` runs a block's forward pass without saving its intermediates, keeps only the block's input, and reruns the forward pass when the backward pass reaches it. I count the bytes autograd saves for the backward pass with a hook, on eight residual blocks, with and without it:

```python
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
blocks = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256))
                             for _ in range(8))
x = torch.randn(64, 256)
weights = {p.untyped_storage().data_ptr() for p in blocks.parameters()}
forward_calls = [0]
for blk in blocks:
    blk.register_forward_pre_hook(lambda *_: forward_calls.__setitem__(0, forward_calls[0] + 1))


def train_step(use_checkpoint):
    """Bytes of activations autograd keeps for the backward pass, block forward calls, and the gradients."""
    saved, forward_calls[0] = [], 0

    def pack(t):                                          # called for every tensor saved for backward
        if t.untyped_storage().data_ptr() not in weights:
            saved.append(nbytes(t))
        return t

    blocks.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = x
        for blk in blocks:
            out = out + (checkpoint(blk, out, use_reentrant=False) if use_checkpoint else blk(out))
        loss = out.pow(2).mean()
    loss.backward()
    return sum(saved), forward_calls[0], [p.grad for p in blocks.parameters()]


kept, calls, grads = train_step(False)
kept_ckpt, calls_ckpt, grads_ckpt = train_step(True)
assert (kept, kept_ckpt) == (4_784_128, 589_824)          # 64 x 256 floats per block input, plus the loss input
assert (calls, calls_ckpt) == (8, 16)                     # every block runs its forward twice
assert all(torch.equal(g, gc) for g, gc in zip(grads, grads_ckpt))   # bit-identical gradients
```

Stored activations fall from 4,784,128 to 589,824 bytes, about 8 times less, because each block now keeps its 64 × 256 input instead of also its two 64 × 1,024 intermediates. The gradients are bit-identical, and the price is visible in the call count: every block's forward runs twice. A forward pass costs about a third of a training step (the backward pass does twice its FLOPs), so full recomputation adds roughly a third more arithmetic; Korthikanti et al. measured 30 to 40% longer steps. Checkpointing every other block, or only the attention scores, sits in between.

## What DDP, ZeRO and FSDP send
DDP all-reduces the gradients: a reduce-scatter followed by an all-gather, in which each GPU sends \((N-1)/N\) of the gradient twice. ZeRO stages 1 and 2 send the same amount, rearranged: a reduce-scatter of the gradients, each GPU updates its shard, then an all-gather of the updated weights. Stage 3 holds no full copy of the weights, so it all-gathers each layer's weights in the forward pass, again in the backward pass, and reduce-scatters the gradients: three passes instead of two.

```python
def comm_bytes(psi, n_gpus, stage, bytes_per_elem=2):
    """Bytes each GPU sends per optimizer step (ring collectives), after the ZeRO paper's section 7."""
    ring = (n_gpus - 1) / n_gpus                          # a reduce-scatter or all-gather of X elements sends X(N-1)/N
    passes = 3 if stage == 3 else 2                       # all-reduce = reduce-scatter + all-gather; stage 3 gathers twice
    return passes * ring * psi * bytes_per_elem


volume = {st: comm_bytes(7.5e9, 64, st) / GB for st in range(4)}
assert volume[0] == volume[1] == volume[2] and volume[3] / volume[0] == 1.5
assert round(volume[0], 1) == 29.5 and round(volume[3], 1) == 44.3
assert (round(volume[0] / 25, 2), round(volume[3] / 25, 2)) == (1.18, 1.77)   # seconds at an assumed 25 GB/s
step_flops = 8 * 7.5e9 * 8 * 2048                         # 8 x 2,048 tokens per GPU; 8N per token: full recompute
assert round(step_flops / (0.5 * 312e12), 1) == 6.3       # seconds at half an A100's bf16 peak
fsdp_fp32_reduce = (64 - 1) / 64 * 7.5e9 * (4 + 2 + 2)   # fp32 reduce-scatter, two bf16 all-gathers
assert round(fsdp_fp32_reduce / comm_bytes(7.5e9, 64, 0), 6) == 2
```

For the 7.5B model on 64 GPUs with 16-bit gradients, DDP and stages 1 and 2 send about 29.5 GB per GPU per step and stage 3 about 44.3 GB. At an assumed 25 GB/s per GPU between machines, that is about 1.2 and 1.8 seconds, while the compute for one 16,384-token micro-batch takes about 6.3 seconds at half an A100's peak (with full recomputation, which a micro-batch that size needs by the activation formula above, so about \(8N\) FLOPs per token). The communication can hide behind compute only if it overlaps with it, which is why DDP buckets gradients and starts all-reducing during the backward pass, and FSDP prefetches the next layer's weights.

| Scheme | Shards | Bytes per param per GPU (mixed-precision Adam) | Sent per step, relative to DDP |
|---|---|---|---|
| DDP | nothing | 16 | 1 |
| ZeRO-1 | optimizer states | \(4 + 12/N\) | 1 |
| ZeRO-2 | + gradients | \(2 + 14/N\) | 1 |
| ZeRO-3 / FSDP full shard | + weights | \(16/N\) | 1.5 |

The last column assumes every collective moves 16-bit elements.

The PyTorch and DeepSpeed versions, not run here (the site builds on a CPU without a cluster):

```plaintext
# PyTorch FSDP2: shard each transformer block, then the root; launch with torchrun --nproc_per_node=8
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:
    fully_shard(block, mp_policy=mp)          # reshard_after_forward=True: free gathered weights (ZeRO-3-like)
fully_shard(model, mp_policy=mp)              # False keeps them until backward: less traffic, more memory

# DeepSpeed: ds_config.json
{ "train_micro_batch_size_per_gpu": 4, "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },
  "zero_optimization": { "stage": 3, "offload_optimizer": { "device": "cpu" } } }
```

When the model is created in fp32, as FSDP's mixed precision expects, the sharded fp32 weights double as the master copy (the optimizer steps on the shard in its original dtype) and the gradient shards are fp32 too: 4 + 4 + 8 bytes instead of the paper's 2 + 2 + 12, the same 16. With `reduce_dtype=torch.float32`, as above, the gradient reduce-scatter sends 4 bytes per element and the two weight all-gathers 2 each, so each step sends twice what 16-bit DDP does, not 1.5 times. Its hybrid mode shards within a machine and replicates across machines, which keeps the extra all-gathers on the fast links inside a node.

## Gradient accumulation and the effective batch
When a micro-batch is all that fits, run several, let `.grad` add up, and step once. The **effective batch** is micro-batch × accumulation steps × GPUs, and it, not the micro-batch, is what the learning rate was tuned for. Change the GPU count without changing the accumulation and the training run changes too.

```python
def accumulated_grads(model, xb, yb, micro_batches):
    model.zero_grad(set_to_none=True)
    for xm, ym in zip(xb.chunk(micro_batches), yb.chunk(micro_batches)):
        loss = torch.nn.functional.mse_loss(model(xm), ym) / micro_batches   # divide, or the step is 4x too big
        loss.backward()                                   # .grad adds up across calls
    return torch.cat([p.grad.flatten() for p in model.parameters()])


torch.manual_seed(0)
xb, yb = torch.randn(32, 16, dtype=torch.float64), torch.randn(32, 1, dtype=torch.float64)
mlp = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).double()
bn = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.BatchNorm1d(32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).double()
gap = {}
for name, model in (("mlp", mlp), ("batchnorm", bn)):
    full, acc = accumulated_grads(model, xb, yb, 1), accumulated_grads(model, xb, yb, 4)
    gap[name] = ((acc - full).norm() / full.norm()).item()
assert gap["mlp"] < 1e-12                                 # same gradient as one batch of 32
assert 0.6 < gap["batchnorm"] < 0.7                       # batch statistics now come from 8 rows, not 32
```

Four micro-batches of 8 give the gradient of one batch of 32 to rounding, but only when no layer looks across the batch: with BatchNorm the accumulated gradient differs by about 65%, because each micro-batch normalizes with its own mean and variance. Accumulation also changes the traffic: DDP's `no_sync()` skips the all-reduce on all but the last micro-batch, while ZeRO-3 and FSDP must still gather the weights for every micro-batch.

## fp16, bf16 and loss scaling
Both formats have 16 bits; they spend them differently. fp16 has 5 exponent bits: its largest value is 65,504 and its normal range ends at \(2^{-14} \approx 6 \times 10^{-5}\). bf16 has fp32's 8 exponent bits and therefore its range, but only 8 bits of precision. Micikevicius et al. (2018) made fp16 training work with three pieces: an fp32 master copy of the weights, fp32 accumulation in matrix products, and **loss scaling**, multiplying the loss so small gradients land in fp16's range and dividing them back in fp32.

```python
f16, bf16 = torch.finfo(torch.float16), torch.finfo(torch.bfloat16)
assert (f16.max, f16.tiny, f16.eps) == (65504, 2.0 ** -14, 2.0 ** -10)
assert bf16.tiny == torch.finfo(torch.float32).tiny and bf16.eps == 2.0 ** -7  # fp32's range, 8 bits of precision

g = torch.tensor([1e-3, 1e-6, 1e-8, 1e-10])
assert g.to(torch.float16).tolist()[2:] == [0.0, 0.0]                         # two gradients vanish in fp16
assert (g.to(torch.bfloat16) != 0).all()                                      # none vanish in bf16
scaled = (g * 2 ** 16).to(torch.float16)                                      # loss scaling: multiply, then cast
assert (scaled != 0).all() and torch.allclose(scaled.float() / 2 ** 16, g, rtol=0.02)
assert torch.isinf(torch.tensor(2.0 * 2 ** 16).to(torch.float16))             # but a gradient of 2 now overflows

torch.manual_seed(0)                                      # gradients from a real backward pass, mean loss
layers, head = [torch.nn.Linear(512, 512) for _ in range(6)], torch.nn.Linear(512, 10)
hidden, out = [], torch.randn(4096, 512)
for layer in layers:
    out = torch.relu(layer(out))
    out.retain_grad()
    hidden.append(out)
torch.nn.functional.cross_entropy(head(out), torch.randint(0, 10, (4096,))).backward()
act_grads = torch.cat([t.grad.flatten() for t in hidden])
act_grads = act_grads[act_grads != 0]                     # count only gradients the cast destroys, not exact fp32 zeros
lost = {name: (act_grads.to(dtype) == 0).float().mean().item()
        for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16))}
lost["fp16 scaled"] = ((act_grads * 2 ** 16).to(torch.float16) == 0).float().mean().item()
assert 0.08 < lost["fp16"] < 0.12 and lost["bf16"] == 0 and lost["fp16 scaled"] < 1e-5, lost
assert (act_grads.abs() < f16.tiny).float().mean().item() > 0.99              # all below fp16's normal range here

w = torch.tensor(1.0, dtype=torch.bfloat16)
assert w + 1e-3 == w                                                          # a small update to a bf16 weight is lost
master = torch.tensor(1.0)
for _ in range(10):
    master += 1e-3                                                            # the fp32 master copy keeps them
assert master.to(torch.bfloat16).item() == 1.0078125                          # and the 16-bit copy moves one step
```

In a six-layer network with a mean loss over 4,096 examples, about 10% of the activation gradients become zero in fp16 and all of them fall below its normal range, where precision drops; bf16 loses none. Scaling by \(2^{16}\) rescues almost all of them in fp16, but the same scale turns a gradient of 2 into infinity, so fp16 training uses a dynamic scale (PyTorch's `GradScaler`, DeepSpeed's `"loss_scale": 0`) that skips the step and lowers the scale when it overflows. bf16 needs none of this. What it lacks is precision: an update of \(10^{-3}\) to a weight of 1 vanishes, because bf16 numbers near 1 are \(2^{-7}\) apart, so bf16 training still keeps the fp32 master copy: 4 of the 12 optimizer bytes per parameter, beside Adam's two fp32 moments.

## Compute-, memory- or input-bound
A kernel's **arithmetic intensity** is its FLOPs per byte moved to and from GPU memory. The roofline model (Williams, Waterman and Patterson, 2009) says it runs at most at \(\min(\text{peak FLOP/s}, \text{intensity} \times \text{bandwidth})\). On NVIDIA's spec sheet the A100 80GB SXM peaks at 312 TFLOP/s of dense bf16 (624 is the sparse figure) and 2,039 GB/s of memory bandwidth.

```python
PEAK, BW = 312e12, 2.039e12                               # A100 80GB SXM spec sheet: dense bf16 FLOP/s, HBM bytes/s
assert round(PEAK / BW) == 153                            # the ridge: FLOPs per byte needed to stop waiting on memory


def matmul(m, k, n, bytes_per_elem=2):
    """(m x k) @ (k x n): arithmetic intensity, and the best FLOP/s the roofline allows."""
    flops = 2 * m * k * n
    moved = bytes_per_elem * (m * k + k * n + m * n)      # read both inputs, write the output, once
    intensity = flops / moved
    return intensity, min(PEAK, intensity * BW)


big, small = matmul(16384, 4096, 4096), matmul(8, 4096, 4096)
assert round(big[0]) == 1820 and big[1] == PEAK
assert round(small[0], 1) == 8.0 and round(small[1] / 1e12) == 16
add_intensity = 1 / (3 * 2)                               # x + y in bf16: 1 FLOP per element, 3 elements moved
assert round(add_intensity * BW / PEAK * 100, 2) == 0.11  # elementwise work reaches 0.11% of peak at best


def mfu(n_params, tokens_per_s, n_gpus, peak=PEAK):
    return 6 * n_params * tokens_per_s / (n_gpus * peak)  # 6N FLOPs per token: forward 2N, backward 4N


assert mfu(1.3e9, 100_000, 8) == 0.3125
```

A layer's weight matrix applied to 16,384 tokens does about 1,820 FLOPs per byte, far above the ridge of 153, so it is compute-bound. The same matrix applied to 8 tokens does 8 and can reach at most about 16 TFLOP/s, and an elementwise add reaches 0.11% of peak at best. That is why fusing elementwise work into neighbouring kernels (what `torch.compile` and FlashAttention do) pays, and why tiny micro-batches waste a GPU. **MFU** sums it up for a whole job: a hypothetical 1.3B model training at 100,000 tokens a second on 8 A100s reaches 31.25%. It counts \(6N\) FLOPs per token even when checkpointing makes the hardware do about \(8N\), so recomputation cannot inflate it; the PaLM paper, which introduced the term and also counts attention FLOPs in it, reports 46.2% for its 540B model on TPU v4.

| Bound by | Symptom in a profiler | Check |
|---|---|---|
| Compute | Large matrix products fill the timeline | MFU already high; only less work or lower precision helps |
| Memory bandwidth | Elementwise, normalization and softmax kernels take a large share | Fuse kernels; their FLOP/s sit far below the ridge |
| Input | The GPU idles between steps waiting for the next batch | Rerun with one batch reused; if steps get faster, the data loader is the limit |
| Communication | Steps slow down as GPUs are added, at the same work per GPU | Compare step time on 1 GPU and on \(N\) with the same micro-batch |

## Pitfalls
| Problem | Naive fix | How it breaks | Real fix | How it's measured |
|---|---|---|---|---|
| Out of memory on a 7B model before the first step | Halve the batch | Model states (112 GB here) do not depend on the batch | ZeRO-3 / FSDP, offload, or train adapters only ([LoRA](../lora-explained/)) | The calculator above, then `torch.cuda.max_memory_allocated` |
| Out of memory at long sequences | More GPUs under ZeRO-3 | Sharding does not split one GPU's activations | Activation checkpointing, fused attention, smaller micro-batch with accumulation | Peak memory against micro-batch size |
| fp16 loss turns NaN or stops falling | Lower the learning rate | Gradients underflow (about 10% here) or a fixed scale overflows | bf16 where the GPU supports it; dynamic loss scaling for fp16 | Logged loss scale and count of skipped steps |
| Results change with the GPU count | Keep the per-GPU batch fixed | The effective batch, and so the learning-rate fit, changed | Fix the global batch with accumulation; retune if it must change | Log micro-batch × accumulation × GPUs |
| Accumulation trains a different model | Trust the identity | BatchNorm statistics come from each micro-batch (about 65% gradient gap here) | LayerNorm or GroupNorm, or SyncBatchNorm across GPUs | Compare accumulated and full-batch gradients, as above |
| More GPUs, little speed-up | Add more | Communication or the data loader, not compute, was the limit | Overlap, accumulation with `no_sync()`, hybrid sharding, more loader workers | MFU and step time against GPU count |

## What this does not show
Everything here ran on a CPU. The memory and communication numbers are formulas: they ignore temporary buffers, memory fragmentation, the gathered layer that ZeRO-3 holds at any moment, and the dropout masks some models do not have, and the time estimates rest on an assumed network bandwidth and an assumed 50% of peak. The roofline uses spec-sheet peaks, which real kernels do not reach. The fp16 result is one untrained network whose gradients were computed in fp32 and then cast, not computed in fp16; how many gradients underflow depends on the model, the loss and the stage of training. Nothing here measures FSDP or DeepSpeed themselves, so no claim is made about their speed.

What stays open is the choice for a given model and cluster: the same memory can go to a larger micro-batch, less recomputation or less sharding, and which wins depends on the measured step time on that hardware, which is the first number to collect before turning any of these knobs.

Sources: model-state memory and communication volume follow Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" ([SC 2020](https://arxiv.org/abs/1910.02054)); activation memory and recomputation overhead Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" ([2022](https://arxiv.org/abs/2205.05198)); loss scaling and fp32 master weights Micikevicius et al., "Mixed Precision Training" (ICLR 2018); MFU Chowdhery et al., "PaLM" (2022); the roofline Williams, Waterman and Patterson (Communications of the ACM, 2009). FSDP as inspired by ZeRO stage 3 follows the [PyTorch FSDP documentation](https://docs.pytorch.org/docs/2.11/fsdp.html); `fully_shard` and `MixedPrecisionPolicy` follow the [PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), the config keys the DeepSpeed documentation, and the A100 figures NVIDIA's A100 datasheet.
