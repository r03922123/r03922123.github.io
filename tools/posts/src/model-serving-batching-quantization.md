---
title: Serving a vision model: dynamic batching, int8 quantization and export formats
summary: What the FastAPI, ONNX, TensorRT, Triton and quantization lines in an ML engineering job ad mean for a CNN or detector under a latency budget. A queue simulation shows where one-at-a-time serving falls over and what a dynamic batcher's two knobs trade; a toy model shows why int8 weights want one scale per channel; and small PyTorch checks show what exporting a model captures and what it refuses.
group: ML systems and design
order: 8
tags: serving · inference · dynamic batching · quantization · ONNX · TensorRT · Triton
---
Give a model a forward pass that costs 5 ms plus 0.5 ms per example, and a server that runs one request at a time can never answer more than about 182 requests a second. In the simulation below, at 250 requests a second its median wait is already 15 seconds after 80 seconds of traffic and still growing, while the same model behind a dynamic batcher with a maximum batch of 32 answers 99% of requests within 13 ms. Training a detector ends with a checkpoint and a validation score. Serving it adds a **latency budget** (say, 99% of requests answered within 50 ms, written p99 < 50 ms) and a cost per request, and most of the tools named in job ads exist to meet those two numbers.

This post builds three pieces of evidence. I simulate a request queue to show the throughput and tail-latency trade of **dynamic batching**, the scheduling that Triton Inference Server and similar model servers do. I quantize a small trained network to int8 to show why one scale per output channel beats one scale per tensor. And I run small PyTorch checks of what exporting a model (TorchScript, `torch.export`, then ONNX or TensorRT) captures and what it refuses. Fan-out, caching and sharding are in [ML infrastructure](../ml-infrastructure/); the language-model case, with its KV cache and continuous batching, is in [LLM inference](../llm-inference/).

## The words in the job ad
| Term | What it is | What it buys |
|---|---|---|
| FastAPI | A Python web framework: parses the HTTP request, validates the input, calls the model, returns JSON | The service boundary, not speed |
| TorchScript, `torch.export` | Ways to capture a PyTorch model as a graph that runs without the Python source | A fixed artifact other runtimes can load and optimize |
| ONNX | An open file format for computation graphs, with ONNX Runtime as a common engine to run them | One export that several runtimes accept; graph-level fusions |
| TensorRT | NVIDIA's compiler that turns a graph into an engine for one GPU, picking kernels by timing them | Fused kernels, FP16 and INT8 on NVIDIA GPUs |
| Triton Inference Server | A model server: queues requests, batches them, runs several models and copies of a model on one machine | Dynamic batching and GPU sharing without writing a scheduler |
| Quantization | Storing weights (and often activations) in fewer bits, usually int8 | Smaller models, and faster ones where int8 kernels exist |

A typical layout puts a thin web service in front for authentication and input checks, and sends the tensor to a model server that batches. Each layer has its own way to go wrong, and the sections below take them in order of cost.

## Dynamic batching
A GPU forward pass has a cost that does not depend on the batch (launching kernels, reading the weights once, the Python and driver overhead) and a cost per example. The simplest model is \(t(b) = a + c\,b\). With \(a = 5\) ms and \(c = 0.5\) ms, one example takes 5.5 ms, so one-at-a-time serving tops out at \(1000 / 5.5 \approx 182\) requests a second, while batches of 32 take 21 ms and can carry about 1,524 a second. The ratio of fixed to per-example cost is what makes batching pay; it is large on GPUs and for small inputs.

Triton's dynamic batcher has two main knobs. `max_batch_size` caps the batch. `max_queue_delay_microseconds` lets the batcher hold requests for up to that long when it cannot yet form a full batch; the default is 0, meaning it does not wait. When the model is busy, requests queue anyway, and when it frees up the batcher takes what is waiting. The simulation below follows that rule for one model instance: Poisson arrivals, a batch launches when it is full or when its oldest request has waited the delay, and never before the previous batch finishes.

```python
import numpy as np

A, C = 5.0, 0.5                                          # ms: fixed cost per batch, cost per example


def simulate(rate, max_batch, delay_ms, n=20_000, seed=0):
    """One model instance. Returns p50 and p99 latency (ms), mean batch size, fraction of time busy."""
    rng = np.random.default_rng(seed)
    arrive = np.cumsum(rng.exponential(1000 / rate, n))  # Poisson arrivals, in ms
    latency, sizes = np.empty(n), []
    i, free = 0, 0.0
    while i < n:
        start = max(free, arrive[i])                     # model idle and a request waiting
        deadline = max(start, arrive[i] + delay_ms)      # oldest request may wait this long
        j = i + max_batch - 1
        launch = max(start, arrive[j]) if j < n and arrive[j] <= deadline else deadline
        k = min(int(np.searchsorted(arrive, launch, side="right")), i + max_batch)
        free = launch + A + C * (k - i)
        latency[i:k] = free - arrive[i:k]
        sizes.append(k - i)
        i = k
    busy = (A * len(sizes) + C * n) / (free - arrive[0])
    return np.percentile(latency, 50), np.percentile(latency, 99), np.mean(sizes), busy


assert round(1000 / (A + C)) == 182 and round(32_000 / (A + 32 * C)) == 1524   # capacity, requests per second
assert A + 32 * C == 21 and 20_000 / 250 == 80                                  # a full batch in ms; seconds of traffic at 250/s

table = {}
for rate in (50, 170, 250, 1000, 1400, 1600):
    u50, u99, _, _ = simulate(rate, 1, 0)
    b50, b99, mean_b, busy = simulate(rate, 32, 0)
    table[rate] = (round(u50), round(u99), round(b50), round(b99), round(mean_b, 1), round(100 * busy))
    print(rate, table[rate])

assert table == {50: (6, 15, 6, 11, 1.0, 27), 170: (33, 173, 8, 12, 1.3, 72),
                 250: (15296, 30219, 9, 13, 1.7, 87), 1000: (45073, 89226, 16, 26, 10.1, 100),
                 1400: (47910, 94847, 26, 46, 23.8, 100), 1600: (48797, 96604, 400, 747, 31.6, 100)}
```

| Requests per second | One at a time: p50 / p99 (ms) | Max batch 32: p50 / p99 (ms) | Mean batch | Time busy |
|---|---|---|---|---|
| 50 | 6 / 15 | 6 / 11 | 1.0 | 27% |
| 170 | 33 / 173 | 8 / 12 | 1.3 | 72% |
| 250 | 15,296 / 30,219 | 9 / 13 | 1.7 | 87% |
| 1,000 | 45,073 / 89,226 | 16 / 26 | 10.1 | 100% |
| 1,400 | 47,910 / 94,847 | 26 / 46 | 23.8 | 100% |
| 1,600 | 48,797 / 96,604 | 400 / 747 | 31.6 | 100% |

The table shows two cliffs. One-at-a-time serving is fine at 50 requests a second, already has a 173 ms p99 at 170, close to its capacity, where queues grow long, and past 182 its queue grows without bound: the multi-second numbers are not a latency but a backlog that grows the longer the run lasts. The batcher survives because the batch size adapts to load. At 1,000 requests a second the model is busy all the time, yet p99 is 26 ms, because each batch carries about 10 requests and spreads the fixed cost over them. "100% busy" is therefore not the alarm it would be without batching. Batching moves the cliff rather than removing it: at 1,600 requests a second, above the 1,524 that full batches can carry, the batched p99 reaches 747 ms and keeps climbing.

The delay knob trades latency for idle time:

```python
d50_low, _, _, _ = simulate(50, 32, 5)
_, _, mean0, busy0 = simulate(250, 32, 0)
d50, d99, mean5, busy5 = simulate(250, 32, 5)
assert round(d50_low) == 10 and round(table[50][2]) == 6        # light load: the wait is pure added latency
assert (round(d50), round(d99)) == (10, 13)                      # moderate load: p99 barely moves...
assert (round(mean0, 1), round(mean5, 1)) == (1.7, 2.3)          # ...batches grow...
assert (round(100 * busy0), round(100 * busy5)) == (87, 67)      # ...and the model is idle a third of the time
```

At 50 requests a second, a 5 ms delay raises the median from 6 to 10 ms and buys nothing, since there is nobody to wait for. At 250 a second the same delay leaves p99 at 13 ms, grows the mean batch from 1.7 to 2.3 and cuts busy time from 87% to 67%. On one instance with no other work that headroom is wasted, which is why Triton's documentation says to start with the default (no delay), measure, and only then try a larger batch or a non-zero delay. The headroom matters when the GPU is shared with other models or must absorb bursts.

## int8 quantization: one scale per channel
Symmetric int8 quantization stores each weight as an integer \(q = \text{round}(w / s)\) in \([-127, 127]\) and keeps a float scale \(s = \max|w| / 127\). The weight comes back as \(q\,s\), with a rounding error of at most \(s / 2\). **Per-tensor** quantization uses one \(s\) for a whole layer; **per-channel** uses one \(s\) per output channel (a row of a linear layer, a filter of a convolution). One channel with large weights inflates a per-tensor \(s\) for every other channel.

Such channels are not exotic. Folding a batch norm into the convolution before it, which export tools do (below), multiplies each output channel's weights by its own \(\gamma / \sqrt{\sigma^2 + \epsilon}\), and Nagel et al. (2019) report per-channel weight ranges in a folded MobileNetV2 that differ by orders of magnitude. I train a small classifier, then create one such channel without changing the function: ReLU satisfies \(\text{relu}(kx) = k\,\text{relu}(x)\) for \(k > 0\), so multiplying one hidden unit's incoming weights and bias by 100 and dividing its outgoing weights by 100 gives the same outputs.

```python
import copy
import torch
from sklearn.datasets import make_classification

torch.manual_seed(0)
X, y = make_classification(n_samples=4000, n_features=20, n_informative=12, n_classes=4,
                           n_clusters_per_class=1, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(300):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(model(Xtr), ytr).backward()
    opt.step()


def quantize(w, per_channel):
    """Symmetric int8, returned dequantized (q * s) so the float model can run it."""
    amax = w.abs().amax(dim=1, keepdim=True) if per_channel else w.abs().max()
    s = amax / 127
    return torch.clamp(torch.round(w / s), -127, 127) * s


def int8_copy(m, per_channel):
    q = copy.deepcopy(m)
    with torch.no_grad():
        for layer in (q[0], q[2]):
            layer.weight.copy_(quantize(layer.weight, per_channel))
    return q


@torch.no_grad()
def evaluate(reference, m):
    out, ref = m(Xte), reference(Xte)
    return (out.argmax(1) == yte).float().mean().item(), ((out - ref).norm() / ref.norm()).item()


outlier = copy.deepcopy(model)
with torch.no_grad():
    outlier[0].weight[0] *= 100
    outlier[0].bias[0] *= 100
    outlier[2].weight[:, 0] /= 100

acc32, _ = evaluate(model, model)
acc_out, err_out = evaluate(model, outlier)
assert 0.96 < acc32 < 0.98 and acc_out == acc32 and err_out < 1e-6   # same function, new weight ranges
res = {(name, pc): evaluate(m, int8_copy(m, pc)) for name, m in [("plain", model), ("outlier", outlier)]
       for pc in (False, True)}
for key, (acc, err) in res.items():
    print(key, f"accuracy {acc:.3f}, output error {err:.3f}")

assert all(abs(res["plain", pc][0] - acc32) < 0.005 for pc in (False, True))   # no outlier: int8 is free
assert 0.012 < res["plain", False][1] < 0.014 and res["plain", True][1] < 0.01
assert 0.70 < res["outlier", False][0] < 0.75 and res["outlier", False][1] > 0.7
assert acc32 - res["outlier", True][0] < 0.01 and 0.075 < res["outlier", True][1] < 0.095
```

| Weights | float32 accuracy | int8 per-tensor: accuracy, output error | int8 per-channel: accuracy, output error |
|---|---|---|---|
| As trained | about 97% | about 97%, about 1.3% | about 97%, under 1% |
| One channel scaled up 100× | about 97% (same function) | about 72%, over 70% | about 97% (under 1 point lower), about 8% |

Without the outlier, int8 costs nothing measurable on this task: about 1.3% output error per-tensor and under 1% per-channel, with accuracy unchanged. With the outlier, per-tensor quantization drops accuracy from about 97% to about 72%, because the other 63 rows share a scale 100 times too coarse and most of their weights round to a handful of levels. Per-channel brings accuracy back within a point. Its output error is still about 8%, not 1%: the second layer's column for that unit was divided by 100, and per-output-channel scales cannot help a small *input* column. Equalizing the ranges across the two layers (Nagel et al.'s cross-layer equalization) or scaling by activation statistics (AWQ, in [LLM inference](../llm-inference/)) targets exactly that.

PyTorch's own quantized tensors agree with the hand-written version, and an int8 element is a quarter of the size of a float32 one:

```python
w = outlier[0].weight.detach()
scales = (w.abs().amax(dim=1) / 127).double()
q = torch.quantize_per_channel(w, scales, torch.zeros(w.shape[0], dtype=torch.long), 0, torch.qint8)
assert torch.equal(q.dequantize(), quantize(w, per_channel=True))
assert q.int_repr().element_size() * 4 == w.element_size()
```

All of this quantizes weights only. A real int8 deployment also quantizes activations, which needs their ranges: measured once on calibration batches (static quantization), or computed per batch at run time (dynamic quantization). A speed-up needs int8 kernels on the target hardware; the code above dequantizes back to float and is no faster.

## What an export format buys
An exported model is a graph of tensor operations with the Python removed. Capture is what makes it portable, and it is also where bugs hide, because Python control flow and Python integers do not survive capture the way they look in the source. PyTorch marks TorchScript (`torch.jit.trace`, `torch.jit.script`) as deprecated since 2.10 and points to `torch.export`, which is also what `torch.onnx.export` uses by default since 2.9. The two capture styles fail differently on a branch that depends on the data:

```python
import warnings


class Gate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:                  # depends on the values, not the shape
            return x * 2
        return -x


pos, neg = torch.ones(2, 3), -torch.ones(2, 3)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    traced = torch.jit.trace(Gate(), pos)
assert torch.equal(traced(neg), neg * 2) and torch.equal(Gate()(neg), -neg)   # trace froze one branch

try:
    torch.export.export(Gate(), (pos,))
    raise AssertionError("export should refuse a data-dependent branch")
except Exception as e:
    assert "data-dependent" in str(e)

net = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU())
fixed = torch.export.export(net, (torch.randn(4, 8),)).module()
try:
    fixed(torch.randn(16, 8))
    raise AssertionError("a graph exported at batch 4 should reject batch 16")
except Exception:
    pass
batch = torch.export.Dim("batch", min=1, max=64)
dynamic = torch.export.export(net, (torch.randn(4, 8),), dynamic_shapes={"input": {0: batch}}).module()
x = torch.randn(16, 8)
assert torch.allclose(dynamic(x), net(x))
```

Tracing records the operations one example happened to run, so it silently returns the wrong branch for negative inputs; `torch.export` refuses instead, and the fix is to rewrite the branch with `torch.where` or `torch.cond`. Shapes are fixed unless declared otherwise: the graph exported at batch 4 rejects batch 16, and declaring the batch dimension dynamic, with a range, makes it accept any batch the batcher sends. TensorRT asks the same question in its own terms: an **optimization profile** gives a minimum, an optimum and a maximum shape for each dynamic input, and kernels are tuned for the optimum.

Once captured, the graph can be rewritten. **Operator fusion** merges operations so intermediate results do not make a round trip through memory. The simplest case removes an operation entirely: in inference mode a batch norm is a per-channel affine map, so it folds into the convolution before it. ONNX Runtime lists this as "Conv BatchNorm Fusion" among its basic graph optimizations.

```python
conv, bn = torch.nn.Conv2d(3, 8, 3, bias=False), torch.nn.BatchNorm2d(8)
with torch.no_grad():                                    # statistics a trained model would have
    bn.running_mean.uniform_(-1, 1)
    bn.running_var.uniform_(0.5, 2)
    bn.weight.uniform_(0.5, 1.5)
    bn.bias.uniform_(-1, 1)
bn.eval()
scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # one factor per output channel
fused = torch.nn.Conv2d(3, 8, 3)
with torch.no_grad():
    fused.weight.copy_(conv.weight * scale[:, None, None, None])
    fused.bias.copy_(bn.bias - bn.running_mean * scale)
img = torch.randn(2, 3, 16, 16)
assert torch.allclose(fused(img), bn(conv(img)), atol=1e-5)   # two operations become one
```

That `scale` is also where the per-channel weight ranges of the previous section come from. The export path for an image model, not run here (ONNX and TensorRT are not installed in this site's build), looks like this:

```plaintext
# PyTorch -> ONNX, batch dimension dynamic
batch = torch.export.Dim("batch", min=1, max=32)
torch.onnx.export(model.eval(), (torch.randn(1, 3, 224, 224),), "model.onnx",
                  input_names=["images"], dynamic_shapes=({0: batch},))      # dim 0 of the first input

# ONNX -> TensorRT engine, with an optimization profile
trtexec --onnx=model.onnx --saveEngine=model.plan \
        --minShapes=images:1x3x224x224 --optShapes=images:16x3x224x224 --maxShapes=images:32x3x224x224

# Triton: models/classifier/config.pbtxt (and the engine at models/classifier/1/model.plan)
platform: "tensorrt_plan"
max_batch_size: 32
input  [ { name: "images", data_type: TYPE_FP32, dims: [ 3, 224, 224 ] } ]
output [ { name: "logits", data_type: TYPE_FP32, dims: [ 1000 ] } ]
dynamic_batching { }
instance_group [ { count: 1, kind: KIND_GPU } ]
```

In Triton's config, `dims` leaves out the batch dimension, which `max_batch_size` adds. A TensorRT engine is, by default, tied to the TensorRT version and the GPU it was built on, so it is built per target, in CI or at deploy time, not copied from a laptop. ONNX files are portable; engines are not.

## The web layer
Two rules cover most of what goes wrong in front of the model. Load the model once when the process starts, never per request: reading weights from disk and moving them to a GPU can take far longer than a forward pass. FastAPI's documented way is a lifespan function:

```plaintext
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    pixels: list[float]

models = {}

@asynccontextmanager
async def lifespan(app):
    models["clf"] = load_model("model.onnx")      # once per process, before the first request
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(item: Item):                          # plain def: FastAPI runs it in a threadpool
    return {"label": models["clf"].predict(item.pixels)}
```

The second rule: `async def` does not make CPU-bound work concurrent. An async endpoint runs on the event loop, and a forward pass inside it holds the loop until it finishes, so every other request, health checks included, waits. FastAPI runs plain `def` endpoints in a threadpool for this reason, and its documentation points to several worker processes for CPU-bound work. The check below stands a busy loop in for a 20 ms forward pass:

```python
import asyncio
import time


def forward_pass():                       # CPU-bound stand-in: 20 ms of work, holds the thread
    t = time.perf_counter()
    while time.perf_counter() - t < 0.02:
        pass


async def cpu_endpoint():
    forward_pass()


async def io_endpoint():                  # e.g. awaiting a call to a model server
    await asyncio.sleep(0.02)


async def eight_requests(endpoint):
    t = time.perf_counter()
    await asyncio.gather(*(endpoint() for _ in range(8)))
    return time.perf_counter() - t


cpu_s, io_s = asyncio.run(eight_requests(cpu_endpoint)), asyncio.run(eight_requests(io_endpoint))
assert cpu_s >= 0.16 and io_s < 0.1        # CPU work runs one after another; waiting overlaps
```

Eight concurrent requests take at least 160 ms when each does 20 ms of CPU work in an `async def`, eight times one request, while eight that await I/O finish together in a little over 20 ms. This is also an argument for the two-layer layout: if the web service only awaits a call to a model server, async works as intended, and the batching happens where the GPU is.

## Pitfalls
| Problem | Naive fix | How it breaks | Real fix | How it's measured |
|---|---|---|---|---|
| p99 grows with traffic | Add GPUs for each rise in load | One-at-a-time serving wastes the fixed cost on every request; past 182 requests a second here the queue never drains | Dynamic batching, then more instances | p50 and p99 against offered load, from a load generator such as Triton's Performance Analyzer |
| Batcher adds latency at low load | Set a large `max_queue_delay` "for throughput" | +4 ms median at 50 requests a second, nothing gained on one instance | Start with delay 0; add delay only if measured | The delay table above, on real traffic |
| int8 model loses accuracy | Quantize per tensor, blame int8 | One wide channel coarsens all others: about 97% to 72% here | Per-channel weight scales; calibrate activations on real inputs | Task metric on held-out data, not only output error |
| Export gives wrong answers on some inputs | Trust the trace because it ran | Tracing froze one branch | `torch.export`, rewrite branches with `torch.where` or `torch.cond` | Compare exported vs eager outputs on varied inputs |
| Export rejects larger batches | Export at the batch size used in testing | Shapes were captured as constants | Declare the batch dimension dynamic; a TensorRT profile range | Run the artifact at the smallest and largest batch |
| Server slow and unresponsive | Load the model in the handler, or run inference in `async def` | Every request pays the load; one forward pass blocks the loop | Lifespan loading; `def` endpoints, worker processes or a model server | Health-check latency under load |

## What this does not show
The batching numbers come from a cost model, not a GPU: a real \(t(b)\) is not linear (it is flat while the GPU is underused, then steepens, and jumps where a kernel changes tactic), arrivals are burstier than Poisson, and the simulation has one model instance, no network time and no preprocessing, which on image models can cost as much as the forward pass. The quantization result is one small MLP on synthetic data with weight-only, simulated int8, and the outlier was made by hand; it shows the mechanism, not the size of the effect on a real CNN. Nothing here runs ONNX Runtime or TensorRT, so no claim is made about their speed.

What stays open is how these choices interact on real hardware: whether a model's latency budget is better spent on a larger batch, a lower precision or more instances depends on the measured \(t(b)\) curve of that model on that GPU, and that curve is the first thing to measure before choosing any of the knobs above.

Sources: dynamic batching and model configuration follow the [Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html); export and quantized tensors the [PyTorch documentation](https://docs.pytorch.org/docs/stable/onnx.html); graph fusions the [ONNX Runtime documentation](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html); optimization profiles, `trtexec` and engine compatibility the TensorRT documentation; the lifespan pattern and `def` versus `async def` the [FastAPI documentation](https://fastapi.tiangolo.com/advanced/events/). Per-channel weight ranges after batch-norm folding and cross-layer equalization are from Nagel et al., "Data-Free Quantization Through Weight Equalization and Bias Correction" (ICCV 2019).
