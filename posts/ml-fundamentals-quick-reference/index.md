# ML fundamentals: a quick reference

Training a neural network rests on a dozen primitives: the loss, autograd, the optimizer and its schedule, activations, initialization, normalization, regularization and, in a transformer, the softmax and the attention mask. Each one can break a model: a learning rate just over \(2/L\) diverges, 30 sigmoid layers pass back no gradient, and forgetting `model.eval()` makes predictions random. This post gives each a few sentences, the equation that matters and a short block of code whose asserts check the claim, with links to longer posts. All the code runs as one script in a few seconds on a laptop CPU. That speed comes from toy cases, such as a quadratic with curvatures 100 and 1 or a batch of 32 random labels, so the asserts show each mechanism, not how large it is in a full-size model.

## Losses: cross-entropy, MSE and KL
A classifier outputs a distribution \(q\) over classes; training pulls it toward the data's distribution \(p\). Cross-entropy splits into a part the model cannot change and a part it can:
\[ H(p, q) = -\sum_i p_i \log q_i = H(p) + \mathrm{KL}(p \,\|\, q) \]
So minimizing cross-entropy minimizes the KL divergence to the data, and with a one-hot label it is \(-\log q_y\), the negative log-likelihood. KL is non-negative, zero only when \(q = p\), and not symmetric ([entropy and information](../entropy-and-information/)). Mean squared error is the same idea for regression: up to constants, the negative log-likelihood of a Gaussian with fixed variance ([regression as maximum likelihood](../regression-as-maximum-likelihood/)).

```python
import math

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(0)
p = rng.dirichlet(np.ones(5))                     # the data's distribution over 5 classes
q = rng.dirichlet(np.ones(5))                     # the model's
entropy = -(p * np.log(p)).sum()
cross_entropy = -(p * np.log(q)).sum()
kl_pq, kl_qp = (p * np.log(p / q)).sum(), (q * np.log(q / p)).sum()

assert np.isclose(cross_entropy, entropy + kl_pq)
assert kl_pq > 0 and kl_qp > 0 and not np.isclose(kl_pq, kl_qp)       # positive, not symmetric
assert np.isclose(F.kl_div(torch.tensor(np.log(q)), torch.tensor(p), reduction="sum").item(), kl_pq)
```

For a sigmoid output \(\hat p = \sigma(z)\), the cross-entropy gradient at the logit is \(\hat p - y\). Squared error on \(\hat p\) multiplies it by \(2\hat p(1 - \hat p)\), which goes to zero whenever the model is confident, right or wrong, so MSE learns slowest from its worst mistakes ([cross-entropy vs MSE](../cross-entropy-vs-mse/)).

```python
z = torch.tensor(-8.0, requires_grad=True)        # a confident mistake: P(y = 1) is about 0.0003
y = torch.tensor(1.0)
F.binary_cross_entropy_with_logits(z, y).backward()
ce_grad, z.grad = z.grad.item(), None
((torch.sigmoid(z) - y) ** 2).backward()
mse_grad, prob = z.grad.item(), torch.sigmoid(z).item()

assert math.isclose(ce_grad, prob - 1, rel_tol=1e-5)
assert math.isclose(mse_grad, 2 * (prob - 1) * prob * (1 - prob), rel_tol=1e-5)
assert abs(ce_grad) > 1000 * abs(mse_grad)
```

## Gradients: what autograd does
Autograd records each forward operation as a node, with the local derivative of its output with respect to each input. The backward pass visits the nodes in reverse topological order and applies the chain rule: a node's gradient is the sum, over the nodes it feeds, of their gradient times the local derivative, so a value used twice gets two contributions. A scalar version fits in a few lines and matches PyTorch to machine precision:

```python
class Value:
    """A scalar that records how it was computed, for reverse-mode differentiation."""

    def __init__(self, data, parents=(), local_grads=()):
        self.data, self.grad = data, 0.0
        self.parents, self.local_grads = parents, local_grads    # d(self) / d(parent), per parent

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    def backward(self):
        order, seen = [], set()

        def visit(node):                          # topological order: inputs before outputs
            if id(node) not in seen:
                seen.add(id(node))
                for parent in node.parents:
                    visit(parent)
                order.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(order):              # a node's gradient is complete before it is passed back
            for parent, local in zip(node.parents, node.local_grads):
                parent.grad += local * node.grad  # chain rule, summed over every use of parent


a, b = Value(0.7), Value(-1.3)
out = (a * b + a).tanh() * b + a * a              # a is used four times
out.backward()

ta, tb = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.7, -1.3))
(torch.tanh(ta * tb + ta) * tb + ta * ta).backward()
assert math.isclose(a.grad, ta.grad.item(), rel_tol=1e-12)
assert math.isclose(b.grad, tb.grad.item(), rel_tol=1e-12)

w = torch.tensor(3.0, requires_grad=True)
for _ in range(2):
    (w * w).backward()
assert w.grad.item() == 12.0                      # 2w, twice: .grad accumulates until zeroed
```

Reverse mode yields the gradient for every parameter in one backward pass, at a small multiple of the forward cost, which suits one scalar loss and millions of parameters. It needs the forward pass's intermediate values, so training holds activations that inference discards; activation checkpointing recomputes them instead. And since `.grad` accumulates, a training loop zeroes it every step. [ML coding from scratch](../ml-coding-from-scratch/) builds whole models this way.

## Optimizers: SGD, momentum, Adam and AdamW
Gradient descent steps \(\theta \leftarrow \theta - \eta g\), with \(\eta\) below \(2/L\) for the steepest curvature \(L\), so progress along flat directions crawls. **Momentum** keeps a running sum, \(v \leftarrow \beta v + g\), and steps \(\theta \leftarrow \theta - \eta v\): oscillations across steep directions cancel and steps along flat ones add up. With curvatures 100 and 1 it converges five times sooner at the same learning rate, and more than twice as fast as plain descent at its best stable learning rate (momentum's effective step is larger, but plain descent cannot take a larger one without diverging):

```python
def steps_to_converge(lr, beta=0.0, curvature=(100.0, 1.0), tol=1e-6, max_steps=5_000):
    """Heavy-ball descent on 0.5 * sum(c * x**2) from x = 1; None if it diverges or runs out of steps."""
    c = np.array(curvature)
    x, velocity = np.ones_like(c), np.zeros_like(c)
    for step in range(max_steps):
        if np.abs(x).max() < tol:
            return step
        if np.abs(x).max() > 1e6:
            return None
        velocity = beta * velocity + c * x        # c * x is the gradient
        x = x - lr * velocity
    return None


assert steps_to_converge(lr=0.01) > 5 * steps_to_converge(lr=0.01, beta=0.9)   # 1375 steps against 263
best_plain = min(s for s in (steps_to_converge(lr) for lr in np.linspace(0.001, 0.0199, 60)) if s)
assert best_plain > 2 * steps_to_converge(lr=0.01, beta=0.9)                   # about 700 against 263
```

**Adam** (Kingma and Ba, 2015) keeps running means of the gradient and of its square, corrects both for starting at zero, and divides:
\[ m \leftarrow \beta_1 m + (1 - \beta_1) g, \qquad v \leftarrow \beta_2 v + (1 - \beta_2) g^2, \qquad \theta \leftarrow \theta - \eta\, \frac{m / (1 - \beta_1^t)}{\sqrt{v / (1 - \beta_2^t)} + \epsilon} \]
Each coordinate is scaled by its own gradient size, so it moves about \(\eta\) per step whatever the gradient's scale; the first step is exactly \(\eta\) times the gradient's sign, up to \(\epsilon\).

```python
def adam_step(theta, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    """One AdamW step, plain Adam when wd = 0. t counts steps from 1."""
    theta = theta - lr * wd * theta                       # decoupled weight decay
    m = b1 * m + (1 - b1) * g                             # running mean of gradients
    v = b2 * v + (1 - b2) * g * g                         # running mean of squared gradients
    m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)   # undo the pull toward the zero start
    return theta - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


A = rng.normal(size=(6, 6))
A, c = A @ A.T + np.eye(6), rng.normal(size=6)            # loss 0.5 x'Ax - c'x, gradient Ax - c
theta0 = rng.normal(size=6)
w = torch.tensor(theta0, requires_grad=True)
opt = torch.optim.AdamW([w], lr=0.05, weight_decay=0.1)
theta, m, v = theta0, np.zeros(6), np.zeros(6)
for t in range(1, 51):
    opt.zero_grad()
    (0.5 * w @ torch.tensor(A) @ w - torch.tensor(c) @ w).backward()
    opt.step()
    theta, m, v = adam_step(theta, A @ theta - c, m, v, t, lr=0.05, wd=0.1)
assert np.allclose(theta, w.detach().numpy(), atol=1e-10)   # matches PyTorch step for step

g = np.array([1e-4, 3.0, -250.0])
first = adam_step(np.zeros(3), g, np.zeros(3), np.zeros(3), t=1, lr=0.01)[0]
assert np.allclose(first, -0.01 * np.sign(g), rtol=1e-3)    # same size step for all three
```

**AdamW** (Loshchilov and Hutter, 2019) changes how weight decay meets that scaling. An L2 penalty \(\tfrac{\lambda}{2}\lVert\theta\rVert^2\) adds \(\lambda\theta\) to the gradient, which Adam then divides by \(\sqrt{\hat v}\): a weight with no loss gradient shrinks at the full rate \(\eta\) per step, one with large gradients hardly at all. AdamW decays the weights directly, by about \(\eta\lambda\theta\) per step for every weight. With plain SGD the two are the same update.

```python
def decay_effect(decoupled, wd=0.1, lr=0.01, steps=50):
    """How far weight decay moves two weights that start at 1: one gets no loss gradient, one a large one."""
    def run(wd):
        theta, m, v = np.ones(2), np.zeros(2), np.zeros(2)
        for t in range(1, steps + 1):
            g = np.array([0.0, 10.0])                     # gradient of the loss alone
            if decoupled:                                 # AdamW
                theta, m, v = adam_step(theta, g, m, v, t, lr=lr, wd=wd)
            else:                                         # Adam with an L2 penalty in the loss
                theta, m, v = adam_step(theta, g + wd * theta, m, v, t, lr=lr)
        return theta
    return run(0.0) - run(wd)


l2, adamw = decay_effect(decoupled=False), decay_effect(decoupled=True)
assert l2[0] > 0.4 and abs(l2[1]) < 1e-3       # Adam + L2: all or nothing, depending on the gradient
assert 1 < adamw[0] / adamw[1] < 1.5           # AdamW: about the same for both
```

AdamW is the default for transformers; SGD with momentum and a tuned schedule is still common for convolutional networks.

## Learning-rate schedules: warmup and cosine decay
The learning rate is usually the most important hyperparameter, and most runs change it during training. The standard transformer recipe is linear **warmup** to a peak over the first few hundred to few thousand steps, then **cosine decay** to a floor at step \(T\):
\[ \eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi (t - t_w)}{T - t_w}\right), \qquad t \ge t_w \]
Warmup keeps the first updates small while Adam's moment estimates rest on few gradients and a fresh network can produce badly scaled ones. Decay lets the weights settle into a minimum instead of bouncing around it. Warmup-stable-decay schedules hold the peak and decay only at the end, which makes a run easy to extend.

```python
def warmup_cosine(step, peak=3e-4, floor=3e-5, warmup=1_000, total=10_000):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / (total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


lrs = np.array([warmup_cosine(s) for s in range(10_001)])
assert lrs.argmax() == 999 and math.isclose(lrs[1000], 3e-4)
assert np.all(np.diff(lrs[:1000]) > 0) and np.all(np.diff(lrs[1000:]) < 0)
assert math.isclose(lrs[5_500], (3e-4 + 3e-5) / 2) and math.isclose(lrs[-1], 3e-5)
```

## Activation functions
Without nonlinearities, stacked linear layers collapse into one linear map. The choices differ mainly in their gradients:

- **Sigmoid and tanh** saturate: their slope is near zero for large \(|z|\), and the sigmoid's is at most \(1/4\), so stacks of them shrink gradients. They survive as output probabilities and as gates.
- **ReLU**, \(\max(0, z)\), has slope 1 for positive inputs and is cheap. A unit that is negative on every input gets no gradient and never recovers (a dead ReLU, often from too high a learning rate); leaky ReLU keeps a small slope below zero.
- **GELU**, \(z\,\Phi(z)\) with \(\Phi\) the normal CDF, and **SiLU** (Swish), \(z\,\sigma(z)\), are smooth ReLUs with a small dip below zero: GELU in BERT and GPT-2, SiLU inside the SwiGLU layers of LLaMA-style models.

```python
z = torch.linspace(-8, 8, 1601, dtype=torch.float64)
s = torch.sigmoid(z)
assert math.isclose((s * (1 - s)).max().item(), 0.25)           # the slope peaks at z = 0
assert (s * (1 - s))[z.abs() > 5].max() < 0.007                 # saturated: almost no gradient
assert torch.allclose(torch.tanh(z), 2 * torch.sigmoid(2 * z) - 1)    # tanh is a rescaled sigmoid
normal_cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
assert torch.allclose(F.gelu(z), z * normal_cdf)
assert torch.allclose(F.silu(z), z * torch.sigmoid(z))
assert F.gelu(z).min() < 0 and F.silu(z).min() < 0 and F.relu(z).min() == 0
```

## Vanishing and exploding gradients
Backpropagation multiplies one Jacobian per layer, so a typical factor below one makes gradients vanish exponentially with depth and one above makes them explode. Four tools keep it near one:

1. **Initialization** that preserves variance: weights with variance \(2/n_{\text{in}}\) for ReLU, which zeroes half its inputs (He et al., 2015), and \(2/(n_{\text{in}} + n_{\text{out}})\), about \(1/n_{\text{in}}\) for a square layer, for tanh (Glorot and Bengio, 2010).
2. **Residual connections**: \(x + f(x)\) has Jacobian \(I + J_f\), an identity path the layers cannot shrink.
3. **Normalization**, which resets the activation scale at every layer (next section).
4. **Gradient clipping**, which rescales the gradient when its norm passes a threshold: a guard against spikes, not a fix for a bad scale.

With weights of variance \(g^2/n\), each ReLU layer multiplies the mean squared activation by \(g^2/2\). Over 30 layers \(g = 1\) shrinks it by \(2^{-30}\), \(g = 2\) grows it by \(2^{30}\), and only \(g = \sqrt 2\) holds it steady. The backward pass multiplies by the same matrices, so gradients follow.

```python
def output_rms(gain, depth=30, width=512):
    """RMS activation after `depth` ReLU layers whose weights have variance gain**2 / width."""
    torch.manual_seed(0)
    h = torch.randn(64, width)
    for _ in range(depth):
        h = torch.relu(h @ (torch.randn(width, width) * gain / math.sqrt(width)))
    return h.pow(2).mean().sqrt().item()


assert output_rms(1.0) < 1e-3                  # vanished
assert 0.3 < output_rms(math.sqrt(2)) < 3      # He initialization: stays of order one
assert output_rms(2.0) > 1e3                   # exploded


def input_gradient_norm(residual, depth=30, width=64):
    torch.manual_seed(0)
    x = torch.randn(32, width, requires_grad=True)
    h = x
    for _ in range(depth):
        f = torch.sigmoid(h @ (torch.randn(width, width) / math.sqrt(width)))
        h = h + f if residual else f
    h.sum().backward()
    return x.grad.norm().item()


assert input_gradient_norm(residual=False) < 1e-12     # 30 sigmoid layers: nothing gets back
assert input_gradient_norm(residual=True) > 1          # the identity path carries it

torch.manual_seed(0)
layer = torch.nn.Linear(10, 1)
(layer(torch.randn(8, 10)) * 1e3).pow(2).mean().backward()          # a gradient spike
raw = torch.cat([p.grad.flatten() for p in layer.parameters()])
norm_before = torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
clipped = torch.cat([p.grad.flatten() for p in layer.parameters()])
assert math.isclose(norm_before.item(), raw.norm().item(), rel_tol=1e-5)
assert torch.allclose(clipped, raw / raw.norm(), atol=1e-6)          # same direction, norm 1
```

## Normalization: BatchNorm and LayerNorm
Both subtract a mean, divide by a standard deviation and apply a learned scale and shift; they differ in where the statistics come from. **BatchNorm** (Ioffe and Szegedy, 2015) normalizes each feature over the batch, so an example's output depends on its batchmates, and inference uses running averages from training. It suits convolutional networks with large batches, not small batches or variable-length sequences. **LayerNorm** (Ba, Kiros and Hinton, 2016) normalizes each example over its own features, the same way in training and inference. Transformers use it or RMSNorm (no mean subtraction), mostly placed before each sublayer, which trains more stably than the original placement after the residual addition ([BatchNorm vs LayerNorm](../batchnorm-vs-layernorm/)).

```python
torch.manual_seed(0)
x = torch.randn(4, 8) * 3 + 5                                  # 4 examples, 8 features
ln = F.layer_norm(x, (8,))                                     # statistics per example (row)
bn = F.batch_norm(x, None, None, training=True)                # statistics per feature (column)
assert torch.allclose(ln.mean(1), torch.zeros(4), atol=1e-5)
assert torch.allclose(ln.var(1, unbiased=False), torch.ones(4), atol=1e-3)
assert torch.allclose(bn.mean(0), torch.zeros(8), atol=1e-5)
assert torch.allclose(bn.var(0, unbiased=False), torch.ones(8), atol=1e-3)

x_other = x.clone()
x_other[1:] = torch.randn(3, 8)                                # same first example, new batchmates
assert torch.allclose(F.layer_norm(x_other, (8,))[0], ln[0])
assert not torch.allclose(F.batch_norm(x_other, None, None, training=True)[0], bn[0])
```

## Overfitting and regularization
Overfitting is learning noise that does not recur: training loss falls while validation loss turns up. The levers are more or augmented data, early stopping, weight decay, dropout and smaller models ([overfitting playbook](../overfitting-playbook/), [bias-variance tradeoff](../bias-variance-tradeoff/), [L1 and L2 regularization](../regularization-l1-l2/)).

**Dropout** (Srivastava et al., 2014) zeroes each activation with probability \(p\) in training and scales the survivors by \(1/(1-p)\), which keeps the expectation unchanged, so at inference it does nothing. Forgetting `model.eval()` leaves it on and makes predictions random ([dropout](../dropout/)).

```python
torch.manual_seed(0)
dropout = torch.nn.Dropout(p=0.3)
ones = torch.ones(1_000_000)
out = dropout.train()(ones)
kept = out[out > 0]
assert torch.allclose(kept, torch.full_like(kept, 1 / 0.7))     # survivors are scaled up
assert abs((out == 0).float().mean().item() - 0.3) < 0.005
assert abs(out.mean().item() - 1) < 0.005                       # so the expectation is unchanged
assert torch.equal(dropout.eval()(ones), ones)                   # identity at inference
```

## Softmax and temperature
Temperature rescales the logits before the softmax, \(\mathrm{softmax}(z/T)_i = e^{z_i/T} / \sum_j e^{z_j/T}\). As \(T \to 0\) it collapses onto the largest logit (greedy decoding), as \(T \to \infty\) it becomes uniform, and in between its entropy rises steadily with \(T\). Adding a constant to every logit changes nothing, which is how softmax is computed safely: subtract the maximum first ([stable softmax](../stable-softmax-cross-entropy/)).

Attention has a built-in temperature. If \(q\) and \(k\) have \(d\) independent zero-mean, unit-variance components, \(q \cdot k\) has variance \(d\); for large \(d\) the raw scores saturate the softmax, nearly one-hot with tiny gradients, so they are divided by \(\sqrt d\) ([attention scaling](../attention-scaling/)).

```python
def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=float) / T
    e = np.exp(z - z.max())                       # the shift changes nothing and cannot overflow
    return e / e.sum()


logits = np.array([2.0, 1.0, 0.5, -1.0])
H = [-(pr * np.log(pr)).sum() for pr in (softmax(logits, T) for T in (0.1, 0.5, 1, 2, 10, 100))]
assert all(h1 < h2 for h1, h2 in zip(H, H[1:]))                   # hotter is flatter
assert softmax(logits, T=0.01)[0] > 0.999999                      # T -> 0: the argmax
assert np.allclose(softmax(logits, T=1e4), 0.25, atol=1e-3)       # T -> infinity: uniform
assert np.allclose(softmax(logits + 1000), softmax(logits))       # shift invariance
assert np.allclose(softmax(logits), torch.softmax(torch.tensor(logits), 0).numpy())

d = 256
q, keys = rng.normal(size=(500, d)), rng.normal(size=(500, 64, d))   # 500 queries, 64 keys each
scores = np.einsum("qd,qkd->qk", q, keys)
assert abs(scores.var() / d - 1) < 0.05                           # raw scores have variance d
assert np.mean([softmax(s_).max() for s_ in scores]) > 0.8        # unscaled: nearly one-hot
assert np.mean([softmax(s_ / math.sqrt(d)).max() for s_ in scores]) < 0.2   # scaled: spread out
```

## Encoders, decoders and the attention mask
Encoder and decoder transformers differ mainly in one mask. An **encoder** such as BERT lets every token attend to every other, which suits classification and embeddings. A **decoder** such as GPT masks attention so that position \(t\) sees only positions up to \(t\): it trains on every next-token prediction in parallel and generates one token at a time. **Encoder-decoders** (the original transformer, T5) add cross-attention from the decoder to the encoder, for tasks with a separate input such as translation. The causal mask is also why the KV cache is valid: earlier positions never see later ones, so their keys and values never change ([LLM inference](../llm-inference/), [transformers and LLMs](../transformers-and-llms-key-ideas/)).

```python
torch.manual_seed(0)
n, d = 6, 16
x = torch.randn(1, n, d)
Wq, Wk, Wv = (torch.randn(d, d) / math.sqrt(d) for _ in range(3))


def attention(x, causal):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:                                    # position t may look at positions <= t only
        future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, -1) @ v


def moved(causal):
    """Which output positions change when only the last input token changes."""
    x_edit = x.clone()
    x_edit[0, -1] += 1.0
    return ((attention(x_edit, causal) - attention(x, causal)).abs().amax(-1) > 1e-6)[0].tolist()


for causal in (False, True):
    expected = F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, is_causal=causal)
    assert torch.allclose(attention(x, causal), expected, atol=1e-5)
assert moved(causal=False) == [True] * n                   # encoder: every output sees the edit
assert moved(causal=True) == [False] * (n - 1) + [True]    # decoder: only the last position does
```

## When training goes wrong
Failed runs tend to fall into four patterns:

- **Loss explodes or turns into NaN.** Usually too high a learning rate: on a quadratic with curvature \(L\), gradient descent converges only for \(\eta < 2/L\) and diverges geometrically past it. Also check for a hand-written log of a softmax, half-precision overflow and bad batches.
- **Loss falls, then stalls high.** Learning rate too low, or too high to settle; saturated or dead units; a bug in the labels or the loss.
- **Loss never moves.** Gradients do not reach the parameters: a detached tensor, a frozen layer, an optimizer over the wrong parameters.
- **Training loss falls, validation loss rises.** Overfitting.

The best first test of a training loop is to overfit one small batch: a working model memorizes 32 random labels in a few hundred steps. If yours cannot, the bug is in the code, not the data.

```python
assert steps_to_converge(lr=1.9 / 100) is not None     # just under 2 / L: converges
assert steps_to_converge(lr=2.1 / 100) is None         # just over: diverges

torch.manual_seed(0)
inputs, labels = torch.randn(32, 20), torch.randint(0, 4, (32,))   # random labels: only memorizing works
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
for _ in range(300):
    opt.zero_grad()
    loss = F.cross_entropy(model(inputs), labels)
    loss.backward()
    opt.step()
assert loss.item() < 0.05
```

## The short version
| Topic | Remember |
|---|---|
| Cross-entropy | Entropy plus KL to the data; gradient \(\hat p - y\) at the logit |
| MSE on probabilities | An extra \(\hat p(1-\hat p)\) factor: slow when confidently wrong |
| Autograd | One reverse pass for all parameters; `.grad` accumulates |
| Momentum | Steps along flat directions add up, oscillations cancel |
| Adam, AdamW | About \(\eta\) per coordinate per step; AdamW decays the weights directly |
| Warmup, cosine decay | Small steps at the start, and again at the end |
| Activations | The sigmoid's slope is at most 1/4; ReLU units can die |
| Initialization | Variance \(2/n_{\text{in}}\) for ReLU layers |
| Residual connections | An identity path in every Jacobian |
| BatchNorm, LayerNorm | Statistics over the batch, or over the features |
| Dropout | Scale by \(1/(1-p)\) in training, off at inference |
| Temperature | \(T \to 0\) is argmax, \(T \to \infty\) is uniform |
| Attention scaling | \(q \cdot k\) has variance \(d\), so divide by \(\sqrt d\) |
| Encoder, decoder | Bidirectional or causal mask |

For the statistics side of the toolkit, see [MLE and MAP](../statistical-estimation-mle-map/) and [A/B testing](../ab-testing/); for these pieces in a working model, [LoRA from scratch](/posts/lora-from-scratch/).

The topics follow a list of deep learning fundamentals compiled by [Yuan Meng](https://www.yuan-meng.com/). Simon Prince's [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (MIT Press, 2023), chapters 1 to 9, 11 and 12, treats each in depth. The text and code here are original.
