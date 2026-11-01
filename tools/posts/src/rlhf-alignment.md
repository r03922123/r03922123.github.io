---
title: RLHF and alignment: reward models, PPO and DPO
summary: How a language model is tuned towards answers people prefer: a reward model fit to pairwise comparisons, the KL-regularized objective and its closed-form optimum, PPO as the way classic RLHF optimizes it, DPO as the way to skip the reward model, and a toy world where over-optimizing a good-looking reward model makes answers worse. Every claim is checked with asserts.
group: LLMs and generative AI
order: 7
tags: LLM · alignment · RLHF · reward model · PPO · DPO
---
A model fine-tuned on demonstrations can only be as good as the demonstrations, and writing excellent answers is slow and expensive. Judging answers is much easier: shown two responses to the same prompt, most people can say which one is better even if they could not have written either. Reinforcement learning from human feedback (RLHF) turns those judgements into a training signal. Its modern form was introduced for Atari games and simulated robot control by Christiano et al. (2017), applied to language models for style and summarization (Ziegler et al., 2019; Stiennon et al., 2020), and became the standard recipe for assistants with InstructGPT (Ouyang et al., 2022), where labelers preferred the outputs of a 1.3B tuned model to those of the 175B GPT-3, a model more than a hundred times larger.

This post builds the pieces in order: the reward model, the objective it is optimized under, PPO, and DPO, which removes the reward model altogether. A small simulated world runs through all of it, with one deliberate flaw in the reward model, because the most important lesson of RLHF is what happens when you optimize a reward that is only approximately right.

## Three stages
The pipeline from a pretrained model (see [foundation models](../foundation-models/)) has three stages, each with its own data:

| Stage | Data | What is trained | Loss |
|---|---|---|---|
| 1. Supervised fine-tuning (SFT) | prompts with written responses | the policy, starting from the base model | next-token cross-entropy on response tokens |
| 2. Reward model | prompts with two or more responses, ranked by people | a copy of the SFT model with a scalar head | Bradley–Terry (pairwise logistic) |
| 3. RL fine-tuning | prompts only | the policy, starting from the SFT model | expected reward minus a KL penalty, optimized with PPO |

The SFT model matters twice. It is the starting point for stage 3, and it is the **reference** \(\pi_{\text{ref}}\) that the KL penalty measures distance from. Stage 3 needs no human labels at all: the policy writes answers, the reward model scores them, and the labeled data only enters through the reward model.

## Stage 2: a reward model from comparisons
People are poor at absolute scores (is this a 6 or a 7?) but reasonably consistent at comparisons. The Bradley–Terry model (Bradley and Terry, 1952) connects the two: each response \(y\) to prompt \(x\) has a hidden score \(r(x, y)\), and the chance that \(y_w\) is preferred to \(y_l\) depends only on the difference,

\[ P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big), \qquad \mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big). \]

Training is logistic regression on score differences, which is maximum likelihood in the sense of [statistical estimation](../statistical-estimation-mle-map/). Two consequences follow. Adding a constant to every score changes no prediction, so a reward has no absolute zero and no unit; only differences mean anything. And a ranking of \(K\) responses yields \(\binom{K}{2}\) pairs; InstructGPT had labelers rank between 4 and 9 responses and kept all pairs from one prompt in the same batch, so the model did not overfit to repeated responses.

Now the toy world. A "response" is reduced to one property, its length in sentences, from 1 to 60. The SFT model mostly writes short answers (a median of 9 sentences). People actually like answers of about 25 sentences best and dislike much longer ones; that is the true reward, which no one gets to see. Labelers compare pairs of SFT answers and pick one with Bradley–Terry probabilities under the true reward. The reward model is deliberately simple: a score linear in length.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
L = np.arange(1, 61)                                     # possible answers: length in sentences
ref = np.exp(-(np.log(L) - np.log(8)) ** 2 / (2 * 0.4 ** 2))
ref /= ref.sum()                                         # pi_ref: the SFT model, mostly short answers
r_true = 3 * (1 - ((L - 25) / 15) ** 2)                  # what people prefer; hidden from everyone


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def label_pairs(n):
    """Two SFT answers per prompt; a labeler prefers one with Bradley-Terry odds under the true reward."""
    a, b = rng.choice(L, n, p=ref), rng.choice(L, n, p=ref)
    a_wins = rng.random(n) < sigmoid(r_true[a - 1] - r_true[b - 1])
    return np.where(a_wins, a, b), np.where(a_wins, b, a)          # (winner, loser)


def phi(length):
    return length / 10                                   # the reward model's only feature


win, lose = label_pairs(5000)
X = np.r_[phi(win) - phi(lose), phi(lose) - phi(win)][:, None]    # each pair in both orders
y = np.r_[np.ones(len(win)), np.zeros(len(win))]
w = LogisticRegression(fit_intercept=False, C=1e6).fit(X, y).coef_[0, 0]
r_rm = w * phi(L)                                        # the learned reward model

# Only differences matter: shifting every score by a constant changes no preference probability.
pa, pb = r_rm[np.r_[4, 9, 30]], r_rm[np.r_[7, 2, 50]]
assert np.allclose(sigmoid(pa - pb), sigmoid((pa + 5) - (pb + 5)))

# On fresh pairs from the SFT model, the reward model agrees with labelers as often as the truth does.
win_te, lose_te = label_pairs(5000)
agree = lambda r: np.mean(r[win_te - 1] > r[lose_te - 1]) + 0.5 * np.mean(r[win_te - 1] == r[lose_te - 1])
assert w > 0 and abs(agree(r_rm) - agree(r_true)) < 0.01
assert np.median(rng.choice(L, 20000, p=ref)) == 9           # the SFT model's typical answer

# Two labelers judging the same pair agree less often than the true reward agrees with one of them.
a, b = rng.choice(L, 20000, p=ref), rng.choice(L, 20000, p=ref)
p_a = sigmoid(r_true[a - 1] - r_true[b - 1])
labeler_vs_labeler = np.mean(p_a ** 2 + (1 - p_a) ** 2)
assert abs(labeler_vs_labeler - 0.70) < 0.01 and labeler_vs_labeler < agree(r_true) - 0.05
print(f"slope {w:.2f} per 10 sentences; agreement {agree(r_rm):.3f} (true reward {agree(r_true):.3f}, two labelers {labeler_vs_labeler:.3f})")
```

On held-out pairs drawn the way the training pairs were drawn, the linear reward model matches the labelers 77.9% of the time, the same as the true reward itself. Labels are noisy (people disagree), so neither gets close to 100%; in InstructGPT, trained labelers agreed with each other about 73% of the time. In the toy world two labelers agree on only about 70% of pairs, below the 78% the true reward reaches: agreement between labelers is a pessimistic guide to the best achievable accuracy, because each judgement carries noise from both people. By every in-distribution check, this reward model is fine. It has learned "longer is better", which is true for the short answers it was shown and false beyond 25 sentences, where it has almost no data.

## Stage 3: the objective and its optimum
Maximizing the reward model's score alone would push the policy to wherever the reward model is most wrong. RLHF instead maximizes reward minus a penalty for drifting from the reference:

\[ \max_{\pi}\; \mathbb{E}_{x,\; y \sim \pi(\cdot \mid x)}\big[r(x, y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big). \]

The KL divergence is taken under the policy being trained (the reverse direction, see [entropy and information](../entropy-and-information/)), so it is cheap to estimate from the policy's own samples: the average of \(\log \pi(y) - \log \pi_{\text{ref}}(y)\). This objective has an exact maximizer,

\[ \pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\big(r(x, y) / \beta\big), \]

the reference reweighted by exponentiated reward. The two-line proof, and how it turns into DPO, is written out in [Text-to-music in 2026](/posts/text-to-music-2026/#dpo). \(\beta\) is the exchange rate between reward and drift: as \(\beta \to \infty\) the policy stays at \(\pi_{\text{ref}}\); as \(\beta \to 0\) it piles onto whatever the reward model scores highest.

In the toy world the response space is small enough to compute \(\pi^*\) exactly for each \(\beta\) and score it with both the reward model and the hidden true reward:

```python
def tilt(r, beta):
    """The exact optimum of reward minus beta * KL to the reference."""
    p = ref * np.exp((r - r.max()) / beta)
    return p / p.sum()


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


betas = [np.inf, 10, 3, 2, 1.5, 1, 0.5]
rows = []
for beta in betas:
    p = ref if beta == np.inf else tilt(r_rm, beta)
    rows.append((beta, p @ r_rm, p @ r_true, p @ L, kl(p, ref)))
    print(f"beta {beta:>4}: RM {p @ r_rm:6.2f}  true {p @ r_true:6.2f}  length {p @ L:5.1f}  KL {kl(p, ref):5.2f}")

rm_score, true_score = [r[1] for r in rows], [r[2] for r in rows]
assert all(np.diff(rm_score) > 0)                        # the reward model's score only goes up...
best = int(np.argmax(true_score))
assert 0 < best < len(betas) - 1 and betas[best] == 2    # ...the true reward peaks at a middle beta...
assert true_score[-1] < true_score[0] - 5                # ...and with too little KL it is far worse than SFT
assert np.allclose(np.round(true_score, 2), [-0.17, 0.04, 0.59, 0.86, -0.07, -7.21, -11.88])   # the table below
```

| \(\beta\) | Reward model score | True reward | Mean length | KL from SFT |
|---|---|---|---|---|
| \(\infty\) (the SFT model) | 3.50 | −0.17 | 10.2 | 0.00 |
| 10 | 3.73 | 0.04 | 10.9 | 0.01 |
| 3 | 4.59 | 0.59 | 13.4 | 0.21 |
| 2 | 5.93 | **0.86** | 17.2 | 0.78 |
| 1.5 | 9.09 | −0.07 | 26.4 | 2.66 |
| 1 | 17.46 | −7.21 | 50.7 | 9.45 |
| 0.5 | 20.07 | −11.88 | 58.3 | 12.73 |

This is reward **over-optimization**, Goodhart's law in miniature. Up to a KL of about 0.8 nats, optimizing the reward model genuinely improves answers: they get longer, and longer is better in that range. Past it, the policy leaves the region the reward model was trained on, its score keeps climbing, and real quality collapses. Gao, Schulman and Hilton (2023) measured the same shape with real reward models: true (gold) reward first rises with KL from the reference and then falls, and larger reward models delay the turn. Length is not an invented example either; longer answers are a well-known way for RLHF policies to raise reward-model scores.

Two practical points follow. The KL from the reference is the quantity to watch during training, not the reward model's score, which rises regardless. And \(\beta\) cannot be tuned by looking at the reward model; it needs an independent judgement, such as fresh human comparisons against the SFT model.

## How PPO optimizes it
For a real language model there is no table of responses to reweight, so the objective is optimized by sampling. Each step, the policy writes answers to a batch of prompts, the reward model scores them, and the policy is updated to make above-average answers more likely. InstructGPT used PPO (Schulman et al., 2017), which makes plain policy gradients safe to run for several gradient steps on each batch. It clips the probability ratio between the current policy and the one that produced the samples, so an update gains nothing from moving an answer's probability by more than a factor of \(1 \pm \epsilon\) (clipping removes the incentive; it is not a hard limit):

\[ \mathcal{L}_{\text{PPO}} = -\,\mathbb{E}\Big[\min\big(\rho\, A,\; \mathrm{clip}(\rho, 1 - \epsilon, 1 + \epsilon)\, A\big)\Big], \qquad \rho = \frac{\pi_\theta(y \mid x)}{\pi_{\text{old}}(y \mid x)}. \]

The KL term is folded into the reward: each sample earns \(r(x, y) - \beta\,[\log \pi(y \mid x) - \log \pi_{\text{ref}}(y \mid x)]\). In a real implementation that penalty is applied per token, the reward model's score arrives at the final token, and a learned value model supplies the baseline from which the advantage \(A\) is estimated at each position (usually with generalized advantage estimation). So classic RLHF keeps four models in play: the policy, the frozen reference, the reward model and the value model, which is a large part of why it is expensive and fiddly.

The toy version has one prompt and a tabular policy over the 60 lengths. Without a value model, the advantage is the shaped reward minus the batch mean, divided by the batch standard deviation; that group-relative baseline is the idea GRPO (Shao et al., 2024) uses to drop the value model for LLMs. PPO should land on the closed-form \(\pi^*\) for the same \(\beta\):

```python
import torch

torch.manual_seed(0)
beta, eps, batch = 2.0, 0.2, 512
log_ref, R = torch.tensor(np.log(ref)), torch.tensor(r_rm)
logits = log_ref.clone().requires_grad_(True)            # the policy starts as the SFT model
opt = torch.optim.Adam([logits], lr=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / 1200)

for it in range(300):
    with torch.no_grad():                                # 1. sample with the current policy and score
        logp_old = torch.log_softmax(logits, 0)
        a = torch.multinomial(logp_old.exp(), batch, replacement=True)
        shaped = R[a] - beta * (logp_old[a] - log_ref[a])           # reward minus the KL penalty
        adv = (shaped - shaped.mean()) / (shaped.std() + 1e-8)      # group-relative advantage
    for _ in range(4):                                   # 2. several clipped steps on the same batch
        ratio = torch.exp(torch.log_softmax(logits, 0)[a] - logp_old[a])
        loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

pi_ppo = torch.softmax(logits, 0).detach().numpy()
target = tilt(r_rm, beta)
objective = lambda p: p @ r_rm - beta * kl(p, ref)
assert 0.5 * np.abs(pi_ppo - target).sum() < 0.02       # total variation distance to the exact optimum
assert objective(target) - objective(pi_ppo) < 0.01 < objective(pi_ppo) - objective(ref)
```

Why subtracting \(\beta \log(\pi / \pi_{\text{ref}})\) from each sample's reward is enough: the gradient of the objective is \(\mathbb{E}_\pi\big[\nabla \log \pi(y)\,(r - \beta \log \tfrac{\pi}{\pi_{\text{ref}}})\big]\) plus a term \(-\beta\,\mathbb{E}_\pi[\nabla \log \pi(y)]\), and the expectation of a score function is zero. The batch-mean baseline is legitimate for the same reason.

## DPO: skip the reward model
Direct Preference Optimization (Rafailov et al., 2023) starts from the closed-form optimum. Solving it for the reward gives \(r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)\). Substituting that into the Bradley–Terry loss, the intractable \(\log Z(x)\) cancels in the difference, leaving a loss on the policy alone:

\[ \mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big). \]

The policy is its own reward model: \(\beta \log(\pi_\theta / \pi_{\text{ref}})\) is the "implicit reward". Training is an ordinary supervised loop over preference pairs, with two forward passes per response (policy and frozen reference) and no sampling. At initialization \(\pi_\theta = \pi_{\text{ref}}\), every margin is zero and the loss is exactly \(\log 2\).

The claim to check is that DPO and RLHF reach the same policy when the policy can represent the same rewards the reward model could. Here that means a policy of the form \(\log \pi_\eta(y) = \log \pi_{\text{ref}}(y) + \eta\,\phi(y) - \log Z\), trained by DPO on exactly the labeled pairs the reward model saw:

```python
eta = torch.zeros(1, dtype=torch.float64, requires_grad=True)
f = torch.tensor(phi(L), dtype=torch.float64)
iw, il = torch.tensor(win - 1), torch.tensor(lose - 1)
opt = torch.optim.Adam([eta], lr=0.05)

for step in range(500):
    logp = torch.log_softmax(log_ref + eta * f, 0)
    margin = beta * ((logp[iw] - log_ref[iw]) - (logp[il] - log_ref[il]))
    loss = -torch.nn.functional.logsigmoid(margin).mean()
    if step == 0:
        assert abs(loss.item() - np.log(2)) < 1e-12     # policy == reference: every margin is 0
    opt.zero_grad()
    loss.backward()
    opt.step()

pi_dpo = torch.softmax(log_ref + eta * f, 0).detach().numpy()
assert abs(beta * eta.item() - w) < 0.01                 # the implicit reward equals the fitted reward model
assert 0.5 * np.abs(pi_dpo - target).sum() < 0.001       # and the policy equals the RLHF optimum
assert abs(pi_dpo @ r_true - pi_ppo @ r_true) < 0.05     # same policy, same true quality
```

DPO recovered the reward model's slope through the policy, and its policy matches the closed-form optimum and the PPO result. It also inherits the same flaw: it learned "longer is better" from the same pairs, and with a small \(\beta\) it would over-optimize in the same way. Removing the reward model removes a training stage, not the problem of imperfect preference data.

The equivalence holds at the optimum. In practice the two methods differ in ways that matter:

- **Offline versus online.** DPO learns from a fixed set of pairs, usually written by some other model. PPO scores the policy's own fresh samples, so it gets feedback exactly where the policy is going. Iterative or online DPO variants regenerate pairs from the current policy to close this gap.
- **The loss constrains only a margin.** DPO needs the winner's log-ratio to beat the loser's. Both log-probabilities can fall while the gap grows, moving probability onto responses that appear in no pair; this has been observed in practice, and an SFT term on the chosen responses is a common guard.
- **Cost and stability.** DPO needs two models in memory, no sampling and few hyperparameters, which is why it is a common default in open fine-tuning recipes. Careful comparisons have found that a well-tuned PPO can still come out ahead, for example on code generation (Xu et al., 2024), so "DPO replaced PPO" is too strong.

To see DPO applied to a non-text generator with tensor shapes at every step, see the alignment section of [Text-to-music in 2026](/posts/text-to-music-2026/#dpo). On limited hardware both methods are often run with [LoRA](../lora-explained/) adapters rather than full fine-tuning; [LoRA from scratch](/posts/lora-from-scratch/) builds those.

## Feedback from AI: Constitutional AI and RLAIF
Human comparisons are the bottleneck, so a natural question is whether a model can supply them. Constitutional AI (Bai et al., 2022) does this in two phases, guided by a short list of written principles, the constitution:

1. **Supervised phase.** The model answers a harmful prompt, critiques its own answer against a principle, and rewrites it. The revised answers become SFT data.
2. **RL phase.** The model generates pairs of answers, and a model asked which one better follows a principle provides the preference label. A preference model is trained on those AI labels and used for RL exactly as in stage 3. This is RL from AI feedback, RLAIF.

In the original work, AI labels replaced human labels for harmlessness, while helpfulness comparisons still came from people. The critique-and-revise loop alone is only the first half; the scalable part is that the preference data itself is generated by a model.

## Which method, and what goes wrong
| Situation | Reasonable choice |
|---|---|
| A fixed dataset of preference pairs, limited compute | DPO from the SFT model, with a modest \(\beta\) |
| You can label new comparisons as the policy changes | online RLHF with a reward model, or iterative DPO on fresh pairs |
| The reward is checkable, such as unit tests or a known answer | RL directly on that reward (PPO or GRPO); no preference model needed |
| A reward model but no budget for RL fine-tuning | best-of-N sampling with the reward model at inference time |

Pitfalls:

- **Trusting the reward model's score.** It rises whether or not quality does. Track KL from the reference and evaluate against held-out human judgements or a separate, stronger judge.
- **Validating the reward model only in distribution.** The toy reward model matched the labelers as well as the truth did, on pairs like its training pairs. Its failure was entirely in the region the policy would move to. Collecting new comparisons on the tuned policy's own outputs is how the flaw is found.
- **Length and style shortcuts.** Labelers, and reward models trained on them, tend to reward longer, more confident or more agreeable answers. Compare at matched length, or control for it in the reward.
- **Skipping SFT.** Preference tuning sharpens a model that already answers reasonably; it is a poor way to teach the format, and the reference it stays close to should be good.
- **Forgetting that labels are noisy.** Even the true reward matched the toy labelers only 78% of the time, so held-out accuracy has a ceiling well below 100%. Agreement between labelers gives a rough, somewhat pessimistic idea of where it sits; past that point, chasing more accuracy mostly fits noise.

Measuring a tuned model well is its own subject; see [evaluating ML systems](/posts/evaluating-ml-systems/).

## References
- Christiano, P. et al. (2017). Deep reinforcement learning from human preferences. NeurIPS.
- Ziegler, D. et al. (2019). Fine-tuning language models from human preferences. arXiv:1909.08593.
- Stiennon, N. et al. (2020). Learning to summarize from human feedback. NeurIPS.
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. NeurIPS.
- Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Gao, L., Schulman, J. and Hilton, J. (2023). Scaling laws for reward model overoptimization. ICML.
- Rafailov, R. et al. (2023). Direct preference optimization: your language model is secretly a reward model. NeurIPS.
- Bai, Y. et al. (2022). Constitutional AI: harmlessness from AI feedback. arXiv:2212.08073.
- Shao, Z. et al. (2024). DeepSeekMath: pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300.
- Xu, S. et al. (2024). Is DPO superior to PPO for LLM alignment? A comprehensive study. ICML.

This post is a rewrite of study notes built from the papers above; the toy world and all code are original.
