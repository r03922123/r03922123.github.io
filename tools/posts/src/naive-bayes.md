---
title: Naive Bayes: count, smooth, and you have a linear classifier
summary: Bayes' rule plus one bold assumption, that features are independent once you know the class. The multinomial model for text fitted by counting, why smoothing is not optional, the proof that it is a linear classifier in log space, the Bernoulli and Gaussian variants checked against scikit-learn, why its probabilities are overconfident, and when it beats logistic regression.
group: Statistics and experimentation
order: 7
tags: naive Bayes · generative models · text classification · classification
---
Naive Bayes classifies by asking the question backwards. Instead of learning "given this email, how likely is it spam?" directly, it learns "what do spam emails look like, and what do normal ones look like?", then uses Bayes' rule to turn that around. One simplifying assumption makes the backwards question cheap to answer: once you know the class, each feature is treated as independent of the others. The assumption is almost always false, the model is fitted by counting, and it still makes a strong baseline for text.

Below, the text version is built from scratch and checked against scikit-learn, shown to be a linear classifier, extended to Bernoulli and Gaussian features, and then charged for its false assumption: overconfident probabilities, and data sizes where logistic regression wins.

## Bayes' rule as a classifier
For a class \(y\) and features \(x = (x_1, \dots, x_d)\), Bayes' rule says

\[ P(y \mid x) = \frac{P(x \mid y)\, P(y)}{P(x)}. \]

To classify you only need the class with the largest posterior, and \(P(x)\) is the same for every class, so the decision is \(\arg\max_y P(x \mid y) P(y)\). The prior \(P(y)\) is easy: the share of each class in the training data. The hard part is the class-conditional likelihood \(P(x \mid y)\). With \(d\) binary features a full model of it has \(2^d - 1\) free parameters per class, about \(10^{30}\) for \(d = 100\). No dataset fills a table that size.

The **naive** assumption is that the features are conditionally independent given the class:

\[ P(x \mid y) = \prod_{j=1}^{d} P(x_j \mid y). \]

Now each class needs only \(d\) one-feature models. For spam this says that, among spam emails, seeing "free" tells you nothing about whether "money" also appears: plainly wrong, hence the name. [Conditional probability and Bayes' rule](../conditional-probability-bayes/) covers the basics by counting.

Naive Bayes is **generative**: it models how each class produces features, then inverts. Logistic regression is **discriminative**: it models \(P(y \mid x)\) directly. The two turn out to share a functional form, the most useful fact in this post.

## Multinomial naive Bayes for text
Represent a document as a bag of words: \(x_k\) is how many times vocabulary word \(k\) appears, and \(m = \sum_k x_k\) is the document length. The model says each class \(y\) has its own distribution \(\theta_y\) over the vocabulary, and a document of class \(y\) is \(m\) independent draws from it. The counts then follow a multinomial distribution:

\[ P(x \mid y) = \frac{m!}{\prod_k x_k!} \prod_k \theta_{yk}^{\,x_k}. \]

Note what is multinomial here: the likelihood of the words given the class. The prior over classes is a separate, plain categorical distribution \(\pi_y = P(y)\). The coefficient \(m! / \prod_k x_k!\) depends on the document but not on the class, so it cancels in Bayes' rule and can be dropped.

Maximum likelihood gives frequencies (see [MLE and MAP](../statistical-estimation-mle-map/)): \(\pi_y\) is the fraction of documents in class \(y\), and with \(N_{yk}\) the number of times word \(k\) appears in class-\(y\) documents and \(N_y = \sum_k N_{yk}\), the estimate is \(N_{yk} / N_y\). In practice one uses, with \(V\) the vocabulary size,

\[ \hat\theta_{yk} = \frac{N_{yk} + \alpha}{N_y + \alpha V}. \]

The \(\alpha\) is **additive smoothing** (Laplace smoothing when \(\alpha = 1\)), and the next section shows why it is required. It is the posterior mean of \(\theta_y\) under a symmetric Dirichlet(\(\alpha\)) prior, the many-category version of the Beta prior in [Beta for uncertainty](../beta-uncertainty/).

Prediction happens in log space, because a product of hundreds of probabilities below one underflows to zero:

\[ \log P(y \mid x) = \log \pi_y + \sum_k x_k \log \theta_{yk} - \log \sum_{y'} \exp\Big(\log \pi_{y'} + \sum_k x_k \log \theta_{y'k}\Big). \]

The last term is a log-sum-exp, computed stably exactly as in [stable softmax and cross-entropy](../stable-softmax-cross-entropy/). Here is the whole model, fitted on a toy corpus and checked against `MultinomialNB`:

```python
import numpy as np
from scipy.special import logsumexp, expit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

docs = [
    ("win free money now", 1), ("free prize claim now", 1), ("cheap pills free shipping", 1),
    ("claim your cash prize", 1), ("win cash win prize", 1),
    ("meeting moved to tomorrow", 0), ("lunch tomorrow with the team", 0),
    ("project notes for the meeting", 0), ("can we move lunch", 0), ("notes from the team call", 0),
    ("the report is attached", 0),
]
texts, y = [d for d, _ in docs], np.array([c for _, c in docs])
vec = CountVectorizer()
X = vec.fit_transform(texts).toarray()
vocab = vec.get_feature_names_out()


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        counts = np.array([X[y == c].sum(axis=0) for c in self.classes])        # N_yk
        smoothed = counts + self.alpha
        with np.errstate(divide="ignore"):                                      # alpha = 0 allows log(0)
            self.log_theta = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
            self.log_prior = np.log(np.array([(y == c).mean() for c in self.classes]))
        return self

    def joint_log_likelihood(self, X):
        # log pi_y + sum_k x_k log theta_yk; a word that does not occur adds 0, even if its log theta is -inf
        Xb = X[:, None, :]
        with np.errstate(invalid="ignore"):
            terms = np.where(Xb > 0, Xb * self.log_theta[None], 0.0)
        return terms.sum(axis=2) + self.log_prior

    def predict_log_proba(self, X):
        jll = self.joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)


ours = MultinomialNaiveBayes(alpha=1.0).fit(X, y)
ref = MultinomialNB(alpha=1.0).fit(X, y)
assert np.allclose(ours.log_theta, ref.feature_log_prob_)
assert np.allclose(ours.log_prior, ref.class_log_prior_)
assert np.allclose(ours.predict_log_proba(X), ref.predict_log_proba(X))

new = vec.transform(["claim free cash", "team meeting notes", "free lunch tomorrow"]).toarray()
print(np.exp(ours.predict_log_proba(new))[:, 1].round(3))               # P(spam) for each
assert (np.exp(ours.predict_log_proba(new))[:, 1] > 0.5).tolist() == [True, False, False]
```

Training is one pass of counting, \(O(\text{nonzeros})\), with no iterations and no learning rate. Adding more data means adding to the counts, which is why naive Bayes works naturally on streams.

## Why smoothing is not optional
Without smoothing (\(\alpha = 0\)) any word never seen in a class gets \(\theta_{yk} = 0\), so \(\log \theta_{yk} = -\infty\), and a single occurrence of that word rules the class out no matter what the rest of the document says. In the toy corpus "free" never appears in a normal email, so it vetoes the normal class; "lunch" and "team" never appear in spam, so they veto spam. A message containing both kinds of word is impossible under every class, and the unsmoothed posterior is \(0 / 0\):

```python
doc = vec.transform(["free lunch with the team tomorrow"]).toarray()
raw = MultinomialNaiveBayes(alpha=0.0).fit(X, y)
smooth = MultinomialNaiveBayes(alpha=1.0).fit(X, y)

assert (raw.joint_log_likelihood(doc) == -np.inf).all()             # both classes ruled out by one word each
with np.errstate(invalid="ignore"):
    assert np.isnan(raw.predict_log_proba(doc)).all()                # the posterior is undefined
p_spam = np.exp(smooth.predict_log_proba(doc))[0, 1]
assert p_spam < 0.05                                                 # smoothed: five normal words outvote "free"
```

With smoothing the same message gets a spam probability of about 0.03, because five words point to a normal email and one points to spam. This is the zero-frequency problem, and it is not rare: word frequencies have a long tail, so at real vocabulary sizes many words never occur in some class's training documents. Treat \(\alpha\) as a hyperparameter and tune it on held-out data; values between about 0.01 and 1 are common.

## It is a linear classifier
Take two classes, spam (1) and normal (0), and write the log-odds:

\[ \log \frac{P(1 \mid x)}{P(0 \mid x)} = \sum_k x_k \log \frac{\theta_{1k}}{\theta_{0k}} + \log \frac{\pi_1}{\pi_0} = w^\top x + b, \]

with \(w_k = \log(\theta_{1k} / \theta_{0k})\) and \(b = \log(\pi_1 / \pi_0)\). Exponentiate and solve for \(P(1 \mid x)\):

\[ P(1 \mid x) = \frac{1}{1 + e^{-(w^\top x + b)}} = \sigma(w^\top x + b). \]

That is exactly the form of logistic regression. (Mind the parentheses: the bias sits inside the negation, \(e^{-(w^\top x + b)}\), not \(e^{-w^\top x + b}\).) With \(C\) classes the same algebra gives a softmax over \(C\) linear scores. The model families are identical; only the fitting differs. Naive Bayes sets each \(w_k\) from that word's per-class counts (divided by the class totals), never looking at which other words it co-occurs with, while logistic regression chooses all of \(w\) together to maximize the conditional likelihood, so it can shrink a weight whose evidence another feature already supplies.

```python
w = ours.log_theta[1] - ours.log_theta[0]
b = ours.log_prior[1] - ours.log_prior[0]
assert np.allclose(expit(X @ w + b), np.exp(ours.predict_log_proba(X))[:, 1])
assert np.allclose(expit(new @ w + b), ref.predict_proba(new)[:, 1])

top = vocab[np.argsort(w)]
print("most spammy:", top[-4:][::-1], " most normal:", top[:4])
assert {"free", "prize", "win"} <= set(top[-5:])
```

Each weight is readable: a word's contribution to the log-odds, per occurrence. The bias also shows how to handle a shift in base rates. If spam is 1% of traffic in production but 45% of the training set, replace \(b\) with \(\log(0.01 / 0.99)\) and leave \(w\) alone.

## Bernoulli and Gaussian variants
The naive assumption works with any per-feature model. The two other common choices:

- **Bernoulli naive Bayes** uses binary presence features, \(P(x \mid y) = \prod_k p_{yk}^{x_k} (1 - p_{yk})^{1 - x_k}\). Unlike the multinomial model, a word that is **absent** is also evidence: if most spam contains "free", a message without it counts slightly against spam. This tends to suit short texts. It is still linear in \(x\).
- **Gaussian naive Bayes** models each continuous feature as \(x_j \mid y \sim \mathcal{N}(\mu_{yj}, \sigma^2_{yj})\). With a separate variance per class the decision boundary is quadratic; with variances shared across classes it becomes linear. See [the normal distribution](../normal-distribution/) for the density.

Both are a few lines from scratch. The Gaussian one matches scikit-learn once you copy its one detail: it adds \(10^{-9}\) times the largest feature variance (computed over all the data) to every per-class variance, so a feature that is constant within one class cannot cause a division by zero.

```python
from sklearn.datasets import load_iris

# Bernoulli on the toy corpus: presence/absence, smoothing over the two outcomes
B = (X > 0).astype(float)
n_c = np.array([(y == c).sum() for c in (0, 1)])
p = (np.array([B[y == c].sum(axis=0) for c in (0, 1)]) + 1.0) / (n_c[:, None] + 2.0)
jll = B @ np.log(p).T + (1 - B) @ np.log(1 - p).T + np.log(n_c / n_c.sum())
assert np.allclose(jll - logsumexp(jll, axis=1, keepdims=True), BernoulliNB(alpha=1.0).fit(X, y).predict_log_proba(X))

# Gaussian on iris: per-class means and variances, plus sklearn's variance floor
Xi, yi = load_iris(return_X_y=True)
eps = 1e-9 * Xi.var(axis=0).max()
mu = np.array([Xi[yi == c].mean(axis=0) for c in range(3)])
var = np.array([Xi[yi == c].var(axis=0) for c in range(3)]) + eps
prior = np.bincount(yi) / len(yi)
jll = np.stack([np.log(prior[c]) - 0.5 * np.sum(np.log(2 * np.pi * var[c]) + (Xi - mu[c]) ** 2 / var[c], axis=1)
                for c in range(3)], axis=1)
gnb = GaussianNB().fit(Xi, yi)
assert np.allclose(np.exp(jll - logsumexp(jll, axis=1, keepdims=True)), gnb.predict_proba(Xi))
assert (gnb.predict(Xi) == yi).mean() > 0.94
```

## Right decisions, wrong probabilities
Why does a model built on a false assumption classify well? A classifier only has to put the largest score on the right class. It does not need the score to be the true probability. Domingos and Pazzani (1997) showed that naive Bayes can be optimal under zero-one loss in many cases where the independence assumption is badly violated.

The probabilities are another story. When features are correlated, the same evidence is counted several times, and the posterior is pushed toward 0 or 1. The cleanest case is a feature copied \(k\) times: the log-odds from the likelihood are multiplied by \(k\) exactly, while the prior term \(b\) is not. With balanced classes (\(b = 0\)) every decision stays the same and only the probabilities get worse; with unbalanced classes the inflated likelihood also overrides the prior, so some decisions flip as well.

```python
from sklearn.metrics import log_loss

rng = np.random.default_rng(0)
yb = np.repeat([0, 1], 500)                                   # balanced, so log(pi_1 / pi_0) = 0
x1 = rng.normal(loc=np.where(yb == 1, 1.0, -1.0), scale=1.5)[:, None]


def logit(model, X):
    lp = model.predict_log_proba(X)
    return lp[:, 1] - lp[:, 0]


one = GaussianNB().fit(x1, yb)
five = GaussianNB().fit(np.repeat(x1, 5, axis=1), yb)         # the same feature, five times
assert np.allclose(logit(five, np.repeat(x1, 5, axis=1)), 5 * logit(one, x1))
assert (five.predict(np.repeat(x1, 5, axis=1)) == one.predict(x1)).all()
ll1, ll5 = log_loss(yb, one.predict_proba(x1)), log_loss(yb, five.predict_proba(np.repeat(x1, 5, axis=1)))
print(f"log loss: 1 copy {ll1:.3f}, 5 copies {ll5:.3f}")
assert ll5 > 1.5 * ll1

# unbalanced classes: the prior is no longer zero, and five copies of the evidence outvote it
yu = np.repeat([0, 1], [800, 200])
xu = rng.normal(loc=np.where(yu == 1, 1.0, -1.0), scale=1.5)[:, None]
one_u, five_u = GaussianNB().fit(xu, yu), GaussianNB().fit(np.repeat(xu, 5, axis=1), yu)
assert (one_u.predict(xu) != five_u.predict(np.repeat(xu, 5, axis=1))).sum() > 0
```

Real text is full of milder versions: "New" and "York", a unigram and the bigram containing it, a word and its plural. Rank by naive Bayes scores if you like, but do not read them as calibrated probabilities; if you need probabilities, recalibrate on held-out data (Platt scaling or isotonic regression).

## Naive Bayes vs logistic regression
Since both are linear, the real question is which way of choosing \(w\) is better. Ng and Jordan (NIPS 2001) gave the classic answer: naive Bayes approaches its asymptotic error with a number of examples that grows only logarithmically in the number of features, while logistic regression needs a number that grows linearly, but logistic regression's asymptotic error is never higher, and is typically lower when the independence assumption is wrong. So naive Bayes tends to win with little data and lose with a lot.

The simulation below draws 30 binary features that really are independent given the class, then repeats it with five of them duplicated six extra times. It averages test accuracy over 20 training sets for the small sizes (and 2 for the large one, where run-to-run variation is tiny):

```python
from sklearn.linear_model import LogisticRegression

d = 30
p1 = np.random.default_rng(123).uniform(0.1, 0.9, d)
p0 = np.clip(p1 + np.random.default_rng(124).normal(0, 0.15, d), 0.05, 0.95)


def sample(n, rng, copies):
    labels = rng.integers(0, 2, n)
    Xs = (rng.random((n, d)) < np.where(labels[:, None] == 1, p1, p0)).astype(float)
    return np.hstack([Xs] + [Xs[:, :5]] * copies), labels


acc = {}
for copies in (0, 6):
    X_test, y_test = sample(5000, np.random.default_rng(99), copies)
    for n in (20, 60, 3000):
        runs = [sample(n, np.random.default_rng(s), copies) for s in range(20 if n < 1000 else 2)]
        nb = np.mean([(BernoulliNB().fit(a, c).predict(X_test) == y_test).mean() for a, c in runs])
        lr = np.mean([(LogisticRegression(max_iter=1000).fit(a, c).predict(X_test) == y_test).mean() for a, c in runs])
        acc[copies, n] = (nb, lr)
        print(f"copies={copies} n={n:5d}  naive Bayes {nb:.3f}  logistic regression {lr:.3f}")

assert acc[0, 20][0] > acc[0, 20][1] and acc[0, 60][0] > acc[0, 60][1]   # assumption holds, little data: NB ahead
assert abs(acc[0, 3000][0] - acc[0, 3000][1]) < 0.01                      # lots of data: the two meet
assert all(acc[6, n][1] > acc[6, n][0] for n in (20, 60, 3000))           # duplicated features: LR ahead at every size
assert acc[6, 3000][1] - acc[6, 3000][0] > 0.05                           # ... and clearly so with lots of data
```

With independent features naive Bayes leads at 20 and 60 examples and the two meet at 3,000. With the duplicated features naive Bayes over-weights those five and falls behind at every size, and logistic regression, which can split the weight among the copies, ends up about nine points ahead at 3,000 examples. One simulation proves no general law, but this is the pattern Ng and Jordan describe.

## When to use it, and the pitfalls
Reach for naive Bayes when:

- **You need a text baseline in minutes:** spam, topic or intent classification on word counts. It is the number any fancier model has to beat.
- **Labeled data is scarce**, or features are many relative to examples.
- **Training and updates must be cheap.** Fitting is counting, so new data is added to the counts, and prediction is one sparse dot product per class.
- **You want readable weights.** Each word's log-ratio is its evidence.

Watch out for:

1. **Zero counts and underflow.** Always smooth and tune \(\alpha\); always work in log space with log-sum-exp.
2. **Overconfident probabilities.** Use the ranking, and recalibrate before thresholding on probabilities.
3. **Redundant features.** Bigrams on top of unigrams, or several encodings of one signal, inflate their weight. Logistic regression copes better.
4. **The wrong variant.** Multinomial expects counts (TF-IDF works in practice, though it is not a count), Bernoulli expects presence, Gaussian expects roughly bell-shaped features; transform skewed ones such as durations first.
5. **Small classes in text.** Their word estimates are noisy. Complement naive Bayes (Rennie et al., ICML 2003) estimates each class's weights from all the other classes' documents to help here; scikit-learn has it as `ComplementNB`.
6. **A shifted base rate.** If the production class mix differs from training, change \(b\) rather than retraining.

For the bigger picture of when a simple model with a strong assumption beats a flexible one, see [the bias-variance tradeoff](../bias-variance-tradeoff/), and for the discriminative side of the story, [regression as maximum likelihood](../regression-as-maximum-likelihood/).

The derivations follow the treatment of generative learning algorithms in Andrew Ng's Stanford CS229 lecture notes, and the variants follow the scikit-learn naive Bayes documentation; both are rewritten here with new examples and code.
