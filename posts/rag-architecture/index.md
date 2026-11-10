# RAG architecture: retrieval in front of a language model

A language model knows only what was in its training data, as of the day that data was collected. It knows nothing about your product manual, last week's policy change or a customer's account. Retrieval-augmented generation (RAG) is the standard workaround: when a question arrives, search a document collection for the passages most likely to contain the answer, paste them into the prompt, and ask the model to answer from them. The model's weights never change. Updating its knowledge means updating an index.

The name comes from Lewis et al. [1], who trained a retriever and a generator together. In practice the term now covers the looser pattern of any search system feeding any LLM, and that is what this post builds. No model is downloaded, so the generator is left out and every other stage is small enough to check with asserts.

## Two pipelines, one index
A RAG system is really two pipelines that share an index:

```plaintext
offline   documents -> clean -> chunk -> embed / index -> store (text + vectors + metadata)
online    question  -> embed / tokenize -> retrieve top k -> (rerank) -> build prompt -> LLM -> answer + citations
```

The offline side decides what can be found; the online side decides what gets found and what the model does with it. Many quality problems trace back to the offline side, because a passage that was badly cut or never indexed cannot be rescued by a better prompt.

The running example is a made-up help centre for a bike-sharing service: eight short articles and fourteen questions, each labelled with the phrase that answers it.

```python
import re
import numpy as np

DOCS = {
    "unlocking": "To start a ride, scan the QR code on the handlebar with the app. The lock opens within five seconds. "
                 "If the lock stays closed, check that Bluetooth is on and that the bike shows a green light. "
                 "A red light means the bike is reserved by another rider or flagged for repair.",
    "pricing": "A single ride costs one dollar to unlock plus fifteen cents per minute. "
               "The monthly pass costs twenty dollars and covers unlimited rides of up to forty five minutes each. "
               "Minutes beyond the forty five minute limit are charged at the standard per-minute rate.",
    "refunds": "Riders can request a refund within fourteen days of a ride from the trip history screen. "
               "Refunds for rides cut short by a broken bike are approved automatically. "
               "Other refunds are reviewed by the support team within three working days.",
    "errors": "Error E-17 means the dock could not confirm that the bike was returned. Push the bike firmly into the dock until it clicks. "
              "Error E-23 means the battery of an electric bike is below ten percent and the bike cannot be unlocked. "
              "Error E-31 means your payment method was declined.",
    "parking": "End every ride at a dock or inside a marked parking zone shown on the map. "
               "Bikes left outside a zone incur a parking fee of ten dollars. "
               "Repeated parking violations can lead to a suspended account.",
    "ebikes": "Electric bikes have a top assisted speed of twenty five kilometres per hour. "
              "They cost an extra ten cents per minute, and this surcharge is not waived by the monthly pass. "
              "The battery range on a full charge is about sixty kilometres.",
    "accounts": "Riders must be at least sixteen years old to create an account. "
                "Riders under eighteen need a guardian to approve the account by email. "
                "Each account can unlock at most two bikes at the same time, for example when riding with a child.",
    "safety": "Helmets are recommended for all riders and required by law for riders under eighteen in some cities. "
              "Report a crash from the help menu so that the bike is taken out of service. "
              "In an emergency, call local emergency services first.",
}
QUERIES = [                                         # (question, phrase that answers it)
    ("how much does a monthly pass cost", "pass costs twenty dollars"),
    ("what does error E-23 mean", "E-23 means the battery"),
    ("what does error E-31 mean", "E-31 means your payment method was declined"),
    ("the lock will not open after scanning", "check that Bluetooth is on"),
    ("can I get my money back for a ride", "request a refund within fourteen days"),
    ("how long do refund reviews take", "within three working days"),
    ("fine for leaving a bike outside a zone", "parking fee of ten dollars"),
    ("how fast do electric bikes go", "twenty five kilometres per hour"),
    ("minimum age to sign up", "at least sixteen years old"),
    ("how many bikes can one account unlock", "at most two bikes at the same time"),
    ("what does a red light on the bike mean", "red light means the bike is reserved"),
    ("ebike battry range", "about sixty kilometres"),
    ("cost per minute of a ride", "fifteen cents per minute"),
    ("bike not returned to dock error", "E-17 means the dock could not confirm"),
]
assert all(sum(a in t for t in DOCS.values()) == 1 for _, a in QUERIES)   # each answer lives in one article
```

## Chunking: deciding what a search result is
Documents are split into chunks because whole documents are too long to fit several into a prompt and too broad to score well against a narrow question. The chunk is the unit of retrieval, so its size is a real trade-off: small chunks match questions precisely but may carry too little context to answer from; large chunks carry context but dilute the match and cost prompt space.

The simplest splitter cuts every \(n\) words. Its flaw is that the cut ignores meaning, so an answer can straddle two chunks and appear whole in neither. Overlapping windows fix this with a guarantee: if consecutive windows share \(o\) words, any phrase of at most \(o + 1\) words lies entirely inside some window. (A window starts every \(n - o\) words, so a phrase begins at most \(n - o - 1\) words into the window that starts just before it, and still fits if \(n - o - 1 + L \le n\), that is \(L \le o + 1\).) A splitter that packs whole sentences up to a size limit avoids the problem differently, by never cutting a sentence.

```python
def fixed_chunks(text, size, overlap=0):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(len(w) - overlap, 1), size - overlap)]


def sentence_chunks(text, size):
    out, cur = [], []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(" ".join(cur + [s]).split()) > size:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    return out + [" ".join(cur)]


def lost_answers(chunker):
    chunks = [c for t in DOCS.values() for c in chunker(t)]
    return [a for _, a in QUERIES if not any(a in c for c in chunks)]


assert lost_answers(lambda t: fixed_chunks(t, 20)) == ["red light means the bike is reserved", "about sixty kilometres"]
longest = max(len(a.split()) for _, a in QUERIES)                       # 8 words
assert lost_answers(lambda t: fixed_chunks(t, 20, overlap=longest - 1)) == []


def every_phrase_covered(length, size=20, overlap=longest - 1):     # brute force over all phrases in the corpus
    for t in DOCS.values():
        w = t.split()
        windows = [c.split() for c in fixed_chunks(t, size, overlap)]
        for i in range(len(w) - length + 1):
            p = w[i:i + length]
            if not any(c[j:j + length] == p for c in windows for j in range(len(c) - length + 1)):
                return False
    return True


assert every_phrase_covered(longest) and not every_phrase_covered(longest + 1)   # o + 1 is guaranteed, o + 2 is not
assert lost_answers(lambda t: sentence_chunks(t, 25)) == []
n_plain = sum(len(fixed_chunks(t, 20)) for t in DOCS.values())
n_overlap = sum(len(fixed_chunks(t, 20, longest - 1)) for t in DOCS.values())
assert (n_plain, n_overlap) == (22, 26)                                 # overlap costs ~18% more chunks here
```

Twenty-word chunks without overlap lose two of the fourteen answers outright: no retriever, however good, can return a chunk that contains them. Overlap recovers both at the price of more chunks to embed and store. Real systems use the same ideas at larger scale: split on headings and paragraphs first, then sentences, then words, and only fall back to hard cuts when a unit is still too long. A common refinement is **parent-child** chunking: index small chunks for precise matching, but hand the model the larger section each one came from.

One practical trap: libraries differ in what "size" counts. LangChain's `RecursiveCharacterTextSplitter`, for instance, measures characters by default, so `chunk_size=512` is roughly 100 words, not 512 tokens. Check the unit before copying a setting.

## Scoring a chunk: words and vectors
From here on the index holds sentence-packed chunks. There are two families of scorer.

**Lexical scoring** counts shared words, weighted by rarity. BM25 [2] is the standard formula. For a query with terms \(t\) and a chunk \(d\) of length \(|d|\),

\[ \text{BM25}(q, d) = \sum_{t \in q} \text{idf}(t)\, \frac{f_{t,d}\,(k_1 + 1)}{f_{t,d} + k_1\left(1 - b + b\,|d| / \overline{|d|}\right)} , \]

where \(f_{t,d}\) is the term's count in the chunk. The fraction saturates, so the tenth repeat of a word adds little, and \(b\) discounts long chunks. BM25 is exact about rare tokens such as error codes and product names, and blind to synonyms and typos.

**Vector scoring** maps the query and each chunk to vectors and ranks by cosine similarity. In a real system the vectors come from a learned embedding model, a bi-encoder trained so that questions land near the passages that answer them, as in dense passage retrieval [3] and the [two-tower retrieval](../two-tower-retrieval/) models of recommender systems; the training objective is a [contrastive loss](../contrastive-loss/). Downloading one is not possible here, so a TF-IDF vector over character n-grams stands in. It is not semantic, but it has the mechanics that matter (fixed-size normalized vectors compared by dot product) and a different failure profile from BM25: it tolerates inflections and typos.

```python
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = [c for t in DOCS.values() for c in sentence_chunks(t, 25)]


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:-\d+)?", text.lower())                # keeps "e-23" as one token


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf])
        df = Counter(w for c in self.tf for w in c)
        n = len(docs)
        self.idf = {w: np.log(1 + (n - m + 0.5) / (m + 0.5)) for w, m in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        s = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.len.mean())
        for w in set(tokens(query)) & self.idf.keys():
            f = np.array([c[w] for c in self.tf])
            s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s


bm25 = BM25(texts)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts)
E = vec.transform(texts).toarray()                                       # one row per chunk


def vector_scores(query):
    return E @ vec.transform([query]).toarray()[0]


q = "what does error E-23 mean"
assert np.allclose(np.linalg.norm(E, axis=1), 1)                         # rows are unit length, so...
assert np.allclose(vector_scores(q), cosine_similarity(vec.transform([q]), E)[0])   # ...dot product = cosine
assert "E-23 means" in texts[np.argmax(bm25.scores(q))]
```

Normalizing once at indexing time is why vector stores can use plain inner product: for unit vectors, cosine similarity and dot product are the same number, and ranking by Euclidean distance gives the same order too, since \(\lVert u - v \rVert^2 = 2 - 2\,u^\top v\).

## Hybrid retrieval with reciprocal rank fusion
Since the two scorers fail on different questions, run both and merge. Their scores live on different scales, so instead of adding scores, reciprocal rank fusion [4] adds up reciprocal ranks:

\[ \text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}, \qquad k = 60 . \]

A chunk ranked near the top by either retriever scores well, and one ranked well by both scores best. The constant \(k\) flattens the curve so that rank 1 does not dominate rank 3.

```python
def ranked(scores):
    return list(np.argsort(-scores, kind="stable"))


def rrf(rankings, k=60):
    fused = Counter()
    for r in rankings:
        for pos, i in enumerate(r):
            fused[i] += 1 / (k + pos + 1)
    return [i for i, _ in fused.most_common()]


retrievers = {
    "BM25": lambda q: ranked(bm25.scores(q)),
    "char vectors": lambda q: ranked(vector_scores(q)),
    "hybrid": lambda q: rrf([ranked(bm25.scores(q)), ranked(vector_scores(q))]),
}


def evaluate(retrieve):
    first = [next(p for p, i in enumerate(retrieve(q)) if a in texts[i]) + 1 for q, a in QUERIES]
    hit = lambda k: sum(r <= k for r in first) / len(first)
    missed = [q for (q, _), r in zip(QUERIES, first) if r > 3]
    return round(hit(1), 2), round(hit(3), 2), round(np.mean([1 / r for r in first]), 2), max(first), missed


results = {name: evaluate(f) for name, f in retrievers.items()}
assert {name: r[:4] for name, r in results.items()} == {                 # the table below
    "BM25": (0.79, 0.93, 0.87, 8), "char vectors": (0.86, 0.86, 0.89, 6), "hybrid": (0.71, 1.0, 0.83, 3)}
assert results["BM25"][4] == ["how long do refund reviews take"]            # "refund" != "refunds"
assert results["char vectors"][4] == ["the lock will not open after scanning", "minimum age to sign up"]
assert results["hybrid"][1] == 1.0                                        # every answer in the top 3
assert results["hybrid"][1] > max(results["BM25"][1], results["char vectors"][1])
assert results["hybrid"][3] < min(results["BM25"][3], results["char vectors"][3])   # best worst case
```

| Retriever | Hit rate at 1 | Hit rate at 3 | MRR | Worst rank | Missed at 3 |
|---|---|---|---|---|---|
| BM25 | 0.79 | 0.93 | 0.87 | 8 | "how long do refund reviews take" |
| Character n-gram vectors | 0.86 | 0.86 | 0.89 | 6 | "the lock will not open…", "minimum age to sign up" |
| Hybrid (RRF) | 0.71 | 1.00 | 0.83 | 3 | none |

BM25 misses the refund question because it has no stemming: "refund" and "reviews" in the query do not match "refunds" and "reviewed" in the answer. The character vectors catch that, but miss "the lock will not open after scanning", whose answer shares few letter sequences with it. Fusion is the weakest of the three at rank 1 here, but it has by far the best worst case: every answer lands in the top three, which is what matters when the top three all go into the prompt. Fourteen questions are an illustration, not a benchmark (rewording one sentence in an unrelated article moves individual ranks by a place or two), but the pattern is a common one: hybrid retrieval is a cheap, robust default. The question neither scorer ranks first, "minimum age to sign up" against "at least sixteen years old to create an account", is a pure paraphrase, which is the gap a learned embedding model exists to close.

A common next stage is a **reranker**: a cross-encoder that reads the question and each of the top 20 to 100 candidates together and rescores them [9]. It is too slow to run over the whole collection but much more accurate than either first-stage scorer.

## Vector search at scale
Scoring every chunk is exact and fine up to perhaps a million vectors. Beyond that, vector stores use approximate nearest-neighbour (ANN) indexes. The simplest is the **inverted file** (IVF) used in FAISS [5]: cluster the vectors with k-means, store each vector in the list of its nearest centroid, and at query time scan only the `nprobe` lists whose centroids are closest to the query. Graph indexes such as HNSW [6] make a different trade but expose the same knob: more work per query buys higher recall.

```python
rng = np.random.default_rng(0)
unit = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)
topics = rng.normal(size=(50, 32))
X = unit(topics[rng.integers(50, size=10_000)] + 1.5 * rng.normal(size=(10_000, 32))).astype(np.float32)
Qv = unit(topics[rng.integers(50, size=100)] + 1.5 * rng.normal(size=(100, 32))).astype(np.float32)

C = X[rng.choice(len(X), 64, replace=False)]                             # k-means with 64 centroids
for _ in range(8):
    assign = np.argmax(X @ C.T, axis=1)
    C = unit(np.stack([X[assign == j].mean(0) if np.any(assign == j) else C[j] for j in range(len(C))]))
assign = np.argmax(X @ C.T, axis=1)
lists = [np.flatnonzero(assign == j) for j in range(len(C))]

S = Qv @ X.T                                                             # exact scores, for the ground truth
exact = np.argsort(-S, axis=1)[:, :10]


def ivf_search(qi, nprobe, k=10):
    cand = np.concatenate([lists[j] for j in np.argsort(-(C @ Qv[qi]))[:nprobe]])
    return cand[np.argsort(-S[qi, cand])[:k]], len(cand)


curve = {}
for nprobe in (1, 4, 16, 64):
    found = [ivf_search(qi, nprobe) for qi in range(len(Qv))]
    recall = np.mean([len(set(top) & set(exact[qi])) / 10 for qi, (top, _) in enumerate(found)])
    curve[nprobe] = (recall, np.mean([m for _, m in found]) / len(X))

recalls = [r for r, _ in curve.values()]
assert recalls == sorted(recalls) and curve[64][0] == 1.0                 # probing every list is exact search
assert curve[1][1] < 0.03 and 0.45 < curve[1][0] < 0.65                 # 1 list: under 3% of the work, about half the recall
assert curve[16][1] < 0.3 and curve[16][0] > 0.9                         # 16 lists: about a quarter of the work, over 90%
```

Probing one list out of 64 scans under 3% of the vectors and finds about half of the true top 10; probing 16 scans about a quarter and finds over 90%. The right setting is a measured choice: build a ground-truth set with exact search, then pick the cheapest `nprobe` (or HNSW `ef`) that meets a recall target. The practical side of choosing and running a vector store is in [vector databases](../vector-databases/).

## Assembling the prompt
The retrieved chunks go into the prompt with a number each, so the model can cite them and the application can link the citations back to documents. Three rules carry most of the value. Keep the instruction explicit: answer only from the sources, cite them, and say so when they do not contain the answer. Respect a context budget, because every chunk adds cost and latency to prefill (see [LLM inference](../llm-inference/)). And mind order: models use information at the start and end of a long context more reliably than information in the middle [7], so the best chunk should not be buried.

```python
INSTRUCTION = ("Answer the question using only the numbered sources and cite them like [1]. "
               "If the sources do not contain the answer, say that you do not know.")


def build_prompt(question, ranking, budget=120):                          # budget in words, a stand-in for tokens
    used = len(INSTRUCTION.split()) + len(question.split())
    sources = {}
    for i in ranking:
        if used + len(texts[i].split()) <= budget:                       # skip what does not fit, keep rank order
            sources[len(sources) + 1] = texts[i]
            used += len(texts[i].split())
    body = "\n".join(f"[{n}] {t}" for n, t in sources.items())
    return f"{INSTRUCTION}\n\nSources:\n{body}\n\nQuestion: {question}", sources, used


question = "how long do refund reviews take"
ranking = retrievers["hybrid"](question)
assert "three working days" in texts[ranking[2]]                          # the answer chunk is ranked third
prompt, sources, used = build_prompt(question, ranking)
assert used <= 120 and 1 < len(sources) < len(texts)
assert any("three working days" in t for t in sources.values())
tight = build_prompt(question, ranking, budget=90)[1]
assert not any("three working days" in t for t in tight.values())       # the third-ranked chunk no longer fits
```

The last assert is the budget trade-off in miniature. With 120 words the answer chunk, ranked third, makes it into the prompt. With 90 it no longer fits, and its place goes to lower-ranked chunks that happen to be shorter. A reranker that pushes the right chunk to the top makes a tight budget much safer.

## Checking that the answer is grounded
Retrieval can be perfect and the answer still wrong, because the model may ignore the sources or embellish them. An instruction to use only the sources reduces this but does not prevent it, so production systems check answers after generation. Below is the crudest useful check: every sentence must cite a source, and nearly all its content words must appear in the sources it cites. A fabricated detail shows up as unsupported words.

```python
STOP = {"a", "an", "the", "are", "is", "by", "and", "of", "to", "within", "for", "it", "they", "be"}


def unsupported(answer, sources, tolerance=0.2):
    flagged = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        support = set().union(*[set(tokens(sources[n])) for n in cited if n in sources])
        words = {w for w in tokens(re.sub(r"\[\d+\]", "", sentence)) if w not in STOP}
        if not cited or len(words - support) > tolerance * len(words):
            flagged.append(sentence)
    return flagged


n = next(n for n, t in sources.items() if "three working days" in t)
good = f"Refunds are reviewed by the support team within three working days [{n}]."
bad = f"Refunds are reviewed within three working days and paid out within 24 hours [{n}]. Keep your receipt."
assert unsupported(good, sources) == []
assert unsupported(bad, sources) == [f"Refunds are reviewed within three working days and paid out within 24 hours [{n}].",
                                     "Keep your receipt."]
```

Word overlap is a weak proxy: it rejects a correct paraphrase and accepts the right words in the wrong meaning. Stronger checks split the answer into claims and ask an entailment model or a second LLM whether each follows from the cited text, the idea behind the faithfulness score in RAGAS [8].

## Debugging: retrieval first
When answers are bad, find out which half failed before changing anything. Keep a set of questions labelled with the chunks that answer them, as above, and measure retrieval on its own: hit rate or recall at \(k\) (the fraction of relevant chunks in the top \(k\); with one relevant chunk per question the two coincide) and MRR. Then judge answers only on questions where retrieval succeeded. The broader evaluation picture is in [Evaluating ML systems](/posts/evaluating-ml-systems/).

| Symptom | Likely half | Typical fixes |
|---|---|---|
| The answer chunk is not in the top \(k\) | Retrieval | Hybrid search, a better or fine-tuned embedding model, a reranker, query rewriting |
| The answer is split across chunks | Indexing | Structure-aware chunking, overlap, parent-child chunks |
| Right chunk retrieved, wrong answer | Generation | Clearer instruction, fewer and better-ordered chunks, a grounding check, a stronger model |
| Outdated or deleted content is returned | Indexing | Incremental re-indexing on document change, deletes that reach the index, timestamps in metadata |
| Users see content they should not | Retrieval | Filter on access-control metadata inside the search, not after it |

## RAG or fine-tuning?
They solve different problems and are often combined. RAG suits knowledge that changes, must be cited, or must respect permissions: an edit or a deletion takes effect as soon as the index is updated, and every answer can point at its source. Fine-tuning, including the cheap [parameter-efficient](../parameter-efficient-fine-tuning/) kinds such as [LoRA](../lora-explained/), suits behaviour: output format, tone, domain vocabulary, following a house style. It is an unreliable way to add facts, since there is no guarantee a fact seen in training will be recalled correctly, and no reliable way to remove one afterwards short of retraining. Fine-tuning does save prompt tokens, so when the knowledge is small and stable it can win on latency. A model fine-tuned to read sources and cite them well, then served with retrieval, gets both benefits. When retrieval becomes one tool among several that a model calls in a loop, the design moves toward [agents](../llm-agents-langchain-langgraph/).

## Pitfalls
- **Chunks without context.** The sentence chunk "They cost an extra ten cents per minute…" from the example index never says what "they" are. Prepend the document title and section path to each chunk before embedding.
- **Mismatched embeddings.** Queries and chunks must be embedded by the same model version, and changing the model means re-embedding the whole collection.
- **Evaluating on questions the model already knows.** A general-knowledge test set rewards the model's memory, not the retrieval. Use questions that only the documents can answer.

Eugene Yan's [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/) was a starting point for this post; the example corpus, code and measurements here are my own.

## References
1. Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), NeurIPS 2020.
2. Robertson and Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019), Foundations and Trends in Information Retrieval, 2009.
3. Karpukhin et al., [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), EMNLP 2020.
4. Cormack, Clarke and Büttcher, [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114), SIGIR 2009.
5. Johnson, Douze and Jégou, [Billion-scale Similarity Search with GPUs](https://arxiv.org/abs/1702.08734), IEEE Transactions on Big Data, 2019.
6. Malkov and Yashunin, [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320), IEEE TPAMI, 2020.
7. Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), TACL 2024.
8. Es et al., [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), EACL 2024 (demo track).
9. Nogueira and Cho, [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085), 2019.
