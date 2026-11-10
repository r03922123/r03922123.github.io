# Vector databases: nearest-neighbor search at scale

A vector database stores embeddings, one vector per document, image, product or user, and answers one question quickly: given a query vector, which stored vectors are most similar to it? That question sits under semantic search, the retrieval step of [retrieval-augmented generation](../rag-architecture/), and the candidate stage of [two-tower recommenders](../two-tower-retrieval/).

Answering it exactly means comparing the query with every stored vector. That is fine for thousands of vectors and too slow or too expensive for hundreds of millions, so vector databases use **approximate nearest-neighbor (ANN)** search: return most of the true top-k while looking at a small fraction of the data. This post builds the three main ANN ideas in NumPy on the same synthetic data, measures each one against brute force, and then covers what a database adds on top of an index.

## Exact search is one matrix product
Many embedding models are trained with cosine similarity, so the usual first step is to scale every vector to unit length. Then cosine similarity is a plain dot product, and Euclidean distance gives the same ranking, because for unit vectors

\[ \lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\, a^\top b = 2 - 2\, a^\top b . \]

With \(n\) stored vectors of dimension \(d\) as rows of a matrix \(X\), exact search for a batch of queries \(Q\) is the product \(QX^\top\) followed by a top-k per row. The cost is \(O(nd)\) per query, and the whole of \(X\) has to be read for every batch.

The data below is synthetic but not uniform: points are scattered around 100 random centers, roughly like embeddings of documents on 100 topics. That matters, because ANN methods exploit structure, and uniformly random high-dimensional vectors are close to the worst case for all of them.

```python
import heapq
import numpy as np

rng = np.random.default_rng(0)
n, d, k = 10_000, 64, 10


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


centers = rng.standard_normal((100, d))


def sample(m, spread=1.5):
    which = rng.integers(len(centers), size=m)
    return unit(centers[which] + spread * rng.standard_normal((m, d))).astype(np.float32)


X = sample(n)          # the stored vectors
Q = sample(200)        # queries from the same distribution, not copies of stored vectors


def exact_search(Q, X, k):
    S = Q @ X.T                                            # every query against every vector
    top = np.argpartition(-S, k, axis=1)[:, :k]            # unordered top k, O(n) per row
    order = np.argsort(-np.take_along_axis(S, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


truth = exact_search(Q, X, k)
assert (truth[:5] == np.argsort(-(Q[:5] @ X.T), axis=1)[:, :k]).all()   # matches a full sort


def recall_at_k(found, truth):
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)]))


# Cosine (dot product of unit vectors) and Euclidean distance rank identically.
dist2 = ((Q[:, None, :] - X[None, :500, :]) ** 2).sum(-1)
assert np.allclose(dist2, 2 - 2 * Q @ X[:500].T, atol=1e-5)
assert (np.argsort(dist2[0])[:k] == np.argsort(-(Q[0] @ X[:500].T))[:k]).all()

# Memory for exact search over 10 million 768-dimensional float32 vectors:
assert 10_000_000 * 768 * 4 / 1e9 == 30.72                 # GB, read in full for every query batch
```

Every approximate method below is scored by **recall@k**: the fraction of the exact top-k that it returns. Note what that measures. It is fidelity to brute force, not relevance. If the embedding model puts the wrong documents nearest the query, a perfect index returns them faithfully.

If the model was trained with raw, unnormalized dot products, the task is maximum inner product search, which ranks differently from Euclidean distance or cosine; pick the index metric to match how the model was trained.

## Inverted lists: search only the nearby clusters
The first idea is to partition. Run k-means on the stored vectors to get `nlist` centroids, and file each vector in the list of its nearest centroid. This is an **inverted file index (IVF)**: the centroids divide the space into Voronoi cells, and each cell keeps a list of its members. At query time, rank the centroids by distance to the query, open the `nprobe` closest lists, and score only their members exactly.

```python
def kmeans(Z, c, iters=10, seed=0):
    r = np.random.default_rng(seed)
    C = Z[r.choice(len(Z), c, replace=False)].astype(np.float64)
    for _ in range(iters):
        a = (-2 * Z @ C.T + (C ** 2).sum(1)).argmin(1)    # nearest centroid; |z|^2 is constant per row
        counts = np.bincount(a, minlength=c)
        sums = np.zeros_like(C)
        np.add.at(sums, a, Z)
        keep = counts > 0
        C[keep] = sums[keep] / counts[keep, None]
    a = (-2 * Z @ C.T + (C ** 2).sum(1)).argmin(1)
    return C.astype(np.float32), a


nlist = 100
centroids, cell = kmeans(X, nlist)
lists = [np.flatnonzero(cell == j) for j in range(nlist)]


centroid_sq = (centroids ** 2).sum(1)


def ivf_search(q, nprobe, k=10):
    probe = np.argsort(centroid_sq - 2 * centroids @ q)[:nprobe]   # nearest centroids, same rule as assignment
    cand = np.concatenate([lists[j] for j in probe])
    return cand[np.argsort(-(X[cand] @ q))[:k]], len(cand)


ivf = {}
for nprobe in [1, 4, 8, 32]:
    res = [ivf_search(q, nprobe) for q in Q]
    ivf[nprobe] = (recall_at_k([r[0] for r in res], truth), np.mean([r[1] for r in res]) / n)

recalls = [r for r, _ in ivf.values()]
assert recalls == sorted(recalls)                  # more lists opened, never lower recall
assert ivf[1][0] < 0.85 and ivf[32][0] > 0.95
assert ivf[8][1] < 0.1                             # nprobe 8 scores under 10% of the vectors
assert ivf_search(Q[0], nlist)[0].tolist() == truth[0].tolist()   # all lists = exact search
```

On this data the trade-off looks like this (fraction scanned is the share of stored vectors scored exactly, not counting the 100 centroid comparisons):

| nprobe | recall@10 | fraction scanned |
|---|---|---|
| 1 | about 0.76 | 1% |
| 4 | about 0.90 | 4% |
| 8 | about 0.94 | 9% |
| 32 | about 0.99 | 34% |

Recall can only rise with `nprobe`, since a larger probe set is a superset and members are scored exactly. Misses happen when a true neighbor sits just across a cell boundary, which in high dimensions is common. The knobs interact: more lists (`nlist`) make each list shorter, so a fixed `nprobe` is faster but covers less of the space. A common starting point is `nlist` of a few times \(\sqrt{n}\), with `nprobe` tuned at query time, which needs no rebuild.

## Product quantization: shrink the vectors
IVF reduces how many vectors are scored, but each one is still stored in full. Thirty gigabytes of float32 per copy is expensive to hold in RAM, and the scan is limited by memory bandwidth. **Product quantization** (PQ), introduced by Hervé Jégou, Matthijs Douze and Cordelia Schmid (2011), compresses each vector to a few bytes.

Split each \(d\)-dimensional vector into \(m\) sub-vectors of \(d/m\) dimensions. For each sub-space, run k-means with 256 centroids, so that a sub-vector is replaced by the one-byte index of its nearest centroid. A vector becomes \(m\) bytes. With \(d = 64\) and \(m = 8\), that is 8 bytes instead of 256.

The search side is what makes PQ fast. The query is not compressed (this is called asymmetric distance computation). For each sub-space, compute the dot product of the query's sub-vector with all 256 centroids once, giving an \(m \times 256\) table. The approximate score of any stored vector is then \(m\) table lookups and a sum:

\[ q^\top \hat{x} = \sum_{j=1}^{m} q_j^\top \, c_j[\text{code}_j(x)] . \]

```python
m, ks = 8, 256
ds = d // m
books = []
codes = np.empty((n, m), dtype=np.uint8)
for j in range(m):
    book, code = kmeans(X[:, j * ds:(j + 1) * ds], ks, iters=8, seed=j)
    books.append(book)
    codes[:, j] = code


def pq_scores(q):
    tables = np.stack([books[j] @ q[j * ds:(j + 1) * ds] for j in range(m)])   # m x 256, built once per query
    return tables[np.arange(m), codes].sum(axis=1)                             # m lookups per stored vector


# The lookup score is exactly the dot product with the decoded vector.
decoded = np.hstack([books[j][codes[:, j]] for j in range(m)])
assert np.allclose(pq_scores(Q[0]), decoded @ Q[0], atol=1e-4)
assert X.nbytes // codes.nbytes == 32                                          # 256 bytes -> 8 bytes

pq_only, reranked = [], []
for q in Q:
    s = pq_scores(q)
    shortlist = np.argpartition(-s, 100)[:100]                                 # best 100 by compressed score
    pq_only.append(shortlist[np.argsort(-s[shortlist])[:k]])
    reranked.append(shortlist[np.argsort(-(X[shortlist] @ q))[:k]])          # rescore with full vectors

pq_recall, rerank_recall = recall_at_k(pq_only, truth), recall_at_k(reranked, truth)
assert pq_recall < 0.6 < 0.85 < rerank_recall
```

At 32 times compression the compressed ranking alone gets only about 40% of the true top 10, because quantization error is large compared with the small gaps between near neighbors. But about nine in ten of the true neighbors are somewhere in the compressed top 100, so rescoring that shortlist with the full vectors recovers about 90%. That is the standard pattern: a compressed scan to get a shortlist, then an exact rerank, with the full vectors on disk or in cheaper storage.

In practice IVF and PQ are combined. A common refinement encodes each vector's offset from its IVF centroid rather than the vector itself, which spends the code bits where they help. ScaNN, from Ruiqi Guo and colleagues at Google (2020), changes the quantization objective instead: it penalizes the error parallel to the data point more than the error orthogonal to it, because the parallel component is what changes large inner products.

## Navigable graphs: walk towards the query
The third idea needs no partition at all. Link each vector to its nearest neighbors, then answer a query by walking the graph: start somewhere, move to whichever neighbor is closer to the query, and stop when no neighbor improves. A plain greedy walk stalls in local optima, so the search keeps a beam of the best `ef` nodes found so far and keeps expanding until the closest unexpanded candidate is worse than the worst node in the beam.

**HNSW** (Hierarchical Navigable Small World graphs, Yury Malkov and Dmitry Yashunin, 2016) is the graph index that most vector databases offer, and often their default. It inserts vectors one at a time, chooses each node's links with a heuristic that keeps some long-range edges, and stacks sparser layers on top so that the search starts with long jumps and finishes with short ones. The sketch below keeps the core and simplifies the rest: the bottom layer is an exact 12-nearest-neighbor graph with reverse edges added, and a random 1% of nodes stands in for the upper layers by picking the starting point.

```python
M = 12
nbrs = np.empty((n, M), dtype=np.int64)
for start in range(0, n, 1000):                       # exact kNN graph, built in chunks
    S = X[start:start + 1000] @ X.T
    S[np.arange(len(S)), np.arange(start, start + len(S))] = -np.inf
    nbrs[start:start + 1000] = np.argpartition(-S, M, axis=1)[:, :M]
adj = [set(row) for row in nbrs]
for i in range(n):
    for j in nbrs[i]:
        adj[j].add(i)                                  # make edges two-way
adj = [np.fromiter(a, dtype=np.int64) for a in adj]
hubs = rng.choice(n, 100, replace=False)               # stand-in for HNSW's upper layers


def graph_search(q, ef, k=10):
    entry = hubs[np.argmax(X[hubs] @ q)]
    s = float(X[entry] @ q)
    visited, n_scored = {entry}, len(hubs) + 1
    frontier = [(-s, entry)]                           # max-heap of candidates to expand
    beam = [(s, entry)]                                # min-heap of the best ef found so far
    while frontier:
        neg, node = heapq.heappop(frontier)
        if len(beam) >= ef and -neg < beam[0][0]:
            break                                      # nothing left that can improve the beam
        new = [v for v in adj[node] if v not in visited]
        visited.update(new)
        n_scored += len(new)
        for v, sv in zip(new, X[new] @ q if new else []):
            if len(beam) < ef or sv > beam[0][0]:
                heapq.heappush(frontier, (-sv, v))
                heapq.heappush(beam, (sv, v))
                if len(beam) > ef:
                    heapq.heappop(beam)
    return [v for _, v in sorted(beam, reverse=True)[:k]], n_scored


graph = {}
for ef in [10, 40, 160]:
    res = [graph_search(q, ef) for q in Q]
    graph[ef] = (recall_at_k([r[0] for r in res], truth), np.mean([r[1] for r in res]) / n)

assert graph[10][0] < graph[40][0] < graph[160][0]
assert graph[40][0] > 0.85 and graph[40][1] < 0.06     # ~90% recall scoring ~4% of the vectors
assert graph[160][0] > 0.95
```

At `ef = 40` the walk returns about 90% of the true top 10 after scoring about 4% of the vectors, and `ef = 160` reaches about 98% at 13%. At ten thousand vectors that is roughly what IVF achieves for the same work (the graph's count includes its 100 entry points, IVF's leaves out its 100 centroids). The difference shows at scale: the number of steps a well-built navigable graph needs grows slowly with \(n\), which is why HNSW-style graphs sit near the top of public recall-versus-throughput benchmarks of in-memory indexes. The price is memory, since the full vectors and every edge live in RAM, and slow, memory-hungry builds. DiskANN (Subramanya and colleagues, 2019) is a graph index designed to keep most of that on SSD.

## Filters and hybrid search
Real queries come with conditions: only documents this user may see, only products in stock. There are two naive ways to apply them, and both fail somewhere.

- **Post-filtering** runs the ANN search, takes the top candidates, and drops those that fail the filter. If the filter is selective, few or none survive.
- **Pre-filtering** restricts the search to items that pass. With an exact scan that is correct, and when only a small set passes it is also cheap. Inside an index it is harder: a graph walk that may only step on passing nodes can find its path cut off.

```python
allowed = rng.random(n) < 0.02                         # a filter that 2% of items pass

post, pre = [], []
for q in Q:
    cand, _ = ivf_search(q, nprobe=8, k=50)            # ANN top 50, then filter
    post.append(cand[allowed[cand]][:k])
    ids = np.flatnonzero(allowed)                      # filter first, then exact search on the rest
    pre.append(ids[np.argsort(-(X[ids] @ q))[:k]])

assert np.mean([len(p) for p in post]) < 2             # usually one or no results instead of 10
assert all(len(p) == k and allowed[p].all() for p in pre)
```

Several vector databases therefore plan filtered queries: they estimate how selective the filter is, scan the passing items exactly when few pass, and otherwise search the index with the filter applied during traversal and a wider beam.

The second common addition is **hybrid search**. Embeddings are good at paraphrase and weak at exact tokens such as part numbers, error codes, names and rare terms, which keyword search with BM25 handles well. Hybrid search runs both and merges the two ranked lists. Their scores are on different scales, so the simplest robust merge uses only ranks. Reciprocal rank fusion (Gordon Cormack, Charles Clarke and Stefan Büttcher, 2009) scores each document by

\[ \text{RRF}(d) = \sum_{\text{lists } L} \frac{1}{60 + \text{rank}_L(d)} , \]

where the constant 60 is the value from their paper and damps the advantage of the very top ranks.

```python
def rrf(*rankings, c=60):
    score = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1 / (c + rank)
    return sorted(score, key=score.get, reverse=True)


# Query: "error E1043 at startup". Dense search finds startup problems in general;
# keyword search finds the one document that contains the code.
dense = ["boot-slow", "startup-crash", "config-reset", "driver-timeout", "e1043-fix"]
keyword = ["e1043-fix", "error-codes-index", "startup-crash"]
fused = rrf(dense, keyword)
assert fused[:2] == ["startup-crash", "e1043-fix"]     # in both lists, so both rise to the top
assert fused.index("e1043-fix") < dense.index("e1043-fix")
```

The fused top results usually go to a cross-encoder reranker, which reads the query and each candidate together and is too slow to run over the whole collection.

## What the database adds, and how to choose
An index answers top-k queries over a fixed set of vectors. A vector database wraps one or more indexes with the parts a production system needs: storage of the vectors and metadata, inserts, updates and deletes (often as tombstones that are cleaned up when segments are rebuilt), filtered queries, sharding and replication, and backups. Those operations are where the index types differ in practice. IVF centroids are trained once and go stale as the data distribution shifts, and HNSW handles inserts well but deletes poorly.

The choices fall into three groups.

| Option | Examples | Fits when |
|---|---|---|
| A library inside your service | FAISS, ScaNN, hnswlib | you control the infrastructure, the index fits on the machine, and updates come in batches |
| A vector extension to a database you already run | pgvector for PostgreSQL, vector fields in Elasticsearch or OpenSearch | millions of vectors, you need joins or transactions with other data, and you want one system to operate |
| A dedicated vector database | Milvus, Qdrant, Weaviate, Chroma, or managed services such as Pinecone | large or fast-changing collections, heavy filtering, many tenants, or you want the operations handled |

FAISS (Jeff Johnson, Matthijs Douze and Hervé Jégou, at Facebook AI Research, now Meta AI) is a library, not one algorithm: it offers exact, IVF, PQ, HNSW and combined indexes on CPU and GPU, and is a good place to prototype before choosing a service.

A rough guide to the index itself:

1. Under about a million vectors, try exact search first. A batched matrix product on a GPU, or even a CPU, is often fast enough, and recall is 1.
2. If the vectors fit in RAM and latency matters most, use HNSW and tune `ef` at query time.
3. If memory is the constraint, use IVF with PQ and rerank a shortlist with full vectors.
4. If the collection is much larger than RAM, use a disk-based graph index or shard across machines.

## Pitfalls
- **Not measuring recall.** Keep a sample of real queries, compute their exact top-k offline, and track the index's recall as the data grows. Tuning `nprobe` or `ef` without this is guessing.
- **Blaming the index for bad results.** When retrieval quality is poor, first check exact search with the same embeddings. If that is also poor, the problem is the embedding model, the chunking, or the query, and no index setting will fix it.
- **Mixing embedding models.** Vectors from different models, or different versions of one model, live in different spaces. Changing the model means re-embedding and re-indexing everything.
- **Inconsistent normalization.** Normalize stored vectors and queries the same way and match the index metric to how the model was trained.
- **Benchmarking on someone else's data.** Recall and speed depend on the structure of the vectors; uniform random data makes every index look bad, and public benchmark numbers do not transfer. Measure on your own embeddings.

For the retrieval pipeline around the index, see [RAG architecture](../rag-architecture/); for how the embeddings are trained, [contrastive losses](../contrastive-loss/) and [two-tower retrieval](../two-tower-retrieval/). Eugene Yan's essays on [patterns for LLM systems](https://eugeneyan.com/writing/llm-patterns/) and [real-time recommendations](https://eugeneyan.com/writing/real-time-recommendations/) are good further reading on putting these indexes into production.
