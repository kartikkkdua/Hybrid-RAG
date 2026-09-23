# Vector indexes

Once passages are embedded, something has to find the nearest neighbours of a
query vector quickly. The choice of index is a trade between recall, latency,
memory and build time.

## Flat (exact) search

A flat index compares the query against every stored vector. It is exact by
definition, returns perfect recall, and needs no training or tuning. The cost is
linear in corpus size. For tens of thousands of vectors a flat scan in optimised
code takes single-digit milliseconds, which is why small deployments should
simply use it and avoid the complexity of approximate search entirely.

## IVF

An inverted file index clusters vectors with k-means and searches only the
clusters nearest the query. The nprobe parameter sets how many clusters to visit:
raising it increases recall and latency together. IVF needs a training pass over
a representative sample before it can be populated, and its quality degrades if
the data distribution shifts away from the sample it was trained on.

## HNSW

Hierarchical Navigable Small World graphs connect each vector to its neighbours
in a layered graph and walk that graph greedily toward the query. HNSW gives
excellent recall at low latency and, unlike IVF, requires no separate training
step. Its costs are memory, since the graph edges are stored alongside the
vectors, and build time, since every insertion performs its own search. The
parameter m controls how many edges each node keeps, while ef_construction and
ef_search trade accuracy against speed at build and query time respectively.

## Product quantization

Product quantization compresses vectors by splitting them into subvectors and
replacing each with the identifier of the nearest centroid in a learned codebook.
Compression of sixteen to thirty-two times is routine, which is what makes
billion-scale indexes fit in memory. The cost is that distances become
approximate, so PQ is usually paired with a reranking pass over the full-precision
vectors of the top candidates.

## Recall is a tunable, not a constant

Every approximate index has a knob that trades recall for speed. Reporting the
latency of an approximate index without reporting its recall is meaningless,
because any index can be made arbitrarily fast if it is allowed to return the
wrong answers.
