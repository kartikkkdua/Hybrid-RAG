# Text embeddings

An embedding model maps a passage to a fixed-length vector so that semantically
similar passages land close together. Retrieval then becomes a nearest-neighbour
problem in that vector space.

## Bi-encoders

A bi-encoder embeds the query and the document independently. Because the
document vectors do not depend on the query, the whole corpus can be encoded
once, offline, and reused for every subsequent search. That independence is
exactly what makes dense retrieval fast enough to run over millions of passages.
It also caps accuracy: the model never sees the query and the document together,
so it cannot reason about how they interact.

## Pooling

A transformer produces one vector per token, but retrieval needs one vector per
passage. Mean pooling averages the token vectors and is the most common choice.
CLS pooling instead takes the vector of a single special token that the model was
trained to use as a summary. Models are usually trained for one pooling strategy
and degrade noticeably if you swap it at inference time.

## Normalization and similarity

Most retrieval embeddings are L2-normalised, meaning every vector is scaled to
unit length. After normalisation the cosine similarity between two vectors equals
their dot product, which removes a division from the inner loop and lets a search
engine use the cheaper operation. Normalisation also means vector magnitude
carries no information, so a long document cannot dominate simply by being long.

## Dimensionality

Larger embedding dimensions capture more nuance but cost more memory and more
time per comparison. A 384-dimensional model stores 1.5 KB per passage in float32
while a 1536-dimensional model stores 6 KB. At a million passages that is the
difference between 1.5 GB and 6 GB of index. Matryoshka embeddings are trained so
that truncating the vector to a shorter prefix degrades quality gracefully,
letting one model serve several size and cost points.

## Asymmetric search

Queries and documents often look nothing alike: a three-word question against a
four-hundred-word passage. Models trained for asymmetric search use separate
instruction prefixes for the two sides, such as prefixing queries with a phrase
like "represent this sentence for searching relevant passages". Forgetting the
prefix at query time is a common and silent cause of poor retrieval quality.
