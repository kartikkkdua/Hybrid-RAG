# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation grounds a language model's answers in an external
corpus. Instead of relying only on parametric knowledge, the system retrieves
relevant passages at query time and conditions generation on them. This reduces
hallucination and lets the model cite its sources.

## Hybrid retrieval

A hybrid retriever combines sparse lexical search with dense semantic search.
Sparse methods such as BM25 match exact terms and are strong on rare keywords,
acronyms, and identifiers. Dense retrieval encodes text into vectors and matches
on meaning, so it handles paraphrase and synonymy that BM25 misses. Running both
and merging the results recovers documents that either method alone would drop.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion, or RRF, merges several ranked lists using only the rank
positions rather than the raw scores. The fused score of a document is the sum
over all lists of one divided by the quantity k plus the rank of the document in
that list. The constant k, commonly set to 60, dampens the influence of the very
top ranks. Because RRF ignores raw score magnitudes, it combines BM25 and cosine
scores cleanly even though those scores live on completely different scales.

## Cross-encoder reranking

After fusion, a cross-encoder reranker reorders the shortlist. Unlike the
bi-encoder used for dense retrieval, a cross-encoder reads the query and a
candidate passage together in a single forward pass, which yields a much more
accurate relevance score at the cost of higher latency. Reranking only the top
candidates keeps that cost bounded while sharply improving precision at the top.
