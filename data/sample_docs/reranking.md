# Reranking

A retriever optimises for recall over a huge corpus; a reranker optimises for
precision over a small shortlist. Running the two in sequence is a classic
cascade: a cheap model narrows millions of candidates to dozens, an expensive
model orders those dozens correctly.

## Cross-encoders

A cross-encoder concatenates the query and a candidate passage into a single
input and runs them jointly through a transformer, producing one relevance score.
Because every layer can attend across the boundary between query and passage, it
models their interaction directly and is substantially more accurate than any
bi-encoder. The price is that nothing can be precomputed: scoring fifty
candidates means fifty forward passes at query time, which is why reranking is
applied to a shortlist rather than the corpus.

## Late interaction

ColBERT sits between the two extremes. It stores one vector per token rather than
one per passage and computes relevance as a sum of maximum similarities between
query tokens and document tokens. This keeps most of the precomputation benefit
of a bi-encoder while recovering some of the interaction modelling of a
cross-encoder, at the cost of a much larger index.

## Listwise reranking

Pointwise rerankers score each candidate in isolation. Listwise rerankers see
several candidates at once and order them relative to each other, which lets them
exploit comparisons a pointwise model cannot make. Large language models can be
prompted to rerank listwise, though latency and cost usually confine that to
offline evaluation or very small shortlists.

## How deep to rerank

Reranking depth is a direct latency-quality dial. Reranking the top fifty
candidates catches relevant passages the first stage ranked poorly, but costs
roughly five times the compute of reranking the top ten. The right depth depends
on how often the first stage buries a relevant result, which is measurable: look
at recall at fifty versus recall at ten. If they are equal, deeper reranking has
nothing left to find.
