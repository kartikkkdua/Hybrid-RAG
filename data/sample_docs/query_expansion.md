# Query expansion and rewriting

Short queries are ambiguous and underspecified. Expansion techniques rewrite or
enrich the query before retrieval so it better matches the language of the corpus.

## Conversational rewriting

In a multi-turn interface the user's follow-up often cannot stand alone. A
question like "why that value?" depends entirely on the previous turn, and
retrieval has no memory of its own: it sees only the string it is given. The fix
is to condense the conversation and the follow-up into a single self-contained
query before retrieving. Crucially the rewritten query should be used for
retrieval only, while generation still receives the user's original wording, so
the answer addresses what was actually asked rather than the expanded form.

## HyDE

Hypothetical Document Embeddings asks a language model to draft a fake answer to
the query, then embeds that draft and searches with it instead of the query. The
reasoning is that a hypothetical answer shares vocabulary and structure with real
answers, closing the gap between a terse question and a prose passage. It costs
one generation call per query and can mislead retrieval when the model
hallucinates a confidently wrong draft.

## Multi-query expansion

Rather than one rewrite, generate several paraphrases of the query, retrieve for
each, and fuse the result lists. This increases recall at the cost of multiple
retrievals, and pairs naturally with rank fusion since the outputs are already
several ranked lists over the same corpus.

## Pseudo-relevance feedback

A classical technique that needs no language model: run the query, assume the top
few results are relevant, extract their most distinctive terms, and re-run the
query enriched with those terms. It is cheap and often effective, but it amplifies
errors, because if the initial results are wrong the expansion drags the second
query further in the wrong direction.

## When not to expand

Expansion has a cost and a failure mode. A query that already contains a
distinctive identifier, an acronym or a proper noun usually retrieves better
unexpanded, because expansion dilutes the rare term that was doing the work.
Detecting whether a query is self-contained and skipping expansion when it is
keeps the common case both fast and accurate.
