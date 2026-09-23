# Chunking strategies

Retrieval operates on chunks, not documents, so chunking silently determines the
ceiling on everything downstream. A chunk that splits an answer in half cannot be
retrieved correctly no matter how good the embedding model is.

## Fixed-size windows

The simplest strategy slices text every N tokens with a fixed overlap. It is
predictable and trivially parallel, but it cuts through sentences and tables at
arbitrary points. Overlap exists to mitigate exactly this: by repeating the last
fifty to a hundred tokens of the previous window, an answer that straddles a
boundary still appears intact in one of the two chunks.

## Recursive and structural splitting

A better approach splits on the largest natural boundary that fits: paragraphs
first, then sentences, then words. Markdown headings, HTML sections and PDF page
breaks are all strong structural signals. Preserving the heading path in each
chunk's text gives both the retriever and the reader context that the raw
paragraph lacks.

## Semantic chunking

Semantic chunking embeds each sentence and starts a new chunk when consecutive
sentences diverge beyond a similarity threshold. It produces topically coherent
chunks but costs an embedding pass over every sentence at ingestion time, and the
threshold is corpus-specific and fiddly to tune.

## Size trade-offs

Small chunks retrieve precisely but fragment context, so the generator sees
snippets without the surrounding argument. Large chunks preserve context but
dilute the embedding, because a single vector must represent several topics at
once, and they waste tokens in the generation prompt. Somewhere between two
hundred and five hundred tokens is a common compromise.

## Offsets are worth keeping

Recording the exact character offsets of every chunk within its parent document
costs almost nothing at ingestion and enables things that are impossible to
retrofit: highlighting a cited span in the original file, verifying that a quoted
passage genuinely appears where the model claims, and merging adjacent chunks
back into contiguous context.
