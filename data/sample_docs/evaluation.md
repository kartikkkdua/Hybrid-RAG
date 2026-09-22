# Evaluating a RAG system

A RAG system has two halves to evaluate: retrieval quality and answer quality.
Measuring them separately is what turns "it feels better" into a number you can
put on a resume.

## Retrieval metrics

Recall@k asks whether a known relevant chunk appears in the top k results. Mean
Reciprocal Rank, or MRR, rewards placing the first relevant result as high as
possible and equals the average of one over the rank of the first hit. Normalized
Discounted Cumulative Gain, nDCG, credits every relevant result but discounts
those further down the list, so it captures graded relevance and ordering.

## Answer metrics

Faithfulness measures whether the generated answer is actually supported by the
retrieved context; an unfaithful answer contradicts or invents facts not present
in the sources. Answer relevance measures whether the response addresses the
question that was asked. Context precision and context recall describe how much
of the retrieved context was useful and how much of the needed information was
retrieved. The RAGAS framework computes these with an LLM acting as a judge.

## Gold sets

A gold set is a fixed list of questions paired with the identifiers of the chunks
that truly answer them. Freezing the gold set lets you compare configurations
fairly: change the reranker, rerun the same questions, and read the delta in
recall and nDCG directly. Without a gold set, every comparison is anecdote.
