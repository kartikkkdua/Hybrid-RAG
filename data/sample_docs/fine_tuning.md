# Fine-tuning retrieval models

Off-the-shelf embedding models are trained on general web text. Domain corpora
with unusual vocabulary often justify adapting the model rather than accepting
the mismatch.

## Contrastive training

Retrieval encoders are trained contrastively: pull a query and its relevant
passage together in vector space while pushing unrelated passages apart. The
quality of the negatives dominates the outcome. Random negatives are too easy and
teach the model almost nothing after the first epoch. Hard negatives, passages
that a current retriever ranks highly but which are actually irrelevant, are what
force the model to learn fine distinctions.

## In-batch negatives

Treating the other passages in a training batch as negatives makes each example
far cheaper, which is why larger batch sizes usually improve retrieval training.
The subtlety is false negatives: with a large batch, some of those other passages
genuinely answer the query, and penalising them teaches the model the opposite of
what was intended.

## LoRA and parameter-efficient adaptation

Low-Rank Adaptation freezes the base weights and trains small rank-decomposition
matrices injected into the attention layers. It cuts trainable parameters by
orders of magnitude, so adaptation fits on a single consumer GPU, and the
resulting adapter is a few megabytes that can be swapped per domain. QLoRA adds
4-bit quantisation of the frozen base, reducing memory further at a small
quality cost.

## Distillation from a cross-encoder

A strong but slow cross-encoder can teach a fast bi-encoder. Score a large set of
query-passage pairs with the cross-encoder and train the bi-encoder to reproduce
those scores. The student inherits much of the teacher's ranking ability while
keeping the precomputation that makes dense retrieval viable at scale.

## When not to fine-tune

Fine-tuning requires labelled data, a training loop, evaluation infrastructure
and a plan for re-embedding the entire corpus whenever the model changes. Before
paying that cost, exhaust the cheaper options: better chunking, hybrid retrieval,
and a reranker. Those routinely deliver larger gains for a fraction of the effort.
