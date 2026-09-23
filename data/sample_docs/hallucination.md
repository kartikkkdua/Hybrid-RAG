# Grounding and hallucination

A retrieval system that returns the right passage can still produce a wrong
answer, because generation is a separate failure surface from retrieval.

## Why grounded systems still hallucinate

Retrieval narrows what the model sees but does not compel it to use it. A model
can blend parametric knowledge with the provided context, over-generalise from a
partial match, or answer confidently when the context does not actually contain
the answer. Retrieval reduces hallucination; it does not eliminate it.

## Attribution and span verification

Asking the model to cite a chunk identifier alongside each claim is a start, but
a citation is only a claim about a claim. The stronger form is to require a
verbatim quote and then independently check that the quote actually occurs in the
cited chunk. A quote that cannot be located is evidence the model invented it,
and the claim resting on it should be dropped rather than displayed.

Mapping a verified quote back to absolute character offsets in the source
document turns attribution into something a reader can check in one glance,
because the exact span can be highlighted in the original file.

## Refusal as a feature

The most important behaviour of a grounded system is declining to answer. If no
claim can be supported by the retrieved context, returning nothing is strictly
better than returning something plausible. Systems that cannot refuse have no
upper bound on how wrong they can be, and users calibrate their trust to the
worst answer they have seen, not the average one.

## Faithfulness versus relevance

Faithfulness asks whether the answer is supported by the retrieved context.
Relevance asks whether it addresses the question. The two are independent: an
answer can be perfectly faithful to a passage that has nothing to do with the
question, and it can be highly relevant while inventing its supporting facts.
Measuring only one of them hides half the failures.
