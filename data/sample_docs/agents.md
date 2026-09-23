# Agents over retrieval

A single retrieval pass answers a single focused question. Questions that span
several facts, or that need a different search once the first results come back,
call for something with control flow.

## The routing decision

Not every question needs an agent. Decomposing "what constant does RRF use" into
sub-questions wastes latency and tokens to arrive at the same passage a single
search would have found. A router that classifies questions as simple lookups
versus genuine multi-part research keeps the cheap path cheap, and is usually the
highest-value node in the whole graph.

## Decomposition

A comparative or multi-part question is decomposed into independent
sub-questions, each answerable by one search. Evidence from all of them is pooled,
deduplicated by chunk, and passed to a synthesis step. The deduplication matters
because overlapping sub-questions frequently retrieve the same passages, and
paying to put the same text in the context twice is pure waste.

## Self-critique

A critic node inspects the draft before it reaches the user and can reject it.
The most reliable critic is not a language model at all but a hard check: if no
claim in the draft is supported by a verified citation, the draft fails
regardless of how confident it sounds. Cheap deterministic gates should run
before expensive judged ones.

## Escalation and termination

When the critic rejects a draft, the run can escalate: a question that took the
single-hop path is handed to the research path for a second, broader attempt.
Every such loop needs a hard iteration bound. An agent that can retry without
limit will, on some input, retry forever, and the correct behaviour when the
bound is reached is to refuse rather than to ship the last rejected draft.

## Observability of agents

An agent that cannot be inspected cannot be debugged. Recording which node ran,
in what order, with what timing and what decision, turns an opaque loop into
something explainable, and is what makes the difference between an agent you can
operate and a demo.
