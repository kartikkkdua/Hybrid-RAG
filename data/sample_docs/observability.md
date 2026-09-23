# Observability and cost control

A retrieval system that cannot be measured cannot be improved, and one whose cost
is invisible will surprise someone at the end of the month.

## Tracing a query

Every query passes through several stages, and any of them can be the one that is
slow or wrong. A useful trace records, per request: the query as received and the
query actually searched, how many candidates each retrieval stage returned, the
scores at each stage, which passages reached the generator, how many tokens went
in and came out, and which citations verified. Recording per-stage latency
separately is what distinguishes a slow reranker from a slow language model.

## Percentiles, not averages

Mean latency hides the experience of the users having the worst time. The p95 and
p99 are what people actually complain about, and they behave differently from the
mean: adding a reranker might move the mean by ten milliseconds while moving the
p99 by hundreds, because the tail is dominated by the largest shortlists.

## Token accounting

Cost is a product of token counts and per-model prices, both of which change. The
price table belongs in one place in the code, and every call should return the
tokens it consumed so cost can be attributed per query, per feature and per user
rather than estimated from a monthly invoice.

## The cost levers

Three knobs dominate spend. Context size is the largest: passing twenty passages
instead of five quadruples input tokens for every single query. Model choice is
next, with an order of magnitude between the cheapest and most capable models.
Caching is third, and the only one that is free: identical and near-identical
queries are common in practice, and serving them from a cache costs nothing.

## Evaluation is part of observability

Latency and cost dashboards tell you the system is running; they say nothing
about whether it is right. Running a frozen evaluation set on every change, and
failing the build when retrieval quality regresses, is what keeps quality from
eroding quietly while the dashboards stay green.
