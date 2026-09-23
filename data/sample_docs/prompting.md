# Prompting for grounded generation

The prompt is the interface between retrieved evidence and the model's output,
and small changes to it move quality more than most people expect.

## Structuring the context

Passages should be clearly delimited and individually labelled, so the model can
refer to a specific one when citing. Including each passage's identifier and
source in its header is what makes a citation contract enforceable: the model has
something unambiguous to name.

Ordering matters. Models attend unevenly across a long context, with the
beginning and end receiving more weight than the middle, a pattern often called
lost in the middle. Placing the highest-scoring passages at the edges of the
context rather than burying them in the centre measurably helps.

## Structured output contracts

Asking for prose and parsing it afterwards is fragile. Specifying an exact JSON
shape, validating the response against a schema, and retrying once with the
validation error appended turns an unreliable text interface into something
closer to a typed function call. The retry matters: most malformed outputs are
recoverable when the model is shown precisely what was wrong.

## Instructing refusal

Models default to helpfulness, which in a grounded system means answering when
they should decline. The instruction to refuse has to be explicit, and it works
better when the output schema has somewhere to put the refusal, such as a boolean
field, rather than requiring the model to break format to express uncertainty.

## Temperature and determinism

Grounded extraction wants temperature at zero. Sampling adds variation that
serves creative writing and actively harms tasks where the correct output is
determined by the context. Determinism also makes evaluation meaningful, since a
change in the score can be attributed to the change you made rather than to
sampling noise.
