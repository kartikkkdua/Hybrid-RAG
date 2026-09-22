# The Model Context Protocol (MCP)

The Model Context Protocol is an open standard that lets AI applications connect
to external tools and data through a uniform interface. An MCP server exposes
capabilities; an MCP client, embedded in an assistant, discovers and calls them.

## Tools, resources, and prompts

An MCP server can expose three kinds of capability. Tools are callable functions
with typed inputs, such as "search the corpus" or "ingest a document". Resources
are readable data the client can pull in as context. Prompts are reusable message
templates. This project's server exposes four tools: search_corpus, answer_question,
ingest_document, and list_sources, which together let any MCP-aware agent use the
hybrid retriever as a backend.

## Why it matters

Before MCP, every assistant integrated every tool with bespoke glue code. By
standardizing the transport and the schema, MCP lets a single server be reused
across many clients, and a single client consume many servers. That is why wiring
a retriever behind an MCP server, rather than a one-off HTTP endpoint, makes it
immediately usable from Claude Desktop and other MCP hosts.
