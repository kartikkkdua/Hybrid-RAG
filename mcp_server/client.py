"""MCP client for the hybrid-rag server.

Two modes:

  # scripted round-trip demo (no API key needed) — proves the 4 tools work
  python -m mcp_server.client

  # agent mode: Claude decides which tools to call to answer a question
  python -m mcp_server.client --agent "What constant does RRF use?"

The client launches the server itself over stdio.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack

from mcp import Client, StdioServerParameters


def _server_params() -> StdioServerParameters:
    # Launch the server with the SAME interpreter running this client.
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], env=None)


def _text(result) -> str:
    from mcp_types import TextContent

    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


async def scripted_demo():
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(Client(_server_params(), mode="auto"))
        tools = (await client.list_tools()).tools
        print("Connected. Tools:", [t.name for t in tools])

        print("\n[ingest_document]")
        r = await client.call_tool("ingest_document", {
            "text": "Reciprocal Rank Fusion merges ranked lists using 1/(k+rank), with k "
                    "commonly set to 60. It ignores raw score scales.",
            "source": "mcp-demo.md", "title": "RRF note",
        })
        print(" ->", _text(r)[:200])

        print("\n[list_sources]")
        r = await client.call_tool("list_sources", {})
        print(" ->", _text(r)[:200])

        print("\n[search_corpus]  query='what constant does RRF use?'")
        r = await client.call_tool("search_corpus", {"query": "what constant does RRF use?", "top_k": 2})
        data = json.loads(_text(r))
        for res in data.get("results", []):
            print(f"   #{res['rank']} {res['source']} score={res['score']} :: {res['text'][:70]}…")

        print("\n[answer_question]  query='what constant does RRF use?'")
        r = await client.call_tool("answer_question", {"query": "what constant does RRF use?"})
        ans = json.loads(_text(r))
        print("   refused:", ans["refused"])
        print("   answer :", ans["answer"][:160])
        print("   citations:", [(c["source"], c["verified"]) for c in ans["citations"]])


async def agent_mode(question: str):
    import os

    from anthropic import Anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("agent mode needs ANTHROPIC_API_KEY; run without --agent for the scripted demo.")
        return
    anthropic = Anthropic()
    model = os.getenv("GEN_MODEL", "claude-sonnet-5")

    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(Client(_server_params(), mode="auto"))
        tools_response = await client.list_tools()
        available = [{"name": t.name, "description": t.description,
                      "input_schema": t.input_schema} for t in tools_response.tools]
        print("Connected. Tools:", [t["name"] for t in available])

        messages = [{"role": "user", "content": question}]
        for _ in range(5):
            resp = anthropic.messages.create(model=model, max_tokens=1024,
                                             messages=messages, tools=available)
            tool_uses = [c for c in resp.content if c.type == "tool_use"]
            for c in resp.content:
                if c.type == "text" and c.text.strip():
                    print("\nClaude:", c.text)
            if not tool_uses:
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                print(f"[tool] {tu.name}({tu.input})")
                out = await client.call_tool(tu.name, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": _text(out), "is_error": bool(out.is_error)})
            messages.append({"role": "user", "content": results})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="", help="run Claude agent mode with this question")
    args = ap.parse_args()
    if args.agent:
        asyncio.run(agent_mode(args.agent))
    else:
        asyncio.run(scripted_demo())


if __name__ == "__main__":
    main()
