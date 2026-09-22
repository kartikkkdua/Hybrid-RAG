"""Command-line interface.

    hybridrag ingest data/sample_docs      # ingest a file or directory
    hybridrag search "what is RRF?"        # hybrid retrieval, per-stage scores
    hybridrag ask "what is RRF?"           # grounded answer + verified citations
    hybridrag sources                      # list ingested documents
    hybridrag stats
    hybridrag serve                        # run the API
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .service import RAGService

app = typer.Typer(add_completion=False, help="Hybrid-search RAG CLI")
console = Console()


@app.command()
def ingest(path: str):
    """Ingest a file or directory (.txt/.md/.pdf)."""
    svc = RAGService()
    results = svc.ingest_path(path)
    total_chunks = sum(r.n_chunks for r in results)
    for r in results:
        tag = "[yellow]skipped[/]" if r.skipped else f"[green]{r.n_chunks} chunks[/]"
        console.print(f"  {r.source}: {tag}")
    console.print(f"[bold]Ingested {len(results)} file(s), {total_chunks} new chunks.[/]")


@app.command()
def search(query: str, top_k: int = 8, no_rerank: bool = typer.Option(False, "--no-rerank")):
    """Hybrid retrieval with per-stage scoring."""
    svc = RAGService()
    resp = svc.search(query, top_k=top_k, rerank=not no_rerank)
    table = Table(title=f"Results for: {query}  ({resp.latency_ms} ms)")
    table.add_column("#", justify="right")
    table.add_column("source")
    table.add_column("bm25", justify="right")
    table.add_column("dense", justify="right")
    table.add_column("rrf", justify="right")
    table.add_column("rerank", justify="right")
    table.add_column("text")
    for r in resp.results:
        table.add_row(
            str(r.rank), r.source,
            f"{r.bm25_score:.2f}" if r.bm25_score is not None else "-",
            f"{r.dense_score:.3f}" if r.dense_score is not None else "-",
            f"{r.rrf_score:.4f}" if r.rrf_score is not None else "-",
            f"{r.rerank_score:.3f}" if r.rerank_score is not None else "-",
            (r.text[:90] + "…") if len(r.text) > 90 else r.text,
        )
    console.print(table)
    console.print(f"[dim]stages: {resp.stages}[/]")


@app.command()
def ask(query: str, top_k: int = 8, mode: str = "grounded"):
    """Grounded answer with verified inline citations."""
    svc = RAGService()
    ans = svc.answer(query, top_k=top_k, mode=mode)
    console.rule("Answer")
    if ans.refused:
        console.print(f"[yellow]REFUSED:[/] {ans.refusal_reason}")
    else:
        console.print(ans.answer)
    if ans.citations:
        console.rule("Citations")
        for i, c in enumerate(ans.citations, 1):
            mark = "[green]✓[/]" if c.verified else "[red]✗[/]"
            console.print(f"{mark} [{i}] {c.source} (chars {c.char_start}-{c.char_end})")
            console.print(f"    “{c.quote[:120]}”")
    console.rule("Usage")
    u = ans.usage
    console.print(
        f"model={u.model} in={u.input_tokens} out={u.output_tokens} "
        f"cost=${u.cost_usd:.5f} latency={u.latency_ms}ms"
    )
    console.print(f"[dim]verification: {ans.verification}[/]")


@app.command()
def agent(query: str, top_k: int = 8):
    """Answer via the multi-agent graph (router → research → critic)."""
    svc = RAGService()
    if not svc.agent_available():
        console.print("[red]langgraph not installed.[/] Run: make agent")
        raise typer.Exit(1)
    a = svc.agent_answer(query, top_k=top_k)

    console.rule("Agent trace")
    table = Table(show_header=True)
    table.add_column("node")
    table.add_column("detail")
    table.add_column("ms", justify="right")
    for t in a.trace:
        table.add_row(t.node, t.detail, f"{t.ms}")
    console.print(table)
    console.print(f"route=[bold]{a.route}[/] ({a.route_reason})")
    if a.subquestions and len(a.subquestions) > 1:
        console.print("sub-questions:")
        for s in a.subquestions:
            console.print(f"  • {s}")
    if a.escalated:
        console.print("[yellow]escalated:[/] lookup was handed off to the researcher")

    console.rule("Answer")
    if a.refused:
        console.print(f"[yellow]REFUSED:[/] {a.refusal_reason}")
    else:
        console.print(a.answer)
    for i, c in enumerate(a.citations, 1):
        mark = "[green]✓[/]" if c.verified else "[red]✗[/]"
        console.print(f"{mark} [{i}] {c.source} (chars {c.char_start}-{c.char_end})")
    u = a.usage
    console.print(
        f"[dim]iterations={a.iterations} in={u.input_tokens} out={u.output_tokens} "
        f"cost=${u.cost_usd:.5f} wall={u.latency_ms}ms[/]"
    )


@app.command()
def sources():
    """List ingested documents."""
    svc = RAGService()
    table = Table(title="Sources")
    table.add_column("source")
    table.add_column("title")
    table.add_column("chunks", justify="right")
    table.add_column("chars", justify="right")
    for s in svc.list_sources():
        table.add_row(s.source, s.title, str(s.n_chunks), str(s.n_chars))
    console.print(table)


@app.command()
def stats():
    """Show corpus + backend info."""
    svc = RAGService()
    for k, v in svc.stats().items():
        console.print(f"  {k}: [bold]{v}[/]")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("app.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
