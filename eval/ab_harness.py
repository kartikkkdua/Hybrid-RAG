"""Prompt / model A/B harness.

Runs the gold questions through several *answer* variants (model, reranking,
top_k) and reports per-variant cost, tokens and p50/p95 latency — the numbers you
use to justify "we shipped Haiku for retrieval-heavy queries at 1/4 the cost".

Cost/token columns are populated only when ANTHROPIC_API_KEY is set; latency and
verified-citation rate are measured either way (extractive fallback otherwise).

    python -m eval.ab_harness --docs data/sample_docs
    python -m eval.ab_harness --docs data/sample_docs --models claude-sonnet-5,claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.service import RAGService
from eval.run_eval import _pctl, load_gold


def make_variants(models: list[str]) -> list[dict]:
    variants = []
    for m in models:
        variants.append({"name": f"{m} | rerank+", "model": m, "rerank": True, "top_k": 8})
        variants.append({"name": f"{m} | rerank-", "model": m, "rerank": False, "top_k": 8})
    return variants


def run_variant(svc: RAGService, gold: list[dict], variant: dict) -> dict:
    latencies, costs, in_toks, out_toks, verified_rates, refusals = [], [], [], [], [], 0
    for item in gold:
        t0 = time.perf_counter()
        ans = svc.answer(item["question"], top_k=variant["top_k"],
                         rerank=variant["rerank"], mode="grounded",
                         model=variant["model"])
        latencies.append((time.perf_counter() - t0) * 1000)
        costs.append(ans.usage.cost_usd)
        in_toks.append(ans.usage.input_tokens)
        out_toks.append(ans.usage.output_tokens)
        if ans.refused:
            refusals += 1
        v = ans.verification
        if v.get("citations_total"):
            verified_rates.append(v["citations_verified"] / v["citations_total"])
    n = len(gold)
    return {
        "variant": variant["name"],
        "avg_cost_usd": round(statistics.mean(costs), 6) if costs else 0.0,
        "total_cost_usd": round(sum(costs), 5),
        "avg_in_tok": round(statistics.mean(in_toks)) if in_toks else 0,
        "avg_out_tok": round(statistics.mean(out_toks)) if out_toks else 0,
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(_pctl(latencies, 95), 1) if latencies else 0.0,
        "verified_citation_rate": round(statistics.mean(verified_rates), 3) if verified_rates else 0.0,
        "refusal_rate": round(refusals / n, 3) if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--gold", default=str(Path(__file__).parent / "gold.jsonl"))
    ap.add_argument("--models", default="")
    args = ap.parse_args()

    settings = Settings(DB_PATH="storage/ab.db")
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(settings.resolved_db_path()) + suffix)
        if p.exists():
            p.unlink()
    svc = RAGService(settings=settings)
    svc.ingest_path(args.docs)
    gold = load_gold(Path(args.gold))

    models = [m for m in args.models.split(",") if m] or [settings.gen_model]
    if not svc.llm.available:
        print("NOTE: no ANTHROPIC_API_KEY — running extractive fallback; "
              "cost/token columns will be zero, latency + citation rate are real.\n")
    variants = make_variants(models)

    rows = [run_variant(svc, gold, v) for v in variants]
    _print_table(rows)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(__file__).parent / "reports" / f"ab-{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"timestamp": ts, "llm": svc.llm.available, "variants": rows}, indent=2))
    print(f"\nSaved -> {out}")
    svc.close()


def _print_table(rows: list[dict]):
    cols = ["variant", "avg_cost_usd", "avg_in_tok", "avg_out_tok",
            "latency_p50_ms", "latency_p95_ms", "verified_citation_rate", "refusal_rate"]
    widths = {c: max(len(c), max(len(f"{r.get(c,'')}") for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(f"{r.get(c,'')}".ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
