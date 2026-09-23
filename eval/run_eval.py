"""Retrieval evaluation + config comparison table.

Runs the same frozen gold set through four retrieval configurations and reports
recall@5 / MRR / nDCG@10 / latency for each — the delta between rows is the
evidence that hybrid + rerank beats either method alone. Optionally (--judge)
adds RAGAS-style faithfulness and answer-relevance on grounded answers.

    python -m eval.run_eval --docs data/sample_docs
    python -m eval.run_eval --docs data/sample_docs --judge     # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.generation import locate_span
from app.models import Chunk
from app.service import RAGService
from eval import metrics as M

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# The four rows of the comparison table.
CONFIGS = [
    {"name": "dense-only",       "bm25": False, "dense": True,  "rerank": False},
    {"name": "bm25-only",        "bm25": True,  "dense": False, "rerank": False},
    {"name": "hybrid (RRF)",     "bm25": True,  "dense": True,  "rerank": False},
    {"name": "hybrid + rerank",  "bm25": True,  "dense": True,  "rerank": True},
]


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_relevant(gold_item: dict, chunks: list[Chunk]) -> set[str]:
    """A chunk is relevant if it contains any gold answer_span (whitespace-tolerant),
    restricted to the gold item's source when given."""
    spans = gold_item.get("answer_spans") or [gold_item.get("answer_span", "")]
    source = gold_item.get("source")
    relevant: set[str] = set()
    for ch in chunks:
        if source and ch.source != source:
            continue
        if any(locate_span(ch.text, s) for s in spans if s):
            relevant.add(ch.chunk_id)
    return relevant


def evaluate_config(svc: RAGService, gold: list[dict], chunks: list[Chunk],
                    cfg: dict, k_recall=5, k_ndcg=10) -> dict:
    per_query = []
    latencies = []
    unresolved = 0
    for item in gold:
        relevant = resolve_relevant(item, chunks)
        if not relevant:
            unresolved += 1
            continue
        t0 = time.perf_counter()
        resp = svc.search(item["question"], top_k=k_ndcg,
                          bm25=cfg["bm25"], dense=cfg["dense"], rerank=cfg["rerank"])
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [r.chunk_id for r in resp.results]
        per_query.append({
            "recall@5": M.recall_at_k(ranked, relevant, k_recall),
            "hit@5": M.hit_at_k(ranked, relevant, k_recall),
            "mrr": M.reciprocal_rank(ranked, relevant),
            "ndcg@10": M.ndcg_at_k(ranked, relevant, k_ndcg),
        })
    agg = M.aggregate(per_query)
    agg["latency_p50_ms"] = round(statistics.median(latencies), 2) if latencies else 0.0
    agg["latency_p95_ms"] = round(_pctl(latencies, 95), 2) if latencies else 0.0
    agg["n_queries"] = len(per_query)
    agg["unresolved_gold"] = unresolved
    return agg


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[idx]


def run_judge(svc: RAGService, gold: list[dict], cfg: dict) -> dict | None:
    if not svc.llm.available:
        return None
    from eval.ragas_eval import answer_relevance, faithfulness

    faith, rel, cost = [], [], 0.0
    for item in gold:
        ans = svc.answer(item["question"], top_k=8,
                         rerank=cfg["rerank"], mode="grounded")
        if ans.refused:
            continue
        contexts = [r.text for r in ans.retrieved]
        f = faithfulness(ans.answer, contexts, svc.llm)
        r = answer_relevance(item["question"], ans.answer, svc.llm)
        if f:
            faith.append(f["faithfulness"]); cost += f["usage"]["cost_usd"]
        if r:
            rel.append(r["answer_relevance"]); cost += r["usage"]["cost_usd"]
        cost += ans.usage.cost_usd
    return {
        "faithfulness": round(statistics.mean(faith), 3) if faith else None,
        "answer_relevance": round(statistics.mean(rel), 3) if rel else None,
        "judged_cost_usd": round(cost, 5),
        "n_answered": len(faith),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--gold", default=str(Path(__file__).parent / "gold.jsonl"))
    ap.add_argument("--judge", action="store_true", help="also run RAGAS-style answer metrics")
    ap.add_argument("--out-json", default="", help="also write the report to this exact path (for CI)")
    args = ap.parse_args()

    # Fresh, reproducible corpus in a scratch DB.
    settings = Settings(DB_PATH="storage/eval.db")
    db_file = settings.resolved_db_path()
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_file) + suffix)
        if p.exists():
            p.unlink()
    svc = RAGService(settings=settings)
    ingested = svc.ingest_path(args.docs)
    chunks = svc.db.all_chunks()
    gold = load_gold(Path(args.gold))

    print(f"corpus: {len(ingested)} docs, {len(chunks)} chunks | "
          f"embedder={svc.embedder.name} | gold={len(gold)} questions | llm={svc.llm.available}\n")

    rows = []
    for cfg in CONFIGS:
        res = evaluate_config(svc, gold, chunks, cfg)
        res["config"] = cfg["name"]
        rows.append(res)

    _print_table(rows)

    judged = None
    if args.judge:
        if not svc.llm.available:
            print("\n--judge requested but no ANTHROPIC_API_KEY set; skipping answer metrics.")
        else:
            print("\nRunning RAGAS-style judge on 'hybrid + rerank' answers ...")
            judged = run_judge(svc, gold, CONFIGS[-1])
            print(f"  faithfulness={judged['faithfulness']}  "
                  f"answer_relevance={judged['answer_relevance']}  "
                  f"cost=${judged['judged_cost_usd']}")

    payload = _save_report(rows, judged, svc, len(chunks), len(gold))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")
    svc.close()


def _print_table(rows: list[dict]):
    cols = ["config", "recall@5", "mrr", "ndcg@10", "hit@5", "latency_p50_ms", "latency_p95_ms"]
    widths = {c: max(len(c), max(len(f"{r.get(c,'')}") for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(f"{r.get(c,'')}".ljust(widths[c]) for c in cols))


def _save_report(rows, judged, svc, n_chunks, n_gold):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "timestamp": ts,
        "embedder": svc.embedder.name,
        "reranker": svc.settings.rerank_backend,
        "llm_available": svc.llm.available,
        "n_chunks": n_chunks,
        "n_gold": n_gold,
        "retrieval": rows,
        "answer_quality": judged,
    }
    (REPORTS_DIR / f"eval-{ts}.json").write_text(json.dumps(payload, indent=2))

    md = [f"# Eval report {ts}", "",
          f"- embedder: `{svc.embedder.name}`  | reranker: `{svc.settings.rerank_backend}`  "
          f"| gold: {n_gold} questions | chunks: {n_chunks}", "",
          "## Retrieval", "",
          "| config | recall@5 | MRR | nDCG@10 | hit@5 | p50 ms | p95 ms |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r.get('recall@5')} | {r.get('mrr')} | "
                  f"{r.get('ndcg@10')} | {r.get('hit@5')} | {r.get('latency_p50_ms')} | "
                  f"{r.get('latency_p95_ms')} |")
    if judged:
        md += ["", "## Answer quality (hybrid + rerank)", "",
               f"- faithfulness: **{judged['faithfulness']}**",
               f"- answer relevance: **{judged['answer_relevance']}**",
               f"- judged cost: ${judged['judged_cost_usd']}"]
    (REPORTS_DIR / f"eval-{ts}.md").write_text("\n".join(md))
    print(f"\nSaved report -> eval/reports/eval-{ts}.md")
    return payload


if __name__ == "__main__":
    main()
