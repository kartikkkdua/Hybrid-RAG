"""Quality gate: fail CI when retrieval regresses.

Tests prove the code runs; this proves the system still *retrieves well*. It
reads an eval report and compares the primary configuration against
`eval/thresholds.json`, exiting non-zero on any breach.

    python -m eval.run_eval --docs data/sample_docs --out-json eval/reports/ci.json
    python -m eval.check_thresholds eval/reports/ci.json

Treat the thresholds as a ratchet: when a change genuinely improves retrieval,
raise them so the gain can't silently erode later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THRESHOLDS = Path(__file__).resolve().parent / "thresholds.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="path to an eval report JSON")
    ap.add_argument("--thresholds", default=str(THRESHOLDS))
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    spec = json.loads(Path(args.thresholds).read_text())
    target = spec.get("config", "hybrid + rerank")

    row = next((r for r in report.get("retrieval", []) if r.get("config") == target), None)
    if row is None:
        print(f"FAIL: no '{target}' row in {args.report}")
        return 1

    print(f"Quality gate — config: {target}")
    print(f"  corpus: {report.get('n_chunks')} chunks | gold: {report.get('n_gold')} "
          f"questions | embedder: {report.get('embedder')}")
    print()

    failures: list[str] = []
    for metric, floor in (spec.get("min") or {}).items():
        actual = row.get(metric)
        ok = actual is not None and actual >= floor
        print(f"  {'PASS' if ok else 'FAIL'}  {metric:<16} {actual}  (min {floor})")
        if not ok:
            failures.append(f"{metric}={actual} < {floor}")

    for metric, ceiling in (spec.get("max") or {}).items():
        actual = row.get(metric)
        ok = actual is not None and actual <= ceiling
        print(f"  {'PASS' if ok else 'FAIL'}  {metric:<16} {actual}  (max {ceiling})")
        if not ok:
            failures.append(f"{metric}={actual} > {ceiling}")

    print()
    if failures:
        print("QUALITY GATE FAILED: " + "; ".join(failures))
        return 1
    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
