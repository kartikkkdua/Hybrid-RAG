import React from "react";
import { fmtMs, fmtNum, splitOnSpan } from "../lib/text.js";

function Stage({ name, ms, detail, on }) {
  return (
    <div className={"stage" + (on ? "" : " off")}>
      <div className="stage-name">{name}</div>
      <div className="stage-ms">{on ? fmtMs(ms) : "off"}</div>
      {detail && <div className="stage-detail">{detail}</div>}
    </div>
  );
}

function Pipeline({ stages }) {
  if (!stages) return null;
  const { bm25, dense, fusion, rerank } = stages;
  return (
    <div className="pipeline">
      <Stage name="BM25" ms={bm25?.ms} detail={bm25 ? `${bm25.hits} hits` : null} on={!!bm25} />
      <span className="arrow">→</span>
      <Stage name="Dense" ms={dense?.ms} detail={dense ? `${dense.hits} hits` : null} on={!!dense} />
      <span className="arrow">→</span>
      <Stage name="RRF" detail={fusion ? `k=${fusion.k} · ${fusion.candidates}` : null} on={!!fusion} ms={0} />
      <span className="arrow">→</span>
      <Stage
        name="Rerank"
        ms={rerank?.ms}
        detail={rerank?.model && rerank.model !== "disabled" ? rerank.model.split(":")[0] : null}
        on={!!rerank && rerank.model !== "disabled"}
      />
    </div>
  );
}

function ScoreBar({ label, value, max, digits }) {
  if (value == null) return null;
  const pct = max > 0 ? Math.max(2, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="scorebar">
      <span className="sb-label">{label}</span>
      <span className="sb-track">
        <span className="sb-fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="sb-val">{fmtNum(value, digits)}</span>
    </div>
  );
}

export default function Inspector({ retrieval, citations, activeCitation }) {
  if (!retrieval) {
    return (
      <div className="inspector empty">
        <div className="section-title">Retrieval inspector</div>
        <p className="muted tiny">
          Ask something — every passage the retriever considered shows up here with its
          BM25, dense, RRF and rerank scores.
        </p>
      </div>
    );
  }

  const results = retrieval.results || [];
  const citedIds = new Set((citations || []).map((c) => c.chunk_id));
  const maxBm25 = Math.max(...results.map((r) => r.bm25_score || 0), 0.0001);
  const maxRerank = Math.max(...results.map((r) => r.rerank_score || 0), 0.0001);

  return (
    <div className="inspector">
      <div className="section-title">
        Retrieval inspector
        <span className="muted tiny"> · {fmtMs(retrieval.latency_ms)}</span>
      </div>

      <Pipeline stages={retrieval.stages} />

      <div className="passages">
        {results.map((r) => {
          const isCited = citedIds.has(r.chunk_id);
          const isActive = activeCitation && activeCitation.chunk_id === r.chunk_id;
          const split = isActive ? splitOnSpan(r.text, r.char_start, activeCitation) : null;
          return (
            <div
              key={r.chunk_id}
              className={
                "passage" + (isCited ? " cited" : "") + (isActive ? " active" : "")
              }
            >
              <div className="p-head">
                <span className="rank">#{r.rank}</span>
                <span className="p-src" title={r.source}>
                  {r.source}
                </span>
                {isCited && <span className="pill">cited</span>}
              </div>

              <div className="p-scores">
                <ScoreBar label="bm25" value={r.bm25_score} max={maxBm25} digits={2} />
                <ScoreBar label="dense" value={r.dense_score} max={1} digits={3} />
                <ScoreBar label="rerank" value={r.rerank_score} max={maxRerank} digits={3} />
                <div className="rrf-line muted tiny">
                  rrf {fmtNum(r.rrf_score, 4)}
                  {r.bm25_rank ? ` · bm25 #${r.bm25_rank}` : ""}
                  {r.dense_rank ? ` · dense #${r.dense_rank}` : ""}
                </div>
              </div>

              <p className="p-text">
                {split ? (
                  <>
                    {split.before.slice(-200)}
                    <mark>{split.match}</mark>
                    {split.after.slice(0, 200)}
                  </>
                ) : (
                  r.text.slice(0, 280) + (r.text.length > 280 ? "…" : "")
                )}
              </p>
              <div className="muted tiny">
                chars {r.char_start}–{r.char_end}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
