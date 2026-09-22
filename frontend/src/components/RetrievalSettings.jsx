import React from "react";

/**
 * Live retrieval controls. Toggling a stage off and re-asking is the fastest way
 * to *see* what hybrid retrieval buys you — the sidebar scores change with it.
 */
export default function RetrievalSettings({ settings, setSettings, agentAvailable }) {
  const set = (k, v) => setSettings((s) => ({ ...s, [k]: v }));
  const Toggle = ({ k, label, hint }) => (
    <label className="toggle" title={hint}>
      <input type="checkbox" checked={settings[k]} onChange={(e) => set(k, e.target.checked)} />
      <span className="track">
        <span className="thumb" />
      </span>
      <span className="toggle-label">{label}</span>
    </label>
  );

  return (
    <div className="panel-section">
      <div className="section-title">Retrieval</div>
      <Toggle k="bm25" label="BM25 (sparse)" hint="Exact-term lexical matching" />
      <Toggle k="dense" label="Dense (vectors)" hint="Semantic similarity" />
      <Toggle k="rerank" label="Cross-encoder rerank" hint="Reorders the fused shortlist" />

      <div className="slider-row">
        <div className="slider-head">
          <span>top-k</span>
          <b>{settings.topK}</b>
        </div>
        <input
          type="range"
          min="1"
          max="20"
          value={settings.topK}
          onChange={(e) => set("topK", Number(e.target.value))}
        />
      </div>

      {!settings.bm25 && !settings.dense && (
        <div className="warn tiny">Enable at least one retrieval stage.</div>
      )}

      <div className="section-title" style={{ marginTop: 18 }}>
        Mode
      </div>
      {agentAvailable ? (
        <>
          <Toggle
            k="agentMode"
            label="Agent mode"
            hint="Route → decompose → research → self-critique"
          />
          <div className="muted tiny">
            {settings.agentMode
              ? "Multi-agent graph: routes the question, decomposes hard ones, and self-critiques before answering."
              : "Single-pass: one retrieval, one grounded answer, streamed."}
          </div>
        </>
      ) : (
        <div className="muted tiny">
          Agent mode unavailable — install langgraph (<code>make agent</code>).
        </div>
      )}
    </div>
  );
}
