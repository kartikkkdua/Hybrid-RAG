import React, { useEffect, useRef } from "react";
import { fmtMs, fmtUsd } from "../lib/text.js";
import AgentTrace from "./AgentTrace.jsx";

function Citations({ citations, activeCitation, onSelect }) {
  if (!citations?.length) return null;
  return (
    <div className="citations">
      {citations.map((c, i) => {
        const active = activeCitation === c;
        return (
          <button
            key={i}
            className={"cite" + (active ? " active" : "")}
            onClick={() => onSelect(active ? null : c)}
            title="Show this exact span in the source passage"
          >
            <span className="cite-idx">{i + 1}</span>
            <span className={"badge " + (c.verified ? "ok" : "bad")}>
              {c.verified ? "verified" : "unverified"}
            </span>
            <span className="cite-src">{c.source}</span>
            <span className="cite-range">
              {c.char_start}–{c.char_end}
            </span>
            <span className="cite-quote">“{c.quote.slice(0, 110)}”</span>
          </button>
        );
      })}
    </div>
  );
}

function Usage({ usage, verification }) {
  if (!usage) return null;
  const hasTokens = usage.input_tokens > 0 || usage.output_tokens > 0;
  return (
    <div className="usage">
      <span>{usage.model || "—"}</span>
      {hasTokens && (
        <>
          <span>
            {usage.input_tokens}→{usage.output_tokens} tok
          </span>
          <span>{fmtUsd(usage.cost_usd)}</span>
        </>
      )}
      {usage.latency_ms > 0 && <span>{fmtMs(usage.latency_ms)}</span>}
      {verification?.citations_total != null && (
        <span title="verified / total citations">
          {verification.citations_verified}/{verification.citations_total} cited
        </span>
      )}
    </div>
  );
}

export default function Chat({
  messages,
  running,
  activeCitation,
  onSelectCitation,
  onSelectMessage,
  selectedIdx,
  input,
  setInput,
  onAsk,
  onClear,
  suggestions,
  corpusEmpty,
}) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, running]);

  return (
    <div className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="welcome">
            {corpusEmpty ? (
              <>
                <h2>Your corpus is empty</h2>
                <p className="muted">
                  Drop a PDF, TXT or MD file into the panel on the left — or paste
                  some text — and then ask a question about it.
                </p>
              </>
            ) : (
              <>
                <h2>Ask your corpus a question</h2>
                <p className="muted">
                  Answers are grounded in retrieved passages, and every claim carries
                  a citation whose exact character span is verified against the source.
                </p>
                <div className="suggestions">
                  {suggestions.map((s) => (
                    <button key={s} className="chip" onClick={() => setInput(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="msg user">
              <div className="bubble">{m.text}</div>
            </div>
          ) : (
            <div
              key={i}
              className={"msg assistant" + (selectedIdx === i ? " selected" : "")}
              onClick={() => onSelectMessage(i)}
            >
              <div className="bubble">
                {m.cached && (
                  <div className="cache-note" title={
                    m.cacheKind === "semantic"
                      ? "A differently-worded question matched this answer, and retrieval confirmed it uses the same evidence."
                      : "The identical question was asked before."
                  }>
                    <span className="cn-label">cached</span>
                    <span className="cn-kind">{m.cacheKind}</span>
                  </div>
                )}
                {m.rewritten && m.searchQuery && (
                  <div className="rewrite-note" title="Retrieval has no memory, so the follow-up was expanded using the conversation">
                    <span className="rw-label">searched</span>
                    <span className="rw-query">{m.searchQuery}</span>
                  </div>
                )}
                {m.agent && <AgentTrace message={m} />}
                {m.refused ? (
                  <div className="refused">
                    <b>Refused.</b> {m.refusalReason}
                    <div className="muted tiny">
                      The retrieved passages didn't support a verifiable answer — the
                      system declines rather than guessing.
                    </div>
                  </div>
                ) : (
                  <div className="answer-text">
                    {m.text}
                    {m.streaming &&
                      (m.agent ? (
                        <span className="muted">Running the agent graph…</span>
                      ) : (
                        <span className="cursor">▋</span>
                      ))}
                  </div>
                )}
                <Citations
                  citations={m.citations}
                  activeCitation={activeCitation}
                  onSelect={onSelectCitation}
                />
                <Usage usage={m.usage} verification={m.verification} />
              </div>
            </div>
          )
        )}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <textarea
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onAsk();
            }
          }}
          placeholder="Ask a question…  (Enter to send, Shift+Enter for newline)"
        />
        <div className="composer-actions">
          {messages.length > 0 && (
            <button className="btn ghost" onClick={onClear} title="Clear conversation">
              Clear
            </button>
          )}
          <button
            className="btn"
            onClick={onAsk}
            disabled={running || !input.trim() || corpusEmpty}
            title={corpusEmpty ? "Add a document to the corpus first" : "Ask"}
          >
            {running ? "…" : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}
