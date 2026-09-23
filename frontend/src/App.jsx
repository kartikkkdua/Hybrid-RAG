import React, { useCallback, useEffect, useRef, useState } from "react";
import { agentAnswer, getSources, getStats, streamAnswer } from "./api.js";
import Chat from "./components/Chat.jsx";
import Corpus from "./components/Corpus.jsx";
import Inspector from "./components/Inspector.jsx";
import RetrievalSettings from "./components/RetrievalSettings.jsx";
import { extractAnswerField } from "./lib/text.js";

const SUGGESTIONS = [
  "What constant does Reciprocal Rank Fusion use, and why?",
  "How does a cross-encoder differ from a bi-encoder?",
  "What does faithfulness measure in a RAG answer?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState(null);
  const [sources, setSources] = useState([]);
  const [activeCitation, setActiveCitation] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const [theme, setTheme] = useState(
    () => localStorage.getItem("hr-theme") || "dark"
  );
  const [settings, setSettings] = useState({
    topK: 8,
    bm25: true,
    dense: true,
    rerank: true,
    agentMode: false,
  });
  const bufRef = useRef("");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("hr-theme", theme);
    } catch {}
  }, [theme]);

  const refresh = useCallback(() => {
    getStats().then(setStats).catch(() => {});
    getSources()
      .then((d) => setSources(d.sources || []))
      .catch(() => {});
  }, []);
  useEffect(refresh, [refresh]);

  const patchLast = (patch) =>
    setMessages((ms) => {
      const copy = [...ms];
      const i = copy.length - 1;
      if (i >= 0) copy[i] = { ...copy[i], ...patch };
      return copy;
    });

  const ask = () => {
    const q = input.trim();
    if (!q || running) return;
    if (!settings.bm25 && !settings.dense) return;

    setRunning(true);
    setActiveCitation(null);
    bufRef.current = "";
    setMessages((ms) => {
      const next = [
        ...ms,
        { role: "user", text: q },
        {
          role: "assistant",
          text: "",
          citations: [],
          retrieval: null,
          usage: null,
          verification: null,
          refused: false,
          streaming: true,
          agent: settings.agentMode,
        },
      ];
      setSelectedIdx(next.length - 1);
      return next;
    });
    setInput("");

    // Retrieval has no memory of its own: send a recent window of turns so a
    // follow-up ("why that value?") can be condensed into a standalone query.
    const history = messages
      .slice(-6)
      .map((m) => ({ role: m.role, text: (m.text || "").slice(0, 400) }))
      .filter((m) => m.text);

    // Agent mode runs the LangGraph graph (not streamable — it branches), so we
    // await the whole run and then render its trace.
    if (settings.agentMode) {
      agentAnswer(q, settings.topK, history)
        .then((d) => {
          patchLast({
            text: d.answer || "",
            citations: d.citations || [],
            usage: d.usage || null,
            verification: d.verification || null,
            refused: !!d.refused,
            refusalReason: d.refusal_reason || "Could not ground an answer.",
            retrieval: { results: d.retrieved || [], stages: {}, latency_ms: d.usage?.latency_ms },
            searchQuery: d.search_query,
            rewritten: !!d.rewritten,
            trace: d.trace || [],
            route: d.route,
            routeReason: d.route_reason,
            subquestions: d.subquestions || [],
            iterations: d.iterations || 0,
            escalated: !!d.escalated,
            critique: d.critique || "",
            streaming: false,
          });
          setRunning(false);
          refresh();
        })
        .catch((e) => {
          patchLast({
            streaming: false,
            refused: true,
            refusalReason: `Agent request failed: ${e.message}`,
          });
          setRunning(false);
        });
      return;
    }

    streamAnswer(q, settings, {
      onRetrieval: (d) =>
        patchLast({
          retrieval: d,
          searchQuery: d.search_query,
          rewritten: !!d.rewritten,
        }),
      onToken: (t) => {
        bufRef.current += t;
        patchLast({ text: extractAnswerField(bufRef.current) });
      },
      onDone: (d) => {
        patchLast({
          text: d.answer ?? bufRef.current,
          citations: d.citations || [],
          usage: d.usage || null,
          verification: d.verification || null,
          refused: !!d.refused,
          refusalReason: d.refusal_reason || "No verifiable answer in the corpus.",
          streaming: false,
        });
        setRunning(false);
        refresh(); // the corpus may have changed in another tab — don't go stale
      },
      onError: () => {
        patchLast({
          streaming: false,
          refused: true,
          refusalReason: "Connection to the server failed.",
        });
        setRunning(false);
      },
    }, history);
  };

  const selected =
    selectedIdx >= 0 && messages[selectedIdx]?.role === "assistant"
      ? messages[selectedIdx]
      : [...messages].reverse().find((m) => m.role === "assistant");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◈</span>
          <div>
            <div className="title">Hybrid RAG</div>
            <div className="tagline">
              BM25 + dense · RRF · cross-encoder · verified citations
            </div>
          </div>
        </div>
        <div className="chips">
          {stats && (
            <>
              <span className="chip stat">{stats.documents} docs</span>
              <span className="chip stat">{stats.chunks} chunks</span>
              <span className="chip stat" title="Storage backend">
                {stats.backend}
              </span>
              <span className="chip stat" title="Embedding model">
                {stats.embedder}
              </span>
              <span className={"chip " + (stats.llm_available ? "good" : "idle")}>
                {stats.llm_available ? "LLM connected" : "extractive mode"}
              </span>
            </>
          )}
          <button
            className="icon-btn"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            title="Toggle theme"
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar left">
          <Corpus sources={sources} onChanged={refresh} />
          <RetrievalSettings
            settings={settings}
            setSettings={setSettings}
            agentAvailable={!!stats?.agent_available}
          />
        </aside>

        <main className="main">
          <Chat
            messages={messages}
            running={running}
            activeCitation={activeCitation}
            onSelectCitation={setActiveCitation}
            onSelectMessage={setSelectedIdx}
            selectedIdx={selectedIdx}
            input={input}
            setInput={setInput}
            onAsk={ask}
            onClear={() => {
              setMessages([]);
              setActiveCitation(null);
              setSelectedIdx(-1);
            }}
            suggestions={SUGGESTIONS}
            corpusEmpty={!!stats && stats.chunks === 0}
          />
        </main>

        <aside className="sidebar right">
          <Inspector
            retrieval={selected?.retrieval}
            citations={selected?.citations}
            activeCitation={activeCitation}
          />
        </aside>
      </div>
    </div>
  );
}
