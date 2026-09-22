// All calls go to /api/* which Vite proxies to the FastAPI backend in dev and
// which FastAPI serves directly in production.
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function j(path, opts) {
  const r = await fetch(`${BASE}${path}`, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const getStats = () => j("/stats");
export const getSources = () => j("/sources");
export const getDocument = (docId) => j(`/documents/${encodeURIComponent(docId)}`);

export const deleteDocument = (docId) =>
  j(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });

export const ingestText = (text, source, title) =>
  j("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source, title }),
  });

export async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  return j("/upload", { method: "POST", body: fd });
}

export const agentAnswer = (query, topK = 8) =>
  j("/agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });

export const search = (query, opts = {}) =>
  j("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 8, ...opts }),
  });

// Stream an answer via SSE, honouring the live retrieval toggles.
export function streamAnswer(query, settings, { onRetrieval, onToken, onDone, onError }) {
  const p = new URLSearchParams({
    query,
    top_k: String(settings.topK ?? 8),
    rerank: String(settings.rerank ?? true),
    dense: String(settings.dense ?? true),
    bm25: String(settings.bm25 ?? true),
  });
  const es = new EventSource(`${BASE}/answer/stream?${p}`);
  es.addEventListener("retrieval", (e) => onRetrieval(JSON.parse(e.data)));
  es.addEventListener("token", (e) => onToken(JSON.parse(e.data).text));
  es.addEventListener("done", (e) => {
    onDone(JSON.parse(e.data));
    es.close();
  });
  es.onerror = (e) => {
    onError && onError(e);
    es.close();
  };
  return es;
}
