import React, { useRef, useState } from "react";
import { deleteDocument, ingestText, uploadFile } from "../api.js";

export default function Corpus({ sources, onChanged }) {
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState("");
  const [paste, setPaste] = useState("");
  const [showPaste, setShowPaste] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = async (files) => {
    for (const f of Array.from(files)) {
      setBusy(`Ingesting ${f.name}…`);
      try {
        await uploadFile(f);
      } catch (e) {
        setBusy(`Failed: ${f.name}`);
        await new Promise((r) => setTimeout(r, 1500));
      }
    }
    setBusy("");
    onChanged();
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
  };

  const doPaste = async () => {
    if (!paste.trim()) return;
    setBusy("Ingesting…");
    await ingestText(paste, "pasted.md", "Pasted document");
    setPaste("");
    setShowPaste(false);
    setBusy("");
    onChanged();
  };

  const remove = async (docId) => {
    setBusy("Deleting…");
    await deleteDocument(docId);
    setBusy("");
    onChanged();
  };

  return (
    <div className="panel-section">
      <div className="section-title">Corpus</div>

      <div
        className={"dropzone" + (dragging ? " active" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md"
          hidden
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
        <div className="dz-icon">＋</div>
        <div className="dz-text">
          {busy || (
            <>
              <b>Drop files</b> or click
              <div className="muted tiny">PDF · TXT · MD</div>
            </>
          )}
        </div>
      </div>

      <button className="link-btn" onClick={() => setShowPaste((v) => !v)}>
        {showPaste ? "− paste text" : "+ paste text"}
      </button>
      {showPaste && (
        <div className="paste-box">
          <textarea
            rows={4}
            value={paste}
            onChange={(e) => setPaste(e.target.value)}
            placeholder="Paste document text…"
          />
          <button className="btn small" onClick={doPaste}>
            Ingest
          </button>
        </div>
      )}

      <div className="doc-list">
        {sources.length === 0 && <div className="muted tiny pad">No documents yet.</div>}
        {sources.map((s) => (
          <div key={s.doc_id} className="doc-row">
            <div className="doc-meta">
              <div className="doc-name" title={s.source}>
                {s.source}
              </div>
              <div className="muted tiny">
                {s.n_chunks} chunks · {(s.n_chars / 1000).toFixed(1)}k chars
              </div>
            </div>
            <button
              className="icon-btn"
              title="Delete document"
              onClick={() => remove(s.doc_id)}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
