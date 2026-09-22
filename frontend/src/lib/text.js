// Pull the growing value of the JSON "answer" field out of a partial SSE buffer,
// so the user sees clean answer text stream in — not raw JSON. Falls back to the
// raw buffer when the stream is plain text (extractive / no-LLM mode).
export function extractAnswerField(buf) {
  const key = buf.indexOf('"answer"');
  if (key === -1) return buf;
  let i = buf.indexOf('"', buf.indexOf(":", key) + 1);
  if (i === -1) return "";
  i += 1;
  let out = "";
  while (i < buf.length) {
    const c = buf[i];
    if (c === "\\") {
      const n = buf[i + 1];
      out += n === "n" ? "\n" : n === "t" ? "\t" : n === "u" ? "" : n || "";
      i += n === "u" ? 6 : 2;
      continue;
    }
    if (c === '"') break;
    out += c;
    i += 1;
  }
  return out;
}

/**
 * Split a passage's text around a citation's character span.
 *
 * Citations carry ABSOLUTE offsets within the source document; a passage knows
 * its own absolute `char_start`. Subtracting gives the local span, which is how
 * we highlight the exact quote the model cited — the payoff for tracking exact
 * offsets through chunking.
 *
 * Returns {before, match, after} or null when the span isn't inside this passage.
 */
export function splitOnSpan(text, passageStart, citation) {
  if (!citation || citation.char_start < 0) return null;
  const start = citation.char_start - passageStart;
  const end = citation.char_end - passageStart;
  if (start < 0 || end > text.length || end <= start) return null;
  return {
    before: text.slice(0, start),
    match: text.slice(start, end),
    after: text.slice(end),
  };
}

export const fmtMs = (ms) => (ms == null ? "–" : `${Number(ms).toFixed(ms < 10 ? 2 : 0)}ms`);
export const fmtNum = (n, d = 3) => (n == null ? "–" : Number(n).toFixed(d));
export const fmtUsd = (n) => `$${Number(n || 0).toFixed(5)}`;
