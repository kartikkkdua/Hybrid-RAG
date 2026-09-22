import React from "react";
import { fmtMs } from "../lib/text.js";

const NODE_LABEL = {
  router: "Router",
  lookup: "Lookup",
  planner: "Planner",
  researcher: "Researcher",
  synthesizer: "Synthesizer",
  critic: "Critic",
  reject: "Reject",
};

/**
 * Renders the agent's path through the graph. This is the whole point of the
 * multi-agent layer being visible rather than a black box: you can see it route,
 * decompose, escalate and self-critique.
 */
export default function AgentTrace({ message }) {
  const { trace, route, routeReason, subquestions, iterations, escalated, critique } =
    message;
  if (!trace?.length) return null;

  return (
    <div className="agent-trace">
      <div className="at-head">
        <span className="at-title">Agent trace</span>
        <span className={"route-pill " + route}>{route}</span>
        {escalated && (
          <span className="route-pill escalated" title="The critic rejected the first draft">
            escalated
          </span>
        )}
        {iterations > 1 && <span className="muted tiny">{iterations} iterations</span>}
      </div>

      {routeReason && <div className="muted tiny at-reason">{routeReason}</div>}

      <ol className="at-steps">
        {trace.map((t, i) => (
          <li key={i} className={"at-step node-" + t.node}>
            <span className="at-dot" />
            <span className="at-node">{NODE_LABEL[t.node] || t.node}</span>
            <span className="at-detail">{t.detail}</span>
            <span className="at-ms">{fmtMs(t.ms)}</span>
          </li>
        ))}
      </ol>

      {subquestions?.length > 1 && (
        <div className="at-subs">
          <div className="muted tiny">Sub-questions</div>
          <ul>
            {subquestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}

      {critique && <div className="at-critique muted tiny">Critic: {critique}</div>}
    </div>
  );
}
