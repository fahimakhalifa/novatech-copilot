import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Metrics as MetricsData } from "./api";

const AXIS = { fill: "var(--text-2)", fontSize: 12 };
// Respect the OS "reduce motion" setting: no growing-bar animation.
const ANIMATE = typeof window === "undefined" || !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const ROUTE_LABEL: Record<string, string> = {
  operations: "Live systems", knowledge: "Documents", both: "Both agents", smalltalk: "Small talk",
  blocked: "Blocked", invalid: "Invalid", error: "Error",
};
const NODE_LABEL: Record<string, string> = {
  guard: "Guard", supervisor: "Supervisor", knowledge: "Knowledge agent", operations: "Operations agent",
  smalltalk: "Small talk", synthesizer: "Synthesizer", output_guard: "Output guard",
};

const secs = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const pct = (x: number) => `${Math.round(x * 100)}%`;

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

function ChartTip({ active, payload, label, unit }: { active?: boolean; payload?: { value: number }[]; label?: string; unit: string }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip"><b>{label}</b><br />{payload[0].value.toLocaleString()} {unit}</div>;
}

function HBar({ data, unit, format }: { data: { name: string; value: number }[]; unit: string; format: (v: number) => string }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(120, data.length * 36 + 20)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }}>
        <CartesianGrid horizontal={false} stroke="var(--grid)" />
        <XAxis type="number" hide />
        <YAxis type="category" dataKey="name" width={130} tick={AXIS} axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip unit={unit} />} />
        <Bar dataKey="value" isAnimationActive={ANIMATE} fill="var(--series-1)" radius={[0, 4, 4, 0]} maxBarSize={20}>
          <LabelList dataKey="value" position="right" formatter={(v) => format(Number(v))} style={{ fill: "var(--text-2)", fontSize: 12 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function Metrics() {
  const [days, setDays] = useState(7);
  const [data, setData] = useState<MetricsData | null>(null);
  const [error, setError] = useState("");

  const load = () => { setError(""); api.metrics(days).then(setData).catch((e) => setError(e.message)); };
  useEffect(load, [days]);

  if (error) return <div className="page"><div className="notice notice-bad">{error}</div></div>;
  if (!data) return <div className="page muted">Loading…</div>;
  const { live, eval: ev } = data;

  const hourly = live.hourly.map((h) => ({
    name: new Date(`${h.hour}:00:00`).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }),
    value: h.turns,
  }));
  const routes = Object.entries(live.by_route).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ name: ROUTE_LABEL[k] ?? k, value: v }));
  const nodes = Object.entries(live.node_p50_ms).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ name: NODE_LABEL[k] ?? k, value: v }));
  const tools = Object.entries(live.tools).reduce<Record<string, Record<string, number>>>((acc, [key, n]) => {
    const [tool, outcome] = key.split(":");
    (acc[tool] ??= {})[outcome] = n;
    return acc;
  }, {});

  return (
    <div className="page">
      <div className="toolbar">
        <h2>Metrics</h2>
        <div className="tabs" role="tablist" aria-label="Time range">
          {[1, 7, 30].map((d) => (
            <button key={d} role="tab" className="tab" aria-selected={days === d} onClick={() => setDays(d)}>
              {d === 1 ? "24 hours" : `${d} days`}
            </button>
          ))}
        </div>
        <button className="btn btn-small" onClick={load}>Refresh</button>
      </div>

      <div className="tiles">
        <Tile label="Conversation turns" value={live.turns.toLocaleString()} sub={`${live.users} user(s)`} />
        <Tile label="Median latency" value={secs(live.latency_ms.p50)} sub={`p95 ${secs(live.latency_ms.p95)}`} />
        <Tile label="Blocked by guard" value={pct(live.blocked_rate)} sub="of all turns" />
        <Tile label="LLM calls per turn" value={live.llm.calls_per_turn.toString()} sub={`${live.llm.tokens.toLocaleString()} tokens`} />
        <Tile label="Fallback model calls" value={live.llm.fallback_calls.toString()} sub="primary model rate-limited" />
        <Tile label="Document answers not found" value={pct(live.knowledge_not_found_rate)} sub="refused rather than guessed" />
        <Tile label="Errors" value={live.errors.toString()} sub={Object.keys(live.output_guard_findings).length
          ? `output guard: ${Object.entries(live.output_guard_findings).map(([k, v]) => `${k} ×${v}`).join(", ")}` : "no output-guard findings"} />
      </div>

      {live.turns === 0 ? <p className="muted">No turns in this period yet. Chat with the copilot to see live metrics.</p> : (
        <div className="charts">
          <div className="card chart-card">
            <h3>Turns per hour</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={hourly} margin={{ left: -16, right: 8, top: 8, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="name" tick={AXIS} axisLine={false} tickLine={false} minTickGap={24} />
                <YAxis allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTip unit="turns" />} />
                <Bar dataKey="value" isAnimationActive={ANIMATE} fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card chart-card">
            <h3>Turns by route</h3>
            <HBar data={routes} unit="turns" format={(v) => v.toString()} />
          </div>
          <div className="card chart-card">
            <h3>Median time per step</h3>
            <HBar data={nodes} unit="ms" format={(v) => (v >= 1000 ? secs(v) : `${v} ms`)} />
          </div>
          <div className="card chart-card table-wrap">
            <h3>Tool calls</h3>
            <table>
              <thead><tr><th>Tool</th><th>✓ ok</th><th>✕ error</th><th>⛔ denied</th></tr></thead>
              <tbody>
                {Object.entries(tools).map(([tool, o]) => (
                  <tr key={tool}><td><code>{tool}</code></td><td>{o.ok ?? 0}</td><td>{o.error ?? 0}</td><td>{o.denied ?? 0}</td></tr>
                ))}
                {Object.keys(tools).length === 0 && <tr><td colSpan={4} className="muted">No tool calls yet.</td></tr>}
              </tbody>
            </table>
            <p className="small muted">"Error" includes expected refusals such as an order that isn't on the customer's account.</p>
          </div>
        </div>
      )}

      <h2>Latest evaluation</h2>
      {!ev ? <p className="muted">No evaluation run yet. Run <code>python -m scripts.run_eval</code> in <code>backend/</code>.</p> : (
        <>
          <div className="tiles">
            <Tile label="Cases passed" value={`${ev.passed}/${ev.cases}`} sub={`${pct(ev.pass_rate)} · run ${ev.started}`} />
            <Tile label="Groundedness" value={ev.judge.groundedness_mean?.toString() ?? "—"} sub={`of 5 · ${ev.judge.fully_grounded}/${ev.judge.judged} fully grounded`} />
            <Tile label="Relevance" value={ev.judge.relevance_mean?.toString() ?? "—"} sub="of 5 (LLM judge)" />
            <Tile label="Median latency" value={secs(ev.latency_ms.p50)} sub={`p95 ${secs(ev.latency_ms.p95)}`} />
          </div>
          <div className="card table-wrap">
            <table>
              <thead><tr><th>Case</th><th>Category</th><th>Result</th><th>Latency</th></tr></thead>
              <tbody>
                {ev.case_results.map((c) => (
                  <tr key={c.id}>
                    <td><code>{c.id}</code></td>
                    <td className="muted">{c.category}</td>
                    <td>{c.passed ? <span className="badge badge-good">✓ pass</span> : <span className="badge badge-bad">✕ fail</span>}</td>
                    <td>{secs(c.latency_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
