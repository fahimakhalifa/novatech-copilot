import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import Markdown from "react-markdown";
import { api, streamChat, type Citation, type ChatResult, type ConversationSummary, type User } from "./api";

interface Shown {
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  result?: ChatResult;   // only for answers produced in this session (route, latency, tools)
}

const ROUTE_LABEL: Record<string, string> = {
  knowledge: "Documents", operations: "Live systems", both: "Documents + live systems",
  smalltalk: "Small talk", blocked: "Blocked by guard", invalid: "Invalid input",
};

const SUGGESTIONS: Record<User["role"], string[]> = {
  customer: ["Where is my order #1025?", "Can I return headphones I already opened?",
             "What does the NovaPhone X 256GB cost, and does it support wireless charging?"],
  support_agent: ["Look up customer Mila de Boer", "Who has to approve an expense of 1,200 euros?",
                  "What's the status of order #1031?"],
  admin: ["How many vacation days do employees get?", "Is the NovaBook Air 13 in stock?",
          "Look up customer Sophie Bakker"],
};

export default function Chat({ user }: { user: User }) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Shown[]>([]);
  const [input, setInput] = useState("");
  const [steps, setSteps] = useState<string[] | null>(null);   // non-null while a turn runs
  const [error, setError] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  const refreshConversations = () => api.conversations().then(setConversations).catch(() => {});
  useEffect(() => { refreshConversations(); }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, steps]);

  async function open(id: string) {
    setActiveId(id);
    setError("");
    try {
      setMessages(await api.messages(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the conversation.");
    }
  }

  function newChat() { setActiveId(null); setMessages([]); setError(""); }

  async function remove(id: string) {
    await api.deleteConversation(id).catch(() => {});
    if (id === activeId) newChat();
    refreshConversations();
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || steps) return;
    setInput("");
    setError("");
    setMessages((m) => [...m, { role: "user", content: message, citations: [] }]);
    setSteps([]);
    try {
      const result = await streamChat(message, activeId, {
        start: (id) => setActiveId(id),
        step: (label) => setSteps((s) => [...(s ?? []), label]),
      });
      setMessages((m) => [...m, { role: "assistant", content: result.answer, citations: result.citations, result }]);
      refreshConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setSteps(null);
    }
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
  }

  return (
    <div className="chat">
      <aside className="sidebar">
        <button className="btn btn-primary" onClick={newChat}>+ New chat</button>
        {conversations.map((c) => (
          <div key={c.id} className={`conv ${c.id === activeId ? "active" : ""}`}>
            <button className="conv-title" title={c.title} onClick={() => open(c.id)}>{c.title}</button>
            <button className="conv-delete" aria-label={`Delete conversation ${c.title}`} onClick={() => remove(c.id)}>✕</button>
          </div>
        ))}
        {conversations.length === 0 && <p className="small muted">No conversations yet.</p>}
      </aside>

      <section className="thread">
        <div className="messages" aria-live="polite">
          {messages.length === 0 && !steps && (
            <div className="empty">
              <h2>How can I help, {user.username}?</h2>
              <p>Ask about orders, products and prices, or NovaTech policies.</p>
              <div className="suggestions">
                {SUGGESTIONS[user.role].map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}
              </div>
            </div>
          )}
          {messages.map((m, i) => m.role === "user"
            ? <div key={i} className="msg msg-user">{m.content}</div>
            : <Answer key={i} message={m} />)}
          {steps && (
            <div className="progress">
              {steps.map((s, i) => <div key={i} className="step">{s}</div>)}
              <div className="current">Working…</div>
            </div>
          )}
          {error && <div className="notice notice-bad" role="alert">{error}</div>}
          <div ref={bottom} />
        </div>
        <div className="composer">
          <textarea className="input" rows={1} value={input} placeholder="Ask NovaTech Copilot…"
                    onChange={(e) => setInput(e.target.value)} onKeyDown={onKey} maxLength={2000} aria-label="Message" />
          <button className="btn btn-primary" onClick={() => send(input)} disabled={!input.trim() || !!steps}>Send</button>
        </div>
      </section>
    </div>
  );
}

function Answer({ message }: { message: Shown }) {
  const r = message.result;
  return (
    <div className="msg msg-bot">
      <Markdown>{message.content}</Markdown>
      {message.citations.length > 0 && (
        <div className="sources">
          {message.citations.map((c) => (
            <div key={c.id}><b>[{c.id}]</b> {c.title} › {c.section} <span className="muted">({c.source})</span></div>
          ))}
        </div>
      )}
      {r && (
        <div className="meta">
          {r.route && <span className={`badge ${r.route === "blocked" ? "badge-bad" : "badge-accent"}`}>{ROUTE_LABEL[r.route] ?? r.route}</span>}
          <span className="badge">{(r.latency_ms / 1000).toFixed(1)} s</span>
          {r.output_findings.length > 0 && <span className="badge badge-warn">⚠ output guard: {r.output_findings.join(", ")}</span>}
        </div>
      )}
      {r && r.tool_log.length > 0 && (
        <details className="tools">
          <summary>{r.tool_log.length} tool call{r.tool_log.length > 1 ? "s" : ""}</summary>
          {r.tool_log.map((t, i) => (
            <div key={i}>
              {t.outcome === "ok" ? "✓" : t.outcome === "denied" ? "⛔" : "✕"} <code>{t.tool}({JSON.stringify(t.args)})</code>{" "}
              <span className="muted">{t.outcome}{t.detail ? `: ${t.detail}` : ""} · {t.ms} ms</span>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}
