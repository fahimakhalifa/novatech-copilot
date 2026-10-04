// Typed client for the NovaTech Copilot API. Every call goes to /api (proxied to FastAPI).

export type Role = "customer" | "support_agent" | "admin";
export interface User { username: string; role: Role }
export interface Citation { id: number; title: string; section: string; source: string }
export interface ToolLogEntry { tool: string; args: Record<string, unknown>; outcome: "ok" | "error" | "denied"; detail: string; ms: number }
export interface ChatResult {
  conversation_id: string;
  answer: string;
  citations: Citation[];
  route: string | null;
  guard: { label: string; p_blocked: number; reason: string } | null;
  tool_log: ToolLogEntry[];
  output_findings: string[];
  latency_ms: number;
}
export interface Message { role: "user" | "assistant"; content: string; citations: Citation[] }
export interface ConversationSummary { id: string; title: string; updated_at: string }
export interface DocumentInfo {
  path: string; title: string; category: string; access: "public" | "internal";
  last_updated: string; chunks: number; quarantined: { section: string; reason: string }[];
}

const TOKEN_KEY = "novatech.token";
const USER_KEY = "novatech.user";

export const session = {
  get(): { token: string; user: User } | null {
    try {
      const token = sessionStorage.getItem(TOKEN_KEY);
      const user = sessionStorage.getItem(USER_KEY);
      return token && user ? { token, user: JSON.parse(user) } : null;
    } catch { return null; }
  },
  set(token: string, user: User) {
    try { sessionStorage.setItem(TOKEN_KEY, token); sessionStorage.setItem(USER_KEY, JSON.stringify(user)); } catch { /* ignore */ }
  },
  clear() {
    try { sessionStorage.removeItem(TOKEN_KEY); sessionStorage.removeItem(USER_KEY); } catch { /* ignore */ }
  },
};

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(handler: () => void) { onUnauthorized = handler; }

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = session.get()?.token;
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init.headers,
    },
  });
  if (res.status === 401 && path !== "/auth/login") onUnauthorized();
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : "Invalid request.";
    } catch { /* keep statusText */ }
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  login: (username: string, password: string) =>
    request<{ access_token: string; user: User }>("/auth/login", {
      method: "POST", body: JSON.stringify({ username, password }),
    }),
  conversations: () => request<ConversationSummary[]>("/conversations"),
  messages: (id: string) => request<Message[]>(`/conversations/${encodeURIComponent(id)}/messages`),
  deleteConversation: (id: string) => request<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  documents: () => request<DocumentInfo[]>("/documents"),
  uploadDocument: (category: string, filename: string, content: string) =>
    request<{ path: string; access: string; chunks: number; replaced: boolean; quarantined: { section: string; reason: string }[] }>(
      "/documents", { method: "POST", body: JSON.stringify({ category, filename, content }) }),
  deleteDocument: (path: string) => request<void>(`/documents/${path}`, { method: "DELETE" }),
  metrics: (days: number) => request<Metrics>(`/metrics?days=${days}`),
  telegramLinkCode: () => request<{ code: string; expires_in: number }>("/telegram/link-code", { method: "POST" }),
};

export async function streamChat(
  message: string,
  conversationId: string | null,
  on: { start?: (id: string) => void; step?: (label: string) => void },
): Promise<ChatResult> {
  const token = session.get()?.token;
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({ message, ...(conversationId ? { conversation_id: conversationId } : {}) }),
  });
  if (res.status === 401) onUnauthorized();
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* keep */ }
    throw new ApiError(res.status, typeof detail === "string" ? detail : "Request failed.");
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {       // one SSE event ends with a blank line
      const raw = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const event = /^event: (.*)$/m.exec(raw)?.[1];
      const data = JSON.parse(/^data: (.*)$/m.exec(raw)?.[1] ?? "{}");
      if (event === "start") on.start?.(data.conversation_id);
      else if (event === "step") on.step?.(data.label);
      else if (event === "error") throw new ApiError(503, data.detail);
      else if (event === "answer") return data as ChatResult;
    }
  }
  throw new ApiError(0, "The connection closed before the answer arrived.");
}

export interface Metrics {
  live: {
    turns: number; errors: number; users: number; blocked_rate: number; knowledge_not_found_rate: number;
    by_route: Record<string, number>;
    latency_ms: { p50: number; p95: number; max: number };
    node_p50_ms: Record<string, number>;
    llm: { calls: number; tokens: number; calls_per_turn: number; models: Record<string, number>; fallback_calls: number };
    tools: Record<string, number>;
    output_guard_findings: Record<string, number>;
    hourly: { hour: string; turns: number; p50_ms: number }[];
  };
  eval: null | {
    started: string; cases: number; passed: number; pass_rate: number;
    by_category: Record<string, string>;
    latency_ms: { p50: number; p95: number; max: number };
    judge: { judged: number; groundedness_mean: number | null; relevance_mean: number | null; fully_grounded: number };
    case_results: { id: string; category: string; passed: boolean; latency_ms: number }[];
  };
}
