import { lazy, Suspense, useEffect, useState } from "react";
import { api, session, setUnauthorizedHandler, type User } from "./api";
import Chat from "./Chat";
import Login from "./Login";

type Tab = "chat" | "documents" | "metrics";
// Staff-only pages load on demand: the charting library never reaches customers' browsers.
const Documents = lazy(() => import("./Documents"));
const Metrics = lazy(() => import("./Metrics"));

const ROLE_LABEL = { customer: "Customer", support_agent: "Support agent", admin: "Admin" } as const;

export default function App() {
  const [user, setUser] = useState<User | null>(() => session.get()?.user ?? null);
  const [tab, setTab] = useState<Tab>("chat");

  const logout = () => { session.clear(); setUser(null); setTab("chat"); };
  // Any 401 (expired token, account removed) sends the user back to the login page.
  useEffect(() => { setUnauthorizedHandler(logout); }, []);

  if (!user) return <Login onLogin={setUser} />;

  // Tabs are hidden for customers, but that's convenience only: the API enforces the roles.
  const staff = user.role !== "customer";
  const tabs: [Tab, string][] = [["chat", "Chat"], ...(staff ? [["documents", "Documents"], ["metrics", "Metrics"]] as [Tab, string][] : [])];

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark" aria-hidden>N</span>NovaTech Copilot</div>
        <nav className="tabs" role="tablist">
          {tabs.map(([id, label]) => (
            <button key={id} role="tab" className="tab" aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        <div className="topbar-right">
          <TelegramLink />
          <span className="small muted">{user.username}</span>
          <span className="badge badge-accent">{ROLE_LABEL[user.role]}</span>
          <button className="btn btn-small" onClick={logout}>Log out</button>
        </div>
      </header>
      <main className="main">
        {tab === "chat" && <Chat user={user} />}
        <Suspense fallback={<div className="page muted">Loading…</div>}>
          {tab === "documents" && staff && <Documents user={user} />}
          {tab === "metrics" && staff && <Metrics />}
        </Suspense>
      </main>
    </div>
  );
}

function TelegramLink() {
  const [code, setCode] = useState<string | null>(null);
  const [error, setError] = useState("");
  async function getCode() {
    setError("");
    try { setCode((await api.telegramLinkCode()).code); } catch (e) { setError(e instanceof Error ? e.message : "Failed."); }
  }
  if (code) return (
    <span className="badge badge-accent" role="status">
      Send <code>/link {code}</code> to the NovaTech bot within 10 min
      <button className="tab" style={{ padding: "0 4px" }} aria-label="Close" onClick={() => setCode(null)}>✕</button>
    </span>
  );
  return <>
    <button className="btn btn-small" onClick={getCode}>Connect Telegram</button>
    {error && <span className="error small">{error}</span>}
  </>;
}
