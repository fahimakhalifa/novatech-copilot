import { useState, type FormEvent } from "react";
import { api, session, type User } from "./api";

const DEMO_USERS = [
  ["mila", "customer, owns order #1025"],
  ["lucas", "customer, does not own #1025"],
  ["sara", "support agent"],
  ["admin", "admin"],
] as const;

export default function Login({ onLogin }: { onLogin: (u: User) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await api.login(username.trim(), password);
      session.set(res.access_token, res.user);
      onLogin(res.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-wrap">
      <form className="card login" onSubmit={submit}>
        <div className="brand"><span className="brand-mark" aria-hidden>N</span>NovaTech Copilot</div>
        <h1>Sign in</h1>
        <label className="small muted">Username
          <input className="input" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" autoFocus />
        </label>
        <label className="small muted">Password
          <input className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />
        </label>
        {error && <div className="error" role="alert">{error}</div>}
        <button className="btn btn-primary" disabled={busy || !username || !password}>{busy ? "Signing in…" : "Sign in"}</button>
        <div className="demo-users">
          <span className="muted">Demo accounts (passwords: <code>DEMO_PASSWORDS</code> in <code>backend/scripts/seed_db.py</code>)</span>
          {DEMO_USERS.map(([name, desc]) => (
            <button type="button" key={name} onClick={() => setUsername(name)}><b>{name}</b> <span className="muted">· {desc}</span></button>
          ))}
        </div>
      </form>
    </div>
  );
}
