import { useEffect, useState, type ChangeEvent } from "react";
import { api, type DocumentInfo, type User } from "./api";

const CATEGORIES = ["customer", "products", "hr", "internal"];

/** Make a browser file name fit the API's pattern: lowercase, a-z0-9 and "-", ".md". */
function toFilename(name: string): string {
  const stem = name.replace(/\.md$/i, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return `${stem.slice(0, 60) || "document"}.md`;
}

export default function Documents({ user }: { user: User }) {
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [error, setError] = useState("");
  const isAdmin = user.role === "admin";

  const load = () => api.documents().then(setDocs).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  async function remove(path: string) {
    if (!confirm(`Delete ${path}? It is removed from search immediately.`)) return;
    try { await api.deleteDocument(path); load(); } catch (e) { setError(e instanceof Error ? e.message : "Delete failed."); }
  }

  return (
    <div className="page">
      <h2>Documents</h2>
      <p className="muted small" style={{ margin: 0 }}>
        Company documents the copilot answers from. <b>Public</b> documents are visible to customers,
        <b> internal</b> ones to staff only. Sections that look like instructions to an AI are quarantined and never
        reach the model.
      </p>
      {isAdmin && <Upload onDone={load} />}
      {error && <div className="notice notice-bad">{error}</div>}
      <div className="card table-wrap">
        <table>
          <thead>
            <tr><th>Document</th><th>Access</th><th>Sections</th><th>Scanner</th><th>Updated</th>{isAdmin && <th />}</tr>
          </thead>
          <tbody>
            {docs === null && <tr><td colSpan={6} className="muted">Loading…</td></tr>}
            {docs?.map((d) => (
              <tr key={d.path}>
                <td><b>{d.title}</b><div className="small muted">{d.path}</div></td>
                <td><span className={`badge ${d.access === "public" ? "badge-good" : ""}`}>{d.access === "public" ? "🌐 public" : "🔒 internal"}</span></td>
                <td>{d.chunks}</td>
                <td>
                  {d.quarantined.length === 0
                    ? <span className="badge badge-good">✓ clean</span>
                    : d.quarantined.map((q) => (
                        <div key={q.section}><span className="badge badge-warn" title={q.reason}>⚠ quarantined: {q.section}</span></div>
                      ))}
                </td>
                <td className="small muted">{d.last_updated || "—"}</td>
                {isAdmin && <td><button className="btn btn-small btn-danger" onClick={() => remove(d.path)}>Delete</button></td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Upload({ onDone }: { onDone: () => void }) {
  const [category, setCategory] = useState("customer");
  const [file, setFile] = useState<{ name: string; content: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ kind: "good" | "warn" | "bad"; text: string } | null>(null);

  async function pick(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    setResult(null);
    if (!f) return setFile(null);
    setFile({ name: toFilename(f.name), content: await f.text() });
  }

  async function upload() {
    if (!file) return;
    setBusy(true);
    try {
      const r = await api.uploadDocument(category, file.name, file.content);
      const base = `${r.replaced ? "Replaced" : "Added"} ${r.path}: ${r.chunks} section(s), ${r.access}.`;
      setResult(r.quarantined.length
        ? { kind: "warn", text: `${base} Quarantined (not searchable): ${r.quarantined.map((q) => `${q.section} (${q.reason})`).join("; ")}.` }
        : { kind: "good", text: `${base} Searchable now.` });
      onDone();
    } catch (e) {
      setResult({ kind: "bad", text: e instanceof Error ? e.message : "Upload failed." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card upload">
      <h3>Upload a Markdown document</h3>
      <div className="upload-row">
        <select className="input" style={{ width: "auto" }} value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category">
          {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
        </select>
        <input type="file" accept=".md,text/markdown" onChange={pick} aria-label="Markdown file" />
        {file && <span className="small muted">saved as <code>{category}/{file.name}</code></span>}
        <button className="btn btn-primary" onClick={upload} disabled={!file || busy}>{busy ? "Indexing…" : "Upload"}</button>
      </div>
      <p className="small muted" style={{ margin: 0 }}>
        Start the file with frontmatter such as <code>---</code> <code>title: Gift Cards</code> <code>access: public</code> <code>---</code>.
        Without an <code>access</code> value the document is treated as internal.
      </p>
      {result && <div className={`notice notice-${result.kind}`}>{result.text}</div>}
    </div>
  );
}
