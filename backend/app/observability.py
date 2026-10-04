"""Observability: one structured record per chat turn, and the metrics computed from them."""
import json
import logging
import math
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from app.config import BACKEND_DIR, get_settings

log = logging.getLogger("novatech.turns")
_write_lock = threading.Lock()
PRIMARY_MODEL_PREFIX = "gemini-3.5-flash-lite"


class LLMCounter(BaseCallbackHandler):
    """Counts chat-model calls, tokens and model names in one turn."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls, self.input_tokens, self.output_tokens = 0, 0, 0
        self.models: Counter = Counter()

    def on_llm_end(self, response, **kwargs):
        with self.lock:
            for generation in (g for batch in response.generations for g in batch):
                msg = getattr(generation, "message", None)
                self.calls += 1
                if msg is not None:
                    usage = msg.usage_metadata or {}
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                    self.models[msg.response_metadata.get("model_name", "?")] += 1


def turns_file() -> Path:
    return Path(get_settings().logs_dir) / "turns.jsonl"


def turn_record(*, caller, conversation_id: str, result: dict, timings: dict, counter: LLMCounter,
                latency_ms: int, error: str | None = None) -> dict:
    ops = result.get("operations") or {}
    knowledge = result.get("knowledge") or {}
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "user": caller.username, "role": caller.role.value, "conversation": conversation_id,
        "route": result.get("route"), "guard": result.get("guard"),
        "knowledge_status": knowledge.get("status"), "citations": len(result.get("citations", [])),
        "ops_status": ops.get("status"),
        "tools": [{"tool": e["tool"], "outcome": e["outcome"], "ms": e["ms"]} for e in ops.get("tool_log", [])],
        "output_findings": result.get("output_findings", []),
        "llm_calls": counter.calls, "input_tokens": counter.input_tokens,
        "output_tokens": counter.output_tokens, "models": dict(counter.models),
        "node_ms": timings, "latency_ms": latency_ms, "error": error,
    }


def log_turn(record: dict) -> None:
    path = turns_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:   # API endpoints run in a thread pool
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("turn user=%s route=%s latency=%dms llm_calls=%d tools=%s findings=%s error=%s",
             record["user"], record["route"], record["latency_ms"], record["llm_calls"],
             [f"{t['tool']}:{t['outcome']}" for t in record["tools"]], record["output_findings"],
             record["error"])


def read_turns(days: int = 7) -> list[dict]:
    path = turns_file()
    if not path.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:   # a half-written line must not break the dashboard
            continue
        if row.get("ts", "") >= cutoff:
            rows.append(row)
    return rows


def _pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] if ordered else 0


def compute_metrics(turns: list[dict]) -> dict:
    ok = [t for t in turns if not t.get("error")]
    latencies = [t["latency_ms"] for t in ok]
    by_route = Counter(t.get("route") or "error" for t in turns)
    tool_outcomes = Counter(f"{x['tool']}:{x['outcome']}" for t in turns for x in t.get("tools", []))
    models = Counter()
    for t in turns:
        models.update(t.get("models", {}))
    knowledge_turns = [t for t in turns if t.get("knowledge_status")]
    node_ms = defaultdict(list)
    for t in ok:
        for node, ms in (t.get("node_ms") or {}).items():
            node_ms[node].append(ms)
    hourly = defaultdict(list)
    for t in turns:
        hourly[t["ts"][:13]].append(t)   # "2026-09-30T14"
    return {
        "turns": len(turns),
        "errors": len(turns) - len(ok),
        "by_route": dict(by_route),
        "blocked_rate": round(by_route.get("blocked", 0) / len(turns), 3) if turns else 0,
        "latency_ms": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95),
                       "max": max(latencies, default=0)},
        "node_p50_ms": {n: _pct(v, 50) for n, v in sorted(node_ms.items())},
        "llm": {
            "calls": sum(t.get("llm_calls", 0) for t in turns),
            "tokens": sum(t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in turns),
            "calls_per_turn": round(sum(t.get("llm_calls", 0) for t in turns) / len(turns), 2) if turns else 0,
            "models": dict(models),
            "fallback_calls": sum(n for m, n in models.items() if not m.startswith(PRIMARY_MODEL_PREFIX)),
        },
        "tools": dict(tool_outcomes),
        "knowledge_not_found_rate": (round(sum(t["knowledge_status"] == "not_found" for t in knowledge_turns)
                                           / len(knowledge_turns), 3) if knowledge_turns else 0),
        "output_guard_findings": dict(Counter(f for t in turns for f in t.get("output_findings", []))),
        "users": len({t["user"] for t in turns}),
        "hourly": [{"hour": h, "turns": len(ts),
                    "p50_ms": _pct([t["latency_ms"] for t in ts if not t.get("error")], 50)}
                   for h, ts in sorted(hourly.items())],
    }


def latest_eval() -> dict | None:
    """Summary of the most recent eval run, shown on the metrics page."""
    runs = sorted((BACKEND_DIR.parent / "eval" / "results").glob("run-*.json"))
    if not runs:
        return None
    data = json.loads(runs[-1].read_text(encoding="utf-8"))
    return {"started": data["started"], **data["summary"],
            "case_results": [{"id": r["id"], "category": r["category"], "passed": r["passed"],
                              "latency_ms": r["latency_ms"]} for r in data["results"]]}
