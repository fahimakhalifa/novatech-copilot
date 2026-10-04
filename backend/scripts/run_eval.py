"""Evaluation harness: run eval/cases.jsonl through the full graph with real models."""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from app.agents.context import AgentContext
from app.agents.graph import build_chat_graph
from app.config import BACKEND_DIR, get_settings
from app.db.models import Role
from app.db.session import SessionLocal
from app.llm import get_llm
from app.observability import LLMCounter
from app.rag.loader import chunk_document, parse_markdown
from app.rag.retriever import retrieve
from app.rag.vectorstore import get_store
from app.security import guard
from app.tools import load_caller

EVAL_DIR = BACKEND_DIR.parent / "eval"
SECONDS_PER_CALL = 60 / 12
PRIMARY_MODEL = "gemini-3.5-flash-lite"


# --- Checks ----------------------------------------------------------------------------

def check(expect: dict, result: dict, counter: LLMCounter) -> list[str]:
    """Every failed expectation as a readable line (empty list = pass)."""
    answer = result.get("answer", "")
    low = answer.lower()
    tool_log = (result.get("operations") or {}).get("tool_log", [])
    ok_tools = {e["tool"] for e in tool_log if e["outcome"] == "ok"}
    cited = {c["source"] for c in result.get("citations", [])}
    k_status = (result.get("knowledge") or {}).get("status")
    failures = []

    if "route" in expect and result.get("route") not in expect["route"]:
        failures.append(f"route was {result.get('route')!r}, expected one of {expect['route']}")
    for text in expect.get("contains_all", []):
        if text.lower() not in low:
            failures.append(f"missing {text!r}")
    if expect.get("contains_any") and not any(t.lower() in low for t in expect["contains_any"]):
        failures.append(f"contains none of {expect['contains_any']}")
    for text in expect.get("not_contains", []):
        if text.lower() in low:
            failures.append(f"must not contain {text!r}")
    for source in expect.get("cites", []):
        if source not in cited:
            failures.append(f"no citation of {source} (cited: {sorted(cited) or 'nothing'})")
    if "knowledge_status" in expect and k_status != expect["knowledge_status"]:
        failures.append(f"knowledge status was {k_status!r}, expected {expect['knowledge_status']!r}")
    for tool in expect.get("tool_ok", []):
        if tool not in ok_tools:
            failures.append(f"tool {tool} did not run successfully")
    for tool in expect.get("tool_not_ok", []):
        if tool in ok_tools:
            failures.append(f"tool {tool} ran successfully but must not")
    if "max_llm_calls" in expect and counter.calls > expect["max_llm_calls"]:
        failures.append(f"{counter.calls} LLM calls, expected at most {expect['max_llm_calls']}")
    return failures


# --- Judge -----------------------------------------------------------------------------

class Verdict(BaseModel):
    groundedness: int = Field(ge=1, le=5, description="5 = every claim is supported by the sources")
    relevance: int = Field(ge=1, le=5, description="5 = fully and directly answers the question")
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = Field(description="one sentence")


JUDGE_PROMPT = """You are evaluating an AI assistant's answer against the source documents it cited.

Question: {question}

Answer:
{answer}

Cited sources:
{sources}

Score two things from 1 to 5:
- groundedness: are ALL factual claims in the answer supported by the sources? Applying a
  general rule from a source to a specific case (e.g. "most products can be returned" to
  headphones) counts as supported. 5 = fully supported, 3 = some unsupported detail,
  1 = mostly unsupported or contradicts the sources.
- relevance: does the answer directly and completely address the question? 5 = fully.
List every unsupported claim (empty list if none). Give a one-sentence reason."""


def source_texts(citations: list[dict]) -> str:
    """Load the cited sections from disk (not from the system's own output)."""
    root, wanted, out = get_settings().docs_dir, {(c["source"], c["section"]) for c in citations}, []
    for source in sorted({s for s, _ in wanted}):
        for chunk in chunk_document(parse_markdown(root / source, root)):
            if (source, chunk.metadata["section"]) in wanted:
                out.append(chunk.page_content)
    return "\n\n---\n\n".join(out)


def judge(question: str, result: dict) -> dict | None:
    if not result.get("citations"):
        return None
    prompt = JUDGE_PROMPT.format(question=question, answer=result["answer"],
                                 sources=source_texts(result["citations"]))
    try:
        return get_llm("judge", schema=Verdict).invoke(prompt).model_dump()
    except Exception as e:   # a judge outage must not hide the deterministic results
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


# --- Running ---------------------------------------------------------------------------

def run_case(app, case: dict, use_judge: bool) -> dict:
    with SessionLocal() as db:
        ctx = AgentContext(caller=load_caller(db, case["user"]))
    counter = LLMCounter()
    config = {"configurable": {"thread_id": f"eval:{case['id']}:{time.time()}"}, "callbacks": [counter]}
    started, result, turn_ms = time.perf_counter(), {}, 0
    for text in case["turns"]:
        t = time.perf_counter()
        result = app.invoke({"messages": [HumanMessage(text)]}, context=ctx, config=config)
        turn_ms = round((time.perf_counter() - t) * 1000)
    failures = check(case["expect"], result, counter)
    verdict = judge(case["turns"][-1], result) if use_judge and case.get("judge") else None
    return {
        "id": case["id"], "category": case["category"], "user": case["user"], "turns": case["turns"],
        "passed": not failures, "failures": failures,
        "answer": result.get("answer", ""), "route": result.get("route"),
        "citations": [f"{c['source']} > {c['section']}" for c in result.get("citations", [])],
        "latency_ms": turn_ms, "total_ms": round((time.perf_counter() - started) * 1000),
        "llm_calls": counter.calls, "input_tokens": counter.input_tokens,
        "output_tokens": counter.output_tokens, "models": dict(counter.models),
        "judge": verdict,
        "used_fallback": any(m != PRIMARY_MODEL for m in counter.models),
    }


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] if ordered else 0.0


def summarize(results: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in results]
    judged = [r["judge"] for r in results if r["judge"] and "error" not in r["judge"]]
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][0] += r["passed"]
        by_cat[r["category"]][1] += 1
    models = Counter()
    for r in results:
        models.update(r["models"])
    return {
        "cases": len(results), "passed": sum(r["passed"] for r in results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "by_category": {c: f"{p}/{n}" for c, (p, n) in sorted(by_cat.items())},
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
                       "max": max(latencies)},
        "llm_calls_total": sum(r["llm_calls"] for r in results),
        "cases_with_fallback": [r["id"] for r in results if r["used_fallback"]],
        "tokens_total": sum(r["input_tokens"] + r["output_tokens"] for r in results),
        "models": dict(models),
        "judge": {
            "judged": len(judged),
            "groundedness_mean": round(sum(j["groundedness"] for j in judged) / len(judged), 2) if judged else None,
            "relevance_mean": round(sum(j["relevance"] for j in judged) / len(judged), 2) if judged else None,
            "fully_grounded": sum(j["groundedness"] == 5 for j in judged),
        },
    }


def markdown(summary: dict, results: list[dict], started: str) -> str:
    j, lat = summary["judge"], summary["latency_ms"]
    lines = [
        f"# Evaluation run {started}", "",
        f"**{summary['passed']}/{summary['cases']} cases passed ({summary['pass_rate']:.0%})**", "",
        "| Metric | Value |", "|---|---|",
        f"| Pass rate by category | {', '.join(f'{c} {v}' for c, v in summary['by_category'].items())} |",
        f"| Latency (last turn) | p50 {lat['p50'] / 1000:.1f}s, p95 {lat['p95'] / 1000:.1f}s, max {lat['max'] / 1000:.1f}s |",
        f"| Groundedness (judge, 1-5) | {j['groundedness_mean']} mean over {j['judged']} answers, {j['fully_grounded']} fully grounded |",
        f"| Relevance (judge, 1-5) | {j['relevance_mean']} mean |",
        f"| LLM calls / tokens | {summary['llm_calls_total']} calls, {summary['tokens_total']:,} tokens |",
        f"| Models used | {', '.join(f'{m} ×{n}' for m, n in summary['models'].items())} |",
        f"| Cases that needed a fallback model | {', '.join(summary['cases_with_fallback']) or 'none'} |",
        "", "| Case | Category | Result | Route | Latency | LLM calls | Grounded / relevant |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        jr = r["judge"]
        judged = "" if not jr else (jr.get("error", "")[:30] if "error" in jr else f"{jr['groundedness']} / {jr['relevance']}")
        fallback = " (fallback)" if r["used_fallback"] else ""
        lines.append(f"| {r['id']} | {r['category']} | {'PASS' if r['passed'] else '**FAIL**'} | {r['route']} "
                     f"| {r['latency_ms'] / 1000:.1f}s{fallback} | {r['llm_calls']} | {judged} |")
    problems = [r for r in results if not r["passed"] or (r["judge"] and r["judge"].get("unsupported_claims"))]
    if problems:
        lines += ["", "## Failures and unsupported claims", ""]
        for r in problems:
            lines.append(f"### {r['id']}")
            lines += [f"- {f}" for f in r["failures"]]
            if r["judge"] and r["judge"].get("unsupported_claims"):
                lines += [f"- judge: unsupported claim: {c}" for c in r["judge"]["unsupported_claims"]]
            lines += ["", "> " + r["answer"].replace("\n", "\n> "), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cases = [json.loads(line) for line in (EVAL_DIR / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    print("Loading models...")
    retrieve("warm-up", Role.SUPPORT_AGENT)
    guard.probabilities("warm-up")
    app = build_chat_graph(InMemorySaver())
    started = datetime.now().strftime("%Y-%m-%d %H:%M")

    results = []
    try:
        for i, case in enumerate(cases, 1):
            r = run_case(app, case, use_judge=not args.no_judge)
            results.append(r)
            jr = r["judge"]
            j = "" if not jr else (" judge=error" if "error" in jr else f" judge={jr['groundedness']}/{jr['relevance']}")
            print(f"[{i:2}/{len(cases)}] {'PASS' if r['passed'] else 'FAIL'} {r['id']:28} "
                  f"{r['latency_ms'] / 1000:5.1f}s calls={r['llm_calls']}{j}")
            for f in r["failures"]:
                print(f"         - {f}")
            if r["used_fallback"]:
                print(f"         (fallback model used: {r['models']})")
            time.sleep(max(1.0, r["llm_calls"] * SECONDS_PER_CALL - r["total_ms"] / 1000))
    finally:
        get_store().client.close()

    summary = summarize(results)
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    kind = "partial" if args.only else "run"
    (out_dir / f"{kind}-{stamp}.json").write_text(
        json.dumps({"started": started, "summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    if not args.only:
        (out_dir / "latest.md").write_text(markdown(summary, results, started), encoding="utf-8")
    report = f"eval/results/{kind}-{stamp}.json" if args.only else "eval/results/latest.md"
    print(f"\n{summary['passed']}/{summary['cases']} passed. Report: {report}")


if __name__ == "__main__":
    main()
