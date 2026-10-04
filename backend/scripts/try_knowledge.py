"""Ask the knowledge agent a question and watch which nodes run. Needs GOOGLE_API_KEY."""
import argparse
import sys
import time

from app.agents.knowledge import build_knowledge_graph
from app.db.models import Role
from app.rag.retriever import retrieve
from app.rag.vectorstore import get_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--role", default="customer", choices=[r.value for r in Role])
    parser.add_argument("--query", action="append", default=[],
                        help="one search query per topic (the supervisor will produce these)")
    args = parser.parse_args()
    # Windows consoles default to cp1252, which can't print "→" or much LLM output.
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        run(args.question, Role(args.role), args.query)
    finally:
        get_store().client.close()


def run(question: str, role: Role, queries: list[str]) -> None:
    graph = build_knowledge_graph()
    retrieve("warm-up", role)
    inputs = {"question": question, "role": role}
    if queries:
        inputs["queries"] = queries

    start = last = time.perf_counter()
    final: dict = {}
    for update in graph.stream(inputs, stream_mode="updates"):
        for node, out in update.items():
            detail = ""
            if node == "retrieve":
                top = out["chunks"][0].relevance if out["chunks"] else 0.0
                detail = f"top={top:.3f} sufficient={out['sufficient']} queries={out['queries']}"
            elif node == "grade":
                detail = f"kept {len(out['chunks'])} chunk(s)"
            elif node == "rewrite":
                detail = f"new queries={out['queries']}"
            now = time.perf_counter()
            print(f"  → {node:9} {now - last:4.1f}s  {detail}")
            last = now
            final.update(out)

    print(f"\n[{final['status']}] ({time.perf_counter() - start:.1f}s)\n{final['answer']}\n")
    for c in final["citations"]:
        print(f"  [{c['id']}] {c['title']} > {c['section']} ({c['source']})")


if __name__ == "__main__":
    main()
