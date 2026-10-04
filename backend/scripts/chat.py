"""Chat with the full copilot graph from the terminal, with persistent memory."""
import argparse
import sys
import time

from langchain_core.messages import HumanMessage

from app.agents.context import AgentContext
from app.agents.graph import build_chat_graph, open_checkpointer
from app.db.session import SessionLocal
from app.rag.retriever import retrieve
from app.rag.vectorstore import get_store
from app.security import guard
from app.tools import load_caller


def turn(app, text: str, ctx: AgentContext, thread_id: str) -> None:
    start = last = time.perf_counter()
    final: dict = {}
    for update in app.stream({"messages": [HumanMessage(text)]}, context=ctx,
                             config={"configurable": {"thread_id": thread_id}},
                             stream_mode="updates"):
        for node, out in update.items():
            now = time.perf_counter()
            detail = ""
            if node == "guard":
                detail = f"{out['guard']['label']} (p_blocked={out['guard']['p_blocked']})"
            elif node == "supervisor":
                detail = (f"route={out['route']} k={out['knowledge_question']!r} "
                          f"queries={out['search_queries']} ops={out['operations_request']!r}")
            elif node == "knowledge":
                detail = f"{out['knowledge']['status']}, {len(out['knowledge']['citations'])} citation(s)"
            elif node == "operations":
                detail = ", ".join(f"{e['tool']}→{e['outcome']}" for e in out["operations"]["tool_log"])
            elif node == "output_guard" and out["output_findings"]:
                detail = f"findings={out['output_findings']}"
            print(f"    · {node:12} {now - last:4.1f}s  {detail}")
            last = now
            final.update(out)
    print(f"\n  copilot ({time.perf_counter() - start:.1f}s): {final['answer']}")
    for c in final.get("citations", []):
        print(f"    [{c['id']}] {c['title']} > {c['section']} ({c['source']})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="mila", help="mila | lucas | sara | admin")
    parser.add_argument("--thread", default="cli")
    parser.add_argument("--say", action="append", default=[], help="send these messages, then exit")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    with SessionLocal() as db:
        ctx = AgentContext(caller=load_caller(db, args.user))
    thread_id = f"{args.user}:{args.thread}"
    print(f"Loading models... (user={args.user}, role={ctx.caller.role.value}, thread={thread_id})")
    retrieve("warm-up", ctx.caller.role)     # embedding + reranker models
    guard.probabilities("warm-up")           # guard classifier
    app = build_chat_graph(open_checkpointer())

    try:
        messages = args.say or iter(lambda: input("you: ").strip(), "")
        for text in messages:
            if args.say:
                print(f"you: {text}")
            turn(app, text, ctx, thread_id)
    finally:
        get_store().client.close()


if __name__ == "__main__":
    main()
