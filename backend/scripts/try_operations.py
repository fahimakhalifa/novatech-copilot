"""Send a request to the operations agent as a test user. Needs GOOGLE_API_KEY."""
import argparse
import sys
import time

from app.agents.context import AgentContext
from app.agents.operations import build_operations_graph
from app.db.session import SessionLocal
from app.tools import load_caller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("--user", default="mila", help="mila | lucas | sara | admin")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with SessionLocal() as db:
        caller = load_caller(db, args.user)   # what the API will do after login
    graph = build_operations_graph()

    start = last = time.perf_counter()
    final: dict = {"tool_log": []}
    for update in graph.stream({"request": args.request}, context=AgentContext(caller=caller),
                               stream_mode="updates"):
        for node, out in update.items():
            now = time.perf_counter()
            detail = ""
            if node == "plan":
                reply = out["messages"][-1]
                calls = ", ".join(f"{c['name']}({c['args']})" for c in reply.tool_calls)
                detail = f"[{reply.response_metadata.get('model_name', '?')}] {calls or '(answers)'}"
            elif node == "permission_gate":
                detail = f"approved {len(out['approved'])}, denied {len(out['tool_log'])}"
            elif node == "run_tool":
                detail = ", ".join(f"{e['tool']}→{e['outcome']}" for e in out["tool_log"])
            print(f"  → {node:15} {now - last:4.1f}s  {detail}")
            last = now
            for key, value in out.items():
                if key == "tool_log":
                    final["tool_log"] += value
                else:
                    final[key] = value

    print(f"\n[{final['status']}] user={caller.username} role={caller.role.value} "
          f"({time.perf_counter() - start:.1f}s)\n{final['answer']}\n")
    for e in final["tool_log"]:
        print(f"  log: {e['tool']} {e['args']} → {e['outcome']} {e['detail']}")


if __name__ == "__main__":
    main()
