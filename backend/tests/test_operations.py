"""Operations agent: permission gate, tool loop and budget, with a scripted fake LLM."""
from dataclasses import replace

import pytest
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from app import llm
from app.agents import operations as ops
from app.agents.context import AgentContext


class ScriptedModel(GenericFakeChatModel):
    """Replays AIMessages in order. bind_tools is a no-op: tools are only schemas here."""
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, id="c1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


@pytest.fixture
def run_ops(monkeypatch, engine, caller):
    def run(username: str, *turns: AIMessage) -> dict:
        model = ScriptedModel(messages=iter(turns))
        monkeypatch.setattr(ops, "get_llm", lambda role, **kw: model)
        ctx = AgentContext(caller=caller(username), db=lambda: Session(engine))
        return ops.build_operations_graph().invoke({"request": "test"}, context=ctx)
    return run


# --- the gate's decision ---------------------------------------------------------------

def test_gate_approves_allowed_call(caller):
    spec, args = ops.check_call(call("get_order_status", {"order_id": "1025"}), caller("mila"), 0)
    assert (spec.name, args.order_id) == ("get_order_status", 1025)


@pytest.mark.parametrize("tool_call, reason", [
    (call("lookup_customer", {"query": "Mila"}), "not available for your role"),
    (call("drop_database", {}), "Unknown tool"),
    (call("get_order_status", {"order_id": "#1025"}), "Invalid argument order_id"),
    (call("get_order_status", {"order_id": 1025, "customer_id": 7}), "Invalid argument customer_id"),
])
def test_gate_denies(caller, tool_call, reason):
    assert reason in ops.check_call(tool_call, caller("mila"), 0)


def test_gate_lets_staff_look_up_customers(caller):
    assert not isinstance(ops.check_call(call("lookup_customer", {"query": "Mila"}), caller("sara"), 0), str)


def test_gate_limits_calls_per_round(caller):
    c = call("calculate", {"expression": "1+1"})
    assert not isinstance(ops.check_call(c, caller("mila"), ops.MAX_CALLS_PER_ROUND - 1), str)
    assert "Too many" in ops.check_call(c, caller("mila"), ops.MAX_CALLS_PER_ROUND)


# --- full subgraph with a scripted model ------------------------------------------------

def test_lucas_gets_not_found_through_the_whole_loop(run_ops):
    out = run_ops("lucas",
                  AIMessage("", tool_calls=[call("get_order_status", {"order_id": 1025})]),
                  AIMessage("Sorry, I couldn't find that order."))
    [entry] = out["tool_log"]
    assert (entry["outcome"], entry["detail"]) == ("error", "No order #1025 was found on your account.")
    assert out["status"] == "done"


def test_mila_gets_her_order(run_ops):
    out = run_ops("mila",
                  AIMessage("", tool_calls=[call("get_order_status", {"order_id": 1025})]),
                  AIMessage("It has shipped."))
    assert out["tool_log"][0]["outcome"] == "ok"


def test_invented_staff_tool_never_runs_for_a_customer(run_ops, monkeypatch):
    """Even if a (hallucinating or injected) model calls lookup_customer, it never executes."""
    ran = []
    spy = replace(ops.TOOLS["lookup_customer"], run=lambda *a: ran.append(a))
    monkeypatch.setitem(ops.TOOLS, "lookup_customer", spy)
    out = run_ops("mila",
                  AIMessage("", tool_calls=[call("lookup_customer", {"query": "Mila de Boer"})]),
                  AIMessage("I can't do that."))
    assert ran == []
    assert out["tool_log"][0]["outcome"] == "denied"


def test_runaway_tool_loop_stops_at_budget(run_ops):
    looping = [AIMessage("", tool_calls=[call("calculate", {"expression": "1+1"}, id=f"c{i}")])
               for i in range(ops.MAX_ROUNDS + 2)]
    out = run_ops("mila", *looping)
    assert out["status"] == "out_of_budget"
    assert out["answer"] == ops.BUDGET_TEXT
    assert len(out["tool_log"]) == ops.MAX_ROUNDS - 1   # the last round is refused before running


# --- model fallback chain ---------------------------------------------------------------

class RateLimited(GenericFakeChatModel):
    def _generate(self, *args, **kwargs):
        raise ModelRateLimitError("429 quota")


def test_rate_limited_primary_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "model_chain", lambda role: ["primary", "fallback"])
    monkeypatch.setattr(llm, "_model", lambda name, last: RateLimited(messages=iter([])) if name == "primary"
                        else GenericFakeChatModel(messages=iter([AIMessage("from fallback")])))
    assert llm.get_llm("agent").invoke("hi").text == "from fallback"
