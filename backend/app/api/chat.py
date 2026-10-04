"""Chat and conversation endpoints."""
import json
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.context import AgentContext
from app.api.auth import current_caller, get_db
from app.db.models import Conversation
from app.observability import LLMCounter, log_turn, turn_record
from app.security import guard
from app.tools import CallerContext

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

UNAVAILABLE = ("The assistant is temporarily unavailable (the free model quota may be used up). "
               "Please try again in a minute.")
STEP_LABELS = {
    "guard": "Checked your message",
    "smalltalk": "Wrote a reply",
    "supervisor": "Understood your request",
    "knowledge": "Searched NovaTech documents",
    "operations": "Checked live systems",
    "synthesizer": "Put the answer together",
    "output_guard": "Final safety check",
}


class RateLimiter:
    """Sliding window per user."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max, self.window = max_requests, window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()   # endpoints run in a thread pool

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.max:
                retry = int(hits[0] + self.window - now) + 1
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                                    f"Too many messages. Try again in {retry} s.",
                                    headers={"Retry-After": str(retry)})
            hits.append(now)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=guard.MAX_INPUT_CHARS)
    # Only the id of one of the caller's own conversations; omit it to start a new one.
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,40}$")


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[dict]
    route: str | None
    guard: dict | None
    tool_log: list[dict]
    output_findings: list[str]
    latency_ms: int


def _own_conversation(db: Session, caller: CallerContext, conversation_id: str) -> Conversation:
    conv = db.scalar(select(Conversation).where(Conversation.id == conversation_id,
                                                Conversation.username == caller.username))
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found.")
    return conv


def _open_conversation(db: Session, caller: CallerContext, req: ChatRequest) -> Conversation:
    if req.conversation_id:
        conv = _own_conversation(db, caller, req.conversation_id)
    else:
        now = datetime.now()
        conv = Conversation(id=secrets.token_urlsafe(9), username=caller.username,
                            title=req.message.strip()[:80], created_at=now, updated_at=now)
        db.add(conv)
    conv.updated_at = datetime.now()
    db.commit()
    return conv


class Turn:
    """One chat turn: runs the graph step by step, records per-step timings and LLM usage, and writes the observability record when it ends, successfully or not."""

    def __init__(self, request: Request, caller: CallerContext, conv: Conversation, message: str):
        self.graph, self.caller, self.conv_id = request.app.state.graph, caller, conv.id
        self.counter = LLMCounter()
        self.inputs = {"messages": [HumanMessage(message)]}
        self.kwargs = {"context": AgentContext(caller=caller, db=request.app.state.session_factory),
                       "config": {"configurable": {"thread_id": conv.thread_id},
                                  "callbacks": [self.counter]}}
        self.result: dict = {}
        self.timings: dict[str, int] = {}
        self.latency_ms = 0

    def steps(self):
        """Yield each node's name as it finishes."""
        started = last = time.perf_counter()
        error = None
        try:
            for update in self.graph.stream(self.inputs, stream_mode="updates", **self.kwargs):
                for node, out in update.items():
                    now = time.perf_counter()
                    self.timings[node] = round((now - last) * 1000)
                    last = now
                    self.result.update(out or {})
                    yield node
        except GeneratorExit:            # streaming client went away mid-turn
            error = "client disconnected"
            raise
        except Exception as e:
            error = type(e).__name__
            raise
        finally:
            self.latency_ms = round((time.perf_counter() - started) * 1000)
            log_turn(turn_record(caller=self.caller, conversation_id=self.conv_id, result=self.result,
                                 timings=self.timings, counter=self.counter,
                                 latency_ms=self.latency_ms, error=error))

    def response(self) -> ChatResponse:
        r = self.result
        return ChatResponse(conversation_id=self.conv_id, answer=r["answer"],
                            citations=r.get("citations", []), route=r.get("route"),
                            guard=r.get("guard"), tool_log=(r.get("operations") or {}).get("tool_log", []),
                            output_findings=r.get("output_findings", []), latency_ms=self.latency_ms)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, caller: CallerContext = Depends(current_caller),
         db: Session = Depends(get_db)) -> ChatResponse:
    request.app.state.rate_limiter.check(caller.username)
    turn = Turn(request, caller, _open_conversation(db, caller, req), req.message)
    try:
        for _ in turn.steps():
            pass
    except Exception:
        log.exception("chat turn failed: user=%s conversation=%s", caller.username, turn.conv_id)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE)
    return turn.response()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request, caller: CallerContext = Depends(current_caller),
                db: Session = Depends(get_db)) -> StreamingResponse:
    """Server-sent events: one "step" event per finished node, then "answer"."""
    request.app.state.rate_limiter.check(caller.username)
    turn = Turn(request, caller, _open_conversation(db, caller, req), req.message)

    def events():
        yield _sse("start", {"conversation_id": turn.conv_id})
        try:
            for node in turn.steps():
                yield _sse("step", {"node": node, "label": STEP_LABELS.get(node, node)})
        except Exception:
            log.exception("streamed chat turn failed: user=%s conversation=%s", caller.username, turn.conv_id)
            yield _sse("error", {"detail": UNAVAILABLE})
            return
        yield _sse("answer", turn.response().model_dump())

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- Conversations ---------------------------------------------------------------------

@router.get("/conversations")
def list_conversations(caller: CallerContext = Depends(current_caller),
                       db: Session = Depends(get_db)) -> list[dict]:
    rows = db.scalars(select(Conversation).where(Conversation.username == caller.username)
                      .order_by(Conversation.updated_at.desc())).all()
    return [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in rows]


@router.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: str, request: Request,
                          caller: CallerContext = Depends(current_caller),
                          db: Session = Depends(get_db)) -> list[dict]:
    conv = _own_conversation(db, caller, conversation_id)
    state = request.app.state.graph.get_state({"configurable": {"thread_id": conv.thread_id}})
    return [{"role": "user" if m.type == "human" else "assistant", "content": m.text,
             "citations": m.additional_kwargs.get("citations", [])}
            for m in state.values.get("messages", [])]


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, request: Request,
                        caller: CallerContext = Depends(current_caller),
                        db: Session = Depends(get_db)) -> None:
    conv = _own_conversation(db, caller, conversation_id)
    request.app.state.graph.checkpointer.delete_thread(conv.thread_id)   # the memory itself
    db.delete(conv)
    db.commit()
