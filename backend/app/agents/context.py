"""Run-scoped context shared by every node of the agent graphs."""
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.tools import CallerContext


@dataclass(frozen=True)
class AgentContext:
    caller: CallerContext
    db: Callable[[], Session] = SessionLocal   # tests pass an in-memory session factory
