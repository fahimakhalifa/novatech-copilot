"""FastAPI application."""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import observability
from app.api import auth, chat, documents, telegram
from app.config import get_settings
from app.db.models import Base, Role
from app.db.session import SessionLocal, engine
from app.tools import CallerContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=200)


def create_app(*, graph=None, warmup: bool = True) -> FastAPI:
    """`graph` and `warmup=False` let tests use an in-memory graph without loading models."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(engine)   # adds missing tables only; never drops data
        if app.state.graph is None:
            from app.agents.graph import build_chat_graph, open_checkpointer
            app.state.graph = build_chat_graph(open_checkpointer())
        if warmup:
            from app.rag.retriever import retrieve
            from app.security import guard
            retrieve("warm-up", Role.CUSTOMER)
            guard.probabilities("warm-up")
            log.info("models loaded, API ready")
        yield
        if warmup:
            from app.rag.vectorstore import get_store
            get_store().client.close()

    app = FastAPI(title="NovaTech Copilot API", version="1.0", lifespan=lifespan)
    app.state.graph = graph
    app.state.session_factory = SessionLocal
    app.state.rate_limiter = chat.RateLimiter(max_requests=10, window_seconds=60)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/auth/login", tags=["auth"])
    def login(req: LoginRequest, db: Session = Depends(auth.get_db)) -> dict:
        user = auth.authenticate(db, req.username, req.password)
        if user is None:
            # One message for both cases, so the endpoint doesn't reveal which usernames exist.
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password.")
        return {"access_token": auth.create_token(user.username), "token_type": "bearer",
                "user": {"username": user.username, "role": user.role.value}}

    @app.get("/api/auth/me", tags=["auth"])
    def me(caller: CallerContext = Depends(auth.current_caller)) -> dict:
        return {"username": caller.username, "role": caller.role.value}

    @app.get("/api/metrics", tags=["meta"])
    def metrics(days: int = Query(7, ge=1, le=90),
                _: CallerContext = Depends(auth.require_staff)) -> dict:
        """Live metrics from the per-turn log plus the latest evaluation run."""
        return {"live": observability.compute_metrics(observability.read_turns(days)),
                "eval": observability.latest_eval()}

    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(telegram.router)
    return app


app = create_app()
