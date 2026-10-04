"""Telegram channel: account linking and the bot's chat endpoint."""
import hmac
import logging
import secrets
import threading
import time
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import current_caller, get_db
from app.api.chat import ChatResponse, Turn, UNAVAILABLE
from app.config import get_settings
from app.db.models import AppUser, Conversation
from app.security import guard
from app.tools import CallerContext

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telegram", tags=["telegram"])

CODE_TTL_SECONDS = 600
_codes: dict[str, tuple[str, float]] = {}      # code -> (username, expires at)
_codes_lock = threading.Lock()


# --- Web side: a logged-in user asks for a link code ------------------------------------

@router.post("/link-code")
def link_code(caller: CallerContext = Depends(current_caller)) -> dict:
    code = secrets.token_hex(4).upper()          # 8 hex characters, about 4 billion options
    with _codes_lock:
        now = time.time()
        for c in [c for c, (_, exp) in _codes.items() if exp < now]:
            del _codes[c]
        _codes[code] = (caller.username, now + CODE_TTL_SECONDS)
    return {"code": code, "expires_in": CODE_TTL_SECONDS}


# --- Bot side: authenticated with the service secret ------------------------------------

def require_bot(x_bot_secret: str = Header(default="")) -> None:
    expected = get_settings().bot_secret
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Telegram is not configured.")
    if not hmac.compare_digest(x_bot_secret, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid bot secret.")


class LinkRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20)
    telegram_user_id: int


class BotChatRequest(BaseModel):
    telegram_user_id: int
    chat_id: int
    message: str = Field(min_length=1, max_length=guard.MAX_INPUT_CHARS)


class TelegramUser(BaseModel):
    telegram_user_id: int


class TelegramChat(BaseModel):
    telegram_user_id: int
    chat_id: int


@router.post("/link", dependencies=[Depends(require_bot)])
def link(req: LinkRequest, db: Session = Depends(get_db)) -> dict:
    with _codes_lock:
        entry = _codes.pop(req.code.strip().upper(), None)   # single use, even if it fails below
    if entry is None or entry[1] < time.time():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "That code is invalid or has expired.")
    username = entry[0]
    # A Telegram account links to at most one user: move it if it was linked elsewhere.
    for other in db.scalars(select(AppUser).where(AppUser.telegram_user_id == req.telegram_user_id)):
        other.telegram_user_id = None
    db.flush()
    user = db.scalar(select(AppUser).where(AppUser.username == username))
    user.telegram_user_id = req.telegram_user_id
    db.commit()
    return {"username": user.username, "role": user.role.value}


def _linked_caller(db: Session, telegram_user_id: int) -> CallerContext:
    user = db.scalar(select(AppUser).where(AppUser.telegram_user_id == telegram_user_id))
    if user is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not linked")
    return CallerContext(username=user.username, role=user.role, customer_id=user.customer_id)


def _telegram_conversation(db: Session, caller: CallerContext, chat_id: int) -> Conversation:
    conv_id = f"tg{chat_id}-{caller.username}"[:40]
    conv = db.get(Conversation, conv_id)
    now = datetime.now()
    if conv is None:
        conv = Conversation(id=conv_id, username=caller.username, title="Telegram chat",
                            created_at=now, updated_at=now)
        db.add(conv)
    conv.updated_at = now
    db.commit()
    return conv


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_bot)])
def bot_chat(req: BotChatRequest, request: Request, db: Session = Depends(get_db)) -> ChatResponse:
    caller = _linked_caller(db, req.telegram_user_id)
    request.app.state.rate_limiter.check(caller.username)   # shared with the web chat
    turn = Turn(request, caller, _telegram_conversation(db, caller, req.chat_id), req.message)
    try:
        for _ in turn.steps():
            pass
    except Exception:
        log.exception("telegram turn failed: user=%s chat=%s", caller.username, req.chat_id)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, UNAVAILABLE)
    return turn.response()


@router.post("/reset", dependencies=[Depends(require_bot)])
def reset(req: TelegramChat, request: Request, db: Session = Depends(get_db)) -> dict:
    """Forget this Telegram chat's conversation (memory included)."""
    caller = _linked_caller(db, req.telegram_user_id)
    conv = db.get(Conversation, f"tg{req.chat_id}-{caller.username}"[:40])
    if conv is not None:
        request.app.state.graph.checkpointer.delete_thread(conv.thread_id)
        db.delete(conv)
        db.commit()
    return {"ok": True}


@router.post("/unlink", dependencies=[Depends(require_bot)])
def unlink(req: TelegramUser, db: Session = Depends(get_db)) -> dict:
    for user in db.scalars(select(AppUser).where(AppUser.telegram_user_id == req.telegram_user_id)):
        user.telegram_user_id = None
    db.commit()
    return {"ok": True}
