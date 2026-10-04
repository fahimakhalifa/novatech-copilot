"""Login, password hashing and signed tokens, using only the standard library."""
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import AppUser, Role
from app.tools import CallerContext

log = logging.getLogger(__name__)

TOKEN_TTL_SECONDS = 8 * 3600
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}   # standard interactive-login parameters


# --- Passwords -------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


# --- Tokens ----------------------------------------------------------------------------

_fallback_secret = secrets.token_hex(32)


def _secret() -> bytes:
    configured = get_settings().app_secret
    if not configured:
        # Tokens then stop working after a restart, which is safe but annoying.
        log.warning("APP_SECRET is not set: using a random per-process secret")
    return (configured or _fallback_secret).encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def create_token(username: str, now: float | None = None) -> str:
    payload = _b64(json.dumps({"sub": username,
                               "exp": int((now or time.time()) + TOKEN_TTL_SECONDS)}).encode())
    signature = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}"


def read_token(token: str, now: float | None = None) -> str | None:
    """The username in a valid, unexpired token, else None."""
    try:
        payload, signature = token.split(".")
        expected = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        claims = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if claims.get("exp", 0) < (now or time.time()):
        return None
    return claims.get("sub")


# --- FastAPI dependencies --------------------------------------------------------------

def get_db():
    """Overridden in tests with an in-memory session factory."""
    from app.db.session import SessionLocal
    with SessionLocal() as db:
        yield db


_bearer = HTTPBearer(auto_error=False)
_UNAUTHORIZED = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in or session expired.",
                              headers={"WWW-Authenticate": "Bearer"})


def authenticate(db: Session, username: str, password: str) -> AppUser | None:
    user = db.scalar(select(AppUser).where(AppUser.username == username))
    if user is None:
        # Hash anyway, so "unknown user" and "wrong password" take the same time.
        verify_password(password, hash_password("timing-equaliser"))
        return None
    return user if verify_password(password, user.password_hash) else None


def current_caller(creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
                   db: Session = Depends(get_db)) -> CallerContext:
    username = read_token(creds.credentials) if creds else None
    if not username:
        raise _UNAUTHORIZED
    user = db.scalar(select(AppUser).where(AppUser.username == username))
    if user is None:                      # account removed after the token was issued
        raise _UNAUTHORIZED
    return CallerContext(username=user.username, role=user.role, customer_id=user.customer_id)


def require_staff(caller: CallerContext = Depends(current_caller)) -> CallerContext:
    if not caller.is_staff:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Staff only.")
    return caller


def require_admin(caller: CallerContext = Depends(current_caller)) -> CallerContext:
    if caller.role != Role.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admins only.")
    return caller
