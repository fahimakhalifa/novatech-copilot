"""Who is calling a tool, and the errors a tool may raise."""
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AppUser, Role


@dataclass(frozen=True)
class CallerContext:
    username: str
    role: Role
    customer_id: int | None = None   # set only for customers: whose data is "theirs"

    def __post_init__(self):
        if self.role == Role.CUSTOMER and self.customer_id is None:
            raise ValueError(f"customer '{self.username}' has no customer_id")

    @property
    def is_staff(self) -> bool:
        return self.role in (Role.SUPPORT_AGENT, Role.ADMIN)


def load_caller(db: Session, username: str) -> CallerContext:
    user = db.scalar(select(AppUser).where(AppUser.username == username))
    if user is None:
        raise LookupError(f"unknown user '{username}'")
    return CallerContext(username=user.username, role=user.role, customer_id=user.customer_id)


class ToolError(Exception):
    """A failure whose message is safe to show the user (and the LLM)."""


class InvalidInput(ToolError):
    pass


class NotFound(ToolError):
    pass


class PermissionDenied(ToolError):
    pass
