"""Customer lookup tool. Staff only, with PII masked in code."""
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Customer
from app.tools.context import CallerContext, InvalidInput, NotFound, PermissionDenied
from app.tools.products import MAX_RESULTS, like_pattern

MAX_QUERY_LEN = 100


class OrderSummary(BaseModel):
    order_id: int
    status: str
    total_eur: float


class CustomerInfo(BaseModel):
    customer_id: int
    name: str
    email_masked: str
    phone_masked: str
    city: str
    novaplus_member: bool
    recent_orders: list[OrderSummary]


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[0]}***@{domain}"


def mask_phone(phone: str) -> str:
    digits = [c for c in phone if c.isdigit()]
    return "*** *** " + "".join(digits[-3:])


def lookup_customer(db: Session, caller: CallerContext, query: str) -> list[CustomerInfo]:
    """Find customers by id, exact email, or (part of) name."""
    if not caller.is_staff:
        raise PermissionDenied("Customer lookup is available to NovaTech support staff only.")

    query = (query or "").strip()
    if not query or len(query) > MAX_QUERY_LEN:
        raise InvalidInput("Please give a customer id, email address or name.")

    stmt = select(Customer).options(selectinload(Customer.orders))
    if query.isdigit():
        stmt = stmt.where(Customer.id == int(query))
    elif "@" in query:
        # Exact match only: a partial email search would let staff fish for addresses.
        stmt = stmt.where(func.lower(Customer.email) == query.lower())
    else:
        stmt = stmt.where(Customer.name.ilike(like_pattern(query), escape="\\"))

    customers = db.scalars(stmt.order_by(Customer.id).limit(MAX_RESULTS)).all()
    if not customers:
        raise NotFound(f"No customer matching '{query}' was found.")

    return [
        CustomerInfo(
            customer_id=c.id, name=c.name,
            email_masked=mask_email(c.email), phone_masked=mask_phone(c.phone),
            city=c.city, novaplus_member=c.novaplus_member,
            recent_orders=[OrderSummary(order_id=o.id, status=o.status.value,
                                        total_eur=float(o.total))
                           for o in sorted(c.orders, key=lambda o: o.id, reverse=True)[:5]],
        )
        for c in customers
    ]
