"""Order status tool. Customers see only their own orders; staff see any order."""
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Order, OrderItem
from app.tools.context import CallerContext, InvalidInput, NotFound


class OrderLine(BaseModel):
    product: str
    sku: str
    quantity: int
    unit_price_eur: float


class OrderInfo(BaseModel):
    order_id: int
    status: str
    created_at: datetime
    shipped_at: datetime | None
    delivered_at: datetime | None
    tracking_number: str | None
    total_eur: float
    items: list[OrderLine]
    customer_name: str | None = None   # filled in for staff only


def parse_order_id(order_id: int | str) -> int:
    """Accept 1025, "1025" or "#1025" (the LLM and users write it all three ways)."""
    text = str(order_id).strip().lstrip("#").strip()
    if not text.isdigit() or not 0 < int(text) < 10**9:
        raise InvalidInput(f"'{order_id}' is not a valid order number.")
    return int(text)


def get_order_status(db: Session, caller: CallerContext, order_id: int | str) -> OrderInfo:
    oid = parse_order_id(order_id)

    stmt = (select(Order).where(Order.id == oid)
            .options(selectinload(Order.items).selectinload(OrderItem.product),
                     selectinload(Order.customer)))
    if not caller.is_staff:
        stmt = stmt.where(Order.customer_id == caller.customer_id)

    order = db.scalar(stmt)
    if order is None:
        where = "" if caller.is_staff else " on your account"
        raise NotFound(f"No order #{oid} was found{where}.")

    return OrderInfo(
        order_id=order.id,
        status=order.status.value,
        created_at=order.created_at,
        shipped_at=order.shipped_at,
        delivered_at=order.delivered_at,
        tracking_number=order.tracking_number,
        total_eur=float(order.total),
        items=[OrderLine(product=i.product.name, sku=i.product.sku, quantity=i.quantity,
                         unit_price_eur=float(i.unit_price)) for i in order.items],
        customer_name=order.customer.name if caller.is_staff else None,
    )
