"""Database models for NovaTech's structured business data."""
from datetime import datetime
from enum import Enum

from sqlalchemy import ForeignKey, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Role(str, Enum):
    """Who is talking to the copilot. Enforced in tool code, never in the prompt."""
    CUSTOMER = "customer"          # own orders only, public docs only
    SUPPORT_AGENT = "support_agent"  # any order, customer lookup, internal docs
    ADMIN = "admin"                # everything


class OrderStatus(str, Enum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(200), unique=True)   # Confidential
    phone: Mapped[str] = mapped_column(String(30))                 # Confidential
    city: Mapped[str] = mapped_column(String(80))
    postcode: Mapped[str] = mapped_column(String(10))
    novaplus_member: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime]

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    sku: Mapped[str] = mapped_column(String(30), unique=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    category: Mapped[str] = mapped_column(String(50))
    price: Mapped[float] = mapped_column(Numeric(10, 2))  # EUR, incl. VAT
    stock: Mapped[int]


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)  # the "order #1025" number
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    status: Mapped[OrderStatus]
    created_at: Mapped[datetime]
    shipped_at: Mapped[datetime | None]
    delivered_at: Mapped[datetime | None]
    tracking_number: Mapped[str | None] = mapped_column(String(40))
    total: Mapped[float] = mapped_column(Numeric(10, 2))

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    quantity: Mapped[int]
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2))  # price at time of order

    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Product] = relationship()


class AppUser(Base):
    """Accounts that can talk to the copilot (web login or Telegram link)."""
    __tablename__ = "app_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    role: Mapped[Role]
    # Set only for role=customer: which customer record this user *is*.
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"))
    telegram_user_id: Mapped[int | None] = mapped_column(unique=True)
    # "scrypt$<salt hex>$<hash hex>", see app/api/auth.py. Never the password itself.
    password_hash: Mapped[str | None] = mapped_column(String(200))


class Conversation(Base):
    """A chat thread."""
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)   # random, from the server
    username: Mapped[str] = mapped_column(ForeignKey("app_users.username"), index=True)
    title: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    @property
    def thread_id(self) -> str:
        return f"{self.username}:{self.id}"
