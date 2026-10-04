"""Create and seed NovaTech's database."""
import random
from datetime import datetime, timedelta

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.auth import hash_password
from app.db.models import (AppUser, Base, Customer, Order, OrderItem, OrderStatus,
                           Product, Role)
from app.db.session import engine as default_engine

BASE_DATE = datetime(2026, 9, 29, 12, 0)

# Demo logins for the fictional test users (the DB stores only scrypt hashes).
DEMO_PASSWORDS = {"mila": "mila-demo-2026", "lucas": "lucas-demo-2026",
                  "sara": "sara-demo-2026", "admin": "admin-demo-2026"}

PRODUCTS = [
    # sku, name, category, price, stock
    ("NP-X-128", "NovaPhone X 128GB", "smartphones", 899.00, 42),
    ("NP-X-256", "NovaPhone X 256GB", "smartphones", 999.00, 17),
    ("NP-LITE-128", "NovaPhone Lite 128GB", "smartphones", 449.00, 80),
    ("NB-PRO-14", "NovaBook Pro 14 (16GB/512GB)", "laptops", 1499.00, 12),
    ("NB-PRO-14-32", "NovaBook Pro 14 (32GB/1TB)", "laptops", 1899.00, 5),
    ("NB-AIR-13", "NovaBook Air 13", "laptops", 999.00, 0),       # out of stock on purpose
    ("NT-TAB-11", "NovaTab 11", "tablets", 549.00, 25),
    ("SW-ANC-01", "SoundWave ANC Headphones", "audio", 299.00, 60),
    ("SW-BUDS-02", "SoundWave Buds 2", "audio", 149.00, 110),
    ("AC-CHG-65", "NovaCharge 65W USB-C Charger", "accessories", 39.99, 300),
    ("AC-CASE-NPX", "NovaPhone X Silicone Case", "accessories", 29.99, 150),
    ("AC-CBL-2M", "USB-C to USB-C Cable 2m", "accessories", 14.99, 500),
]

CUSTOMERS = [
    # name, city, postcode, novaplus
    ("Emma de Vries", "Amsterdam", "1012AB", True),
    ("Lucas Janssen", "Rotterdam", "3011CD", False),
    ("Sophie Bakker", "Utrecht", "3511EF", False),
    ("Daan Visser", "Den Haag", "2511GH", True),
    ("Julia Smit", "Eindhoven", "5611IJ", False),
    ("Noah Meijer", "Groningen", "9711KL", False),
    ("Mila de Boer", "Leiden", "2311MN", True),        # customer 7, owns order #1025
    ("Sem Mulder", "Haarlem", "2011OP", False),
    ("Tess de Groot", "Antwerp", "2000", False),
    ("Finn Bos", "Brussels", "1000", True),
    ("Lotte Vos", "Cologne", "50667", False),
    ("Levi Peters", "Paris", "75001", False),
    ("Sara Hendriks", "Luxembourg", "1111", False),
    ("Milan van Dijk", "Delft", "2611QR", True),
    ("Anna Dekker", "Nijmegen", "6511ST", False),
]


def make_email(name: str) -> str:
    first, *rest = name.lower().split()
    return f"{first}.{''.join(rest)}@example.com"


def make_phone(rng: random.Random) -> str:
    return f"+31 6 {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"


def build_order(rng: random.Random, order_id: int, customer: Customer,
                products: list[Product], status: OrderStatus, days_ago: int) -> Order:
    created = BASE_DATE - timedelta(days=days_ago, hours=rng.randint(0, 10))
    picks = rng.sample(products, k=rng.randint(1, 3))
    items = [OrderItem(product=p, quantity=rng.randint(1, 2), unit_price=p.price) for p in picks]

    shipped = delivered = tracking = None
    if status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED, OrderStatus.RETURNED):
        shipped = created + timedelta(days=1)
        tracking = f"NT{rng.randint(10**9, 10**10 - 1)}NL"
    if status in (OrderStatus.DELIVERED, OrderStatus.RETURNED):
        delivered = shipped + timedelta(days=2)

    total = round(sum(float(i.unit_price) * i.quantity for i in items), 2)
    return Order(id=order_id, customer=customer, status=status, created_at=created,
                 shipped_at=shipped, delivered_at=delivered, tracking_number=tracking,
                 total=total, items=items)


def seed(engine: Engine = default_engine) -> None:
    """Recreate all tables on `engine` and fill them. Tests pass an in-memory engine."""
    # A fresh RNG per call, so seeding twice in one process gives identical data.
    rng = random.Random(42)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as db:
        products = [Product(sku=s, name=n, category=c, price=p, stock=st)
                    for s, n, c, p, st in PRODUCTS]
        customers = [
            Customer(id=i, name=n, email=make_email(n), phone=make_phone(rng), city=c,
                     postcode=pc, novaplus_member=plus,
                     created_at=BASE_DATE - timedelta(days=rng.randint(60, 900)))
            for i, (n, c, pc, plus) in enumerate(CUSTOMERS, start=1)
        ]
        db.add_all(products + customers)
        db.flush()

        # Recent orders are more likely to still be processing or shipped.
        orders = []
        for order_id in range(1001, 1041):
            days_ago = (1041 - order_id) * 2
            if days_ago <= 2:
                status = OrderStatus.PROCESSING
            elif days_ago <= 6:
                status = OrderStatus.SHIPPED
            else:
                status = rng.choices(
                    [OrderStatus.DELIVERED, OrderStatus.RETURNED, OrderStatus.CANCELLED],
                    weights=[85, 8, 7])[0]
            customer = rng.choice(customers)
            if order_id == 1025:
                customer, status, days_ago = customers[6], OrderStatus.SHIPPED, 3
            orders.append(build_order(rng, order_id, customer, products, status, days_ago))
        db.add_all(orders)

        db.add_all([
            AppUser(username="mila", role=Role.CUSTOMER, customer_id=7),    # owns #1025
            AppUser(username="lucas", role=Role.CUSTOMER, customer_id=2),   # does NOT own #1025
            AppUser(username="sara", role=Role.SUPPORT_AGENT),
            AppUser(username="admin", role=Role.ADMIN),
        ])
        db.flush()
        for user in db.query(AppUser):
            user.password_hash = hash_password(DEMO_PASSWORDS[user.username])
        db.commit()

        print(f"Seeded {len(products)} products, {len(customers)} customers, "
              f"{len(orders)} orders, 4 app users.")


if __name__ == "__main__":
    seed()
