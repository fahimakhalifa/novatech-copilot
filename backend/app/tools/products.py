"""Product tools: catalogue search and price lookup. Open to every role."""
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.db.models import Product
from app.tools.context import InvalidInput, NotFound

MAX_QUERY_LEN = 100
MAX_RESULTS = 5


class ProductInfo(BaseModel):
    sku: str
    name: str
    category: str
    price_eur: float
    in_stock: bool   # stock counts are internal; everyone sees only availability

    @classmethod
    def of(cls, p: Product) -> "ProductInfo":
        return cls(sku=p.sku, name=p.name, category=p.category,
                   price_eur=float(p.price), in_stock=p.stock > 0)


def _clean_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise InvalidInput("Please give a product name, SKU or category.")
    if len(query) > MAX_QUERY_LEN:
        raise InvalidInput(f"Search text is limited to {MAX_QUERY_LEN} characters.")
    return query


def like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _match_all_words(query: str):
    """Every word must appear in the name, SKU or category (in any order)."""
    return [
        or_(Product.name.ilike(like_pattern(w), escape="\\"),
            Product.sku.ilike(like_pattern(w), escape="\\"),
            Product.category.ilike(like_pattern(w), escape="\\"))
        for w in query.split()
    ]


def search_products(db: Session, query: str, category: str | None = None,
                    max_price: float | None = None) -> list[ProductInfo]:
    """Browse the catalogue, optionally within a category and price ceiling."""
    query = _clean_query(query)
    if max_price is not None and max_price <= 0:
        raise InvalidInput("max_price must be a positive amount in EUR.")

    stmt = select(Product).where(*_match_all_words(query))
    if category:
        stmt = stmt.where(func.lower(Product.category) == category.strip().lower())
    if max_price is not None:
        stmt = stmt.where(Product.price <= max_price)
    products = db.scalars(stmt.order_by(Product.price).limit(MAX_RESULTS)).all()
    return [ProductInfo.of(p) for p in products]


def get_product_price(db: Session, product: str) -> list[ProductInfo]:
    """Price of one product, by SKU or name."""
    product = _clean_query(product)
    exact = db.scalar(select(Product).where(func.upper(Product.sku) == product.upper()))
    if exact:
        return [ProductInfo.of(exact)]

    matches = db.scalars(select(Product).where(*_match_all_words(product))
                         .order_by(Product.price).limit(MAX_RESULTS)).all()
    if not matches:
        raise NotFound(f"No product matching '{product}' was found in the catalogue.")
    return [ProductInfo.of(p) for p in matches]
