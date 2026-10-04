"""What the LLM may call: tool schemas, role policy, and adapters to the business functions."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.db.models import Role
from app.tools import (CallerContext, calculate, get_order_status, get_product_price,
                       lookup_customer, search_products)

ALL_ROLES = frozenset(Role)
STAFF = frozenset({Role.SUPPORT_AGENT, Role.ADMIN})


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchProductsArgs(ToolArgs):
    query: str = Field(max_length=100, description="Words to search for, e.g. 'headphones' or 'novaphone case'.")
    # An enum in the schema means the model picks a real category instead of inventing one.
    category: Literal["smartphones", "laptops", "tablets", "audio", "accessories"] | None = None
    max_price: float | None = Field(default=None, gt=0, description="Upper price limit in EUR.")


class GetProductPriceArgs(ToolArgs):
    product: str = Field(max_length=100, description="SKU (e.g. 'NP-X-128') or product name.")


class GetOrderStatusArgs(ToolArgs):
    order_id: int = Field(gt=0, description="The order number, e.g. 1025 for order #1025.")


class LookupCustomerArgs(ToolArgs):
    query: str = Field(max_length=100, description="Customer id, exact email address, or name.")


class CalculateArgs(ToolArgs):
    expression: str = Field(max_length=200, description="Arithmetic only, e.g. '2 * 899 * 0.9'.")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    roles: frozenset[Role]
    run: Callable[[Session, CallerContext, Any], Any]

    def schema(self) -> dict:
        """OpenAI-style function schema, which LangChain converts for each provider."""
        params = self.args.model_json_schema()
        params.pop("additionalProperties", None)
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": params}}


TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("search_products",
             "Search the product catalogue. Returns name, SKU, category, price in EUR and availability.",
             SearchProductsArgs, ALL_ROLES,
             lambda db, caller, a: search_products(db, a.query, a.category, a.max_price)),
    ToolSpec("get_product_price",
             "Current price and availability of a product, by SKU or name. May return several variants.",
             GetProductPriceArgs, ALL_ROLES,
             lambda db, caller, a: get_product_price(db, a.product)),
    ToolSpec("get_order_status",
             "Status, dates, tracking number, items and total of an order.",
             GetOrderStatusArgs, ALL_ROLES,
             lambda db, caller, a: get_order_status(db, caller, a.order_id)),
    ToolSpec("lookup_customer",
             "Find a customer record (contact details masked) and their recent orders. Staff only.",
             LookupCustomerArgs, STAFF,
             lambda db, caller, a: lookup_customer(db, caller, a.query)),
    ToolSpec("calculate",
             "Evaluate an arithmetic expression. Use it for every calculation.",
             CalculateArgs, ALL_ROLES,
             lambda db, caller, a: {"result": calculate(a.expression)}),
]}


def tools_for(role: Role) -> list[ToolSpec]:
    """Least privilege: the model is only shown the tools this role may use."""
    return [t for t in TOOLS.values() if role in t.roles]


def to_jsonable(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, list):
        return [to_jsonable(r) for r in result]
    return result
