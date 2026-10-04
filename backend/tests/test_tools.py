"""Operations tools: role and ownership checks, input validation, calculator safety."""
import time

import pytest

from app.db.models import Role
from app.tools import (CallerContext, InvalidInput, NotFound, PermissionDenied, calculate,
                       get_order_status, get_product_price, lookup_customer, search_products)


# --- get_order_status: the brief's authorization case ---------------------------------

def test_owner_sees_own_order(db, caller):
    order = get_order_status(db, caller("mila"), 1025)
    assert order.status == "shipped"
    assert order.tracking_number
    assert order.customer_name is None   # customers don't get the staff-only field


def test_lucas_cannot_see_order_1025(db, caller):
    with pytest.raises(NotFound):
        get_order_status(db, caller("lucas"), 1025)


def test_foreign_order_looks_exactly_like_missing_order(db, caller):
    """lucas must not learn whether #1025 exists at all."""
    lucas = caller("lucas")
    with pytest.raises(NotFound) as foreign:
        get_order_status(db, lucas, 1025)
    with pytest.raises(NotFound) as missing:
        get_order_status(db, lucas, 9999)
    assert str(foreign.value).replace("1025", "N") == str(missing.value).replace("9999", "N")


@pytest.mark.parametrize("username", ["sara", "admin"])
def test_staff_see_any_order(db, caller, username):
    order = get_order_status(db, caller(username), 1025)
    assert order.status == "shipped"
    assert order.customer_name == "Mila de Boer"


def test_identity_cannot_be_claimed_through_arguments(db):
    """A customer context built with lucas's customer_id stays lucas, whatever the chat says."""
    lucas = CallerContext(username="lucas", role=Role.CUSTOMER, customer_id=2)
    with pytest.raises(NotFound):
        get_order_status(db, lucas, "#1025")


def test_customer_without_customer_record_is_rejected():
    with pytest.raises(ValueError):
        CallerContext(username="ghost", role=Role.CUSTOMER, customer_id=None)


@pytest.mark.parametrize("raw", [1025, "1025", "#1025", " # 1025 "])
def test_order_id_formats(db, caller, raw):
    assert get_order_status(db, caller("mila"), raw).order_id == 1025


@pytest.mark.parametrize("raw", ["abc", "-5", "0", "1025; DROP TABLE orders", "", "1e3"])
def test_invalid_order_ids(db, caller, raw):
    with pytest.raises(InvalidInput):
        get_order_status(db, caller("sara"), raw)


# --- lookup_customer -------------------------------------------------------------------

@pytest.mark.parametrize("username", ["mila", "lucas"])
def test_customers_cannot_look_up_customers(db, caller, username):
    with pytest.raises(PermissionDenied):
        lookup_customer(db, caller(username), "Mila")


def test_partial_name_returns_every_match(db, caller):
    names = {c.name for c in lookup_customer(db, caller("sara"), "mila")}
    assert names == {"Mila de Boer", "Milan van Dijk"}


def test_staff_lookup_masks_pii(db, caller):
    [mila] = lookup_customer(db, caller("sara"), "Mila de Boer")
    dumped = mila.model_dump_json()
    assert mila.customer_id == 7
    assert mila.email_masked == "m***@example.com"
    assert "mila.deboer" not in dumped             # full email never leaves the tool
    assert mila.phone_masked.startswith("*** *** ")
    assert "postcode" not in dumped and "2311MN" not in dumped
    assert 1025 in [o.order_id for o in mila.recent_orders]


def test_lookup_by_id_and_exact_email(db, caller):
    sara = caller("sara")
    assert lookup_customer(db, sara, "7")[0].name == "Mila de Boer"
    assert lookup_customer(db, sara, "Mila.DeBoer@example.com")[0].customer_id == 7


def test_partial_email_does_not_match(db, caller):
    with pytest.raises(NotFound):
        lookup_customer(db, caller("sara"), "@example.com")


def test_like_wildcards_are_literal(db, caller):
    with pytest.raises(NotFound):
        lookup_customer(db, caller("admin"), "%")


# --- products ---------------------------------------------------------------------------

def test_exact_sku_price(db):
    [p] = get_product_price(db, "np-x-128")
    assert (p.name, p.price_eur, p.in_stock) == ("NovaPhone X 128GB", 899.00, True)


def test_ambiguous_name_returns_all_variants(db):
    skus = {p.sku for p in get_product_price(db, "NovaPhone X")}
    assert {"NP-X-128", "NP-X-256"} <= skus


def test_unknown_product_is_not_found(db):
    with pytest.raises(NotFound):
        get_product_price(db, "toaster")


def test_out_of_stock_product(db):
    [p] = get_product_price(db, "NB-AIR-13")
    assert p.in_stock is False


def test_search_matches_words_in_any_order(db):
    assert [p.sku for p in search_products(db, "novaphone case")] == ["AC-CASE-NPX"]


def test_search_with_category_and_budget(db):
    results = search_products(db, "soundwave", category="audio", max_price=200)
    assert [p.sku for p in results] == ["SW-BUDS-02"]


def test_search_percent_is_not_a_wildcard(db):
    assert search_products(db, "%") == []


@pytest.mark.parametrize("bad", ["", "   ", "x" * 101])
def test_search_rejects_bad_queries(db, bad):
    with pytest.raises(InvalidInput):
        search_products(db, bad)


# --- calculate --------------------------------------------------------------------------

@pytest.mark.parametrize("expr, expected", [
    ("2 * 899 * 0.9", 1618.2),
    ("0.1 + 0.2", 0.3),
    ("round(1499 / 3, 2)", 499.67),
    ("max(899, 999) - min(449, 549)", 550),
    ("-(2 ** 10) % 7", 5),
])
def test_calculate(expr, expected):
    assert calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('dir')",   # code execution
    "open('.env').read()",              # file access
    "(1).__class__",                    # attribute access
    "True + 1",                         # bools are not numbers here
    "'a' * 3",                          # strings
    "x + 1",                            # names
    "1 / 0",
    "(-8) ** 0.5",                      # complex result
    "10 ** 400",                        # float overflow
    "2 ** 1000",                        # exponent limit
    "1 +",
])
def test_calculate_rejects(expr):
    with pytest.raises(InvalidInput):
        calculate(expr)


def test_power_tower_is_rejected_quickly():
    """10**10**10 would hang an eval-based calculator. Here it fails in microseconds."""
    start = time.perf_counter()
    with pytest.raises(InvalidInput):
        calculate("10 ** 10 ** 10")
    assert time.perf_counter() - start < 0.1
