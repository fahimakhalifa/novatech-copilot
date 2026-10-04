"""Input guard: validate the message, then classify it as blocked | smalltalk | task."""
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from app.config import get_settings

log = logging.getLogger(__name__)

MAX_INPUT_CHARS = 2000
BLOCK_THRESHOLD = 0.9

Label = Literal["blocked", "smalltalk", "task", "invalid"]


@dataclass(frozen=True)
class GuardResult:
    label: Label
    p_blocked: float
    reason: str = ""


def clean_input(text: str) -> str:
    """Normalise Unicode and drop control characters (but keep newlines and tabs)."""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()


@lru_cache
def _classifier():
    """Load the adapter once per process; None if it isn't installed."""
    model_dir = get_settings().guard_model_dir
    if not (model_dir / "adapter_config.json").exists():
        log.warning("guard classifier not found at %s: every message is treated as 'task' "
                    "(the permission gate, ingest quarantine and output guard still apply)", model_dir)
        return None
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = json.loads((model_dir / "guard_meta.json").read_text())
    base = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=len(meta["labels"]), dtype=torch.float32)
    model = PeftModel.from_pretrained(base, model_dir).eval()
    return AutoTokenizer.from_pretrained(model_dir), model, meta


def probabilities(text: str) -> dict[str, float] | None:
    loaded = _classifier()
    if loaded is None:
        return None
    import torch
    tokenizer, model, meta = loaded
    with torch.no_grad():
        enc = tokenizer(text, truncation=True, max_length=meta["max_len"], return_tensors="pt")
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    return dict(zip(meta["labels"], probs))


def decide(probs: dict[str, float] | None) -> GuardResult:
    """Turn class probabilities into a decision (pure, unit-tested)."""
    if probs is None:
        return GuardResult("task", 0.0, "classifier unavailable")
    p_blocked = probs["blocked"]
    if p_blocked >= BLOCK_THRESHOLD:
        return GuardResult("blocked", p_blocked, "classifier")
    label = "smalltalk" if probs["smalltalk"] > probs["task"] else "task"
    return GuardResult(label, p_blocked)


def check(text: str) -> GuardResult:
    if not text:
        return GuardResult("invalid", 0.0, "empty message")
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult("invalid", 0.0, f"message longer than {MAX_INPUT_CHARS} characters")
    return decide(probabilities(text))


# --- Canned replies (no LLM) -------------------------------------------------------------

BLOCKED_REPLY = ("I can't help with that request. I can help with your orders, NovaTech "
                 "products and prices, and our policies.")

_THANKS = re.compile(r"\b(thanks?|thank you|thx|cheers|bedankt|dank je|danke)\b", re.I)
_BYE = re.compile(r"\b(bye|goodbye|see you|doei|tot ziens|tschüss)\b", re.I)


_GREETING = re.compile(r"^\W*(hi|hello|hey|hiya|hallo|hoi|good (morning|afternoon|evening))\b", re.I)
_CAPABILITIES = re.compile(r"\bwhat (can|do) you (do|help)|\bhow can you help\b|\bwho are you\b", re.I)
_SHORT = 60   # a canned reply only fits short, purely social messages


def smalltalk_reply(text: str, staff: bool) -> str | None:
    """A canned reply for purely social messages, or None if the reply needs the conversation (e.g. "what did I ask you before?"), which the smalltalk node handles."""
    if len(text) > _SHORT:
        return None
    if _THANKS.search(text):
        return "You're welcome! Is there anything else I can help you with?"
    if _BYE.search(text):
        return "Goodbye! Come back any time."
    if _GREETING.search(text) or _CAPABILITIES.search(text):
        topics = ["orders and deliveries", "products, prices and stock",
                  "returns, shipping and warranty", "NovaPlus membership"]
        if staff:
            topics += ["customer lookups", "internal policies such as HR rules"]
        return (f"Hi! I'm NovaTech Copilot. I can help with {'; '.join(topics[:-1])}; "
                f"and {topics[-1]}. What would you like to know?")
    return None
