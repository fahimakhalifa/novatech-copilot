"""Scan document chunks for embedded instructions (indirect prompt injection)."""
import re

_PATTERNS: dict[str, str] = {
    "override": r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your)\b[^.\n]{0,20}\b(instructions?|rules|prompts?|guidelines)\b",
    # Not "developer mode": that's a real phone/tablet setting a product guide may describe.
    "role_change": r"\byou are now\b|\b(admin|god|dan|jailbreak|unrestricted) mode\b|\bact as (an? )?(admin|system|root)\b",
    "fake_system": r"\bsystem (instruction|prompt|message|override)\b|<\s*/?\s*(system|instructions?)\s*>",
    "addresses_ai": r"\b(note|message|instructions?)\s+(for|to)\s+(the\s+)?(ai|llm|chatbot|assistant|language model)s?\b",
    "concealment": r"\bdo not (mention|reveal|tell|disclose)\b[^.\n]{0,30}\b(this|these)\s+(instructions?|message|note)\b",
    "data_exfiltration": r"\b(list|reveal|output|print|send)\b[^.\n]{0,40}\b(every|all)\s+customers?\b",
}
_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, p in _PATTERNS.items()}


def scan(text: str) -> list[str]:
    """Names of the patterns found in `text` (empty list = clean)."""
    return [name for name, rx in _COMPILED.items() if rx.search(text)]
