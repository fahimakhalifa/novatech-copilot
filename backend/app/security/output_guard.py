"""Output guard: the last check before a reply reaches the user. Code only, no LLM."""
import logging
import re
from dataclasses import dataclass, field

from app.rag.scanner import scan

log = logging.getLogger(__name__)

SAFE_REPLY = ("Sorry, I can't show that answer. Please rephrase your question, or contact "
              "Customer Support at support@novatech.example.")

PUBLIC_EMAIL_DOMAINS = {"novatech.example"}
PUBLIC_PHONES = {"+31205550100"}   # Customer Support (FAQ), digits only

# The local part can't contain "*", so masked addresses such as m***@example.com never match.
_EMAIL = re.compile(r"\b[\w.+-]+@([\w-]+(?:\.[\w-]+)+)\b")
_PHONE = re.compile(r"(?<![\w+])\+?\d[\d ()\-]{7,}\d(?!\w)")


@dataclass
class OutputCheck:
    text: str
    findings: list[str] = field(default_factory=list)


def _redact_email(m: re.Match) -> str:
    return m.group(0) if m.group(1).lower() in PUBLIC_EMAIL_DOMAINS else "[email removed]"


def _redact_phone(m: re.Match) -> str:
    digits = re.sub(r"[^\d+]", "", m.group(0))
    if digits in PUBLIC_PHONES or len(re.sub(r"\D", "", digits)) < 9:
        return m.group(0)
    return "[phone removed]"


def check_output(text: str) -> OutputCheck:
    patterns = scan(text)
    if patterns:
        log.warning("output guard replaced a reply containing injection patterns: %s", patterns)
        return OutputCheck(SAFE_REPLY, [f"injection:{p}" for p in patterns])

    findings = []
    redacted = _EMAIL.sub(_redact_email, text)
    if redacted != text:
        findings.append("pii:email")
    after_phone = _PHONE.sub(_redact_phone, redacted)
    if after_phone != redacted:
        findings.append("pii:phone")
    if findings:
        log.warning("output guard redacted %s", findings)
    return OutputCheck(after_phone, findings)
