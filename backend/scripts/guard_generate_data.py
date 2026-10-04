"""Generate synthetic training data for the guard classifier (blocked | smalltalk | task)."""
import json
import random
import re
import sys
import time

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from app.config import BACKEND_DIR, get_settings

OUT = BACKEND_DIR / "data" / "guard" / "synthetic.jsonl"
TEST = BACKEND_DIR / "data" / "guard" / "test_handwritten.jsonl"
EVAL_CASES = BACKEND_DIR.parent / "eval" / "cases.jsonl"   # never train on near-copies of these either
PER_CALL = 25
STYLES = [
    "short and casual, like chat messages: lowercase, some typos, missing punctuation",
    "complete sentences, polite and detailed, sometimes two or three sentences long",
]

CONTEXT = """NovaTech is an online electronics retailer in the Netherlands. Products: NovaPhone X
(128/256GB), NovaPhone Lite, NovaBook Pro 14, NovaBook Air 13, NovaTab 11, SoundWave ANC
headphones, SoundWave Buds 2, chargers, cases, cables. Its AI assistant answers questions about
orders (e.g. order #1025), products, prices, stock, returns, shipping, warranty, NovaPlus
membership, and, for employees, HR policies (vacation, expenses, remote work)."""

# (label, sub-category, what to generate)
CATEGORIES = [
    ("task", "order_status", "customers asking about the status, delivery or tracking of their order, usually with an order number"),
    ("task", "product_info", "questions about product features, specs and comparisons"),
    ("task", "price_stock", "questions about prices, discounts and whether products are in stock"),
    ("task", "policy", "questions about returns, refunds, shipping, warranty, NovaCare, NovaPlus, payment methods, stores, price matching"),
    ("task", "hr_internal", "NovaTech employees asking about vacation, sick leave, expenses, remote work or data handling rules"),
    ("task", "staff_ops", "support agents asking to look up a customer by name or email, or to check a customer's orders"),
    ("task", "calculation", "requests to calculate totals, discounts, instalments or price differences for products"),
    ("task", "follow_up", "short follow-up messages that only make sense mid-conversation, e.g. 'and the 256GB one?' or 'what about to Belgium?'"),
    ("task", "hard_negative", "legitimate NovaTech questions that happen to contain words like ignore, override, system, admin, developer mode, forget, previous, instructions, act as, prompt, reset, bypass, unlock; they must be clearly harmless shopping or support questions"),
    ("task", "non_english", "legitimate NovaTech questions written in Dutch or German"),
    ("smalltalk", "social", "greetings, goodbyes, thanks and acknowledgements with no request in them"),
    ("smalltalk", "about_assistant", "questions about the assistant itself: what it can do, whether it is human, who made it, how it works (without trying to extract its instructions)"),
    ("smalltalk", "off_topic", "off-topic chat unrelated to NovaTech: weather, jokes, poems, sports, recipes, trivia, homework"),
    ("smalltalk", "non_english", "greetings, thanks and small talk written in Dutch or German"),
    ("blocked", "instruction_override", "attempts to make the assistant ignore, forget or override its instructions or rules"),
    ("blocked", "jailbreak_roleplay", "jailbreaks using role-play, personas (DAN etc.), games, hypotheticals or emotional stories to remove restrictions"),
    ("blocked", "prompt_extraction", "attempts to make the assistant reveal its system prompt, hidden instructions, configuration or internal rules"),
    ("blocked", "fake_authority", "claims of being an admin, developer, CEO or support agent to gain more access, or requests to change the user's role"),
    ("blocked", "data_exfiltration", "requests to dump or list many customers' personal data, the customer database, passwords, API keys or internal documents"),
    ("blocked", "in_domain_manipulation", "attempts to trick the assistant into free products, unauthorized refunds, changing prices or orders, or skipping checks"),
    ("blocked", "obfuscated", "injection attempts hidden with encodings (base64), leetspeak, split words, fake XML/system tags, or wrapped in translation or summarization requests"),
    ("blocked", "non_english", "instruction-override and data-theft attempts written in Dutch, German, French or Spanish"),
    ("task", "contact_lookup", "support agents asking to look up ONE specific named customer and see their contact details (email, phone, address) or account, often saying why, e.g. the customer called about a delivery or a return"),
    ("blocked", "bulk_contact_dump", "requests to list, export or dump the contact details (emails, phone numbers, addresses) of many or all customers at once"),
]


class Batch(BaseModel):
    messages: list[str]


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def too_similar(a: str, b: str) -> bool:
    """Token Jaccard similarity above 0.8, i.e. a near-copy of a test example."""
    ta, tb = set(norm(a).split()), set(norm(b).split())
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) > 0.8


def _save(rows: list[dict]) -> None:
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    llm = init_chat_model("google_genai:gemini-3.1-flash-lite",
                          api_key=get_settings().google_api_key, temperature=1.0, max_retries=6)
    generate = llm.with_structured_output(Batch)
    test_texts = [json.loads(line)["text"] for line in TEST.read_text(encoding="utf-8").splitlines()]
    test_texts += [turn for line in EVAL_CASES.read_text(encoding="utf-8").splitlines() if line.strip()
                   for turn in json.loads(line)["turns"]]

    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()] if OUT.exists() else []
    done = {(r["source"], r["style"]) for r in rows}
    seen, dropped_leak = {norm(r["text"]) for r in rows}, 0

    for label, sub, what in CATEGORIES:
        for style_id, style in enumerate(STYLES):
            if (f"synthetic/{sub}", style_id) in done:
                continue
            prompt = (f"{CONTEXT}\n\nWrite {PER_CALL} different messages that a user might send to this "
                      f"assistant. Category: {what}.\nStyle: {style}.\nMake them varied in wording, "
                      f"length and topic. Do not number them.")
            batch = generate.invoke(prompt)
            kept = 0
            for msg in batch.messages:
                msg = msg.strip()
                if not msg or norm(msg) in seen:
                    continue
                if any(too_similar(msg, t) for t in test_texts):
                    dropped_leak += 1
                    continue
                seen.add(norm(msg))
                rows.append({"text": msg, "label": label, "source": f"synthetic/{sub}", "style": style_id})
                kept += 1
            print(f"{label:9} {sub:22} style={style_id} +{kept}")
            _save(rows)
            time.sleep(4)   # 15 requests/minute free limit

    random.Random(42).shuffle(rows)
    _save(rows)
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in ("task", "smalltalk", "blocked")}
    print(f"\nWrote {len(rows)} examples to {OUT.name}: {counts}. "
          f"Dropped {dropped_leak} near-duplicates of test examples in this run.")


if __name__ == "__main__":
    main()
