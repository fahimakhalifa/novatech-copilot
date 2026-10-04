"""NovaTech Copilot on Telegram: a thin client of the API."""
import logging
import re

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # it logs every URL, including the bot token
log = logging.getLogger("telegram_bot")

HELP = ("Hi! I'm NovaTech Copilot. To use me here, link your NovaTech account first:\n"
        "1. Sign in on the NovaTech Copilot website and click \"Connect Telegram\".\n"
        "2. Send me /link followed by the code, e.g. /link A1B2C3D4\n\n"
        "Then just ask about your orders, products and prices, or our policies.\n"
        "/reset starts a new conversation, /unlink disconnects your account.")
NOT_LINKED = "Your Telegram account isn't linked yet.\n\n" + HELP


def api() -> httpx.AsyncClient:
    s = get_settings()
    return httpx.AsyncClient(base_url=s.api_url, headers={"X-Bot-Secret": s.bot_secret}, timeout=120)


def to_plain(markdown: str) -> str:
    """Telegram's Markdown parser rejects unbalanced symbols, so answers go out as plain text: bold markers are dropped and "* item" bullets become "• item"."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", markdown)
    return re.sub(r"^\s*[*-]\s+", "• ", text, flags=re.MULTILINE)


def format_answer(data: dict) -> str:
    parts = [to_plain(data["answer"])]
    if data.get("citations"):
        parts.append("Sources:\n" + "\n".join(f"[{c['id']}] {c['title']} › {c['section']}" for c in data["citations"]))
    return "\n\n".join(parts)


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Send /link followed by the code from the website, e.g. /link A1B2C3D4")
        return
    async with api() as client:
        r = await client.post("/api/telegram/link", json={"code": context.args[0],
                                                          "telegram_user_id": update.effective_user.id})
    if r.status_code == 200:
        await update.message.reply_text(f"Linked to NovaTech account '{r.json()['username']}'. Ask me anything!")
    else:
        await update.message.reply_text(r.json().get("detail", "Linking failed."))


async def reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    async with api() as client:
        r = await client.post("/api/telegram/reset", json={"telegram_user_id": update.effective_user.id,
                                                           "chat_id": update.effective_chat.id})
    await update.message.reply_text("Started a new conversation." if r.status_code == 200 else NOT_LINKED)


async def unlink(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    async with api() as client:
        await client.post("/api/telegram/unlink", json={"telegram_user_id": update.effective_user.id})
    await update.message.reply_text("Your NovaTech account is no longer linked to this Telegram account.")


async def message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    async with api() as client:
        r = await client.post("/api/telegram/chat", json={"telegram_user_id": update.effective_user.id,
                                                          "chat_id": update.effective_chat.id,
                                                          "message": update.message.text})
    if r.status_code == 200:
        reply = format_answer(r.json())
    elif r.status_code == 403:
        reply = NOT_LINKED
    else:
        reply = r.json().get("detail", "Something went wrong. Please try again.")
    # Telegram's limit is 4096 characters per message.
    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i:i + 4000])


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Without a handler, a failure is only a traceback in the log and the user gets no reply."""
    log.warning("update failed: %s: %s", type(context.error).__name__, context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "Sorry, I couldn't reach the assistant just now. Please try again in a moment.")


def main() -> None:
    s = get_settings()
    if not s.telegram_bot_token or not s.bot_secret:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and BOT_SECRET in backend/.env first.")
    app = Application.builder().token(s.telegram_bot_token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("link", link))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("unlink", unlink))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message))
    app.add_error_handler(on_error)
    log.info("bot running (API at %s)", s.api_url)
    app.run_polling()


if __name__ == "__main__":
    main()
