"""Telegram approvals bot (§10 + §12 + §20 + §21).

Commands:
- /start    -> introduce, link decision-trace base URL
- /ping     -> latency check
- /health   -> hits API /health
- /pending  -> list pending approvals; one-tap approve/deny inline keyboard

The bot is intentionally thin; the heavy lifting (Pydantic validation, audit
write, LangGraph signalling) happens server-side in apps/api.
"""
from __future__ import annotations

import os

import httpx

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_OPERATOR_CHAT_ID", "")


async def _api_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            return (await c.get(f"{API_BASE}/health")).status_code == 200
    except Exception:
        return False


async def _list_pending() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{API_BASE}/approvals/pending")
            return r.json().get("pending", [])
    except Exception:
        return []


def main() -> None:
    if not TOKEN:
        print("TELEGRAM_BOT_TOKEN not set — skipping bot start.")
        return

    try:
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
        from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
    except ImportError:
        print("python-telegram-bot not installed; run `make setup` first.")
        return

    async def start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "ai-investing online. /ping /health /pending"
        )

    async def ping(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("pong")

    async def health(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        ok = await _api_health()
        await update.message.reply_text("API OK" if ok else "API DOWN")

    async def pending(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        items = await _list_pending()
        if not items:
            await update.message.reply_text("no pending approvals")
            return
        for item in items:
            kb = InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton("✅ approve", callback_data=f"approve:{item['decision_id']}"),
                    InlineKeyboardButton("⛔ deny",    callback_data=f"deny:{item['decision_id']}"),
                ]]
            )
            txt = (
                f"{item.get('side','?').upper()} {item.get('qty','?')} {item.get('symbol','?')}\n"
                f"{item.get('thesis','')}"
            )
            await update.message.reply_text(txt, reply_markup=kb)

    async def on_callback(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        await q.answer()
        action, did = q.data.split(":", 1)
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(f"{API_BASE}/approvals/{did}", json={"approve": action == "approve"})
        await q.edit_message_text(f"{action}d {did}")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("pending", pending))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.run_polling()


if __name__ == "__main__":
    main()
