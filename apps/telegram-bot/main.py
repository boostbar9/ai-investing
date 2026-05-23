"""Telegram approvals bot.

Phase 0 stub: responds to /ping. In Phase 4, this owns the human-approval signal
flowing back into the LangGraph/Temporal workflow.
"""
from __future__ import annotations

import os


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM_BOT_TOKEN not set — skipping bot start (Phase 0).")
        return
    # Real implementation in Phase 4.
    print("Telegram bot would start here.")


if __name__ == "__main__":
    main()
