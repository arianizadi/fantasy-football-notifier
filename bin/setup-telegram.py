#!/usr/bin/env python3
"""Resolve TELEGRAM_CHAT_ID from getUpdates and write it into .env.

Message the bot once from your phone first; Telegram only exposes a chat id
after the user has initiated a conversation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in .env")
        return 1

    response = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20)
    response.raise_for_status()
    chats: dict[int, str] = {}
    for update in response.json().get("result", []):
        message = update.get("message") or update.get("channel_post")
        if message:
            chat = message["chat"]
            chats[chat["id"]] = chat.get("first_name") or chat.get("title") or ""

    if not chats:
        print("No chats found. Open Telegram, message your bot, then re-run this.")
        return 1

    chat_id = next(iter(chats))
    if len(chats) > 1:
        print(f"Multiple chats found: {chats}. Using {chat_id}.")

    env_path = ROOT / ".env"
    lines = [
        line
        for line in env_path.read_text().splitlines()
        if not line.startswith("TELEGRAM_CHAT_ID=")
    ]
    lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
    env_path.write_text("\n".join(lines) + "\n")
    env_path.chmod(0o600)
    print(f"Wrote TELEGRAM_CHAT_ID={chat_id} ({chats[chat_id]}) to .env")

    send = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        timeout=20,
        json={
            "chat_id": chat_id,
            "text": "Fantasy notifier connected. You will get alerts here.",
        },
    )
    print("test message sent:", send.ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
