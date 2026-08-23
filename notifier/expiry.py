"""Disabled legacy expiry; Telegram's native chat TTL owns retention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .history import MessageHistory


@dataclass(frozen=True)
class ExpiryResult:
    deleted: list[int]
    failed: list[int]
    too_old: list[int]


def expire_due_messages(
    store: MessageHistory,
    ttl_seconds: int,
    delete: Callable[[int], bool],
) -> ExpiryResult:
    """No-op compatibility shim.

    A bot-side timer races Telegram's chat-level seven-day timer and cannot
    implement seven days anyway because the Bot API deletion window is only 48
    hours. Keeping this callable but inert makes an old cron deployment safe.
    """
    del store, ttl_seconds, delete
    return ExpiryResult(deleted=[], failed=[], too_old=[])
