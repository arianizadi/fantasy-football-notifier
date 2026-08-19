"""Retry-safe expiry of Telegram messages created by this bot."""

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
    """Delete due messages while retaining transient failures for a later retry."""
    deletable, too_old = store.due(ttl_seconds)
    deleted = [message_id for message_id in deletable if delete(message_id)]
    failed = [message_id for message_id in deletable if message_id not in deleted]

    # Successful deletions and messages beyond Telegram's deletion window are
    # terminal. Failed requests remain tracked so the next cron run retries them.
    store.forget(deleted + too_old)
    store.save()
    return ExpiryResult(deleted=deleted, failed=failed, too_old=too_old)
