"""Legacy message-id history retained for state-file compatibility.

Telegram's native chat auto-delete timer now owns retention. The custom bot
expiry path is intentionally disabled in :mod:`notifier.expiry`; this reader
remains so an older state file does not break an upgrade or rollback.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from .logging_utils import structured_log

TELEGRAM_DELETE_WINDOW_SECONDS = 48 * 60 * 60
MAX_TRACKED = 5000


class MessageHistory:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._sent: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            self._sent = {
                str(k): float(v) for k, v in json.loads(self._path.read_text()).items()
            }
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            structured_log(logging.WARNING, "history.unreadable", error=str(error))

    def save(self) -> None:
        try:
            temporary = self._path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self._sent, separators=(",", ":")))
            os.replace(temporary, self._path)
        except OSError as error:
            structured_log(logging.WARNING, "history.write_failed", error=str(error))

    def record(self, message_id: int) -> None:
        if message_id and message_id > 0:
            self._sent[str(message_id)] = time.time()
            if len(self._sent) > MAX_TRACKED:
                for key in sorted(self._sent, key=self._sent.get)[: len(self._sent) - MAX_TRACKED]:
                    del self._sent[key]

    def due(self, ttl_seconds: int) -> tuple[list[int], list[int]]:
        """Return (deletable, expired_past_window)."""
        now = time.time()
        deletable, too_old = [], []
        for raw, sent_at in self._sent.items():
            age = now - sent_at
            if age < ttl_seconds:
                continue
            (too_old if age >= TELEGRAM_DELETE_WINDOW_SECONDS else deletable).append(int(raw))
        return sorted(deletable), sorted(too_old)

    def forget(self, message_ids: list[int]) -> None:
        for message_id in message_ids:
            self._sent.pop(str(message_id), None)

    def __len__(self) -> int:
        return len(self._sent)
