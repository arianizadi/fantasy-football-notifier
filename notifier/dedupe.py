"""Suppress repeat alerts.

Two layers:
  1. GUID - the same feed item seen on a later poll.
  2. Content fingerprint - the same event reported again with different
     wording within a time window (a practice report followed by a beat
     writer's confirmation of the same thing).

State persists to disk so a restart mid-Sunday does not replay old news.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

from .logging_utils import structured_log
from .models import NewsItem

DEFAULT_WINDOW_SECONDS = 30 * 60
# A tweet and the RotoWire write-up of the same event are worded completely
# differently, so text similarity cannot catch the duplicate. After
# classification we know (player, event_type), which is stable across sources.
SEMANTIC_WINDOW_SECONDS = 90 * 60
MAX_TRACKED_ENTRIES = 5000
STOPWORDS = frozenset({"the", "a", "an", "is", "was", "for", "of", "to", "in", "on", "at", "with"})


def fingerprint(item: NewsItem) -> str:
    """Hash the meaningful words of a headline so rewordings collide."""
    text = re.sub(r"[^a-z0-9 ]", " ", item.fingerprint_text().lower())
    tokens = sorted({token for token in text.split() if token and token not in STOPWORDS})
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:32]


class SeenStore:
    def __init__(self, path: Path, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> None:
        self._path = path
        self._window = window_seconds
        self._guids: dict[str, float] = {}
        self._fingerprints: dict[str, float] = {}
        self._semantic: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text())
            self._guids = {str(k): float(v) for k, v in payload.get("guids", {}).items()}
            self._fingerprints = {
                str(k): float(v) for k, v in payload.get("fingerprints", {}).items()
            }
            self._semantic = {str(k): float(v) for k, v in payload.get("semantic", {}).items()}
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            structured_log(logging.WARNING, "dedupe.state_unreadable", error=str(error))

    def save(self) -> None:
        self._prune()
        payload = {
            "guids": self._guids,
            "fingerprints": self._fingerprints,
            "semantic": self._semantic,
        }
        try:
            temporary = self._path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(temporary, self._path)
        except OSError as error:
            structured_log(logging.WARNING, "dedupe.state_write_failed", error=str(error))

    def _prune(self) -> None:
        now = time.time()
        # GUIDs are kept far longer than the fingerprint window: a feed item
        # can reappear hours later and must not re-alert.
        guid_ttl = max(self._window, 24 * 60 * 60)
        self._guids = {k: v for k, v in self._guids.items() if now - v < guid_ttl}
        self._fingerprints = {k: v for k, v in self._fingerprints.items() if now - v < self._window}
        self._semantic = {
            k: v for k, v in self._semantic.items() if now - v < SEMANTIC_WINDOW_SECONDS
        }

        for store in (self._guids, self._fingerprints, self._semantic):
            if len(store) > MAX_TRACKED_ENTRIES:
                for key in sorted(store, key=store.get)[: len(store) - MAX_TRACKED_ENTRIES]:
                    del store[key]

    def is_new(self, item: NewsItem) -> bool:
        now = time.time()
        if item.guid in self._guids:
            return False
        digest = fingerprint(item)
        recent = self._fingerprints.get(digest)
        if recent is not None and (now - recent) < self._window:
            # Record the GUID so the duplicate is cheap to reject next poll.
            self._guids[item.guid] = now
            return False
        return True

    def record(self, item: NewsItem) -> None:
        now = time.time()
        self._guids[item.guid] = now
        self._fingerprints[fingerprint(item)] = now

    @staticmethod
    def semantic_key(player_name: str, event_type: str) -> str:
        from .matcher import compact_key

        return f"{compact_key(player_name)}|{event_type}"

    def is_semantically_new(self, player_name: str, event_type: str) -> bool:
        """False when this (player, event) already alerted from another source."""
        if not player_name:
            return True
        seen_at = self._semantic.get(self.semantic_key(player_name, event_type))
        return seen_at is None or (time.time() - seen_at) >= SEMANTIC_WINDOW_SECONDS

    def record_semantic(self, player_name: str, event_type: str) -> None:
        if player_name:
            self._semantic[self.semantic_key(player_name, event_type)] = time.time()

    def prime(self, items: list[NewsItem]) -> None:
        """Mark existing items as seen without alerting (first run)."""
        for item in items:
            self.record(item)
        self.save()
