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
import tempfile
import threading
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

# Ordered from weak/uncertain participation notes to definitive absences.
# Within the 90-minute semantic window a strictly worse status is a meaningful
# update and should alert even when the model chooses the same event type.
STATUS_PATTERNS = (
    (
        "season_out",
        100,
        re.compile(
            r"\b(season[-\s]ending|out\s+for\s+the\s+season|"
            r"(?:torn|tore)\s+(acl|achilles))\b",
            re.I,
        ),
    ),
    (
        "injured_reserve",
        90,
        re.compile(r"\b(injured\s+reserve|reserve/injured|placed\s+on\s+ir)\b", re.I),
    ),
    (
        "inactive",
        80,
        re.compile(r"\b(inactive|ruled\s+out|will\s+not\s+play)\b", re.I),
    ),
    ("doubtful", 60, re.compile(r"\bdoubtful\b", re.I)),
    ("questionable", 50, re.compile(r"\bquestionable\b", re.I)),
    ("dnp", 40, re.compile(r"\b(dnp|did\s+not\s+practice)\b", re.I)),
    ("limited", 30, re.compile(r"\b(limited|limited\s+participant)\b", re.I)),
    (
        "cleared",
        20,
        re.compile(
            r"\b(activated|cleared|full\s+participant|returned\s+to\s+practice)\b",
            re.I,
        ),
    ),
)


def event_status(item: NewsItem, event_type: str) -> str:
    text = f"{item.headline} {item.body}"
    for label, _rank, pattern in STATUS_PATTERNS:
        if pattern.search(text):
            return label
    return event_type


def _status_rank(status: str) -> int:
    for label, rank, _pattern in STATUS_PATTERNS:
        if status == label:
            return rank
    return 0


def _status_is_meaningfully_new(previous: str, current: str) -> bool:
    """Allow definitive worsening and a later clearance through early dedupe."""
    if not current or current == previous:
        return False
    if current == "cleared" and _status_rank(previous) >= _status_rank("limited"):
        return True
    return _status_rank(current) > _status_rank(previous)


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
        self._guid_statuses: dict[str, str] = {}
        self._fingerprint_statuses: dict[str, str] = {}
        self._semantic: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()
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
            self._guid_statuses = {
                str(key): str(value)
                for key, value in payload.get("guidStatuses", {}).items()
            }
            self._fingerprint_statuses = {
                str(key): str(value)
                for key, value in payload.get("fingerprintStatuses", {}).items()
            }
            semantic: dict[str, dict[str, object]] = {}
            for key, value in payload.get("semantic", {}).items():
                if isinstance(value, dict):
                    semantic[str(key)] = {
                        "seen_at": float(value.get("seen_at", 0)),
                        "severity": int(value["severity"])
                        if value.get("severity") is not None
                        else None,
                        "status": str(value.get("status") or ""),
                    }
                else:
                    # Backward compatibility: old stores only had timestamps.
                    semantic[str(key)] = {
                        "seen_at": float(value),
                        "severity": None,
                        "status": "",
                    }
            self._semantic = semantic
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            structured_log(logging.WARNING, "dedupe.state_unreadable", error=str(error))

    def save(self) -> bool:
        with self._lock:
            self._prune()
            payload = {
                "guids": self._guids,
                "fingerprints": self._fingerprints,
                "guidStatuses": self._guid_statuses,
                "fingerprintStatuses": self._fingerprint_statuses,
                "semantic": self._semantic,
            }
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{self._path.name}.",
                    suffix=".tmp",
                    dir=self._path.parent,
                    text=True,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "w") as handle:
                        json.dump(payload, handle, separators=(",", ":"))
                    os.replace(temporary, self._path)
                finally:
                    temporary.unlink(missing_ok=True)
                return True
            except OSError as error:
                structured_log(logging.WARNING, "dedupe.state_write_failed", error=str(error))
                return False

    def _prune(self) -> None:
        now = time.time()
        # GUIDs are kept far longer than the fingerprint window: a feed item
        # can reappear hours later and must not re-alert.
        guid_ttl = max(self._window, 24 * 60 * 60)
        self._guids = {k: v for k, v in self._guids.items() if now - v < guid_ttl}
        self._fingerprints = {k: v for k, v in self._fingerprints.items() if now - v < self._window}
        self._guid_statuses = {
            key: value for key, value in self._guid_statuses.items() if key in self._guids
        }
        self._fingerprint_statuses = {
            key: value
            for key, value in self._fingerprint_statuses.items()
            if key in self._fingerprints
        }
        self._semantic = {
            key: value
            for key, value in self._semantic.items()
            if now - float(value.get("seen_at", 0)) < SEMANTIC_WINDOW_SECONDS
        }

        for store in (self._guids, self._fingerprints, self._semantic):
            if len(store) > MAX_TRACKED_ENTRIES:
                if store is self._semantic:
                    ordering = lambda key: float(store[key].get("seen_at", 0))
                else:
                    ordering = store.get
                for key in sorted(store, key=ordering)[: len(store) - MAX_TRACKED_ENTRIES]:
                    del store[key]

    def is_new(self, item: NewsItem) -> bool:
        # Deliberately pure: previews/dry-runs must not advance state, and a
        # duplicate check must never make a failed delivery unretryable.
        with self._lock:
            now = time.time()
            status = event_status(item, "")
            if item.guid in self._guids:
                return _status_is_meaningfully_new(
                    self._guid_statuses.get(item.guid, ""),
                    status,
                )
            digest = fingerprint(item)
            recent = self._fingerprints.get(digest)
            if recent is None or (now - recent) >= self._window:
                return True
            return _status_is_meaningfully_new(
                self._fingerprint_statuses.get(digest, ""),
                status,
            )

    def record(self, item: NewsItem) -> None:
        with self._lock:
            now = time.time()
            digest = fingerprint(item)
            status = event_status(item, "")
            self._guids[item.guid] = now
            self._fingerprints[digest] = now
            self._guid_statuses[item.guid] = status
            self._fingerprint_statuses[digest] = status

    @staticmethod
    def semantic_key(player_name: str, event_type: str) -> str:
        from .matcher import compact_key

        return f"{compact_key(player_name)}|{event_type}"

    def is_semantically_new(
        self,
        player_name: str,
        event_type: str,
        severity: int | None = None,
        status: str = "",
    ) -> bool:
        """False for a repeat, True for a new event or meaningful escalation."""
        if not player_name:
            return True
        with self._lock:
            previous = self._semantic.get(self.semantic_key(player_name, event_type))
            if previous is None:
                return True
            if (time.time() - float(previous.get("seen_at", 0))) >= SEMANTIC_WINDOW_SECONDS:
                return True

            old_severity = previous.get("severity")
            if severity is not None and old_severity is not None and severity > old_severity:
                return True
            old_status = str(previous.get("status") or "")
            if status and old_status and _status_rank(status) > _status_rank(old_status):
                return True
            return False

    def record_semantic(
        self,
        player_name: str,
        event_type: str,
        severity: int | None = None,
        status: str = "",
    ) -> None:
        if player_name:
            with self._lock:
                self._semantic[self.semantic_key(player_name, event_type)] = {
                    "seen_at": time.time(),
                    "severity": severity,
                    "status": status,
                }

    def prime(self, items: list[NewsItem]) -> None:
        """Mark existing items as seen without alerting (first run)."""
        for item in items:
            self.record(item)
        self.save()
