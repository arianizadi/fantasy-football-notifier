"""Durable pending Telegram deliveries.

An alert is written here before its first send attempt.  It is removed only
after Telegram returns a message id, so a transient rejection or process
restart cannot turn an already-seen feed item into a permanently lost alert.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .dedupe import fingerprint
from .logging_utils import structured_log
from .models import Alert, Classification, LeagueRef, NewsItem
from .plays import Beneficiary, DepthEntry, LeaguePlays, TeamContext

OUTBOX_FILENAME = "pending-alerts.json"
SOURCE_PRIORITY = {"twitter": 0, "rotowire": 1}
RETRY_DELAYS_SECONDS = (15, 60, 5 * 60, 15 * 60)
MAX_TRACKED = 1000


@dataclass
class PendingDelivery:
    delivery_id: str
    alert: Alert
    queued_at: float
    attempts: int = 0
    next_attempt_at: float = 0.0
    last_error: str = ""


def _delivery_id(alert: Alert) -> str:
    value = f"{alert.item.guid}|{alert.item.player_name}|{alert.classification.event_type}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _league(payload: dict[str, Any]) -> LeagueRef:
    return LeagueRef(**payload)


def _news_item(payload: dict[str, Any]) -> NewsItem:
    published = payload.get("published_at")
    return NewsItem(
        source=str(payload.get("source") or ""),
        guid=str(payload.get("guid") or ""),
        player_name=str(payload.get("player_name") or ""),
        headline=str(payload.get("headline") or ""),
        body=str(payload.get("body") or ""),
        url=str(payload.get("url") or ""),
        published_at=datetime.fromisoformat(published) if published else None,
    )


def _classification(payload: dict[str, Any]) -> Classification:
    return Classification(
        event_type=str(payload.get("event_type") or "other"),
        severity=int(payload.get("severity") or 3),
        fantasy_impact=str(payload.get("fantasy_impact") or ""),
        is_actionable=bool(payload.get("is_actionable", False)),
        raw=dict(payload.get("raw") or {}),
    )


def _beneficiary(payload: dict[str, Any]) -> Beneficiary:
    return Beneficiary(
        name=str(payload.get("name") or ""),
        position=str(payload.get("position") or ""),
        depth_order=payload.get("depth_order"),
        state=str(payload.get("state") or "free_agent"),
        fantasy_team=str(payload.get("fantasy_team") or ""),
    )


def _depth_entry(payload: dict[str, Any]) -> DepthEntry:
    ownership = {
        str(key): (str(value[0]), str(value[1]))
        for key, value in (payload.get("ownership") or {}).items()
        if isinstance(value, (list, tuple)) and len(value) == 2
    }
    return DepthEntry(
        name=str(payload.get("name") or ""),
        position=str(payload.get("position") or ""),
        depth_order=payload.get("depth_order"),
        search_rank=payload.get("search_rank"),
        is_subject=bool(payload.get("is_subject", False)),
        ownership=ownership,
        sleeper_injury_status=str(payload.get("sleeper_injury_status") or ""),
        sleeper_status=str(payload.get("sleeper_status") or ""),
    )


def _league_plays(payload: dict[str, Any]) -> LeaguePlays:
    return LeaguePlays(
        league=_league(payload["league"]),
        subject_state=str(payload.get("subject_state") or "free_agent"),
        subject_owner=str(payload.get("subject_owner") or ""),
        beneficiaries=[_beneficiary(value) for value in payload.get("beneficiaries", [])],
        bench_options=[str(value) for value in payload.get("bench_options", [])],
    )


def _team_context(payload: dict[str, Any] | None) -> TeamContext | None:
    if not payload:
        return None
    refreshed = payload.get("player_index_refreshed_at")
    return TeamContext(
        team=str(payload.get("team") or ""),
        subject_position=str(payload.get("subject_position") or ""),
        same_position=[_depth_entry(value) for value in payload.get("same_position", [])],
        adjacent=[_depth_entry(value) for value in payload.get("adjacent", [])],
        player_index_refreshed_at=(
            datetime.fromisoformat(refreshed) if refreshed else None
        ),
    )


def _alert_from_dict(payload: dict[str, Any]) -> Alert:
    return Alert(
        item=_news_item(payload["item"]),
        classification=_classification(payload["classification"]),
        tier=str(payload.get("tier") or "league"),
        per_league=[_league_plays(value) for value in payload.get("per_league", [])],
        context=_team_context(payload.get("context")),
        all_leagues=[_league(value) for value in payload.get("all_leagues", [])],
        availability_refresh_failed=bool(
            payload.get("availability_refresh_failed", False)
        ),
        delivery_delayed=bool(payload.get("delivery_delayed", False)),
    )


def _alert_to_dict(alert: Alert) -> dict[str, Any]:
    payload = asdict(alert)
    if alert.item.published_at is not None:
        payload["item"]["published_at"] = alert.item.published_at.isoformat()
    if alert.context is not None and alert.context.player_index_refreshed_at is not None:
        payload["context"]["player_index_refreshed_at"] = (
            alert.context.player_index_refreshed_at.isoformat()
        )
    return payload


class DeliveryOutbox:
    def __init__(self, state_dir: Path) -> None:
        self._path = state_dir / OUTBOX_FILENAME
        self._pending: dict[str, PendingDelivery] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text())
        except (TypeError, ValueError, json.JSONDecodeError, OSError) as error:
            structured_log(logging.WARNING, "outbox.unreadable", error=str(error))
            return
        if not isinstance(payload, dict):
            structured_log(logging.WARNING, "outbox.unreadable", error="root is not an object")
            return

        for value in payload.get("pending", []):
            if not isinstance(value, dict):
                structured_log(
                    logging.WARNING,
                    "outbox.entry_unreadable",
                    error="entry is not an object",
                )
                continue
            try:
                delivery = PendingDelivery(
                    delivery_id=str(value["delivery_id"]),
                    alert=_alert_from_dict(value["alert"]),
                    queued_at=float(value.get("queued_at", 0)),
                    attempts=int(value.get("attempts", 0)),
                    next_attempt_at=float(value.get("next_attempt_at", 0)),
                    last_error=str(value.get("last_error") or ""),
                )
                self._pending[delivery.delivery_id] = delivery
            except (KeyError, TypeError, ValueError) as error:
                # One malformed legacy entry must not discard every other
                # pending alert in the file.
                structured_log(logging.WARNING, "outbox.entry_unreadable", error=str(error))

    def _save(self) -> None:
        payload = {
            "pending": [
                {
                    "delivery_id": delivery.delivery_id,
                    "alert": _alert_to_dict(delivery.alert),
                    "queued_at": delivery.queued_at,
                    "attempts": delivery.attempts,
                    "next_attempt_at": delivery.next_attempt_at,
                    "last_error": delivery.last_error,
                }
                for delivery in self._pending.values()
            ]
        }
        try:
            temporary = self._path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(temporary, self._path)
        except OSError as error:
            structured_log(logging.ERROR, "outbox.write_failed", error=str(error))
            raise

    def add(self, alert: Alert) -> PendingDelivery:
        """Persist an alert before its first Telegram attempt."""
        with self._lock:
            delivery_id = _delivery_id(alert)
            existing = self._pending.get(delivery_id)
            if existing is not None:
                return existing
            delivery = PendingDelivery(
                delivery_id=delivery_id,
                alert=alert,
                queued_at=time.time(),
            )
            self._pending[delivery_id] = delivery
            if len(self._pending) > MAX_TRACKED:
                # Never silently discard pending work. Refuse to accept more so
                # the source item remains unseen and can be retried later.
                self._pending.pop(delivery_id, None)
                raise RuntimeError("delivery outbox is full")
            try:
                self._save()
            except OSError:
                # A send is allowed only after persistence succeeds.
                self._pending.pop(delivery_id, None)
                raise
            return delivery

    def due(self, now: float | None = None) -> list[PendingDelivery]:
        with self._lock:
            stamp = time.time() if now is None else now
            ready = [value for value in self._pending.values() if value.next_attempt_at <= stamp]
            return sorted(
                ready,
                key=lambda value: (
                    value.queued_at,
                    SOURCE_PRIORITY.get(value.alert.item.source, 9),
                ),
            )

    def contains_item(self, item: NewsItem) -> bool:
        with self._lock:
            digest = fingerprint(item)
            return any(
                value.alert.item.guid == item.guid
                or fingerprint(value.alert.item) == digest
                for value in self._pending.values()
            )

    def mark_failed(
        self,
        delivery_id: str,
        error: str = "send_failed",
        *,
        retry_after: int = 0,
    ) -> None:
        with self._lock:
            delivery = self._pending.get(delivery_id)
            if delivery is None:
                return
            delivery.attempts += 1
            delay = RETRY_DELAYS_SECONDS[
                min(delivery.attempts - 1, len(RETRY_DELAYS_SECONDS) - 1)
            ]
            delay = max(delay, max(0, int(retry_after)))
            delivery.next_attempt_at = time.time() + delay
            delivery.last_error = error
            self._save()

    def remove(self, delivery_id: str) -> None:
        with self._lock:
            removed = self._pending.pop(delivery_id, None)
            if removed is None:
                return
            try:
                self._save()
            except OSError:
                # Keep the in-memory view consistent with the still-pending
                # on-disk state so a later retry can retire it safely.
                self._pending[delivery_id] = removed
                raise

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)
