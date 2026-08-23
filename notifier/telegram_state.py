"""Small durable state store for Telegram threads, feedback, and digests."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .logging_utils import structured_log
from .matcher import compact_key
from .models import Alert

STATE_VERSION = 1
MAX_FEEDBACK = 1000
ALERT_RETENTION_DAYS = 8


def alert_token(guid: str) -> str:
    """Stable short identifier that fits comfortably in callback_data."""
    return hashlib.sha256(guid.encode("utf-8")).hexdigest()[:16]


def feedback_markup(guid: str, selected: str = "") -> dict[str, Any]:
    """Telegram inline keyboard; callback_data stays below the 64-byte cap."""
    return feedback_markup_for_token(alert_token(guid), selected=selected)


def feedback_markup_for_token(token: str, selected: str = "") -> dict[str, Any]:
    """Build the feedback keyboard when the alert token is already known."""
    buttons = (
        ("useful", "Useful"),
        ("wrong", "Wrong"),
        ("noisy", "Too noisy"),
    )
    return {
        "inline_keyboard": [
            [
                {
                    "text": ("✓ " if selected == value else "") + label,
                    "callback_data": f"feedback:{token}:{value}",
                }
                for value, label in buttons
            ]
        ]
    }


def _digest_action(alert: Alert) -> str:
    bases = [plays.league.short_label for plays in alert.per_league]
    counts = Counter(base.casefold() for base in bases)
    actions: list[str] = []
    for plays in alert.per_league:
        base = plays.league.short_label
        label = base
        if counts[base.casefold()] > 1:
            provider = (plays.league.provider or "league").upper()
            suffix = plays.league.league_id[-4:] if plays.league.league_id else ""
            label = f"{base} ({provider} {suffix})".strip()
        moves = []
        if not alert.availability_refresh_failed:
            moves = [f"ADD {player.name}" for player in plays.claimable[:2]]
        if plays.bench_options:
            moves.append(f"START {plays.bench_options[0]}")
        if moves:
            actions.append(f"{label}: {', '.join(moves)}")
    if actions:
        return " | ".join(actions)[:300]

    if alert.availability_refresh_failed:
        return "League availability refresh failed; no ADD recommendation was shown."

    event_type = (
        alert.classification.event_type.strip().lower().replace("-", "_").replace(" ", "_")
    )
    if alert.tier == "preseason":
        if event_type == "return":
            return "Confirm full practice and Week 1 status before changing draft value."
        if event_type in {"injury", "inactive", "suspension"}:
            return "Recheck official availability before drafting this player."
        return "Reassess draft value after confirming the report and role."
    if event_type == "return":
        return "Confirm active status and expected workload before lineup lock."
    if event_type in {"injury", "inactive", "suspension", "release"}:
        return "Verify official availability and league context before acting."
    if event_type == "practice_report":
        return "Wait for the final official game-status report before acting."
    if event_type in {"trade", "signing", "depth_chart", "usage"}:
        return "Confirm the official role and depth chart before making a move."
    return "Verify the report and player context before making a move."


class TelegramState:
    """Atomic JSON state shared by the sender and Telegram control thread."""

    def __init__(self, path: Path, *, thread_hours: int = 168) -> None:
        self._path = path
        self._thread_seconds = thread_hours * 3600
        self._lock = threading.RLock()
        self._payload: dict[str, Any] = self._empty()
        self._load()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "updateOffset": 0,
            "threads": {},
            "alerts": [],
            "feedback": {},
            "lastDigestDate": "",
            "lastTelegramSuccess": 0.0,
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text())
            if payload.get("version") != STATE_VERSION:
                raise ValueError("unsupported state version")
            for key, default in self._empty().items():
                payload.setdefault(key, default)
            self._payload = payload
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            structured_log(logging.WARNING, "telegram.state_unreadable", error=str(error))

    def _save_locked(self) -> None:
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
                    json.dump(self._payload, handle, separators=(",", ":"))
                os.replace(temporary, self._path)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError as error:
            structured_log(logging.WARNING, "telegram.state_write_failed", error=str(error))

    @property
    def update_offset(self) -> int:
        with self._lock:
            return int(self._payload.get("updateOffset") or 0)

    def set_update_offset(self, offset: int) -> None:
        with self._lock:
            if offset <= int(self._payload.get("updateOffset") or 0):
                return
            self._payload["updateOffset"] = int(offset)
            self._save_locked()

    def previous_message_id(self, player_name: str, *, now: float | None = None) -> int | None:
        """Return the latest alert for a player while its Telegram message exists."""
        key = compact_key(player_name)
        if not key:
            return None
        stamp = time.time() if now is None else now
        with self._lock:
            entry = (self._payload.get("threads") or {}).get(key)
            if not isinstance(entry, dict):
                return None
            try:
                message_id = int(entry["messageId"])
                sent_at = float(entry["sentAt"])
            except (KeyError, TypeError, ValueError):
                return None
            if stamp - sent_at >= self._thread_seconds:
                del self._payload["threads"][key]
                self._save_locked()
                return None
            return message_id

    def record_sent(self, alert: Alert, message_id: int) -> str:
        """Advance a player's reply chain and add the alert to the daily digest."""
        token = alert_token(alert.item.guid)
        now = time.time()
        player = (alert.item.player_name or "League news").strip()
        record = {
            "token": token,
            "messageId": int(message_id),
            "sentAt": now,
            "player": player,
            "eventType": alert.classification.event_type,
            "severity": alert.classification.severity,
            "tier": alert.tier,
            "headline": alert.item.headline[:220],
            "action": _digest_action(alert),
            "publishedAt": (
                alert.item.published_at.isoformat()
                if alert.item.published_at is not None
                else ""
            ),
        }
        with self._lock:
            key = compact_key(alert.item.player_name)
            if key:
                self._payload["threads"][key] = {
                    "messageId": int(message_id),
                    "sentAt": now,
                    "player": alert.item.player_name,
                }
            alerts = self._payload["alerts"]
            alerts.append(record)
            cutoff = now - (ALERT_RETENTION_DAYS * 24 * 3600)
            self._payload["alerts"] = [
                entry
                for entry in alerts
                if float(entry.get("sentAt") or 0) >= cutoff
            ]
            self._payload["lastTelegramSuccess"] = now
            self._save_locked()
        return token

    def mark_telegram_success(self) -> None:
        with self._lock:
            self._payload["lastTelegramSuccess"] = time.time()
            self._save_locked()

    @property
    def last_telegram_success(self) -> float:
        with self._lock:
            return float(self._payload.get("lastTelegramSuccess") or 0)

    def record_feedback(self, token: str, verdict: str) -> bool:
        if verdict not in {"useful", "wrong", "noisy"}:
            return False
        with self._lock:
            alerts = {str(entry.get("token")): entry for entry in self._payload["alerts"]}
            if token not in alerts:
                return False
            feedback = self._payload["feedback"]
            feedback[token] = {
                "verdict": verdict,
                "recordedAt": time.time(),
                "messageId": alerts[token].get("messageId"),
                "player": alerts[token].get("player"),
                "eventType": alerts[token].get("eventType"),
                "severity": alerts[token].get("severity"),
            }
            if len(feedback) > MAX_FEEDBACK:
                keep = sorted(
                    feedback,
                    key=lambda key: float(feedback[key].get("recordedAt") or 0),
                    reverse=True,
                )[:MAX_FEEDBACK]
                self._payload["feedback"] = {key: feedback[key] for key in keep}
            self._save_locked()
        return True

    def feedback_verdict(self, token: str) -> str:
        with self._lock:
            entry = (self._payload.get("feedback") or {}).get(token) or {}
            return str(entry.get("verdict") or "")

    def digest_due(self, now: datetime, *, hour: int, timezone_name: str) -> bool:
        local = now.astimezone(ZoneInfo(timezone_name))
        with self._lock:
            return (
                local.hour >= hour
                and self._payload.get("lastDigestDate") != local.date().isoformat()
            )

    def mark_digest_sent(self, now: datetime, *, timezone_name: str) -> None:
        local = now.astimezone(ZoneInfo(timezone_name))
        with self._lock:
            self._payload["lastDigestDate"] = local.date().isoformat()
            self._save_locked()

    def format_digest(
        self,
        now: datetime,
        *,
        timezone_name: str,
        hours: int = 24,
    ) -> str:
        """Render sent alerts from the last day, highest urgency first."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cutoff = now.timestamp() - timedelta(hours=hours).total_seconds()
        with self._lock:
            recent = [
                dict(entry)
                for entry in self._payload.get("alerts", [])
                if float(entry.get("sentAt") or 0) >= cutoff
            ]

        recent.sort(
            key=lambda entry: (
                -int(entry.get("severity") or 0),
                -float(entry.get("sentAt") or 0),
            )
        )
        local = now.astimezone(ZoneInfo(timezone_name))
        lines = [f"<b>Daily fantasy action digest · {local:%b %-d}</b>"]
        if not recent:
            lines += ["", "No alerts in the last 24 hours."]
            return "\n".join(lines)

        for entry in recent[:12]:
            severity = int(entry.get("severity") or 0)
            player = html.escape(str(entry.get("player") or "League news"), quote=False)
            event_type = html.escape(
                str(entry.get("eventType") or "news").replace("_", " "), quote=False
            )
            impact = html.escape(str(entry.get("action") or ""), quote=False)
            lines += ["", f"<b>{severity}/5 · {player}</b> · {event_type}"]
            published_raw = str(entry.get("publishedAt") or "")
            if published_raw:
                try:
                    published = datetime.fromisoformat(published_raw)
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                    lines[-1] += f" · reported {published.astimezone(ZoneInfo(timezone_name)):%-I:%M %p}"
                except ValueError:
                    pass
            if impact:
                lines.append(impact)
        if len(recent) > 12:
            lines += ["", f"+ {len(recent) - 12} more alerts"]
        return "\n".join(lines)
