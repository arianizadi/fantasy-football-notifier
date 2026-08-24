"""Small durable state store for Telegram threads, feedback, and digests."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import logging
import os
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .dedupe import (
    event_facts_equivalent,
    role_decision_status,
    semantic_event_fact_signature,
    semantic_event_status,
    semantic_event_type,
)
from .logging_utils import structured_log
from .matcher import compact_key
from .models import Alert, NewsItem, report_revision_identity
from .plays import normalized_event_type

STATE_VERSION = 1
ROLE_METADATA_MIGRATION_VERSION = 1
MAX_FEEDBACK = 1000
MAX_FEEDBACK_TARGETS = 2000
ALERT_RETENTION_DAYS = 8
EDIT_WINDOW_SECONDS = 6 * 60 * 60
SCHEDULED_REPORT_RETENTION_DAYS = 30
MAX_SCHEDULED_REPORTS = 120


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _visible_units(markup: str) -> int:
    parser = _VisibleTextParser()
    parser.feed(markup)
    parser.close()
    return len("".join(parser.parts).encode("utf-16-le")) // 2


@dataclass(frozen=True)
class TelegramEditTarget:
    """A previously sent message that is safe to update in place."""

    message_id: int
    token: str
    severity: int


def alert_token(report: NewsItem | str) -> str:
    """Stable short report-revision identifier for Telegram callback data.

    String input retains the legacy GUID-only helper behavior for callers
    reading old state. New alerts always pass ``NewsItem`` so a feed changing
    text under a reused GUID receives a distinct token.
    """
    if isinstance(report, NewsItem):
        return report_revision_identity(report)[:16]
    return hashlib.sha256(report.encode("utf-8")).hexdigest()[:16]


def feedback_markup(report: NewsItem | str, selected: str = "") -> dict[str, Any]:
    """Telegram inline keyboard; callback_data stays below the 64-byte cap."""
    return feedback_markup_for_token(alert_token(report), selected=selected)


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
        moves: list[str] = []
        if not alert.availability_refresh_failed:
            options = [player.name for player in plays.claimable[:2]]
            if options:
                if len(options) > 1:
                    moves.append(f"PICKUP OPTIONS {' / '.join(options)}")
                else:
                    moves.append(f"ADD OPTION {options[0]}")
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
            "roleMetadataMigration": ROLE_METADATA_MIGRATION_VERSION,
            "updateOffset": 0,
            "threads": {},
            "alerts": [],
            "feedback": {},
            "feedbackTargets": {},
            "lastDigestDate": "",
            "scheduledReports": {},
            "lastTelegramSuccess": 0.0,
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text())
            if payload.get("version") != STATE_VERSION:
                raise ValueError("unsupported state version")
            try:
                migration_version = int(payload.get("roleMetadataMigration") or 0)
            except (TypeError, ValueError):
                migration_version = 0
            for key, default in self._empty().items():
                payload.setdefault(key, default)
            self._payload = payload
            if migration_version < ROLE_METADATA_MIGRATION_VERSION:
                self._migrate_role_metadata_locked()
                self._payload["roleMetadataMigration"] = (
                    ROLE_METADATA_MIGRATION_VERSION
                )
                self._save_locked()
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            structured_log(logging.WARNING, "telegram.state_unreadable", error=str(error))

    def _migrate_role_metadata_locked(self) -> bool:
        """Canonicalize legacy role threads before the first post-deploy alert.

        Older state saved model labels and generic Week markers.  The latest
        headline and player are enough to prove an explicit starter decision,
        so migrate only that narrow case; all other legacy entries continue to
        fail closed.
        """
        def role_metadata(
            player: str,
            headline: str,
            prior_event: str,
        ) -> dict[str, str] | None:
            if not player or not headline:
                return None
            item = NewsItem(
                source="state",
                guid="",
                player_name=player,
                headline=headline,
                body=headline,
                url="",
                published_at=None,
            )
            if not role_decision_status(item):
                return None
            event_type = semantic_event_type(item, prior_event)
            if event_type != "depth_chart":
                return None
            return {
                "eventType": event_type,
                "eventStatus": semantic_event_status(item, event_type),
                "eventFactSignature": semantic_event_fact_signature(
                    item,
                    event_type,
                ),
            }

        changed = False
        alerts = self._payload.get("alerts") or []
        threads = self._payload.get("threads") or {}
        now = time.time()
        for thread_key, entry in list(threads.items()):
            if not isinstance(entry, dict):
                continue
            player = str(entry.get("player") or "").strip()
            headline = str(entry.get("latestHeadline") or "").strip()
            prior_event = str(entry.get("eventType") or "other")
            metadata = role_metadata(player, headline, prior_event)

            message_id = entry.get("messageId")
            token = entry.get("token")
            if metadata is not None:
                if any(entry.get(key) != value for key, value in metadata.items()):
                    entry.update(metadata)
                    changed = True
                for digest in reversed(alerts):
                    if not isinstance(digest, dict):
                        continue
                    if (
                        digest.get("messageId") != message_id
                        or digest.get("token") != token
                    ):
                        continue
                    if any(
                        digest.get(key) != value
                        for key, value in metadata.items()
                    ):
                        digest.update(metadata)
                        changed = True
                    break
                continue

            # Production may have received generic commentary immediately
            # after the real role decision, leaving the per-player pointer on
            # an uneditable ``other`` message. Recover the newest provable role
            # alert from the digest, but only when the current pointer itself
            # carries no concrete fact and the candidate is still editable.
            if (
                normalized_event_type(prior_event) != "other"
                or str(entry.get("eventFactSignature") or "")
                not in {"", "unspecified"}
            ):
                continue
            for digest in reversed(alerts):
                if not isinstance(digest, dict):
                    continue
                candidate_player = str(digest.get("player") or "").strip()
                candidate_headline = str(digest.get("headline") or "").strip()
                if compact_key(candidate_player) != compact_key(player):
                    continue
                candidate_metadata = role_metadata(
                    candidate_player,
                    candidate_headline,
                    str(digest.get("eventType") or "other"),
                )
                if candidate_metadata is None:
                    continue
                try:
                    candidate_message_id = int(digest["messageId"])
                    candidate_sent_at = float(digest["sentAt"])
                    candidate_severity = int(digest["severity"])
                except (KeyError, TypeError, ValueError):
                    continue
                candidate_token = str(digest.get("token") or "")
                age = now - candidate_sent_at
                if age < 0 or age >= EDIT_WINDOW_SECONDS or not candidate_token:
                    continue
                if any(
                    digest.get(key) != value
                    for key, value in candidate_metadata.items()
                ):
                    digest.update(candidate_metadata)
                    changed = True
                threads[thread_key] = {
                    "messageId": candidate_message_id,
                    "sentAt": candidate_sent_at,
                    "player": candidate_player,
                    "token": candidate_token,
                    "eventType": candidate_metadata["eventType"],
                    "severity": candidate_severity,
                    "eventStatus": candidate_metadata["eventStatus"],
                    "eventFactSignature": candidate_metadata[
                        "eventFactSignature"
                    ],
                    "latestHeadline": candidate_headline[:220],
                }
                changed = True
                break
        return changed

    def _save_locked(self) -> bool:
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
            return True
        except OSError as error:
            structured_log(logging.WARNING, "telegram.state_write_failed", error=str(error))
            return False

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

    def coalescing_target(
        self,
        alert: Alert,
        *,
        now: float | None = None,
    ) -> TelegramEditTarget | None:
        """Return an existing message only for a true same-state corroboration.

        Old state files intentionally fail closed. Without every piece of
        event metadata, the sender cannot prove that an edit would preserve an
        urgent state transition, so it sends a normal reply instead.
        """
        key = compact_key(alert.item.player_name)
        if not key:
            return None
        stamp = time.time() if now is None else now
        current_event = semantic_event_type(
            alert.item,
            alert.classification.event_type,
            str(alert.classification.raw.get("event_type") or ""),
        )
        current_status = semantic_event_status(alert.item, current_event)
        current_fact_signature = semantic_event_fact_signature(alert.item, current_event)
        current_severity = int(alert.classification.severity)

        with self._lock:
            entry = (self._payload.get("threads") or {}).get(key)
            if not isinstance(entry, dict):
                return None
            required = {
                "messageId",
                "sentAt",
                "token",
                "eventType",
                "severity",
                "eventStatus",
                "eventFactSignature",
                "latestHeadline",
            }
            if not required.issubset(entry):
                return None
            try:
                message_id = int(entry["messageId"])
                sent_at = float(entry["sentAt"])
                token = str(entry["token"])
                previous_event = normalized_event_type(str(entry["eventType"]))
                previous_severity = int(entry["severity"])
                previous_status = str(entry["eventStatus"])
                previous_fact_signature = str(entry["eventFactSignature"])
            except (TypeError, ValueError):
                return None
            age = stamp - sent_at
            if age < 0 or age >= EDIT_WINDOW_SECONDS or not token:
                return None
            if current_event != previous_event:
                return None
            if not current_status or current_status != previous_status:
                return None
            if not event_facts_equivalent(
                previous_fact_signature,
                current_fact_signature,
                status=current_status,
            ):
                return None
            same_role_fact = (
                current_event == "depth_chart"
                and current_status.startswith("role_")
                and current_fact_signature.startswith("role:")
            )
            same_trade_fact = (
                current_event == "trade"
                and event_facts_equivalent(
                    previous_fact_signature,
                    current_fact_signature,
                    status=current_status,
                )
            )
            if (
                current_severity > previous_severity
                and not same_role_fact
                and not same_trade_fact
            ):
                return None

            # The digest record must be editable too. Otherwise a Telegram
            # edit would leave the digest describing an older alert.
            digest_record = next(
                (
                    record
                    for record in reversed(self._payload.get("alerts") or [])
                    if isinstance(record, dict)
                    and record.get("messageId") == message_id
                    and record.get("token") == token
                ),
                None,
            )
            digest_required = {
                "eventType",
                "severity",
                "eventStatus",
                "eventFactSignature",
                "headline",
            }
            if not isinstance(digest_record, dict) or not digest_required.issubset(
                digest_record
            ):
                return None
            return TelegramEditTarget(
                message_id=message_id,
                token=token,
                severity=previous_severity,
            )

    def _register_feedback_target_locked(
        self,
        alert: Alert,
        *,
        token: str,
        message_id: int,
        recorded_at: float,
    ) -> None:
        """Retain report identity even after its message is edited again."""
        targets = self._payload.setdefault("feedbackTargets", {})
        targets[token] = {
            "messageId": int(message_id),
            "player": (alert.item.player_name or "League news").strip(),
            "eventType": semantic_event_type(
                alert.item,
                alert.classification.event_type,
                str(alert.classification.raw.get("event_type") or ""),
            ),
            "severity": alert.classification.severity,
            "headline": alert.item.headline[:220],
            "recordedAt": recorded_at,
        }
        cutoff = recorded_at - (ALERT_RETENTION_DAYS * 24 * 3600)

        def target_timestamp(value: object) -> float:
            if not isinstance(value, dict):
                return 0.0
            try:
                return float(value.get("recordedAt") or 0)
            except (TypeError, ValueError):
                return 0.0

        targets = {
            key: value
            for key, value in targets.items()
            if isinstance(value, dict)
            and target_timestamp(value) >= cutoff
        }
        if len(targets) > MAX_FEEDBACK_TARGETS:
            keep = sorted(
                targets,
                key=lambda key: target_timestamp(targets[key]),
                reverse=True,
            )[:MAX_FEEDBACK_TARGETS]
            targets = {key: targets[key] for key in keep}
        self._payload["feedbackTargets"] = targets

    def record_sent(self, alert: Alert, message_id: int) -> str:
        """Advance a player's reply chain and add the alert to the daily digest."""
        token = alert_token(alert.item)
        now = time.time()
        player = (alert.item.player_name or "League news").strip()
        normalized_event = semantic_event_type(
            alert.item,
            alert.classification.event_type,
            str(alert.classification.raw.get("event_type") or ""),
        )
        status = semantic_event_status(alert.item, normalized_event)
        fact_signature = semantic_event_fact_signature(alert.item, normalized_event)
        record = {
            "token": token,
            "messageId": int(message_id),
            "sentAt": now,
            "player": player,
            "eventType": normalized_event,
            "severity": alert.classification.severity,
            "eventStatus": status,
            "eventFactSignature": fact_signature,
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
                    "token": token,
                    "eventType": normalized_event,
                    "severity": alert.classification.severity,
                    "eventStatus": status,
                    "eventFactSignature": fact_signature,
                    "latestHeadline": alert.item.headline[:220],
                }
            self._register_feedback_target_locked(
                alert,
                token=token,
                message_id=message_id,
                recorded_at=now,
            )
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

    def record_edited(
        self,
        alert: Alert,
        message_id: int,
        expected_token: str,
    ) -> bool:
        """Atomically point message state and feedback at the displayed report.

        Telegram is edited before this method runs.  Returning ``False`` tells
        delivery to retain the outbox entry and retry; it must never mark the
        new report delivered while local state still describes the old text.
        """
        key = compact_key(alert.item.player_name)
        if not key:
            return False
        now = time.time()
        normalized_event = semantic_event_type(
            alert.item,
            alert.classification.event_type,
            str(alert.classification.raw.get("event_type") or ""),
        )
        status = semantic_event_status(alert.item, normalized_event)
        fact_signature = semantic_event_fact_signature(alert.item, normalized_event)
        new_token = alert_token(alert.item)
        published_at = (
            alert.item.published_at.isoformat()
            if alert.item.published_at is not None
            else ""
        )

        with self._lock:
            thread = (self._payload.get("threads") or {}).get(key)
            if not isinstance(thread, dict):
                return False
            try:
                thread_message_id = int(thread.get("messageId"))
            except (TypeError, ValueError):
                return False
            if (
                thread_message_id != int(message_id)
                or str(thread.get("token") or "") != expected_token
            ):
                return False

            digest_record = next(
                (
                    record
                    for record in reversed(self._payload.get("alerts") or [])
                    if isinstance(record, dict)
                    and record.get("messageId") == int(message_id)
                    and record.get("token") == expected_token
                ),
                None,
            )
            if digest_record is None:
                return False

            previous_payload = copy.deepcopy(self._payload)
            # Keep sentAt stable so repeated edits cannot extend the six-hour
            # coalescing window. Rotate the token because the visible text now
            # represents this report; the old target and any old vote remain
            # in feedbackTargets/feedback as historical evidence.
            thread.update(
                {
                    "player": alert.item.player_name,
                    "token": new_token,
                    "eventType": normalized_event,
                    "severity": alert.classification.severity,
                    "eventStatus": status,
                    "eventFactSignature": fact_signature,
                    "latestHeadline": alert.item.headline[:220],
                    "updatedAt": now,
                }
            )
            digest_record.update(
                {
                    "token": new_token,
                    "player": (alert.item.player_name or "League news").strip(),
                    "eventType": normalized_event,
                    "severity": alert.classification.severity,
                    "eventStatus": status,
                    "eventFactSignature": fact_signature,
                    "tier": alert.tier,
                    "headline": alert.item.headline[:220],
                    "action": _digest_action(alert),
                    "publishedAt": published_at,
                    "updatedAt": now,
                }
            )
            self._register_feedback_target_locked(
                alert,
                token=new_token,
                message_id=message_id,
                recorded_at=now,
            )
            self._payload["lastTelegramSuccess"] = now
            try:
                saved = self._save_locked()
            except Exception as error:  # noqa: BLE001 - roll back in-memory state
                structured_log(
                    logging.WARNING,
                    "telegram.edit_state_write_failed",
                    errorType=type(error).__name__,
                )
                saved = False
            if not saved:
                self._payload = previous_payload
                return False
            return True

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
            target = (self._payload.get("feedbackTargets") or {}).get(token)
            if not isinstance(target, dict):
                # Backward compatibility for state written before targets were
                # retained separately from the current digest record.
                target = next(
                    (
                        entry
                        for entry in reversed(self._payload.get("alerts") or [])
                        if isinstance(entry, dict) and str(entry.get("token")) == token
                    ),
                    None,
                )
            if not isinstance(target, dict):
                return False
            feedback = self._payload["feedback"]
            feedback[token] = {
                "verdict": verdict,
                "recordedAt": time.time(),
                "messageId": target.get("messageId"),
                "player": target.get("player"),
                "eventType": target.get("eventType"),
                "severity": target.get("severity"),
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

    def register_scheduled_report(
        self,
        key: str,
        *,
        kind: str,
        parts: tuple[str, ...] | list[str],
        notify_first: bool = False,
    ) -> bool:
        """Persist a multipart report before its first Telegram request.

        If the daemon restarts after only some parts were accepted, the exact
        original content and the accepted message ids remain available. A
        regenerated waiver pool or recap can therefore never make a retry
        duplicate part one or silently skip a changed part.
        """
        report_key = str(key or "").strip()
        normalized_parts = [str(part) for part in parts if str(part).strip()]
        if not report_key or not normalized_parts or len(normalized_parts) > 20:
            return False
        if any(_visible_units(part) > 4096 for part in normalized_parts):
            return False

        now = time.time()
        with self._lock:
            reports = self._payload.setdefault("scheduledReports", {})
            existing = reports.get(report_key)
            if isinstance(existing, dict):
                message_ids = existing.get("messageIds") or []
                completed = float(existing.get("completedAt") or 0) > 0
                if (
                    not completed
                    and isinstance(message_ids, list)
                    and not message_ids
                ):
                    # Nothing reached Telegram yet, so an availability refresh
                    # may safely replace a stale rendering. Once any part is
                    # accepted, the exact original sequence wins so a restart
                    # resumes rather than mixing two report versions.
                    previous_payload = copy.deepcopy(self._payload)
                    existing.update(
                        {
                            "kind": str(kind or "report")[:40],
                            "parts": normalized_parts,
                            "notifyFirst": bool(notify_first),
                            "updatedAt": now,
                        }
                    )
                    if not self._save_locked():
                        self._payload = previous_payload
                        return False
                return True
            previous_payload = copy.deepcopy(self._payload)
            reports[report_key] = {
                "kind": str(kind or "report")[:40],
                "parts": normalized_parts,
                "messageIds": [],
                "notifyFirst": bool(notify_first),
                "createdAt": now,
                "completedAt": 0.0,
            }
            self._prune_scheduled_reports_locked(now)
            if not self._save_locked():
                self._payload = previous_payload
                return False
            return True

    def _prune_scheduled_reports_locked(self, now: float) -> None:
        reports = self._payload.get("scheduledReports") or {}
        cutoff = now - (SCHEDULED_REPORT_RETENTION_DAYS * 24 * 3600)

        def timestamp(value: object) -> float:
            if not isinstance(value, dict):
                return 0.0
            try:
                return float(value.get("completedAt") or value.get("createdAt") or 0)
            except (TypeError, ValueError):
                return 0.0

        kept = {
            key: value
            for key, value in reports.items()
            if isinstance(value, dict) and timestamp(value) >= cutoff
        }
        if len(kept) > MAX_SCHEDULED_REPORTS:
            newest = sorted(kept, key=lambda key: timestamp(kept[key]), reverse=True)[
                :MAX_SCHEDULED_REPORTS
            ]
            kept = {key: kept[key] for key in newest}
        self._payload["scheduledReports"] = kept

    def next_scheduled_report_part(
        self,
        key: str,
    ) -> tuple[int, str, int | None, bool] | None:
        """Return the next unsent part and its reply-chain parent."""
        with self._lock:
            entry = (self._payload.get("scheduledReports") or {}).get(key)
            if not isinstance(entry, dict) or float(entry.get("completedAt") or 0) > 0:
                return None
            parts = entry.get("parts") or []
            message_ids = entry.get("messageIds") or []
            if not isinstance(parts, list) or not isinstance(message_ids, list):
                return None
            index = len(message_ids)
            if index >= len(parts):
                return None
            try:
                reply_to = int(message_ids[-1]) if message_ids else None
            except (TypeError, ValueError):
                reply_to = None
            return index, str(parts[index]), reply_to, bool(entry.get("notifyFirst"))

    def record_scheduled_report_part(
        self,
        key: str,
        *,
        part_index: int,
        message_id: int,
    ) -> bool | None:
        """Commit one accepted part; return completion, or ``None`` on failure."""
        now = time.time()
        with self._lock:
            entry = (self._payload.get("scheduledReports") or {}).get(key)
            if not isinstance(entry, dict):
                return None
            parts = entry.get("parts") or []
            message_ids = entry.get("messageIds") or []
            if not isinstance(parts, list) or not isinstance(message_ids, list):
                return None
            if int(part_index) != len(message_ids) or len(message_ids) >= len(parts):
                return None
            previous_payload = copy.deepcopy(self._payload)
            message_ids.append(int(message_id))
            entry["messageIds"] = message_ids
            entry["updatedAt"] = now
            completed = len(message_ids) == len(parts)
            if completed:
                entry["completedAt"] = now
            self._payload["lastTelegramSuccess"] = now
            if not self._save_locked():
                self._payload = previous_payload
                return None
            return completed

    def scheduled_report_completed(self, key: str) -> bool:
        with self._lock:
            entry = (self._payload.get("scheduledReports") or {}).get(key)
            return isinstance(entry, dict) and float(entry.get("completedAt") or 0) > 0

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
