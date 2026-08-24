"""Telegram commands, feedback callbacks, and the scheduled daily digest."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import requests

from .config import Config
from .logging_utils import structured_log
from .notify import API_BASE, send_plain
from .telegram_state import TelegramState, feedback_markup_for_token

LONG_POLL_SECONDS = 25
REQUEST_TIMEOUT = (5, LONG_POLL_SECONDS + 10)


@dataclass(frozen=True)
class ScheduledReport:
    """One idempotent, already-formatted outbound report."""

    key: str
    kind: str
    parts: tuple[str, ...]
    notify_first: bool = False
    expires_at: datetime | None = None


class TelegramControl:
    """Run Bot API long polling independently of the breaking-news loop."""

    def __init__(
        self,
        config: Config,
        state: TelegramState,
        *,
        status_provider: Callable[[], str],
        player_provider: Callable[[str], str],
        search_provider: Callable[[str], str] | None = None,
        feedback_provider: Callable[[str, str], bool] | None = None,
        daily_recap_provider: Callable[[datetime], Sequence[str]] | None = None,
        scheduled_report_provider: Callable[[datetime], Sequence[ScheduledReport]]
        | None = None,
    ) -> None:
        self._config = config
        self._state = state
        self._status_provider = status_provider
        self._player_provider = player_provider
        self._search_provider = search_provider
        self._feedback_provider = feedback_provider
        self._daily_recap_provider = daily_recap_provider
        self._scheduled_report_provider = scheduled_report_provider
        self._session = requests.Session()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._auto_delete_seconds: int | None = None
        self._last_poll_success = 0.0
        self._last_poll_error_at = 0.0
        self._last_poll_error = ""

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="telegram-control",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._session.close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=LONG_POLL_SECONDS + 12)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def auto_delete_seconds(self) -> int | None:
        return self._auto_delete_seconds

    @property
    def status_label(self) -> str:
        if not self._config.telegram_controls_enabled:
            return "disabled (digest scheduler only)"
        if not self.is_alive:
            return "stopped"
        if self._last_poll_error_at > self._last_poll_success:
            return "error (getUpdates; see service log)"
        if self._last_poll_success:
            return "healthy"
        return "starting"

    def _url(self, method: str) -> str:
        return f"{API_BASE}/bot{self._config.telegram_bot_token}/{method}"

    def _run(self) -> None:
        if self._config.dry_run:
            return
        self._refresh_chat_settings()
        if not self._config.telegram_controls_enabled:
            # Sending a digest does not consume updates, so it can safely run
            # even when OpenClaw or another process owns getUpdates.
            while not self._stop.is_set():
                self._maybe_send_digest()
                self._maybe_send_scheduled_reports()
                self._stop.wait(30)
            return

        self._set_commands()
        while not self._stop.is_set():
            try:
                response = self._session.get(
                    self._url("getUpdates"),
                    params={
                        "offset": self._state.update_offset,
                        "timeout": LONG_POLL_SECONDS,
                        "allowed_updates": '["message","callback_query"]',
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                self._last_poll_success = time.time()
                self._last_poll_error = ""
                updates = response.json().get("result") or []
                for update in updates:
                    self._handle_update(update)
                    self._state.set_update_offset(int(update.get("update_id") or 0) + 1)
            except (requests.RequestException, KeyError, TypeError, ValueError) as error:
                self._last_poll_error_at = time.time()
                self._last_poll_error = str(error)[:160]
                structured_log(logging.WARNING, "telegram.control_error", error=str(error))
                self._stop.wait(5)
            finally:
                # Digest timing must not depend on a healthy getUpdates
                # connection. A 409 conflict or transient outage cannot block
                # the scheduled outbound message.
                self._maybe_send_digest()
                self._maybe_send_scheduled_reports()

    def _set_commands(self) -> None:
        commands = [
            {"command": "status", "description": "Check source and delivery health"},
            {"command": "player", "description": "Look up a player: /player Kittle"},
            {"command": "news", "description": "Search saved reports: /news Kittle"},
            {"command": "digest", "description": "Show the 24-hour football recap"},
            {"command": "help", "description": "Show bot commands"},
        ]
        try:
            response = self._session.post(
                self._url("setMyCommands"),
                json={"commands": commands},
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            structured_log(logging.WARNING, "telegram.commands_failed", error=str(error))

    def _refresh_chat_settings(self) -> None:
        try:
            response = self._session.get(
                self._url("getChat"),
                params={"chat_id": self._config.telegram_chat_id},
                timeout=15,
            )
            response.raise_for_status()
            result = response.json().get("result") or {}
            self._auto_delete_seconds = int(result.get("message_auto_delete_time") or 0)
        except (requests.RequestException, AttributeError, TypeError, ValueError) as error:
            structured_log(logging.WARNING, "telegram.chat_settings_failed", error=str(error))

    def _authorized_chat(self, payload: dict) -> bool:
        chat = payload.get("chat") or {}
        return str(chat.get("id") or "") == str(self._config.telegram_chat_id)

    def _handle_update(self, update: dict) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            self._handle_callback(callback)
            return

        message = update.get("message")
        if not isinstance(message, dict) or not self._authorized_chat(message):
            return
        text = str(message.get("text") or "").strip()
        if not text.startswith("/"):
            return

        first, _, argument = text.partition(" ")
        command = first.split("@", 1)[0].lower()
        if command == "/status":
            self._reply_from_provider("status", self._status_provider)
        elif command == "/player":
            query = argument.strip()
            if not query:
                self._reply("Usage: <code>/player Kittle</code>")
            else:
                self._reply_from_provider(
                    "player lookup",
                    lambda: self._player_provider(query),
                )
        elif command == "/news":
            query = argument.strip()
            if not query:
                self._reply("Usage: <code>/news Kittle</code>")
            elif self._search_provider is None:
                self._reply("Saved-news search is unavailable.")
            else:
                self._reply_from_provider(
                    "saved-news search",
                    lambda: self._search_provider(query),
                )
        elif command == "/digest":
            try:
                parts = self._daily_recap_parts(datetime.now(timezone.utc))
            except Exception:  # failure was logged by the provider wrapper
                self._reply("Daily recap is temporarily unavailable.")
                return
            if len(parts) == 1:
                self._reply(parts[0])
            else:
                self._reply_many(parts)
        elif command in {"/help", "/start"}:
            self._reply(
                "<b>Fantasy notifier commands</b>\n"
                "/status - source and delivery health\n"
                "/player Kittle - depth, status, and league ownership\n"
                "/news Kittle - search the saved tweet/report journal\n"
                "/digest - big news, smaller moves, and a learning note\n\n"
                "Use the buttons under an alert to mark it useful, wrong, or too noisy."
            )

    def _reply_from_provider(self, label: str, provider: Callable[[], str]) -> None:
        try:
            text = provider()
        except Exception as error:  # noqa: BLE001 - do not poison getUpdates offset
            structured_log(
                logging.ERROR,
                "telegram.command_failed",
                command=label,
                error=str(error),
            )
            self._reply(f"{label.capitalize()} is temporarily unavailable.")
            return
        self._reply(text)

    def _handle_callback(self, callback: dict) -> None:
        message = callback.get("message") or {}
        callback_id = str(callback.get("id") or "")
        if not self._authorized_chat(message):
            self._answer_callback(callback_id, "This chat is not authorized.", alert=True)
            return

        data = str(callback.get("data") or "")
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != "feedback":
            self._answer_callback(callback_id, "Unknown action.", alert=True)
            return
        _, token, verdict = parts
        state_recorded = self._state.record_feedback(token, verdict)
        durable_recorded = False
        if self._feedback_provider is not None:
            try:
                durable_recorded = self._feedback_provider(token, verdict)
            except Exception as error:  # noqa: BLE001 - callback must still be acknowledged
                structured_log(logging.WARNING, "telegram.feedback_store_failed", error=str(error))
        if not state_recorded and not durable_recorded:
            self._answer_callback(callback_id, "That alert is no longer in local history.")
            return

        labels = {"useful": "Useful", "wrong": "Wrong", "noisy": "Too noisy"}
        self._answer_callback(callback_id, f"Recorded: {labels[verdict]}")
        message_id = message.get("message_id")
        if message_id is not None:
            try:
                self._session.post(
                    self._url("editMessageReplyMarkup"),
                    json={
                        "chat_id": self._config.telegram_chat_id,
                        "message_id": int(message_id),
                        "reply_markup": feedback_markup_for_token(token, selected=verdict),
                    },
                    timeout=15,
                ).raise_for_status()
            except (requests.RequestException, TypeError, ValueError) as error:
                structured_log(logging.WARNING, "telegram.feedback_markup_failed", error=str(error))

    def _answer_callback(self, callback_id: str, text: str, *, alert: bool = False) -> None:
        if not callback_id:
            return
        try:
            self._session.post(
                self._url("answerCallbackQuery"),
                json={"callback_query_id": callback_id, "text": text, "show_alert": alert},
                timeout=15,
            ).raise_for_status()
        except requests.RequestException as error:
            structured_log(logging.WARNING, "telegram.callback_answer_failed", error=str(error))

    def _reply(self, text: str) -> int | None:
        message_id = send_plain(self._session, self._config, text)
        if message_id is not None and message_id >= 0:
            self._state.mark_telegram_success()
        return message_id

    def _reply_many(self, parts: Sequence[str]) -> None:
        previous: int | None = None
        for part in parts:
            message_id = send_plain(
                self._session,
                self._config,
                str(part),
                reply_to=previous,
            )
            if message_id is None or message_id < 0:
                return
            self._state.mark_telegram_success()
            previous = message_id

    def _daily_recap_parts(self, now: datetime) -> tuple[str, ...]:
        if self._daily_recap_provider is not None:
            try:
                parts = tuple(str(part) for part in self._daily_recap_provider(now))
                if parts:
                    return parts
            except Exception as error:  # noqa: BLE001 - scheduled retry is safe
                structured_log(
                    logging.WARNING,
                    "telegram.daily_recap_failed",
                    errorType=type(error).__name__,
                    error=str(error)[:160],
                )
                raise
        return (
            self._state.format_digest(
                now,
                timezone_name=self._config.daily_digest_timezone,
            ),
        )

    def _deliver_registered_report(self, report: ScheduledReport) -> bool:
        """Send only unsent parts, committing each accepted message id."""
        while True:
            expires_at = report.expires_at
            if expires_at is not None:
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) >= expires_at.astimezone(timezone.utc):
                    structured_log(
                        logging.INFO,
                        "telegram.scheduled_report_expired",
                        reportKind=report.kind,
                        reportKey=report.key,
                    )
                    return False
            pending = self._state.next_scheduled_report_part(report.key)
            if pending is None:
                return self._state.scheduled_report_completed(report.key)
            part_index, text, reply_to, notify_first = pending
            message_id = send_plain(
                self._session,
                self._config,
                text,
                silent=not (notify_first and part_index == 0),
                reply_to=reply_to,
            )
            if message_id is None or message_id < 0:
                return False
            completed = self._state.record_scheduled_report_part(
                report.key,
                part_index=part_index,
                message_id=message_id,
            )
            if completed is None:
                structured_log(
                    logging.ERROR,
                    "telegram.scheduled_report_state_failed",
                    reportKind=report.kind,
                    reportKey=report.key,
                    part=part_index + 1,
                )
                return False
            structured_log(
                logging.INFO,
                "telegram.scheduled_report_part_sent",
                reportKind=report.kind,
                reportKey=report.key,
                part=part_index + 1,
                messageId=message_id,
            )
            if completed:
                return True

    def _maybe_send_digest(self) -> None:
        if self._config.dry_run or not self._config.daily_digest_enabled:
            return
        now = datetime.now(timezone.utc)
        if not self._state.digest_due(
            now,
            hour=self._config.daily_digest_hour,
            timezone_name=self._config.daily_digest_timezone,
        ):
            return
        local = now.astimezone(ZoneInfo(self._config.daily_digest_timezone))
        key = f"daily-recap:{local.date().isoformat()}"
        try:
            parts = self._daily_recap_parts(now)
        except Exception:  # failure was logged by the provider wrapper
            return
        report = ScheduledReport(
            key=key,
            kind="daily_recap",
            parts=parts,
            notify_first=False,
        )
        if not self._state.register_scheduled_report(
            report.key,
            kind=report.kind,
            parts=report.parts,
            notify_first=report.notify_first,
        ):
            structured_log(
                logging.ERROR,
                "telegram.scheduled_report_registration_failed",
                reportKind=report.kind,
                reportKey=report.key,
                partCount=len(report.parts),
            )
            return
        if self._deliver_registered_report(report):
            self._state.mark_digest_sent(
                now,
                timezone_name=self._config.daily_digest_timezone,
            )

    def _maybe_send_scheduled_reports(self) -> None:
        if (
            self._config.dry_run
            or self._scheduled_report_provider is None
        ):
            return
        try:
            reports = tuple(
                self._scheduled_report_provider(datetime.now(timezone.utc))
            )
        except Exception as error:  # noqa: BLE001 - retry on the next scheduler tick
            structured_log(
                logging.WARNING,
                "telegram.scheduled_reports_failed",
                errorType=type(error).__name__,
                error=str(error)[:160],
            )
            return
        for report in reports:
            if not report.parts:
                continue
            if not self._state.register_scheduled_report(
                report.key,
                kind=report.kind,
                parts=report.parts,
                notify_first=report.notify_first,
            ):
                structured_log(
                    logging.ERROR,
                    "telegram.scheduled_report_registration_failed",
                    reportKind=report.kind,
                    reportKey=report.key,
                    partCount=len(report.parts),
                )
                continue
            self._deliver_registered_report(report)
