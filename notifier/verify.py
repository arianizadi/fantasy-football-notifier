"""Second-opinion pass, delivered as a threaded reply to the fast alert.

Runs on a background thread so the fast path is never blocked: the Flash alert
goes out in ~2s, and the verification lands underneath it whenever it finishes.

Measured on this task, the verifier model is not more accurate than Flash
(9/12 vs 11/12 on a labelled set), so this is not "the smart model checking the
dumb one". Its value is that two independent reads disagreeing marks a call as
genuinely ambiguous and worth your own eyes.
"""

from __future__ import annotations

import html
import json
import logging
import queue
import threading
from dataclasses import dataclass

import requests

from .classify import _extract_json
from .openrouter import chat
from .stealth import StealthReviewer
from .config import Config
from .logging_utils import structured_log
from .models import Classification, NewsItem

REQUEST_TIMEOUT = 90
QUEUE_MAXSIZE = 200

SYSTEM_PROMPT = """You are double-checking another model's fantasy football call.
Return ONLY JSON: {"severity":int,"agree":bool,"note":str}
severity: your own independent 1-5 rating (1=noise, 5=season-defining)
agree: whether the other model's severity was reasonable
note: under 120 chars, only what a fantasy manager should do differently.
Judge fantasy consequence, not dramatic wording."""


@dataclass(frozen=True)
class VerifyJob:
    item: NewsItem
    first: Classification
    message_id: int


class Verifier:
    """Background worker that posts second opinions as Telegram replies."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # Free, optional, and entirely self-disabling when no stealth model
        # is published. Never allowed to affect the alert or the paid verifier.
        self.stealth = (
            StealthReviewer(config.openrouter_api_key)
            if config.stealth_review_enabled
            else None
        )
        self._queue: queue.Queue[VerifyJob | None] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._session = requests.Session()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.verify_enabled:
            return
        self._thread = threading.Thread(target=self._run, name="verifier", daemon=True)
        self._thread.start()
        stealth_model = self.stealth.model() if self.stealth is not None else None
        structured_log(
            logging.INFO,
            "verify.started",
            model=self.config.verify_model,
            stealthModel=stealth_model,
            stealthEnabled=bool(stealth_model),
        )

    def submit(self, job: VerifyJob) -> None:
        if not self.config.verify_enabled or self._thread is None:
            return
        # Only double-check calls that actually matter.
        if job.first.severity < self.config.verify_min_severity:
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            structured_log(logging.WARNING, "verify.queue_full", player=job.item.player_name)

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._process(job)
            except Exception as error:  # noqa: BLE001 - worker must survive
                structured_log(
                    logging.ERROR,
                    "verify.failed",
                    error=str(error),
                    errorType=type(error).__name__,
                )

    def _process(self, job: VerifyJob) -> None:
        from .notify import send_reply

        prompt = (
            f"Headline: {job.item.player_name}: {job.item.headline}\n"
            f"Report: {job.item.body[:500]}\n\n"
            f"Other model said: severity {job.first.severity}/5, "
            f"event {job.first.event_type}, impact: {job.first.fantasy_impact}"
        )
        # chat() negotiates the reasoning policy per model: DeepSeek returns
        # no content unless reasoning is disabled, while ox-alpha rejects the
        # request outright if it is ("Reasoning is mandatory for this endpoint").
        response = chat(
            self._session,
            self.config.openrouter_api_key,
            self.config.verify_model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code in (404, 400):
            # A stealth model withdrawn at the end of its free window 404s.
            # Verification is a nice-to-have; never let it break the alert path.
            structured_log(
                logging.WARNING,
                "verify.model_unavailable",
                model=self.config.verify_model,
                httpStatus=response.status_code,
                hint="Re-run bin/eval-models.py to pick a live model.",
            )
            return
        response.raise_for_status()

        content = response.json()["choices"][0]["message"].get("content")
        if not content:
            structured_log(
                logging.WARNING, "verify.empty_content", model=self.config.verify_model
            )
            return
        try:
            parsed = _extract_json(content)
        except (ValueError, TypeError):
            # Weaker free models routinely return prose. Skip rather than
            # posting a reply built from a fallback that means nothing.
            structured_log(
                logging.WARNING,
                "verify.unparseable",
                model=self.config.verify_model,
                preview=str(content)[:120],
            )
            return

        try:
            second = max(1, min(5, int(parsed.get("severity", job.first.severity))))
        except (TypeError, ValueError):
            second = job.first.severity
        note = str(parsed.get("note") or "").strip()[:160]
        gap = abs(second - job.first.severity)

        text = format_reply(job.first.severity, second, note)

        # Best-effort third opinion. Any failure just omits the line.
        stealth_severity = None
        if self.stealth is not None:
            try:
                result = self.stealth.review(
                    f"{job.item.player_name}: {job.item.headline}",
                    job.item.body,
                    job.first.severity,
                )
            except Exception as error:  # noqa: BLE001 - must never break delivery
                structured_log(
                    logging.DEBUG, "stealth.review_failed", error=str(error)
                )
                result = None
            if result is not None:
                stealth_severity, stealth_note = result
                label = (self.stealth.model() or "stealth").split("/")[-1]
                text += f"\n3rd ({html.escape(label, quote=False)}): {stealth_severity}/5"
                if stealth_note:
                    text += f" - {html.escape(stealth_note, quote=False)}"

        send_reply(self._session, self.config, text, job.message_id)
        structured_log(
            logging.INFO,
            "verify.replied",
            player=job.item.player_name,
            firstSeverity=job.first.severity,
            secondSeverity=second,
            stealthSeverity=stealth_severity,
            gap=gap,
        )


def format_reply(first_severity: int, second_severity: int, note: str = "") -> str:
    """Format safe Telegram HTML from the verifier's untrusted model output."""
    gap = abs(second_severity - first_severity)
    if gap == 0:
        text = f"2nd opinion: agrees, {second_severity}/5."
    elif gap == 1:
        text = f"2nd opinion: {second_severity}/5 (vs {first_severity}/5)."
    else:
        text = (
            f"2nd opinion: <b>disagrees, {second_severity}/5</b> vs "
            f"{first_severity}/5 - check this one yourself."
        )
    if note:
        text += f"\n{html.escape(note, quote=False)}"
    return text
