"""Second-opinion pass, delivered as a threaded reply to the fast alert.

Runs on a background thread so the fast path is never blocked: the Flash alert
goes out in ~2s, and the verification lands underneath it whenever it finishes.

Measured on this task, the verifier model is not more accurate than Flash
(9/12 vs 11/12 on a labelled set), so this is not "the smart model checking the
dumb one". Its value is that two independent reads disagreeing marks a call as
genuinely ambiguous and worth your own eyes.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass

import requests

from .classify import API_URL, _extract_json
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
        self._queue: queue.Queue[VerifyJob | None] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._session = requests.Session()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.config.verify_enabled:
            return
        self._thread = threading.Thread(target=self._run, name="verifier", daemon=True)
        self._thread.start()
        structured_log(logging.INFO, "verify.started", model=self.config.verify_model)

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
        response = self._session.post(
            API_URL,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.verify_model,
                "temperature": 0,
                "max_tokens": 400,
                "response_format": {"type": "json_object"},
                # Same reasoning-token trap as the fast path; see classify.py.
                "reasoning": {"enabled": False},
                # Default routing sprays across ~6 providers and the tail
                # reached 8.9s. Pinning by throughput held max at 1.5s --
                # a 5.8x better worst case, which matters more than median
                # for a breaking-news alert.
                "provider": {"sort": "throughput"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        parsed = _extract_json(response.json()["choices"][0]["message"]["content"])

        try:
            second = max(1, min(5, int(parsed.get("severity", job.first.severity))))
        except (TypeError, ValueError):
            second = job.first.severity
        note = str(parsed.get("note") or "").strip()[:160]
        gap = abs(second - job.first.severity)

        model_label = self.config.verify_model.split("/")[-1]
        if gap == 0:
            text = f"2nd opinion: agrees, {second}/5."
        elif gap == 1:
            text = f"2nd opinion: {second}/5 (vs {job.first.severity}/5)."
        else:
            # A two-point split means the call is genuinely ambiguous.
            text = (
                f"2nd opinion: <b>disagrees, {second}/5</b> vs "
                f"{job.first.severity}/5 - check this one yourself."
            )
        if note:
            text += f"\n{note}"

        send_reply(self._session, self.config, text, job.message_id)
        structured_log(
            logging.INFO,
            "verify.replied",
            player=job.item.player_name,
            firstSeverity=job.first.severity,
            secondSeverity=second,
            gap=gap,
        )
