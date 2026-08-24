"""OpenRouter embeddings used as a guarded Telegram coalescing assistant.

Vectors never replace the deterministic event/status/fact checks.  They only
help recognize paraphrases after those checks prove that an edit cannot hide a
status reversal, new condition, timetable, transaction destination, or severity
escalation.  Every raw report remains in :mod:`notifier.event_store`.
"""

from __future__ import annotations

import hashlib
import html
import logging
import math
import re
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Iterable

import requests

from .dedupe import (
    event_facts_equivalent,
    semantic_event_fact_signature,
    semantic_event_status,
    semantic_event_type,
)
from .event_store import EventStore, classification_direction
from .health import HEALTH
from .logging_utils import structured_log
from .matcher import compact_key
from .models import Alert, Classification, NewsItem, report_revision_identity

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
INPUT_VERSION = "news-report-v1"
MAX_INPUT_CHARACTERS = 1800
MAX_PENDING_FUTURES = 2000
EMBEDDING_WORKERS = 4
CIRCUIT_FAILURES = 2
CIRCUIT_OPEN_SECONDS = 60

_URL = re.compile(r"https?://\S+", re.I)
_HANDLE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_SPACE = re.compile(r"\s+")


class EmbeddingUnavailable(RuntimeError):
    """The optional embedding provider could not return a safe vector."""


@dataclass(frozen=True)
class EmbeddingVector:
    model: str
    provider: str
    dimensions: int
    input_version: str
    input_hash: str
    values: tuple[float, ...]
    blob: bytes
    prompt_tokens: int = 0


@dataclass(frozen=True)
class EmbeddingMatch:
    row: dict[str, Any]
    score: float
    reason: str


@dataclass(frozen=True)
class EmbeddingServiceStatus:
    enabled: bool
    mode: str
    model: str
    dimensions: int
    embedded: int
    matches: int
    failures: int
    prompt_tokens: int
    last_provider: str


def canonical_embedding_text(item: NewsItem) -> str:
    """Stable provider-neutral text; player identity remains a metadata gate."""

    def clean(value: str) -> str:
        value = html.unescape(value or "")
        value = _URL.sub(" ", value)
        value = _HANDLE.sub(" ", value)
        return _SPACE.sub(" ", value).strip()

    headline = clean(item.headline)
    body = clean(item.body)
    parts: list[str] = []
    if headline:
        parts.append(f"Headline: {headline}")
    if body and body.casefold() != headline.casefold():
        parts.append(f"Report: {body}")
    if not parts:
        parts.append("Report unavailable")
    return "\n".join(parts)[:MAX_INPUT_CHARACTERS]


def embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_vector(values: Iterable[float], *, dimensions: int) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if len(vector) != dimensions:
        raise ValueError(
            f"embedding dimensions: expected {dimensions} values, received {len(vector)}"
        )
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding contains non-finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding has zero or invalid magnitude")
    return tuple(value / norm for value in vector)


def pack_vector(values: tuple[float, ...]) -> bytes:
    if not values:
        raise ValueError("cannot pack an empty embedding")
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes, *, dimensions: int) -> tuple[float, ...]:
    if dimensions <= 0 or len(blob) != dimensions * 4:
        raise ValueError("embedding byte length does not match its dimensions")
    values = struct.unpack(f"<{dimensions}f", blob)
    return normalize_vector(values, dimensions=dimensions)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("embedding contains non-finite values")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("embedding has zero magnitude")
    score = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, float(score)))


def news_item_from_row(row: dict[str, Any]) -> NewsItem:
    published = row.get("published_at")
    return NewsItem(
        source=str(row.get("source") or ""),
        guid=str(row.get("guid") or ""),
        player_name=str(row.get("player_name") or ""),
        headline=str(row.get("headline") or ""),
        body=str(row.get("body") or ""),
        url=str(row.get("url") or ""),
        published_at=datetime.fromisoformat(str(published)) if published else None,
        subject_confident=bool(row.get("subject_confident", True)),
    )


def alert_from_row(row: dict[str, Any]) -> Alert:
    try:
        severity = int(row.get("severity") or 0)
    except (TypeError, ValueError):
        severity = 0
    direction = str(row.get("direction") or "unknown")
    return Alert(
        item=news_item_from_row(row),
        classification=Classification(
            event_type=str(row.get("event_type") or "other"),
            severity=severity,
            fantasy_impact=str(row.get("summary") or ""),
            is_actionable=bool(row.get("is_actionable", False)),
            raw={"direction": direction},
        ),
        tier=str(row.get("tier") or "league"),
    )


def embedding_transition_guard(
    current: Alert,
    previous: dict[str, Any],
    *,
    score: float,
    threshold: float,
) -> tuple[bool, str]:
    """Prove that similarity may edit a prior alert without hiding new facts."""
    if not math.isfinite(score) or not math.isfinite(threshold):
        return False, "invalid_similarity"
    if score < threshold:
        return False, "below_threshold"
    if not current.item.subject_confident or not bool(
        previous.get("subject_confident", True)
    ):
        return False, "uncertain_subject"
    if compact_key(current.item.player_name) != compact_key(
        str(previous.get("player_name") or "")
    ):
        return False, "different_player"
    if previous.get("feedback") in {"wrong", "noisy"}:
        return False, "rejected_prior"
    try:
        previous_message_id = int(previous.get("telegram_message_id") or 0)
    except (TypeError, ValueError):
        previous_message_id = 0
    if previous_message_id <= 0 or not str(previous.get("alert_token") or ""):
        return False, "no_editable_message"

    previous_item = news_item_from_row(previous)
    current_event = semantic_event_type(
        current.item,
        current.classification.event_type,
        str(current.classification.raw.get("event_type") or ""),
    )
    previous_event = semantic_event_type(
        previous_item,
        str(previous.get("event_type") or "other"),
    )
    if not current_event or current_event != previous_event:
        return False, "event_transition"

    current_direction = classification_direction(current.classification)
    previous_direction = str(previous.get("direction") or "unknown").strip().lower()
    if (
        current_direction == "unknown"
        or previous_direction == "unknown"
        or current_direction != previous_direction
    ):
        return False, "direction_transition"

    try:
        previous_severity = int(previous.get("severity") or 0)
    except (TypeError, ValueError):
        previous_severity = 0
    if int(current.classification.severity) > previous_severity:
        return False, "severity_escalation"

    current_status = semantic_event_status(current.item, current_event)
    previous_status = semantic_event_status(previous_item, previous_event)
    if not current_status or current_status != previous_status:
        return False, "status_transition"

    current_facts = semantic_event_fact_signature(current.item, current_event)
    previous_facts = semantic_event_fact_signature(previous_item, previous_event)
    facts_equivalent = event_facts_equivalent(
        previous_facts,
        current_facts,
        status=current_status,
    )
    # Dense similarity is allowed to prove a paraphrase only when neither
    # report claims a concrete condition/timetable/destination. If either side
    # has a structured fact, the deterministic equivalence check must pass.
    if not facts_equivalent and not (
        previous_facts == "unspecified" and current_facts == "unspecified"
    ):
        return False, "fact_transition"
    # Telegram edits replace the older alert. A terse follow-up must not erase
    # a richer report merely because both describe the same core event.
    previous_text = canonical_embedding_text(previous_item)
    current_text = canonical_embedding_text(current.item)
    if len(current_text) < len(previous_text) * 0.70:
        return False, "information_regression"
    return True, "safe_paraphrase"


class OpenRouterEmbeddingClient:
    """Small fail-fast client with a circuit breaker; no retry delays alerts."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        *,
        timeout_seconds: int = 8,
        post: Any = requests.post,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.dimensions = int(dimensions)
        self.timeout_seconds = int(timeout_seconds)
        self._post = post
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0

    def _before_request(self) -> None:
        with self._lock:
            if time.monotonic() < self._open_until:
                raise EmbeddingUnavailable("embedding circuit is open")

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= CIRCUIT_FAILURES:
                self._open_until = time.monotonic() + CIRCUIT_OPEN_SECONDS

    def embed_many(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        if not self.api_key:
            raise EmbeddingUnavailable("OpenRouter embedding key is unavailable")
        self._before_request()
        try:
            response = self._post(
                OPENROUTER_EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "Fantasy Football Notifier",
                },
                json={
                    "model": self.model,
                    "input": texts,
                    "dimensions": self.dimensions,
                    "encoding_format": "float",
                },
                timeout=(2, self.timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
            raw_data = payload.get("data")
            if not isinstance(raw_data, list) or len(raw_data) != len(texts):
                raise ValueError("embedding response count mismatch")
            if not all(isinstance(entry, dict) for entry in raw_data):
                raise ValueError("embedding response entries must be objects")
            ordered = sorted(raw_data, key=lambda entry: int(entry.get("index", -1)))
            if [int(entry.get("index", -1)) for entry in ordered] != list(
                range(len(texts))
            ):
                raise ValueError("embedding response indices are invalid")
            prompt_tokens = int((payload.get("usage") or {}).get("prompt_tokens") or 0)
            tokens_each, token_remainder = divmod(prompt_tokens, len(texts))
            vectors: list[EmbeddingVector] = []
            for position, (text, entry) in enumerate(
                zip(texts, ordered, strict=True)
            ):
                normalized = normalize_vector(
                    entry.get("embedding") or (),
                    dimensions=self.dimensions,
                )
                vectors.append(
                    EmbeddingVector(
                        model=self.model,
                        provider="openrouter",
                        dimensions=self.dimensions,
                        input_version=INPUT_VERSION,
                        input_hash=embedding_input_hash(text),
                        values=normalized,
                        blob=pack_vector(normalized),
                        prompt_tokens=tokens_each + int(position < token_remainder),
                    )
                )
            self._record_success()
            return vectors
        except EmbeddingUnavailable:
            raise
        except Exception as error:  # noqa: BLE001 - optional provider fails open
            self._record_failure()
            status = getattr(getattr(error, "response", None), "status_code", None)
            structured_log(
                logging.WARNING,
                "embeddings.request_failed",
                errorType=type(error).__name__,
                status=status,
            )
            raise EmbeddingUnavailable("embedding provider request failed") from error

    def embed_one(self, text: str) -> EmbeddingVector:
        return self.embed_many([text])[0]


class EmbeddingService:
    """Asynchronously archive vectors and annotate only proven-safe coalesces."""

    def __init__(
        self,
        store: EventStore,
        *,
        api_key: str,
        mode: str = "off",
        model: str = "qwen/qwen3-embedding-8b",
        dimensions: int = 512,
        threshold: float = 0.90,
        window_hours: int = 6,
        timeout_seconds: int = 8,
        wait_ms: int = 250,
        client: OpenRouterEmbeddingClient | None = None,
    ) -> None:
        self.store = store
        self.mode = mode if mode in {"off", "shadow", "coalesce"} else "off"
        self.model = model
        self.dimensions = int(dimensions)
        parsed_threshold = float(threshold)
        self.threshold = (
            parsed_threshold
            if math.isfinite(parsed_threshold) and 0.5 <= parsed_threshold <= 1.0
            else 1.0
        )
        self.window_hours = int(window_hours)
        self.wait_seconds = max(0.0, float(wait_ms) / 1000.0)
        self.enabled = self.mode != "off" and bool(api_key.strip())
        self.client = client or OpenRouterEmbeddingClient(
            api_key,
            model,
            dimensions,
            timeout_seconds=timeout_seconds,
        )
        self._executor = (
            ThreadPoolExecutor(
                max_workers=EMBEDDING_WORKERS,
                thread_name_prefix="embeddings",
            )
            if self.enabled
            else None
        )
        self._lock = threading.RLock()
        self._futures: dict[str, Future[EmbeddingVector]] = {}
        self._embedded = 0
        self._matches = 0
        self._failures = 0
        self._prompt_tokens = 0
        self._last_provider = ""

    @classmethod
    def from_config(cls, store: EventStore, config: Any) -> "EmbeddingService":
        # Dry-run promises no external calls or durable writes. It must stay
        # inert even if production's .env has coalescing enabled.
        mode = (
            "off"
            if bool(getattr(config, "dry_run", False))
            else str(getattr(config, "embedding_mode", "off"))
        )
        return cls(
            store,
            api_key=str(getattr(config, "openrouter_api_key", "")),
            mode=mode,
            model=str(
                getattr(config, "embedding_model", "qwen/qwen3-embedding-8b")
            ),
            dimensions=int(getattr(config, "embedding_dimensions", 512)),
            threshold=float(
                getattr(config, "embedding_similarity_threshold", 0.90)
            ),
            window_hours=int(getattr(config, "embedding_window_hours", 6)),
            timeout_seconds=int(getattr(config, "embedding_timeout_seconds", 8)),
            wait_ms=int(getattr(config, "embedding_wait_ms", 250)),
        )

    def _row_vector(self, item: NewsItem) -> EmbeddingVector | None:
        try:
            row = self.store.get(item)
            if not row or not row.get("embedding"):
                return None
            text = canonical_embedding_text(item)
            if (
                row.get("embedding_model") != self.model
                or int(row.get("embedding_dimensions") or 0) != self.dimensions
                or row.get("embedding_input_version") != INPUT_VERSION
                or row.get("embedding_input_hash") != embedding_input_hash(text)
            ):
                return None
            values = unpack_vector(bytes(row["embedding"]), dimensions=self.dimensions)
        except Exception:  # noqa: BLE001 - corrupt cache/database fails open
            return None
        return EmbeddingVector(
            model=self.model,
            provider=str(row.get("embedding_provider") or "openrouter"),
            dimensions=self.dimensions,
            input_version=INPUT_VERSION,
            input_hash=embedding_input_hash(text),
            values=values,
            blob=bytes(row["embedding"]),
        )

    def enqueue(self, item: NewsItem) -> None:
        if not self.enabled or self._executor is None:
            return
        if self._row_vector(item) is not None:
            return
        report_id = report_revision_identity(item)
        text = canonical_embedding_text(item)
        with self._lock:
            existing = self._futures.get(report_id)
            if existing is not None and not existing.cancelled():
                return
            if len(self._futures) >= MAX_PENDING_FUTURES:
                self._failures += 1
                return
            try:
                future = self._executor.submit(self.client.embed_one, text)
            except RuntimeError:
                self._failures += 1
                return
            self._futures[report_id] = future
        future.add_done_callback(
            lambda result, target=item: self._store_completed(target, result)
        )

    def _store_completed(self, item: NewsItem, future: Future[EmbeddingVector]) -> None:
        report_id = report_revision_identity(item)
        try:
            vector = future.result()
            stored = self.store.store_embedding(
                item,
                vector.model,
                vector.blob,
                provider=vector.provider,
                dimensions=vector.dimensions,
                input_version=vector.input_version,
                input_hash=vector.input_hash,
            )
            if not stored:
                raise RuntimeError("embedding row disappeared before storage")
            with self._lock:
                self._embedded += 1
                self._prompt_tokens += vector.prompt_tokens
                self._last_provider = vector.provider
            HEALTH.mark(
                "embeddings",
                ok=True,
                detail=f"{self.model} · {self.mode}",
            )
        except Exception as error:  # noqa: BLE001 - vectors never block alerts
            with self._lock:
                self._failures += 1
            HEALTH.mark(
                "embeddings",
                ok=False,
                detail=type(error).__name__,
            )
        finally:
            with self._lock:
                if self._futures.get(report_id) is future:
                    self._futures.pop(report_id, None)

    def _current_vector(self, item: NewsItem) -> EmbeddingVector | None:
        cached = self._row_vector(item)
        if cached is not None:
            return cached
        report_id = report_revision_identity(item)
        with self._lock:
            future = self._futures.get(report_id)
        if future is None:
            self.enqueue(item)
            with self._lock:
                future = self._futures.get(report_id)
        if future is None:
            return None
        try:
            return future.result(timeout=self.wait_seconds)
        except Exception:  # noqa: BLE001 - optional similarity fails open
            return None

    def annotate(
        self,
        alert: Alert,
        *,
        active_message_id: int = 0,
        active_alert_token: str = "",
    ) -> Alert:
        """Attach an edit hint when the vector and structured guard both agree."""
        if (
            not self.enabled
            or active_message_id <= 0
            or not active_alert_token
        ):
            return alert
        current = self._current_vector(alert.item)
        if current is None:
            return alert
        try:
            rows = self.store.recent_embedded_for_player(
                alert.item.player_name,
                model=current.model,
                dimensions=current.dimensions,
                input_version=current.input_version,
                exclude_report_id=report_revision_identity(alert.item),
                active_message_id=active_message_id,
                active_alert_token=active_alert_token,
                since_hours=self.window_hours,
                limit=20,
            )
        except Exception:  # noqa: BLE001 - optional similarity fails open
            return alert

        candidates: list[EmbeddingMatch] = []
        for row in rows:
            try:
                previous_item = news_item_from_row(row)
                previous_text = canonical_embedding_text(previous_item)
                if (
                    row.get("embedding_provider") != current.provider
                    or not row.get("embedding_at")
                    or row.get("embedding_input_hash")
                    != embedding_input_hash(previous_text)
                ):
                    continue
                previous = unpack_vector(
                    bytes(row.get("embedding") or b""),
                    dimensions=current.dimensions,
                )
                score = cosine_similarity(current.values, previous)
                safe, reason = embedding_transition_guard(
                    alert,
                    row,
                    score=score,
                    threshold=self.threshold,
                )
            except Exception:  # noqa: BLE001 - one corrupt candidate fails open
                continue
            candidates.append(
                EmbeddingMatch(
                    row=row,
                    score=score,
                    reason=reason if safe else f"blocked:{reason}",
                )
            )

        if not candidates:
            return alert
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        top = candidates[0]
        best_safe = top if top.reason == "safe_paraphrase" else None
        structured_log(
            logging.INFO if best_safe is not None else logging.DEBUG,
            "embeddings.similarity_checked",
            player=alert.item.player_name,
            topScore=round(top.score, 4),
            topDecision=top.reason,
            safeScore=round(best_safe.score, 4) if best_safe is not None else None,
            mode=self.mode,
        )
        if best_safe is None or self.mode == "shadow":
            return alert
        try:
            message_id = int(best_safe.row.get("telegram_message_id") or 0)
        except (TypeError, ValueError):
            return alert
        token = str(best_safe.row.get("alert_token") or "")
        if message_id <= 0 or not token:
            return alert
        with self._lock:
            self._matches += 1
        return replace(
            alert,
            embedding_match_message_id=message_id,
            embedding_match_token=token,
            embedding_similarity=best_safe.score,
            embedding_model=current.model,
        )

    def status(self) -> EmbeddingServiceStatus:
        with self._lock:
            return EmbeddingServiceStatus(
                enabled=self.enabled,
                mode=self.mode,
                model=self.model,
                dimensions=self.dimensions,
                embedded=self._embedded,
                matches=self._matches,
                failures=self._failures,
                prompt_tokens=self._prompt_tokens,
                last_provider=self._last_provider,
            )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
