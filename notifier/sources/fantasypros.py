"""Budgeted, cached FantasyPros consensus-ranking snapshots.

Breaking news never waits for this module.  ``refresh`` is intended for one
background worker, while ``lookup`` and ``signal`` are memory-only reads that
perform no HTTP calls.  FantasyPros is therefore supporting evidence for a
Sleeper-derived pickup candidate, not a trigger or veto for an alert.

The public API permits bulk consensus-ranking requests.  Four snapshots cover
the formats used by the notifier: PPR/HALF x WAIVER/ROS.  A persistent rolling
request ledger is reserved *before* every HTTP request so restarts and failed
requests cannot accidentally evade the configured daily budget.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..logging_utils import structured_log
from ..matcher import compact_key

BASE_URL = "https://api.fantasypros.com/public/v2/json"
CONSENSUS_PATH = "nfl/{season}/consensus-rankings"
CACHE_FILENAME = "fantasypros-cache.json"
CACHE_VERSION = 1

PROVIDER_DAILY_LIMIT = 500
DEFAULT_APP_DAILY_CAP = 425
DEFAULT_REFRESH_SECONDS = 2 * 60 * 60
DEFAULT_MAX_STALE_SECONDS = 12 * 60 * 60
MIN_REQUEST_INTERVAL_SECONDS = 1.05
ROLLING_WINDOW_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 20
CORPUS_REQUEST_BUCKET = "corpus"

SUPPORTED_SCORING = ("PPR", "HALF", "STD")
DEFAULT_SCORING = ("PPR", "HALF")
RANKING_TYPES = ("WAIVER", "ROS")
SOURCE_NAME = "FantasyPros consensus rankings"

# FantasyPros' v2 OpenAPI exposes both ``WW`` and ``WAIVER`` for waiver-wire
# consensus rankings.  ``WW`` is the provider's longstanding request/response
# code; keep the clearer ``WAIVER`` label inside our cache and public API.
_PROVIDER_RANKING_TYPE = {
    "WAIVER": "WW",
    "ROS": "ROS",
}
_RESPONSE_RANKING_TYPES = {
    "WAIVER": frozenset({"WW", "WAIVER"}),
    "ROS": frozenset({"ROS"}),
}
_KNOWN_PROVIDER_RANKING_TYPES = frozenset(
    {
        "WW",
        "WAIVER",
        "ROS",
        "DRAFT",
        "PRESEASON",
        "SLEEPERS",
        "ADP",
        "BEST",
        "PROSPECT",
        "PRO",
        "DEVY",
        "ROOKIES",
        "DYNADP",
        "RKADP",
        "BESTADP",
        "DYNASTY",
        "PRE",
        "DRAFTERS",
        "MOCK",
    }
)

_TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WSH": "WAS",
    "WSN": "WAS",
}
_POSITION_ALIASES = {
    "D/ST": "DST",
    "DEF": "DST",
}


@dataclass(frozen=True)
class FantasyProsRanking:
    """One fresh ranking row with enough metadata for visible attribution."""

    player_id: int | None
    player_name: str
    team: str
    position: str
    scoring: str
    ranking_type: str
    rank: int
    pos_rank: str
    tier: int | None
    owned_espn: float | None
    owned_yahoo: float | None
    season: int
    updated_at: datetime
    fetched_at: datetime
    source: str = SOURCE_NAME
    source_url: str = ""

    @property
    def attribution(self) -> str:
        return (
            f"{self.source} · {self.scoring} {self.ranking_type} · "
            f"updated {self.updated_at.isoformat()}"
        )


@dataclass(frozen=True)
class FantasyProsSignal:
    """Combined WAIVER/ROS evidence for one player and scoring format."""

    player_name: str
    team: str
    position: str
    scoring: str
    waiver_rank: int | None
    waiver_pos_rank: str
    ros_rank: int | None
    ros_pos_rank: str
    updated_at: datetime
    fetched_at: datetime
    waiver_updated_at: datetime | None = None
    ros_updated_at: datetime | None = None
    source: str = SOURCE_NAME
    source_url: str = ""

    @property
    def attribution(self) -> str:
        return (
            f"{self.source} · {self.scoring} · "
            f"updated {self.updated_at.isoformat()}"
        )


@dataclass(frozen=True)
class FantasyProsStatus:
    """Secret-free operational state suitable for logs or a status command."""

    enabled: bool
    closed: bool
    refreshing: bool
    ledger_trusted: bool
    requests_used: int
    request_cap: int
    datasets_cached: tuple[str, ...]
    datasets_fresh: tuple[str, ...]
    last_success_at: datetime | None
    last_error: str
    next_refresh_in_seconds: float


_REQUEST_ERROR_CODES = frozenset(
    {
        "budget_exhausted",
        "budget_reserved",
        "cache_write_failed",
        "closed",
        "dataset_unavailable",
        "invalid_request",
        "invalid_response",
        "ledger_untrusted",
        "request_failed",
        "request_limit",
    }
)


class FantasyProsRequestError(RuntimeError):
    """A public provider error that contains only a stable, secret-free code."""

    def __init__(self, code: str, *, request_reserved: bool = False) -> None:
        safe_code = code if code in _REQUEST_ERROR_CODES else "request_failed"
        super().__init__(safe_code)
        self.code = safe_code
        self.request_reserved = bool(request_reserved)


class _RefreshFailure(FantasyProsRequestError):
    """An intentionally sanitized refresh failure."""


class _DatasetUnavailable(ValueError):
    """The provider returned a valid envelope but not the requested dataset."""


def _dataset_key(scoring: str, ranking_type: str) -> str:
    return f"{scoring}:{ranking_type}"


def _normalize_scoring(value: str) -> str:
    candidate = str(value or "").strip().upper()
    if candidate not in SUPPORTED_SCORING:
        raise ValueError(f"Unsupported FantasyPros scoring format: {candidate or 'empty'}")
    return candidate


def _normalize_ranking_type(value: str) -> str:
    candidate = str(value or "").strip().upper()
    if candidate not in RANKING_TYPES:
        raise ValueError(f"Unsupported FantasyPros ranking type: {candidate or 'empty'}")
    return candidate


def _normalize_team(value: Any) -> str:
    candidate = str(value or "").strip().upper()
    return _TEAM_ALIASES.get(candidate, candidate)


def _normalize_position(value: Any) -> str:
    candidate = str(value or "").strip().upper()
    return _POSITION_ALIASES.get(candidate, candidate)


def _optional_int(value: Any, *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    result = int(number)
    if positive and result <= 0:
        return None
    return result


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _request_bucket_name(value: Any) -> str | None:
    """Return one safe persistent-ledger bucket name, or ``None``."""
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        return None
    if not value[0].isascii() or not value[0].isalnum():
        return None
    if not all(
        character.isascii()
        and (character.isalnum() or character in {"_", "-", "."})
        for character in value
    ):
        return None
    return value


def _utc_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, timezone.utc)


class FantasyProsCache:
    """Thread-safe persistent cache for bulk FantasyPros ranking snapshots.

    Passing an empty API key disables refreshes but still permits reads from an
    existing cache.  An injected session is borrowed and is not closed by this
    object; an internally-created session is owned and closed by ``close``.
    """

    def __init__(
        self,
        state_dir: Path,
        api_key: str,
        season: int,
        *,
        session: Any | None = None,
        app_daily_cap: int = DEFAULT_APP_DAILY_CAP,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        max_stale_seconds: float = DEFAULT_MAX_STALE_SECONDS,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(app_daily_cap, bool) or not 1 <= app_daily_cap < PROVIDER_DAILY_LIMIT:
            raise ValueError(
                f"FantasyPros app_daily_cap must be between 1 and "
                f"{PROVIDER_DAILY_LIMIT - 1}"
            )
        if refresh_seconds <= 0 or max_stale_seconds <= 0 or request_timeout <= 0:
            raise ValueError("FantasyPros time intervals must be positive")

        self.cache_path = Path(state_dir) / CACHE_FILENAME
        self._api_key = str(api_key or "").strip()
        self._season = int(season)
        self._app_daily_cap = int(app_daily_cap)
        self._refresh_seconds = float(refresh_seconds)
        self._max_stale_seconds = float(max_stale_seconds)
        # Never permit a configuration override to violate the provider's
        # one-request-per-second boundary.
        self._min_request_interval = max(
            MIN_REQUEST_INTERVAL_SECONDS,
            float(min_request_interval),
        )
        self._request_timeout = float(request_timeout)
        self._clock = clock
        self._sleep = sleep
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None

        self._state_lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._datasets: dict[str, dict[str, Any]] = {}
        self._request_times: list[float] = []
        self._request_buckets: dict[str, list[float]] = {}
        self._ledger_trusted = True
        self._refreshing = False
        self._closed = False
        self._session_closed = False
        self._last_error = ""
        self._load_cache()

    @property
    def enabled(self) -> bool:
        with self._state_lock:
            return bool(self._api_key) and not self._closed and self._ledger_trusted

    @property
    def season(self) -> int:
        return self._season

    @property
    def app_daily_cap(self) -> int:
        return self._app_daily_cap

    def _recent_requests(self, now: float) -> list[float]:
        cutoff = now - ROLLING_WINDOW_SECONDS
        return [timestamp for timestamp in self._request_times if timestamp > cutoff]

    def _recent_bucket_requests(self, bucket: str, now: float) -> list[float]:
        cutoff = now - ROLLING_WINDOW_SECONDS
        return [
            timestamp
            for timestamp in self._request_buckets.get(bucket, ())
            if timestamp > cutoff
        ]

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._ledger_trusted = False
            self._last_error = "cache_unreadable"
            structured_log(
                logging.WARNING,
                "fantasypros.cache_unreadable",
                reason="invalid_json_or_io",
            )
            return

        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            self._ledger_trusted = False
            self._last_error = "cache_version_invalid"
            structured_log(
                logging.WARNING,
                "fantasypros.cache_unreadable",
                reason="invalid_version",
            )
            return

        now = self._clock()
        raw_requests = payload.get("requests")
        parsed_requests: list[float] = []
        if not isinstance(raw_requests, list):
            self._ledger_trusted = False
            self._last_error = "ledger_invalid"
        else:
            for raw_timestamp in raw_requests:
                timestamp = _optional_float(raw_timestamp)
                if timestamp is None or timestamp <= 0 or timestamp > now + 60:
                    self._ledger_trusted = False
                    self._last_error = "ledger_invalid"
                    parsed_requests = []
                    break
                parsed_requests.append(timestamp)
            if self._ledger_trusted:
                self._request_times = sorted(
                    timestamp
                    for timestamp in parsed_requests
                    if timestamp > now - ROLLING_WINDOW_SECONDS
                )

        # Version-1 ledgers predate named request buckets, so a missing field is
        # an empty, valid bucket map.  Once present, every bucket reservation
        # must also exist in the authoritative global request ledger.
        raw_buckets = payload.get("request_buckets", {})
        parsed_buckets: dict[str, list[float]] = {}
        if not isinstance(raw_buckets, dict):
            self._ledger_trusted = False
            self._last_error = "ledger_invalid"
        elif self._ledger_trusted:
            global_counts = Counter(parsed_requests)
            bucket_counts: Counter[float] = Counter()
            for raw_name, raw_timestamps in raw_buckets.items():
                bucket_name = _request_bucket_name(raw_name)
                if bucket_name is None or not isinstance(raw_timestamps, list):
                    self._ledger_trusted = False
                    self._last_error = "ledger_invalid"
                    break
                timestamps: list[float] = []
                for raw_timestamp in raw_timestamps:
                    timestamp = _optional_float(raw_timestamp)
                    if timestamp is None or timestamp <= 0 or timestamp > now + 60:
                        self._ledger_trusted = False
                        self._last_error = "ledger_invalid"
                        break
                    timestamps.append(timestamp)
                    bucket_counts[timestamp] += 1
                    if bucket_counts[timestamp] > global_counts[timestamp]:
                        self._ledger_trusted = False
                        self._last_error = "ledger_invalid"
                        break
                if not self._ledger_trusted:
                    break
                parsed_buckets[bucket_name] = sorted(
                    timestamp
                    for timestamp in timestamps
                    if timestamp > now - ROLLING_WINDOW_SECONDS
                )
            if self._ledger_trusted:
                self._request_buckets = parsed_buckets

        raw_datasets = payload.get("datasets")
        if isinstance(raw_datasets, dict):
            for raw_key, raw_dataset in raw_datasets.items():
                dataset = self._validate_cached_dataset(raw_dataset)
                if dataset is None:
                    continue
                expected_key = _dataset_key(
                    dataset["scoring"], dataset["ranking_type"]
                )
                if str(raw_key) == expected_key:
                    self._datasets[expected_key] = dataset

        if not self._ledger_trusted:
            structured_log(
                logging.WARNING,
                "fantasypros.ledger_untrusted",
                reason=self._last_error,
            )

    def _validate_cached_dataset(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        try:
            scoring = _normalize_scoring(raw.get("scoring"))
            ranking_type = _normalize_ranking_type(raw.get("ranking_type"))
        except ValueError:
            return None
        if _optional_int(raw.get("season")) != self._season:
            return None
        fetched_at = _optional_float(raw.get("fetched_at"))
        if fetched_at is None or fetched_at <= 0:
            return None
        source_updated_at = _optional_float(raw.get("source_updated_at"))
        raw_players = raw.get("players")
        if not isinstance(raw_players, list):
            return None

        players: list[dict[str, Any]] = []
        for raw_player in raw_players:
            player = self._validate_cached_player(raw_player)
            if player is not None:
                players.append(player)
        if not players:
            return None
        return {
            "season": self._season,
            "scoring": scoring,
            "ranking_type": ranking_type,
            "fetched_at": fetched_at,
            "source_updated_at": source_updated_at,
            "players": players,
        }

    @staticmethod
    def _validate_cached_player(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        name_key = compact_key(name)
        rank = _optional_int(raw.get("rank"), positive=True)
        if not name_key or rank is None:
            return None
        raw_positions = raw.get("positions")
        if not isinstance(raw_positions, list):
            raw_positions = []
        positions = [
            normalized
            for value in raw_positions
            if (normalized := _normalize_position(value))
        ]
        if not positions:
            fallback_position = _normalize_position(raw.get("position"))
            if fallback_position:
                positions = [fallback_position]
        return {
            "player_id": _optional_int(raw.get("player_id"), positive=True),
            "name": name,
            "name_key": name_key,
            "team": _normalize_team(raw.get("team")),
            "positions": list(dict.fromkeys(positions)),
            "rank": rank,
            "pos_rank": str(raw.get("pos_rank") or "").strip().upper(),
            "tier": _optional_int(raw.get("tier"), positive=True),
            "owned_espn": _optional_float(raw.get("owned_espn")),
            "owned_yahoo": _optional_float(raw.get("owned_yahoo")),
        }

    def _state_payload(
        self,
        *,
        datasets: dict[str, dict[str, Any]] | None = None,
        requests_: list[float] | None = None,
        request_buckets_: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        return {
            "version": CACHE_VERSION,
            "requests": list(self._request_times if requests_ is None else requests_),
            "request_buckets": {
                name: list(timestamps)
                for name, timestamps in (
                    self._request_buckets
                    if request_buckets_ is None
                    else request_buckets_
                ).items()
            },
            "datasets": dict(self._datasets if datasets is None else datasets),
        }

    def _write_state_locked(
        self,
        *,
        datasets: dict[str, dict[str, Any]] | None = None,
        requests_: list[float] | None = None,
        request_buckets_: dict[str, list[float]] | None = None,
    ) -> None:
        """Atomically persist sanitized state.  Caller must hold state lock."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.cache_path.name}.",
            suffix=".tmp",
            dir=self.cache_path.parent,
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    self._state_payload(
                        datasets=datasets,
                        requests_=requests_,
                        request_buckets_=request_buckets_,
                    ),
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _set_error(self, code: str) -> None:
        with self._state_lock:
            self._last_error = code

    def _reserve_request(
        self,
        *,
        request_ceiling: int | None = None,
        request_bucket: str = "",
        request_bucket_limit: int | None = None,
    ) -> float:
        """Pace and durably reserve one global and optional bucket request."""
        while True:
            with self._state_lock:
                if self._closed:
                    raise _RefreshFailure("closed")
                if not self._ledger_trusted:
                    raise _RefreshFailure("ledger_untrusted")
                now = self._clock()
                recent = self._recent_requests(now)
                if len(recent) >= self._app_daily_cap:
                    raise _RefreshFailure("budget_exhausted")
                if request_ceiling is not None and len(recent) >= request_ceiling:
                    raise _RefreshFailure("budget_reserved")
                recent_bucket = (
                    self._recent_bucket_requests(request_bucket, now)
                    if request_bucket
                    else []
                )
                if (
                    request_bucket_limit is not None
                    and len(recent_bucket) >= request_bucket_limit
                ):
                    raise _RefreshFailure("request_limit")
                last_request = max(recent, default=None)
                delay = (
                    self._min_request_interval - (now - last_request)
                    if last_request is not None
                    else 0.0
                )
                if delay <= 0:
                    reservation = now
                    reserved = [*recent, reservation]
                    reserved_buckets = {
                        name: timestamps
                        for name in self._request_buckets
                        if (
                            timestamps := self._recent_bucket_requests(name, now)
                        )
                    }
                    if request_bucket:
                        reserved_buckets[request_bucket] = [
                            *recent_bucket,
                            reservation,
                        ]
                    try:
                        self._write_state_locked(
                            requests_=reserved,
                            request_buckets_=reserved_buckets,
                        )
                    except OSError:
                        raise _RefreshFailure("cache_write_failed") from None
                    self._request_times = reserved
                    self._request_buckets = reserved_buckets
                    return reservation
            # Epoch floats have sub-microsecond resolution.  A calculated
            # remainder smaller than that resolution would otherwise sleep
            # without advancing the clock and spin forever.  One millisecond
            # of safety margin is negligible relative to the 1.05 s policy.
            self._sleep(max(delay, 0.001))

    def get_json(
        self,
        relative_path: str,
        *,
        params: Mapping[str, Any] | None = None,
        request_ceiling: int | None = None,
        request_bucket: str = "",
        request_bucket_limit: int | None = None,
    ) -> Any:
        """Perform one serialized, budgeted GET against the FantasyPros API.

        ``relative_path`` is deliberately confined to ``BASE_URL`` so callers
        cannot redirect the API-key header to another host. The request uses
        the same lock, durable rolling ledger, daily cap, and minimum cadence
        as ranking refreshes. Optional global and named-bucket ceilings are
        checked in the same critical section that persists the reservation.
        Provider and transport exception text is never propagated because it
        may contain request headers or credentials.
        """
        try:
            path = str(relative_path or "").strip()
        except Exception:
            raise FantasyProsRequestError("invalid_request") from None
        segments = path.split("/")
        if (
            not path
            or path.startswith("/")
            or any(segment in {"", ".", ".."} for segment in segments)
            or not all(
                character.isascii()
                and (character.isalnum() or character in {"/", "_", "-", "."})
                for character in path
            )
        ):
            raise FantasyProsRequestError("invalid_request")
        try:
            request_params = dict(params or {})
        except Exception:
            raise FantasyProsRequestError("invalid_request") from None
        normalized_ceiling = (
            _optional_int(request_ceiling, positive=True)
            if request_ceiling is not None
            else None
        )
        normalized_bucket = ""
        if isinstance(request_bucket, str) and request_bucket:
            normalized_bucket = _request_bucket_name(request_bucket) or ""
        normalized_bucket_limit = (
            _optional_int(request_bucket_limit, positive=True)
            if request_bucket_limit is not None
            else None
        )
        if (
            (request_ceiling is not None and normalized_ceiling is None)
            or (
                normalized_ceiling is not None
                and normalized_ceiling > self._app_daily_cap
            )
            or not isinstance(request_bucket, str)
            or (request_bucket and not normalized_bucket)
            or (request_bucket_limit is not None and normalized_bucket_limit is None)
            or (normalized_bucket_limit is not None and not normalized_bucket)
        ):
            raise FantasyProsRequestError("invalid_request")

        # Serialize every provider GET with ranking refreshes. This preserves
        # one request-start cadence and lets close() wait for active work.
        with self._refresh_lock:
            with self._state_lock:
                if self._closed or not self._api_key:
                    raise FantasyProsRequestError("closed")
            try:
                self._reserve_request(
                    request_ceiling=normalized_ceiling,
                    request_bucket=normalized_bucket,
                    request_bucket_limit=normalized_bucket_limit,
                )
            except _RefreshFailure as error:
                raise FantasyProsRequestError(error.code) from None
            except Exception:
                raise FantasyProsRequestError("request_failed") from None

            # Read transport details only after the durable reservation. They
            # remain local variables and are never copied into state or errors.
            with self._state_lock:
                if self._closed or not self._api_key:
                    raise FantasyProsRequestError("closed", request_reserved=True)
                api_key = self._api_key
                session = self._session
                timeout = self._request_timeout
            try:
                response = session.get(
                    f"{BASE_URL}/{path}",
                    headers={"x-api-key": api_key},
                    params=request_params,
                    timeout=timeout,
                )
                status_code = int(getattr(response, "status_code", 200))
            except Exception:
                raise FantasyProsRequestError(
                    "request_failed", request_reserved=True
                ) from None
            if not 200 <= status_code < 300:
                raise FantasyProsRequestError(
                    "request_failed", request_reserved=True
                )
            try:
                return response.json()
            except Exception:
                raise FantasyProsRequestError(
                    "invalid_response", request_reserved=True
                ) from None

    def _fetch_dataset(self, scoring: str, ranking_type: str) -> dict[str, Any]:
        self._reserve_request()
        url = f"{BASE_URL}/{CONSENSUS_PATH.format(season=self._season)}"
        provider_ranking_type = _PROVIDER_RANKING_TYPE[ranking_type]
        # Read the key only after the durable request reservation.  It is used
        # exclusively as an HTTP header and is never copied into state/errors.
        with self._state_lock:
            api_key = self._api_key
        try:
            response = self._session.get(
                url,
                headers={"x-api-key": api_key},
                params={
                    "position": "ALL",
                    "scoring": scoring,
                    "type": provider_ranking_type,
                },
                timeout=self._request_timeout,
            )
            status_code = int(getattr(response, "status_code", 200))
            if not 200 <= status_code < 300:
                raise _RefreshFailure("request_failed")
            payload = response.json()
        except _RefreshFailure:
            raise
        except Exception:
            # Do not propagate provider/session text: custom transports can put
            # headers in exception strings, including the API key.
            raise _RefreshFailure("request_failed") from None

        try:
            return self._parse_dataset(payload, scoring, ranking_type)
        except _DatasetUnavailable:
            raise _RefreshFailure("dataset_unavailable") from None
        except (TypeError, ValueError, OverflowError):
            raise _RefreshFailure("invalid_response") from None

    def _parse_dataset(
        self,
        payload: Any,
        scoring: str,
        ranking_type: str,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("invalid response root")
        # These fields are the official identity of an NFL rankings response.
        # Compare their semantic scalar values, but never infer or default a
        # missing field from the request: a mismatched response must not be
        # cached under the requested dataset key.
        text_identity = {
            "sport": "NFL",
            "scoring": scoring,
            "position_id": "ALL",
        }
        for field, expected in text_identity.items():
            if field not in payload:
                raise ValueError(f"missing {field}")
            reported = str(payload[field] or "").strip().upper()
            if reported != expected:
                raise ValueError(f"unexpected {field}")

        # The OpenAPI models ``year`` as a string, while other FantasyPros v2
        # endpoints and historical examples sometimes emit the same scalar as
        # an integer. Numeric equivalence is unambiguous; missing, fractional,
        # boolean, and mismatched seasons still fail closed.
        if _optional_int(payload.get("year"), positive=True) != self._season:
            raise ValueError("unexpected year")

        if "ranking_type_name" not in payload:
            raise ValueError("missing ranking_type_name")
        reported_ranking_type = str(
            payload.get("ranking_type_name") or ""
        ).strip().upper()
        if reported_ranking_type not in _RESPONSE_RANKING_TYPES[ranking_type]:
            if reported_ranking_type in _KNOWN_PROVIDER_RANKING_TYPES:
                # The live API can fall back to another valid ranking family
                # when the requested seasonal dataset is not published yet.
                # Recognize that state, but never relabel (for example) DRAFT
                # rows as ROS evidence.
                raise _DatasetUnavailable("requested ranking dataset unavailable")
            raise ValueError("unexpected ranking_type_name")
        raw_players = payload.get("players")
        if not isinstance(raw_players, list):
            raise ValueError("missing players")
        if not raw_players:
            # The current API represents an unpublished waiver dataset as a
            # successful, correctly identified envelope with an empty array.
            # That is not cacheable ranking evidence, but it is also not a
            # malformed transport response.
            raise _DatasetUnavailable("requested ranking dataset is empty")

        players: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for raw_player in raw_players:
            player = self._parse_api_player(raw_player)
            if player is None:
                continue
            identity: tuple[Any, ...]
            if player["player_id"] is not None:
                identity = ("id", player["player_id"])
            else:
                identity = (
                    "name",
                    player["name_key"],
                    player["team"],
                    tuple(player["positions"]),
                )
            if identity in seen:
                continue
            seen.add(identity)
            players.append(player)
        if not players:
            raise ValueError("no usable players")

        fetched_at = self._clock()
        source_updated_at = _optional_float(payload.get("last_updated_ts"))
        if (
            source_updated_at is None
            or source_updated_at <= 0
            or source_updated_at > fetched_at + 5 * 60
        ):
            source_updated_at = None
        return {
            "season": self._season,
            "scoring": scoring,
            "ranking_type": ranking_type,
            "fetched_at": fetched_at,
            "source_updated_at": source_updated_at,
            "players": players,
        }

    @staticmethod
    def _parse_api_player(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("player_name") or "").strip()
        name_key = compact_key(name)
        rank = _optional_int(raw.get("rank_ecr"), positive=True)
        if not name_key or rank is None:
            return None

        raw_positions: Any = (
            raw.get("player_positions") or raw.get("player_position_id") or ""
        )
        if isinstance(raw_positions, list):
            position_parts = raw_positions
        else:
            position_parts = str(raw_positions).split(",")
        positions = [
            normalized
            for value in position_parts
            if (normalized := _normalize_position(value))
        ]
        if not positions:
            pos_rank = str(raw.get("pos_rank") or "").strip().upper()
            inferred = "".join(character for character in pos_rank if character.isalpha())
            if inferred:
                positions = [_normalize_position(inferred)]
        return {
            "player_id": _optional_int(raw.get("player_id"), positive=True),
            "name": name,
            "name_key": name_key,
            "team": _normalize_team(raw.get("player_team_id")),
            "positions": list(dict.fromkeys(positions)),
            "rank": rank,
            "pos_rank": str(raw.get("pos_rank") or "").strip().upper(),
            "tier": _optional_int(raw.get("tier"), positive=True),
            "owned_espn": _optional_float(raw.get("player_owned_espn")),
            "owned_yahoo": _optional_float(raw.get("player_owned_yahoo")),
        }

    def _due_keys(self, scoring_formats: tuple[str, ...], force: bool) -> list[str]:
        now = self._clock()
        due: list[str] = []
        with self._state_lock:
            for scoring in scoring_formats:
                for ranking_type in RANKING_TYPES:
                    key = _dataset_key(scoring, ranking_type)
                    dataset = self._datasets.get(key)
                    fetched_at = (
                        _optional_float(dataset.get("fetched_at"))
                        if dataset is not None
                        else None
                    )
                    if (
                        force
                        or fetched_at is None
                        or now - fetched_at >= self._refresh_seconds
                    ):
                        due.append(key)
        return due

    def refresh(
        self,
        scoring_formats: Iterable[str] = DEFAULT_SCORING,
        *,
        force: bool = False,
    ) -> bool:
        """Refresh due snapshots, isolating known unpublished datasets.

        A valid provider envelope can explicitly report that one requested
        ranking family is not published.  That outcome is isolated to its
        dataset so the remaining bulk probes can still yield useful evidence.
        Transport, budget, persistence, and malformed-response failures remain
        batch-fatal: any snapshots collected before them are discarded.
        """
        requested = {_normalize_scoring(value) for value in scoring_formats}
        normalized = tuple(
            scoring
            for scoring in SUPPORTED_SCORING
            if scoring in requested
        )
        if not normalized:
            raise ValueError("At least one FantasyPros scoring format is required")

        with self._state_lock:
            if not self._api_key or self._closed or not self._ledger_trusted:
                return False

        with self._refresh_lock:
            with self._state_lock:
                if not self._api_key or self._closed or not self._ledger_trusted:
                    return False
                self._refreshing = True
            try:
                due_keys = self._due_keys(normalized, force)
                if not due_keys:
                    return True

                pending: dict[str, dict[str, Any]] = {}
                unavailable: list[str] = []
                for key in due_keys:
                    scoring, ranking_type = key.split(":", 1)
                    try:
                        pending[key] = self._fetch_dataset(scoring, ranking_type)
                    except _RefreshFailure as error:
                        if error.code != "dataset_unavailable":
                            raise
                        # Dataset keys are selected from fixed internal enums,
                        # so they are safe to expose in operational logs.
                        unavailable.append(key)

                result = (
                    "partial_dataset_unavailable"
                    if pending and unavailable
                    else "dataset_unavailable"
                    if unavailable
                    else ""
                )
                with self._state_lock:
                    if self._closed:
                        raise _RefreshFailure("closed")
                    if pending:
                        # Publish one cache generation only after every
                        # non-fatal probe finishes. Existing snapshots for an
                        # unavailable key remain untouched.
                        committed = {**self._datasets, **pending}
                        try:
                            self._write_state_locked(datasets=committed)
                        except OSError:
                            raise _RefreshFailure("cache_write_failed") from None
                        self._datasets = committed
                    self._last_error = result

                if unavailable:
                    structured_log(
                        logging.WARNING,
                        "fantasypros.refresh_incomplete",
                        reason=result,
                        datasetCount=len(pending),
                        unavailableCount=len(unavailable),
                        unavailableDatasets=tuple(unavailable),
                        requestUsage=self.request_usage(),
                        requestCap=self._app_daily_cap,
                    )
                    return False

                structured_log(
                    logging.INFO,
                    "fantasypros.cache_refreshed",
                    datasetCount=len(pending),
                    requestUsage=self.request_usage(),
                    requestCap=self._app_daily_cap,
                )
                return True
            except _RefreshFailure as error:
                self._set_error(error.code)
                structured_log(
                    logging.WARNING,
                    "fantasypros.refresh_failed",
                    reason=error.code,
                    requestUsage=self.request_usage(),
                    requestCap=self._app_daily_cap,
                )
                return False
            finally:
                with self._state_lock:
                    self._refreshing = False

    def _freshness_timestamp(self, dataset: dict[str, Any]) -> float | None:
        # Never make an old provider snapshot appear current merely because it
        # was downloaded again. Missing ``last_updated_ts`` means the ranking
        # can stay cached for diagnostics but cannot annotate an alert.
        return _optional_float(dataset.get("source_updated_at"))

    def _dataset_is_fresh(self, dataset: dict[str, Any], now: float) -> bool:
        as_of = self._freshness_timestamp(dataset)
        return as_of is not None and max(0.0, now - as_of) <= self._max_stale_seconds

    def _lookup_locked(
        self,
        player_name: str,
        *,
        scoring: str,
        ranking_type: str,
        team: str,
        position: str,
        now: float,
    ) -> FantasyProsRanking | None:
        """Resolve one row while the caller holds ``_state_lock``."""
        name_key = compact_key(player_name)
        if not name_key:
            return None
        dataset = self._datasets.get(_dataset_key(scoring, ranking_type))
        if dataset is None or not self._dataset_is_fresh(dataset, now):
            return None
        candidates = [
            candidate
            for candidate in dataset["players"]
            if candidate["name_key"] == name_key
        ]
        if not candidates:
            return None

        wanted_team = _normalize_team(team)
        if wanted_team:
            # Secondary-provider lag is not enough evidence to attach a rank
            # to a differently-teamed player. Omit the optional annotation.
            candidates = [
                candidate
                for candidate in candidates
                if candidate["team"] == wanted_team
            ]
            if not candidates:
                return None

        wanted_position = _normalize_position(position)
        if wanted_position:
            candidates = [
                candidate
                for candidate in candidates
                if wanted_position in candidate["positions"]
            ]
            if not candidates:
                return None

        if len(candidates) != 1:
            return None
        player = candidates[0]
        source_timestamp = self._freshness_timestamp(dataset)
        fetched_timestamp = _optional_float(dataset.get("fetched_at"))
        if source_timestamp is None or fetched_timestamp is None:
            return None
        return FantasyProsRanking(
            player_id=player["player_id"],
            player_name=player["name"],
            team=player["team"],
            position=(player["positions"][0] if player["positions"] else ""),
            scoring=scoring,
            ranking_type=ranking_type,
            rank=player["rank"],
            pos_rank=player["pos_rank"],
            tier=player["tier"],
            owned_espn=player["owned_espn"],
            owned_yahoo=player["owned_yahoo"],
            season=self._season,
            updated_at=_utc_datetime(source_timestamp),
            fetched_at=_utc_datetime(fetched_timestamp),
            source_url=(
                f"{BASE_URL}/{CONSENSUS_PATH.format(season=self._season)}"
            ),
        )

    def lookup(
        self,
        player_name: str,
        *,
        scoring: str,
        ranking_type: str,
        team: str = "",
        position: str = "",
    ) -> FantasyProsRanking | None:
        """Read one cached ranking without network, disk, or lock waits."""
        scoring = _normalize_scoring(scoring)
        ranking_type = _normalize_ranking_type(ranking_type)
        # Refresh holds this lock while durably reserving quota and atomically
        # publishing a completed batch. Alert processing must never queue
        # behind that fsync: contention simply means no optional annotation.
        if not self._state_lock.acquire(False):
            return None
        try:
            return self._lookup_locked(
                player_name,
                scoring=scoring,
                ranking_type=ranking_type,
                team=team,
                position=position,
                now=self._clock(),
            )
        finally:
            self._state_lock.release()

    @staticmethod
    def _same_player(
        first: FantasyProsRanking,
        second: FantasyProsRanking,
    ) -> bool:
        """Confirm two ranking rows before combining their evidence."""
        if first.player_id is not None and second.player_id is not None:
            return first.player_id == second.player_id
        first_team = _normalize_team(first.team)
        second_team = _normalize_team(second.team)
        first_position = _normalize_position(first.position)
        second_position = _normalize_position(second.position)
        return bool(first_team and first_position) and (
            compact_key(first.player_name) == compact_key(second.player_name)
            and first_team == second_team
            and first_position == second_position
        )

    def signal(
        self,
        player_name: str,
        *,
        scoring: str,
        team: str = "",
        position: str = "",
    ) -> FantasyProsSignal | None:
        """Combine one cache generation without network, disk, or lock waits."""
        scoring = _normalize_scoring(scoring)
        # Resolve both rows while holding one nonblocking read lock so a
        # background refresh cannot publish between WAIVER and ROS.
        if not self._state_lock.acquire(False):
            return None
        try:
            now = self._clock()
            waiver = self._lookup_locked(
                player_name,
                scoring=scoring,
                ranking_type="WAIVER",
                team=team,
                position=position,
                now=now,
            )
            ros = self._lookup_locked(
                player_name,
                scoring=scoring,
                ranking_type="ROS",
                team=team,
                position=position,
                now=now,
            )
        finally:
            self._state_lock.release()
        if waiver is not None and ros is not None and not self._same_player(waiver, ros):
            return None
        available = [entry for entry in (waiver, ros) if entry is not None]
        if not available:
            return None
        identity = waiver or ros
        assert identity is not None
        return FantasyProsSignal(
            player_name=identity.player_name,
            team=identity.team,
            position=identity.position,
            scoring=scoring,
            waiver_rank=waiver.rank if waiver else None,
            waiver_pos_rank=waiver.pos_rank if waiver else "",
            ros_rank=ros.rank if ros else None,
            ros_pos_rank=ros.pos_rank if ros else "",
            # The oldest contributing timestamp is the conservative single
            # "as of" date for the combined annotation.
            updated_at=min(entry.updated_at for entry in available),
            fetched_at=min(entry.fetched_at for entry in available),
            waiver_updated_at=waiver.updated_at if waiver else None,
            ros_updated_at=ros.updated_at if ros else None,
            source_url=identity.source_url,
        )

    def request_usage(self, *, bucket: str = "") -> int:
        """Return rolling global usage or one safe named-bucket count."""
        with self._state_lock:
            now = self._clock()
            if not bucket:
                return len(self._recent_requests(now))
            normalized = _request_bucket_name(bucket)
            if normalized is None:
                raise ValueError("Invalid FantasyPros request bucket")
            return len(self._recent_bucket_requests(normalized, now))

    def seconds_until_refresh(
        self,
        scoring_formats: Iterable[str] = DEFAULT_SCORING,
    ) -> float:
        requested = tuple(_normalize_scoring(value) for value in scoring_formats)
        with self._state_lock:
            if self._closed or not self._api_key or not self._ledger_trusted:
                return math.inf
            now = self._clock()
            remaining: list[float] = []
            for scoring in requested:
                for ranking_type in RANKING_TYPES:
                    dataset = self._datasets.get(_dataset_key(scoring, ranking_type))
                    if dataset is None:
                        return 0.0
                    fetched_at = _optional_float(dataset.get("fetched_at"))
                    if fetched_at is None:
                        return 0.0
                    remaining.append(
                        max(0.0, self._refresh_seconds - (now - fetched_at))
                    )
            return min(remaining, default=0.0)

    def status(self) -> FantasyProsStatus:
        with self._state_lock:
            now = self._clock()
            cached = tuple(sorted(self._datasets))
            fresh = tuple(
                key
                for key in cached
                if self._dataset_is_fresh(self._datasets[key], now)
            )
            fetched = [
                timestamp
                for dataset in self._datasets.values()
                if (timestamp := _optional_float(dataset.get("fetched_at"))) is not None
            ]
            return FantasyProsStatus(
                enabled=bool(self._api_key) and not self._closed and self._ledger_trusted,
                closed=self._closed,
                refreshing=self._refreshing,
                ledger_trusted=self._ledger_trusted,
                requests_used=len(self._recent_requests(now)),
                request_cap=self._app_daily_cap,
                datasets_cached=cached,
                datasets_fresh=fresh,
                last_success_at=_utc_datetime(max(fetched)) if fetched else None,
                last_error=self._last_error,
                next_refresh_in_seconds=self.seconds_until_refresh(),
            )

    def close(self) -> None:
        """Stop future work and safely close an internally-owned session."""
        with self._state_lock:
            if self._closed and self._session_closed:
                return
            self._closed = True
        # An active refresh owns this lock.  Marking closed first makes it stop
        # before its next request; waiting here prevents closing a live session.
        with self._refresh_lock:
            with self._state_lock:
                self._api_key = ""
                if self._session_closed:
                    return
                self._session_closed = True
            if self._owns_session:
                self._session.close()


# The alias reads naturally where the object is treated as an external client,
# while retaining ``FantasyProsCache`` for pipeline integration.
FantasyProsClient = FantasyProsCache
