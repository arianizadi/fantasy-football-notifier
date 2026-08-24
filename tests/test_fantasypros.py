from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from notifier.sources.fantasypros import (
    CACHE_FILENAME,
    MIN_REQUEST_INTERVAL_SECONDS,
    FantasyProsCache,
    FantasyProsSignal,
)
from notifier.models import LeagueRef
from notifier.pipeline import (
    Notifier,
    _fantasypros_failure_delay,
    _fantasypros_retry_delay,
)
from notifier.plays import Beneficiary, LeaguePlays


class FakeClock:
    def __init__(self, now: float = 2_000_000_000.0) -> None:
        self._now = now
        self._lock = threading.Lock()
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self._now += seconds

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(
        self,
        clock: FakeClock,
        *,
        fail_on_call: int | None = None,
        rank_offset: int = 0,
        source_age: float = 0,
        players: list[dict[str, Any]] | None = None,
        players_by_type: dict[str, list[dict[str, Any]]] | None = None,
        ranking_type_overrides: dict[str, str | None] | None = None,
        response_overrides: dict[str, Any] | None = None,
        omitted_response_fields: set[str] | None = None,
        raise_secret: str = "",
    ) -> None:
        self.clock = clock
        self.fail_on_call = fail_on_call
        self.rank_offset = rank_offset
        self.source_age = source_age
        self.players = players
        self.players_by_type = players_by_type or {}
        self.ranking_type_overrides = ranking_type_overrides or {}
        self.response_overrides = response_overrides or {}
        self.omitted_response_fields = omitted_response_fields or set()
        self.raise_secret = raise_secret
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._lock = threading.Lock()

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        with self._lock:
            call_number = len(self.calls) + 1
            self.calls.append({"url": url, "started": self.clock(), **kwargs})
        if self.raise_secret:
            raise RuntimeError(self.raise_secret)
        if self.fail_on_call == call_number:
            return FakeResponse({"message": "not available"}, status_code=500)

        params = kwargs["params"]
        provider_ranking_type = params["type"]
        ranking_type = (
            "WAIVER"
            if provider_ranking_type in {"WW", "WAIVER"}
            else provider_ranking_type
        )
        scoring = params["scoring"]
        type_offset = 0 if ranking_type == "WAIVER" else 100
        scoring_offset = 0 if scoring == "PPR" else 10
        default_players = (
            self.players
            if self.players is not None
            else [
                {
                    "player_id": 101,
                    "player_name": "DJ Moore",
                    "player_team_id": "CHI",
                    "player_position_id": "WR",
                    "player_positions": "WR",
                    "rank_ecr": 7 + type_offset + scoring_offset + self.rank_offset,
                    "pos_rank": f"WR{3 + scoring_offset}",
                    "tier": 2,
                    "player_owned_espn": 73.5,
                    "player_owned_yahoo": 75.0,
                }
            ]
        )
        players = self.players_by_type.get(ranking_type, default_players)
        payload = {
            "sport": "NFL",
            "year": "2026",
            "scoring": scoring,
            "position_id": params["position"],
            "ranking_type_name": self.ranking_type_overrides.get(
                ranking_type, provider_ranking_type
            ),
            "last_updated_ts": int(self.clock() - self.source_age),
            "players": players,
        }
        payload.update(self.response_overrides)
        for field in self.omitted_response_fields:
            payload.pop(field, None)
        return FakeResponse(payload)

    def close(self) -> None:
        self.closed = True


def _client(
    state_dir: Path,
    clock: FakeClock,
    session: FakeSession,
    **kwargs: Any,
) -> FantasyProsCache:
    return FantasyProsCache(
        state_dir,
        "fp-test-secret-value",
        2026,
        session=session,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_refresh_fetches_four_bulk_snapshots_with_pacing_and_safe_cache(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(force=True) is True

    assert len(session.calls) == 4
    assert [call["params"] for call in session.calls] == [
        {"position": "ALL", "scoring": "PPR", "type": "WW"},
        {"position": "ALL", "scoring": "PPR", "type": "ROS"},
        {"position": "ALL", "scoring": "HALF", "type": "WW"},
        {"position": "ALL", "scoring": "HALF", "type": "ROS"},
    ]
    starts = [call["started"] for call in session.calls]
    assert all(
        later - earlier >= MIN_REQUEST_INTERVAL_SECONDS
        for earlier, later in zip(starts, starts[1:])
    )
    assert all(call["headers"] == {"x-api-key": "fp-test-secret-value"} for call in session.calls)

    signal = cache.signal(
        "D.J. Moore Jr.", scoring="PPR", team="CHI", position="WR"
    )
    assert signal is not None
    assert signal.waiver_rank == 7
    assert signal.waiver_pos_rank == "WR3"
    assert signal.ros_rank == 107
    assert signal.source == "FantasyPros consensus rankings"
    assert "FantasyPros" in signal.attribution

    payload_text = (tmp_path / CACHE_FILENAME).read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    assert "fp-test-secret-value" not in payload_text
    assert len(payload["requests"]) == 4
    assert set(payload["datasets"]) == {
        "PPR:WAIVER",
        "PPR:ROS",
        "HALF:WAIVER",
        "HALF:ROS",
    }


def test_cache_remains_readable_without_a_key_and_lookup_never_fetches(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    writer_session = FakeSession(clock)
    writer = _client(tmp_path, clock, writer_session)
    assert writer.refresh(force=True)
    calls_after_refresh = len(writer_session.calls)

    assert writer.lookup(
        "DJ Moore", scoring="PPR", ranking_type="WAIVER"
    ) is not None
    assert writer.signal("DJ Moore", scoring="PPR") is not None
    assert len(writer_session.calls) == calls_after_refresh

    reader_session = FakeSession(clock)
    reader = FantasyProsCache(
        tmp_path,
        "",
        2026,
        session=reader_session,
        clock=clock,
        sleep=clock.sleep,
    )
    assert reader.enabled is False
    assert reader.refresh(force=True) is False
    assert reader.signal("DJ Moore", scoring="PPR") is not None
    assert reader_session.calls == []


def test_provider_staleness_suppresses_signal_without_network_call(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock, source_age=13 * 60 * 60)
    cache = _client(tmp_path, clock, session, max_stale_seconds=12 * 60 * 60)

    assert cache.refresh(force=True)
    request_count = len(session.calls)

    assert cache.lookup(
        "DJ Moore", scoring="PPR", ranking_type="WAIVER"
    ) is None
    assert cache.signal("DJ Moore", scoring="PPR") is None
    assert len(session.calls) == request_count
    assert cache.status().datasets_fresh == ()


def test_missing_provider_timestamp_never_uses_fetch_time_as_freshness(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    payload = session.get(
        "unused",
        params={"position": "ALL", "type": "WAIVER", "scoring": "PPR"},
        headers={},
    ).json()
    payload.pop("last_updated_ts")
    dataset = cache._parse_dataset(payload, "PPR", "WAIVER")
    with cache._state_lock:
        cache._datasets["PPR:WAIVER"] = dataset

    assert cache.lookup(
        "DJ Moore", scoring="PPR", ranking_type="WAIVER"
    ) is None


@pytest.mark.parametrize("reported_type", ["Rest of Season", None])
def test_response_ranking_type_must_match_the_requested_dataset(
    tmp_path: Path,
    reported_type: str | None,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        ranking_type_overrides={"WAIVER": reported_type},
    )
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is False
    assert len(session.calls) == 1
    assert cache.status().datasets_cached == ()
    assert cache.status().last_error == "invalid_response"


@pytest.mark.parametrize("reported_type", ["ROS", "draft"])
def test_known_provider_fallback_does_not_block_valid_sibling_dataset(
    tmp_path: Path,
    reported_type: str,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        ranking_type_overrides={"WAIVER": reported_type},
    )
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is False
    assert len(session.calls) == 2
    assert cache.status().datasets_cached == ("PPR:ROS",)
    assert cache.status().last_error == "partial_dataset_unavailable"

    on_disk = json.loads((tmp_path / CACHE_FILENAME).read_text(encoding="utf-8"))
    assert set(on_disk["datasets"]) == {"PPR:ROS"}


def test_current_empty_waiver_envelope_is_unavailable_and_never_cached(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        players=[],
        ranking_type_overrides={"WAIVER": "waiver"},
        response_overrides={"last_updated_ts": None},
    )
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is False
    assert len(session.calls) == 2
    assert cache.status().datasets_cached == ()
    assert cache.status().last_error == "dataset_unavailable"


def test_unpublished_waivers_allow_all_four_bulk_probes_and_cache_both_ros(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock, players_by_type={"WAIVER": []})
    cache = _client(tmp_path, clock, session)

    with caplog.at_level(logging.WARNING, logger="fantasy-news-notifier"):
        assert cache.refresh(force=True) is False

    assert len(session.calls) == 4
    assert cache.status().datasets_cached == ("HALF:ROS", "PPR:ROS")
    assert cache.status().last_error == "partial_dataset_unavailable"
    assert '"event":"fantasypros.refresh_incomplete"' in caplog.text
    assert '"reason":"partial_dataset_unavailable"' in caplog.text
    assert '"unavailableCount":2' in caplog.text
    assert "fp-test-secret-value" not in caplog.text


def test_partial_refresh_updates_valid_dataset_and_preserves_old_unavailable_one(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    first = _client(tmp_path, clock, FakeSession(clock))
    assert first.refresh(("PPR",), force=True) is True

    second_session = FakeSession(
        clock,
        rank_offset=900,
        players_by_type={"WAIVER": []},
    )
    second = _client(tmp_path, clock, second_session)

    assert second.refresh(("PPR",), force=True) is False
    assert second.status().last_error == "partial_dataset_unavailable"
    assert second.status().datasets_cached == ("PPR:ROS", "PPR:WAIVER")

    waiver = second.lookup("DJ Moore", scoring="PPR", ranking_type="WAIVER")
    ros = second.lookup("DJ Moore", scoring="PPR", ranking_type="ROS")
    assert waiver is not None and waiver.rank == 7
    assert ros is not None and ros.rank == 1007

    on_disk = json.loads((tmp_path / CACHE_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["datasets"]["PPR:WAIVER"]["players"][0]["rank"] == 7
    assert on_disk["datasets"]["PPR:ROS"]["players"][0]["rank"] == 1007


def test_hard_failure_after_unavailable_dataset_aborts_without_partial_commit(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        fail_on_call=2,
        players_by_type={"WAIVER": []},
    )
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is False
    assert len(session.calls) == 2
    assert cache.status().datasets_cached == ()
    assert cache.status().last_error == "request_failed"


@pytest.mark.parametrize("reported_type", ["WW", "WAIVER", "ww", "waiver"])
def test_documented_waiver_type_aliases_map_to_the_internal_dataset(
    tmp_path: Path,
    reported_type: str,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        ranking_type_overrides={"WAIVER": reported_type},
    )
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is True
    assert cache.status().last_error == ""
    assert cache.status().datasets_cached == ("PPR:ROS", "PPR:WAIVER")
    assert session.calls[0]["params"]["type"] == "WW"


def test_integer_year_is_accepted_only_when_it_matches_requested_season(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock, response_overrides={"year": 2026})
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is True
    assert cache.status().datasets_cached == ("PPR:ROS", "PPR:WAIVER")


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("sport", "MLB"),
        ("year", "2025"),
        ("scoring", "HALF"),
        ("position_id", "RB"),
    ],
)
def test_response_identity_mismatch_is_never_cached(
    tmp_path: Path,
    field: str,
    mismatched_value: str,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock, response_overrides={field: mismatched_value})
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is False
    assert len(session.calls) == 1
    assert cache.status().datasets_cached == ()
    assert cache.status().last_error == "invalid_response"


@pytest.mark.parametrize(
    "field",
    ["sport", "year", "scoring", "position_id", "ranking_type_name"],
)
def test_missing_response_identity_field_is_never_cached(
    tmp_path: Path,
    field: str,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock, omitted_response_fields={field})
    cache = _client(tmp_path, clock, session)

    assert cache.refresh(("PPR",), force=True) is False
    assert len(session.calls) == 1
    assert cache.status().datasets_cached == ()
    assert cache.status().last_error == "invalid_response"


def test_failed_batch_keeps_all_prior_datasets_and_counts_failed_request(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    first_session = FakeSession(clock)
    first = _client(tmp_path, clock, first_session)
    assert first.refresh(force=True)
    original = first.signal("DJ Moore", scoring="PPR")
    assert original is not None and original.waiver_rank == 7

    failing_session = FakeSession(clock, fail_on_call=2, rank_offset=900)
    second = _client(tmp_path, clock, failing_session)

    assert second.refresh(force=True) is False
    preserved = second.signal("DJ Moore", scoring="PPR")
    assert preserved is not None
    assert preserved.waiver_rank == 7
    assert preserved.ros_rank == 107
    assert len(failing_session.calls) == 2
    assert second.request_usage() == 6
    assert second.status().last_error == "request_failed"

    on_disk = json.loads((tmp_path / CACHE_FILENAME).read_text(encoding="utf-8"))
    ppr_waiver = on_disk["datasets"]["PPR:WAIVER"]["players"][0]
    assert ppr_waiver["rank"] == 7
    assert len(on_disk["requests"]) == 6


def test_request_budget_is_persisted_and_enforced_across_restart(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    first_session = FakeSession(clock)
    first = _client(tmp_path, clock, first_session, app_daily_cap=2)

    assert first.refresh(force=True) is False
    assert len(first_session.calls) == 2
    assert first.status().last_error == "budget_exhausted"

    second_session = FakeSession(clock)
    second = _client(tmp_path, clock, second_session, app_daily_cap=2)
    assert second.refresh(force=True) is False
    assert second_session.calls == []
    assert second.request_usage() == 2


def test_untrusted_ledger_fails_closed_without_sending_request(tmp_path: Path) -> None:
    (tmp_path / CACHE_FILENAME).write_text(
        json.dumps({"version": 1, "requests": "unknown", "datasets": {}}),
        encoding="utf-8",
    )
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    assert cache.enabled is False
    assert cache.refresh(force=True) is False
    assert session.calls == []
    assert cache.status().ledger_trusted is False
    assert cache.status().last_error == "ledger_invalid"


def test_cache_write_failure_prevents_the_reserved_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    def fail_replace(_source: Any, _destination: Any) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr("notifier.sources.fantasypros.os.replace", fail_replace)

    assert cache.refresh(force=True) is False
    assert session.calls == []
    assert cache.request_usage() == 0
    assert cache.status().last_error == "cache_write_failed"


def test_concurrent_refreshes_collapse_to_one_bulk_batch(tmp_path: Path) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: cache.refresh(), range(2)))

    assert results == [True, True]
    assert len(session.calls) == 4
    assert cache.request_usage() == 4


def test_two_hour_bulk_schedule_uses_forty_eight_requests_per_day(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    for cycle in range(12):
        if cycle:
            clock.advance(2 * 60 * 60)
        assert cache.refresh() is True

    assert len(session.calls) == 48
    assert cache.request_usage() == 48


@pytest.mark.parametrize(
    "reason",
    ["dataset_unavailable", "partial_dataset_unavailable"],
)
def test_unpublished_dataset_retries_on_healthy_two_hour_cadence(reason: str) -> None:
    assert _fantasypros_failure_delay(reason, 5, 2 * 60 * 60) == 2 * 60 * 60


def test_hard_fantasypros_failure_keeps_exponential_backoff() -> None:
    assert _fantasypros_failure_delay("request_failed", 3, 2 * 60 * 60) == 60 * 60


def test_failed_refresh_backoff_is_bounded_and_deterministic() -> None:
    assert [_fantasypros_retry_delay(failures) for failures in range(1, 9)] == [
        15 * 60,
        30 * 60,
        60 * 60,
        2 * 60 * 60,
        4 * 60 * 60,
        6 * 60 * 60,
        6 * 60 * 60,
        6 * 60 * 60,
    ]


def test_team_alias_and_position_disambiguate_normalized_name_collision(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    players = [
        {
            "player_id": 1,
            "player_name": "Josh Allen",
            "player_team_id": "BUF",
            "player_positions": "QB",
            "rank_ecr": 5,
            "pos_rank": "QB2",
        },
        {
            "player_id": 2,
            "player_name": "Josh Allen",
            "player_team_id": "JAX",
            "player_positions": "DE",
            "rank_ecr": 205,
            "pos_rank": "DE5",
        },
    ]
    session = FakeSession(clock, players=players)
    cache = _client(tmp_path, clock, session)
    assert cache.refresh(force=True)

    assert cache.lookup(
        "Josh Allen", scoring="PPR", ranking_type="ROS"
    ) is None
    defender = cache.lookup(
        "Josh Allen",
        scoring="PPR",
        ranking_type="ROS",
        team="JAC",
        position="DE",
    )
    assert defender is not None
    assert defender.player_id == 2
    assert defender.team == "JAX"
    assert defender.position == "DE"


def test_team_and_position_constraints_fail_closed_on_provider_disagreement(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)
    assert cache.refresh(force=True)

    assert cache.lookup(
        "DJ Moore",
        scoring="HALF",
        ranking_type="WAIVER",
        team="CAR",
        position="WR",
    ) is None
    assert cache.lookup(
        "DJ Moore",
        scoring="HALF",
        ranking_type="WAIVER",
        team="CHI",
        position="RB",
    ) is None
    ranking = cache.lookup(
        "DJ Moore",
        scoring="HALF",
        ranking_type="WAIVER",
        team="CHI",
        position="WR",
    )
    assert ranking is not None


def test_signal_never_merges_waiver_and_ros_rows_with_different_player_ids(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    common = {
        "player_name": "Chris Smith",
        "player_team_id": "ARI",
        "player_positions": "RB",
        "rank_ecr": 25,
        "pos_rank": "RB10",
    }
    session = FakeSession(
        clock,
        players_by_type={
            "WAIVER": [{**common, "player_id": 101}],
            "ROS": [{**common, "player_id": 202}],
        },
    )
    cache = _client(tmp_path, clock, session)
    assert cache.refresh(("PPR",), force=True)

    assert cache.lookup(
        "Chris Smith", scoring="PPR", ranking_type="WAIVER"
    ) is not None
    assert cache.lookup(
        "Chris Smith", scoring="PPR", ranking_type="ROS"
    ) is not None
    assert cache.signal("Chris Smith", scoring="PPR") is None


def test_signal_uses_strict_normalized_identity_when_ids_are_missing(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(
        clock,
        players=[
            {
                "player_name": "D.J. Moore Jr.",
                "player_team_id": "CHI",
                "player_positions": "WR",
                "rank_ecr": 25,
                "pos_rank": "WR12",
            }
        ],
    )
    cache = _client(tmp_path, clock, session)
    assert cache.refresh(("PPR",), force=True)

    signal = cache.signal(
        "DJ Moore", scoring="PPR", team="CHI", position="WR"
    )
    assert signal is not None
    assert signal.waiver_rank == 25
    assert signal.ros_rank == 25


def test_alert_path_cache_reads_fail_closed_instead_of_waiting_for_state_lock(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    class ContendedLock:
        def __init__(self) -> None:
            self.acquire_modes: list[bool] = []

        def acquire(self, blocking: bool = True) -> bool:
            self.acquire_modes.append(blocking)
            return False

        def release(self) -> None:
            raise AssertionError("unacquired lock must not be released")

    contended = ContendedLock()
    cache._state_lock = contended  # type: ignore[assignment]

    assert cache.lookup(
        "DJ Moore", scoring="PPR", ranking_type="WAIVER"
    ) is None
    assert cache.signal("DJ Moore", scoring="PPR") is None
    assert contended.acquire_modes == [False, False]


def test_provider_or_transport_error_never_exposes_api_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "fp-test-secret-value"
    clock = FakeClock()
    session = FakeSession(clock, raise_secret=secret)
    cache = _client(tmp_path, clock, session)

    with caplog.at_level("WARNING"):
        assert cache.refresh(force=True) is False

    assert secret not in caplog.text
    assert secret not in repr(cache.status())
    assert cache.status().last_error == "request_failed"


def test_close_is_idempotent_and_borrowed_session_remains_owned_by_caller(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    session = FakeSession(clock)
    cache = _client(tmp_path, clock, session)

    cache.close()
    cache.close()

    assert cache.status().closed is True
    assert cache.status().enabled is False
    assert cache.refresh(force=True) is False
    assert session.closed is False
    assert session.calls == []


def test_pipeline_enrichment_is_memory_only_and_preserves_sleeper_order() -> None:
    stamp = datetime(2026, 8, 23, 18, tzinfo=timezone.utc)
    signals = {
        "Michael Carter": FantasyProsSignal(
            player_name="Michael Carter",
            team="ARI",
            position="RB",
            scoring="HALF",
            waiver_rank=34,
            waiver_pos_rank="RB34",
            ros_rank=55,
            ros_pos_rank="RB55",
            updated_at=stamp,
            fetched_at=stamp,
        ),
        "Bam Knight": FantasyProsSignal(
            player_name="Bam Knight",
            team="ARI",
            position="RB",
            scoring="HALF",
            waiver_rank=41,
            waiver_pos_rank="RB41",
            ros_rank=63,
            ros_pos_rank="RB63",
            updated_at=stamp,
            fetched_at=stamp,
        ),
    }
    cache = Mock()
    cache.signal.side_effect = lambda name, **_kwargs: signals.get(name)
    notifier = Notifier.__new__(Notifier)
    notifier.fantasypros = cache
    plays = LeaguePlays(
        league=LeagueRef("sleeper", "1", "Home", "Mine"),
        subject_state="mine",
        subject_owner="Mine",
        scoring_format="HALF",
        beneficiaries=[
            Beneficiary("Michael Carter", "RB", 2, "free_agent", pro_team="ARI"),
            Beneficiary("Bam Knight", "RB", 3, "free_agent", pro_team="ARI"),
        ],
    )

    enriched = notifier._enrich_fantasypros([plays])[0]

    assert [candidate.name for candidate in enriched.beneficiaries] == [
        "Michael Carter",
        "Bam Knight",
    ]
    assert enriched.beneficiaries[0].fantasypros_waiver_rank == 34
    assert enriched.beneficiaries[1].fantasypros_ros_pos_rank == "RB63"
    assert cache.signal.call_count == 2
    cache.refresh.assert_not_called()


def test_pipeline_omits_optional_context_when_cache_raises_without_dropping_alert(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "fp-secret-must-not-appear"
    cache = Mock()
    cache.signal.side_effect = RuntimeError(secret)
    notifier = Notifier.__new__(Notifier)
    notifier.fantasypros = cache
    plays = LeaguePlays(
        league=LeagueRef("sleeper", "1", "Home", "Mine"),
        subject_state="mine",
        subject_owner="Mine",
        scoring_format="HALF",
        beneficiaries=[
            Beneficiary("Michael Carter", "RB", 2, "free_agent", pro_team="ARI"),
            Beneficiary("Bam Knight", "RB", 3, "free_agent", pro_team="ARI"),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="fantasy-news-notifier"):
        enriched = notifier._enrich_fantasypros([plays])

    assert enriched == [plays]
    assert cache.signal.call_count == 2
    assert caplog.text.count("fantasypros.enrichment_skipped") == 1
    assert secret not in caplog.text
