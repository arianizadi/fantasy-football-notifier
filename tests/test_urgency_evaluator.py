from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "bin" / "eval-urgency.py"
SPEC = importlib.util.spec_from_file_location("urgency_evaluator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)

NOW = datetime(2026, 8, 24, 20, tzinfo=timezone.utc)


def _row(
    index: int,
    player: str,
    *,
    rule_level: str = "act_today",
    level: str | None = None,
    basis: str = "rules",
    delta: int = 0,
    tier: str = "mine",
    action_context: str = "mine_starter",
    action_available: bool = False,
    availability_verified: bool = True,
    subject_confident: bool = True,
    feedback: str | None = None,
    received_at: datetime | None = None,
    support_ids: tuple[str, ...] = (),
    support_count: int | None = None,
    score: float | None = None,
) -> dict:
    timestamp = received_at or NOW - timedelta(hours=10 - index)
    return {
        "id": index,
        "report_id": f"report-{index}",
        "guid": f"guid-{index}",
        "source": "test",
        "player_name": player,
        "headline": f"{player} injury update",
        "body": "Status update",
        "url": "https://example.test/report",
        "subject_confident": int(subject_confident),
        "feedback": feedback,
        "published_at": timestamp.isoformat(),
        "received_at": timestamp.isoformat(),
        "event_type": "injury",
        "severity": 4,
        "tier": tier,
        "embedding_model": "provider/model",
        "embedding_provider": "openrouter",
        "embedding_dimensions": 2,
        "embedding_input_version": evaluator.INPUT_VERSION,
        "urgency_rule_level": rule_level,
        "urgency_level": level or rule_level,
        "urgency_reason_codes": "[]",
        "urgency_basis": basis,
        "urgency_embedding_delta": delta,
        "urgency_embedding_score": score,
        "urgency_embedding_support_count": (
            len(support_ids) if support_count is None else support_count
        ),
        "urgency_embedding_report_ids": list(support_ids),
        "urgency_policy_version": evaluator.POLICY_VERSION,
        "urgency_action_available": int(action_available),
        "urgency_roster_relevant": 1,
        "urgency_availability_verified": int(availability_verified),
        "urgency_event_type": "injury",
        "urgency_direction": "negative",
        "urgency_event_status": "questionable",
        "urgency_action_context": action_context,
        "urgency_subject_is_starter": int(action_context == "mine_starter"),
    }


def _valid_lift_fixture() -> tuple[list[dict], dict[str, tuple[float, ...]], dict]:
    prior_one = _row(1, "Player One")
    prior_two = _row(2, "Player Two")
    prior_three = _row(3, "Player Three")
    current = _row(
        4,
        "Current Player",
        rule_level="monitor",
        level="act_today",
        basis="rules+embedding_lift",
        delta=1,
        action_available=True,
        support_ids=("report-1", "report-2", "report-3"),
        score=1.0,
    )
    rows = [prior_one, prior_two, prior_three, current]
    vectors = {
        "report-1": (1.0, 0.0),
        "report-2": (0.98, 0.05),
        "report-3": (0.95, 0.1),
        "report-4": (1.0, 0.0),
    }
    return rows, vectors, current


def _comparable(
    rows: list[dict], vectors: dict[str, tuple[float, ...]]
) -> list[dict]:
    cutoff = NOW - timedelta(days=365)
    return [
        row
        for row in rows
        if evaluator._is_eligible_comparable(row, vectors=vectors, cutoff=cutoff)
    ]


def test_rank_delta_must_equal_persisted_embedding_delta() -> None:
    row = _row(
        1,
        "Mismatch",
        rule_level="monitor",
        level="act_today",
        basis="rules+embedding_lift",
        delta=0,
        action_available=True,
    )

    violations = evaluator._assessment_violations(
        row,
        [],
        {},
        threshold=0.70,
        min_neighbors=2,
    )

    assert "rank_delta_mismatch:report-1" in violations
    assert "invalid_lift_shape:report-1" in violations


def test_valid_persisted_lift_replays_against_compatible_live_support() -> None:
    rows, vectors, current = _valid_lift_fixture()
    comparable = _comparable(rows, vectors)

    violations = evaluator._assessment_violations(
        current,
        comparable,
        vectors,
        threshold=0.70,
        min_neighbors=2,
    )

    assert violations == []
    neighbors = evaluator._candidate_neighbors(
        current,
        comparable,
        vectors,
        threshold=0.70,
    )
    assert [row["report_id"] for _score, row in neighbors] == [
        "report-1",
        "report-2",
        "report-3",
    ]
    assert evaluator._usable_lift_selection(
        current,
        neighbors,
        min_neighbors=2,
    )


def test_support_recording_caps_at_five_when_neighbor_minimum_is_higher() -> None:
    priors = [_row(index, f"Player {index}") for index in range(1, 7)]
    current = _row(
        7,
        "Current Player",
        rule_level="monitor",
        level="act_today",
        basis="rules+embedding_lift",
        delta=1,
        action_available=True,
        support_ids=tuple(f"report-{index}" for index in range(1, 6)),
        score=1.0,
    )
    rows = [*priors, current]
    vectors = {
        **{
            f"report-{index}": (1.0 - (index - 1) * 0.01, (index - 1) * 0.01)
            for index in range(1, 7)
        },
        "report-7": (1.0, 0.0),
    }
    comparable = _comparable(rows, vectors)

    violations = evaluator._assessment_violations(
        current,
        comparable,
        vectors,
        threshold=0.70,
        min_neighbors=6,
    )

    assert violations == []


def test_lift_rejects_wrong_support_ids_count_score_and_context() -> None:
    rows, vectors, current = _valid_lift_fixture()
    invalid = deepcopy(current)
    invalid["urgency_embedding_report_ids"] = [
        "report-1",
        "report-1",
        "missing",
    ]
    invalid["urgency_embedding_support_count"] = 4
    invalid["urgency_embedding_score"] = 0.8
    invalid["tier"] = "league"
    invalid["urgency_action_context"] = "league"
    rows[-1] = invalid
    comparable = _comparable(rows, vectors)

    violations = evaluator._assessment_violations(
        invalid,
        comparable,
        vectors,
        threshold=0.70,
        min_neighbors=2,
    )

    assert "invalid_lift_context:report-4" in violations
    assert "invalid_lift_support_count:report-4" in violations
    assert "incompatible_lift_support:report-4" in violations


def test_lift_rejects_score_that_does_not_match_selected_vectors() -> None:
    rows, vectors, current = _valid_lift_fixture()
    current["urgency_embedding_score"] = 0.8
    comparable = _comparable(rows, vectors)

    violations = evaluator._assessment_violations(
        current,
        comparable,
        vectors,
        threshold=0.70,
        min_neighbors=2,
    )

    assert "lift_score_mismatch:report-4" in violations


def test_lift_requires_verified_available_action_and_exact_basis() -> None:
    rows, vectors, current = _valid_lift_fixture()
    current["urgency_basis"] = "rules"
    current["urgency_action_available"] = 0
    current["urgency_availability_verified"] = 0
    comparable = _comparable(rows, vectors)

    violations = evaluator._assessment_violations(
        current,
        comparable,
        vectors,
        threshold=0.70,
        min_neighbors=2,
    )

    assert "invalid_lift_basis:report-4" in violations
    assert "unverified_lift_action:report-4" in violations


def test_calibration_excludes_uncertain_feedback_and_out_of_window_rows() -> None:
    rows, vectors, current = _valid_lift_fixture()
    rows[0]["subject_confident"] = 0
    rows[1]["feedback"] = "wrong"
    rows[2]["feedback"] = "noisy"
    old = _row(
        5,
        "Old Player",
        received_at=NOW - timedelta(days=366),
    )
    rows.insert(3, old)
    vectors["report-5"] = (1.0, 0.0)
    comparable = _comparable(rows, vectors)

    assert [row["report_id"] for row in comparable] == ["report-4"]
    assert (
        evaluator._candidate_neighbors(
            current,
            comparable,
            vectors,
            threshold=0.70,
        )
        == []
    )


def test_readiness_requires_eligible_rows_and_usable_lift_evidence() -> None:
    eligible_live = [_row(index, f"Player {index}") for index in range(1, 51)]

    ready_without_lift, players = evaluator._readiness(
        eligible_live,
        [],
        min_neighbors=2,
    )
    ready_with_lift, _players = evaluator._readiness(
        eligible_live,
        [eligible_live[-1]],
        min_neighbors=2,
    )

    assert len(players) == 50
    assert ready_without_lift is False
    assert ready_with_lift is True


def test_readiness_does_not_count_unverified_live_volume() -> None:
    unverified = [
        _row(
            index,
            f"Unverified {index}",
            availability_verified=False,
        )
        for index in range(1, 51)
    ]
    usable = _row(51, "Usable")

    ready, players = evaluator._readiness(
        [*unverified, usable],
        [usable],
        min_neighbors=2,
    )

    assert ready is False
    assert players == {"usable"}
