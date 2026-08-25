#!/usr/bin/env python3
"""Audit urgency persistence and evidence readiness across the full archive.

This validates storage, vector provenance, and safety invariants. Historical
rule labels are not outcome ground truth, so the report deliberately does not
claim urgency accuracy.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from notifier.config import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_URGENCY_EMBEDDING_MIN_NEIGHBORS,
    DEFAULT_URGENCY_EMBEDDING_HISTORY_DAYS,
    DEFAULT_URGENCY_EMBEDDING_THRESHOLD,
)
from notifier.embeddings import (  # noqa: E402
    INPUT_VERSION,
    canonical_embedding_text,
    cosine_similarity,
    embedding_input_hash,
    news_item_from_row,
    unpack_vector,
)
from notifier.event_store import read_all_reports  # noqa: E402
from notifier.matcher import compact_key  # noqa: E402
from notifier.urgency import (  # noqa: E402
    MAX_HISTORY_CANDIDATES,
    POLICY_VERSION,
    urgency_from_row,
    urgency_rank,
)

MIN_LIVE_EVIDENCE_ROWS = 50
MIN_LIVE_EVIDENCE_PLAYERS = 25
MAX_RECORDED_SUPPORT = 5
ALLOWED_LIFT_CONTEXTS = frozenset(
    {
        ("mine", "mine_starter"),
        ("mine", "mine_bench"),
        ("claimable", "claimable"),
    }
)


def _headline(row: dict) -> str:
    return str(row.get("headline") or "").replace("\n", " ")[:72]


def _proxy_level(row: dict) -> str:
    urgency = urgency_from_row(row)
    if urgency is not None:
        return urgency.rule_level
    try:
        return "monitor" if int(row.get("severity") or 0) >= 3 else "fyi"
    except (TypeError, ValueError):
        return "fyi"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_id(row: dict[str, Any]) -> int:
    try:
        return int(row.get("id") or 0)
    except (TypeError, ValueError):
        return 0


def _precedes(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Match EventStore.recent_urgency_candidates chronology exactly."""
    previous_published = _parse_datetime(previous.get("published_at"))
    current_published = _parse_datetime(current.get("published_at"))
    if previous_published is not None and current_published is not None:
        return previous_published < current_published or (
            previous_published == current_published
            and _row_id(previous) < _row_id(current)
        )
    return _row_id(previous) < _row_id(current)


def _is_live_assessment(row: dict[str, Any]) -> bool:
    urgency = urgency_from_row(row)
    return bool(
        urgency is not None
        and urgency.policy_version == POLICY_VERSION
        and urgency.basis != "archive_replay"
    )


def _is_eligible_live_evidence(row: dict[str, Any]) -> bool:
    urgency = urgency_from_row(row)
    return bool(
        urgency is not None
        and _is_live_assessment(row)
        and urgency.availability_verified
    )


def _is_eligible_comparable(
    row: dict[str, Any],
    *,
    vectors: dict[str, tuple[float, ...]],
    cutoff: datetime,
) -> bool:
    """Whether a row can participate in the production candidate query."""
    report_id = str(row.get("report_id") or "")
    received_at = _parse_datetime(row.get("received_at"))
    urgency = urgency_from_row(row)
    return bool(
        report_id in vectors
        and received_at is not None
        and received_at >= cutoff
        and row.get("subject_confident") == 1
        and row.get("feedback") not in {"wrong", "noisy"}
        and compact_key(str(row.get("player_name") or ""))
        and urgency is not None
        and urgency.canonical_event_type
        and urgency.direction.strip().lower() not in {"", "unknown"}
        and urgency.event_status
        and urgency.action_context
        and str(row.get("tier") or "")
        and str(row.get("embedding_provider") or "")
    )


def _same_candidate_context(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    previous_urgency = urgency_from_row(previous)
    current_urgency = urgency_from_row(current)
    if previous_urgency is None or current_urgency is None:
        return False
    return bool(
        previous_urgency.canonical_event_type
        == current_urgency.canonical_event_type
        and previous_urgency.direction.strip().lower()
        == current_urgency.direction.strip().lower()
        and previous_urgency.event_status == current_urgency.event_status
        and previous_urgency.action_context == current_urgency.action_context
        and str(previous.get("tier") or "league")
        == str(current.get("tier") or "league")
        and previous.get("embedding_model") == current.get("embedding_model")
        and previous.get("embedding_provider") == current.get("embedding_provider")
        and int(previous.get("embedding_dimensions") or 0)
        == int(current.get("embedding_dimensions") or 0)
        and previous.get("embedding_input_version")
        == current.get("embedding_input_version")
    )


def _candidate_neighbors(
    current: dict[str, Any],
    comparable: list[dict[str, Any]],
    vectors: dict[str, tuple[float, ...]],
    *,
    threshold: float,
) -> list[tuple[float, dict[str, Any]]]:
    """Reconstruct the exact candidate pool the live service would inspect."""
    current_report_id = str(current.get("report_id") or "")
    current_player = compact_key(str(current.get("player_name") or ""))
    current_vector = vectors[current_report_id]
    query_rows = [
        row
        for row in comparable
        if str(row.get("report_id") or "") != current_report_id
        and _same_candidate_context(row, current)
    ]
    query_rows.sort(
        key=lambda row: (
            _parse_datetime(row.get("received_at")) or datetime.min.replace(tzinfo=timezone.utc),
            _row_id(row),
        ),
        reverse=True,
    )

    # The SQL query first limits the context match to 1,000 recent rows. The
    # store then keeps at most 200 rows that precede the current report.
    older: list[dict[str, Any]] = []
    for row in query_rows[:1000]:
        if _precedes(row, current):
            older.append(row)
        if len(older) >= MAX_HISTORY_CANDIDATES:
            break

    best_by_player: dict[str, tuple[float, dict[str, Any]]] = {}
    for previous in older:
        previous_player = compact_key(str(previous.get("player_name") or ""))
        if not previous_player or previous_player == current_player:
            continue
        try:
            score = cosine_similarity(
                current_vector,
                vectors[str(previous["report_id"])],
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(score) or score < threshold:
            continue
        prior = best_by_player.get(previous_player)
        if prior is None or score > prior[0]:
            best_by_player[previous_player] = (score, previous)
    return sorted(best_by_player.values(), key=lambda value: value[0], reverse=True)


def _lift_pools(
    neighbors: list[tuple[float, dict[str, Any]]],
) -> tuple[list[tuple[float, dict[str, Any]]], list[tuple[float, dict[str, Any]]]]:
    live_rows = [value for value in neighbors if _is_live_assessment(value[1])]
    lift_pool = [
        value
        for value in live_rows
        if (urgency := urgency_from_row(value[1])) is not None
        and urgency.availability_verified
        and urgency_rank(urgency.rule_level) >= urgency_rank("act_today")
    ]
    return live_rows, lift_pool


def _current_can_consume_lift(row: dict[str, Any]) -> bool:
    urgency = urgency_from_row(row)
    tier_context = (
        str(row.get("tier") or "league"),
        urgency.action_context if urgency is not None else "",
    )
    return bool(
        urgency is not None
        and _is_live_assessment(row)
        and urgency.rule_level == "monitor"
        and urgency.action_available
        and urgency.availability_verified
        and tier_context in ALLOWED_LIFT_CONTEXTS
    )


def _usable_lift_selection(
    current: dict[str, Any],
    neighbors: list[tuple[float, dict[str, Any]]],
    *,
    min_neighbors: int,
) -> list[tuple[float, dict[str, Any]]]:
    if not _current_can_consume_lift(current):
        return []
    live_rows, lift_pool = _lift_pools(neighbors)
    minimum = max(3, int(min_neighbors))
    if (
        len(lift_pool) < minimum
        or not live_rows
        or len(lift_pool) / len(live_rows) < 0.75
    ):
        return []
    return lift_pool[:MAX_RECORDED_SUPPORT]


def _validate_persisted_lift(
    row: dict[str, Any],
    comparable: list[dict[str, Any]],
    vectors: dict[str, tuple[float, ...]],
    *,
    threshold: float,
    min_neighbors: int,
) -> list[str]:
    """Prove a saved lift could have been produced by the live policy."""
    urgency = urgency_from_row(row)
    if urgency is None:
        return []
    report_id = str(row.get("report_id") or "")
    delta = urgency_rank(urgency.level) - urgency_rank(urgency.rule_level)
    claims_lift = bool(
        delta > 0
        or urgency.embedding_delta != 0
        or urgency.basis == "rules+embedding_lift"
    )
    if not claims_lift:
        return []

    violations: list[str] = []
    if not (
        urgency.rule_level == "monitor"
        and urgency.level == "act_today"
        and delta == 1
        and urgency.embedding_delta == 1
    ):
        violations.append(f"invalid_lift_shape:{report_id}")
    if urgency.basis != "rules+embedding_lift":
        violations.append(f"invalid_lift_basis:{report_id}")
    if not urgency.availability_verified or not urgency.action_available:
        violations.append(f"unverified_lift_action:{report_id}")
    if (
        str(row.get("tier") or "league"),
        urgency.action_context,
    ) not in ALLOWED_LIFT_CONTEXTS:
        violations.append(f"invalid_lift_context:{report_id}")
    if row.get("subject_confident") != 1:
        violations.append(f"uncertain_subject_lift:{report_id}")
    if row not in comparable:
        violations.append(f"ineligible_lift_current:{report_id}")
        return violations

    support_ids = tuple(urgency.embedding_report_ids)
    support_count = int(urgency.embedding_support_count)
    if (
        support_count != len(support_ids)
        or len(set(support_ids)) != len(support_ids)
        or report_id in support_ids
        or support_count < 1
        or support_count > MAX_RECORDED_SUPPORT
    ):
        violations.append(f"invalid_lift_support_count:{report_id}")

    score = urgency.embedding_score
    if (
        score is None
        or not math.isfinite(score)
        or score < threshold
        or score > 1.0
    ):
        violations.append(f"invalid_lift_score:{report_id}")

    neighbors = _candidate_neighbors(
        row,
        comparable,
        vectors,
        threshold=threshold,
    )
    selected = _usable_lift_selection(
        row,
        neighbors,
        min_neighbors=min_neighbors,
    )
    expected_ids = tuple(str(previous.get("report_id") or "") for _, previous in selected)
    if not selected or support_ids != expected_ids:
        violations.append(f"incompatible_lift_support:{report_id}")
    else:
        expected_score = max(value for value, _previous in selected)
        if score is None or not math.isclose(
            score,
            expected_score,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            violations.append(f"lift_score_mismatch:{report_id}")
    return violations


def _assessment_violations(
    row: dict[str, Any],
    comparable: list[dict[str, Any]],
    vectors: dict[str, tuple[float, ...]],
    *,
    threshold: float,
    min_neighbors: int,
) -> list[str]:
    urgency = urgency_from_row(row)
    if urgency is None:
        return []
    report_id = str(row.get("report_id") or "")
    delta = urgency_rank(urgency.level) - urgency_rank(urgency.rule_level)
    violations: list[str] = []
    if delta != urgency.embedding_delta:
        violations.append(f"rank_delta_mismatch:{report_id}")
    if delta < 0:
        violations.append(f"downgrade:{report_id}")
    if delta > 1 or urgency.embedding_delta > 1 or urgency.embedding_delta < 0:
        violations.append(f"multi_band_lift:{report_id}")
    if urgency.embedding_delta > 0 and urgency.level == "act_now":
        violations.append(f"embedding_act_now:{report_id}")
    if urgency.embedding_delta > 0 and urgency.basis != "rules+embedding_lift":
        violations.append(f"unsupported_lift:{report_id}")
    if row.get("subject_confident") != 1 and urgency.embedding_delta > 0:
        violations.append(f"uncertain_subject_lift:{report_id}")
    if urgency.basis == "archive_replay" and urgency.embedding_delta != 0:
        violations.append(f"archive_replay_lift:{report_id}")
    if not (
        urgency.canonical_event_type
        and urgency.direction
        and urgency.event_status
        and urgency.action_context
    ):
        violations.append(f"missing_context:{report_id}")
    violations.extend(
        _validate_persisted_lift(
            row,
            comparable,
            vectors,
            threshold=threshold,
            min_neighbors=min_neighbors,
        )
    )
    return violations


def _readiness(
    eligible_live_assessed: list[dict[str, Any]],
    usable_lift_rows: list[dict[str, Any]],
    *,
    min_neighbors: int,
) -> tuple[bool, set[str]]:
    eligible_live_assessed = [
        row for row in eligible_live_assessed if _is_eligible_live_evidence(row)
    ]
    live_players = {
        compact_key(str(row.get("player_name") or ""))
        for row in eligible_live_assessed
    }
    live_players.discard("")
    ready = bool(
        len(eligible_live_assessed) >= max(MIN_LIVE_EVIDENCE_ROWS, min_neighbors)
        and len(live_players) >= max(MIN_LIVE_EVIDENCE_PLAYERS, min_neighbors)
        and usable_lift_rows
    )
    return ready, live_players


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-neighbors", type=int)
    parser.add_argument("--history-days", type=int)
    parser.add_argument("--show", type=int, default=15)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    model = os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    dimensions = int(
        os.environ.get("EMBEDDING_DIMENSIONS", "").strip()
        or DEFAULT_EMBEDDING_DIMENSIONS
    )
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(
            os.environ.get("URGENCY_EMBEDDING_THRESHOLD", "").strip()
            or DEFAULT_URGENCY_EMBEDDING_THRESHOLD
        )
    )
    min_neighbors = max(
        2,
        int(args.min_neighbors)
        if args.min_neighbors is not None
        else int(
            os.environ.get("URGENCY_EMBEDDING_MIN_NEIGHBORS", "").strip()
            or DEFAULT_URGENCY_EMBEDDING_MIN_NEIGHBORS
        ),
    )
    history_days = max(
        7,
        int(args.history_days)
        if args.history_days is not None
        else int(
            os.environ.get("URGENCY_EMBEDDING_HISTORY_DAYS", "").strip()
            or DEFAULT_URGENCY_EMBEDDING_HISTORY_DAYS
        ),
    )
    if not math.isfinite(threshold) or threshold < 0.5 or threshold > 1.0:
        print("threshold must be a finite value between 0.5 and 1.0", file=sys.stderr)
        return 2

    state_dir = Path(os.environ.get("NOTIFIER_STATE_DIR", "").strip() or root / "state")
    rows = read_all_reports(state_dir)

    vectors: dict[str, tuple[float, ...]] = {}
    invalid_vectors: list[dict] = []
    invalid_metadata: list[dict] = []
    embedded = 0
    for row in rows:
        if (
            not row.get("embedding")
            or row.get("embedding_model") != model
            or int(row.get("embedding_dimensions") or 0) != dimensions
            or row.get("embedding_input_version") != INPUT_VERSION
        ):
            continue
        embedded += 1
        try:
            item = news_item_from_row(row)
            metadata_valid = bool(
                row.get("embedding_provider")
                and row.get("embedding_at")
                and row.get("embedding_input_hash")
                == embedding_input_hash(canonical_embedding_text(item))
            )
            if not metadata_valid:
                invalid_metadata.append(row)
            vector = unpack_vector(
                bytes(row["embedding"]), dimensions=dimensions
            )
            if metadata_valid:
                vectors[str(row["report_id"])] = vector
        except (TypeError, ValueError):
            invalid_vectors.append(row)

    classified = [
        row
        for row in rows
        if row.get("event_type") and row.get("severity") is not None
    ]
    assessed = [row for row in classified if urgency_from_row(row) is not None]
    cutoff = datetime.now(timezone.utc) - timedelta(days=history_days)
    comparable = [
        row
        for row in assessed
        if _is_eligible_comparable(row, vectors=vectors, cutoff=cutoff)
    ]
    eligible_live_assessed = [
        row for row in comparable if _is_eligible_live_evidence(row)
    ]
    violations: list[str] = []
    for row in assessed:
        violations.extend(
            _assessment_violations(
                row,
                comparable,
                vectors,
                threshold=threshold,
                min_neighbors=min_neighbors,
            )
        )

    # Measure cross-player support coverage without treating model severity as
    # human ground truth. This is calibration only; repeated same-player
    # stories are excluded so they cannot make accuracy look artificially high.
    evidence_rows: list[tuple[float, int, float, dict, list[dict]]] = []
    usable_lift_rows: list[dict[str, Any]] = []
    for current in comparable:
        neighbors = _candidate_neighbors(
            current,
            comparable,
            vectors,
            threshold=threshold,
        )
        if len(neighbors) < min_neighbors:
            continue
        same = [row for _score, row in neighbors if _proxy_level(row) == _proxy_level(current)]
        ratio = len(same) / len(neighbors)
        evidence_rows.append(
            (
                neighbors[0][0],
                len(neighbors),
                ratio,
                current,
                [row for _score, row in neighbors],
            )
        )
        if _usable_lift_selection(
            current,
            neighbors,
            min_neighbors=min_neighbors,
        ):
            usable_lift_rows.append(current)

    source_counts = Counter(str(row.get("source") or "") for row in rows)
    urgency_counts = Counter(
        (urgency_from_row(row).level if urgency_from_row(row) is not None else "unassessed")
        for row in rows
    )
    basis_counts = Counter(
        (urgency_from_row(row).basis if urgency_from_row(row) is not None else "unassessed")
        for row in rows
    )
    supported = [row for row in evidence_rows if row[2] >= 0.75]
    print(
        f"archive reports={len(rows)} classified={len(classified)} "
        f"urgency={len(assessed)}/{len(classified)} policy={POLICY_VERSION}"
    )
    print(
        f"vectors embedded={embedded}/{len(rows)} valid={len(vectors)} "
        f"invalid={len(invalid_vectors)} bad_metadata={len(invalid_metadata)}"
    )
    print("sources:", " ".join(f"{key}={value}" for key, value in source_counts.items()))
    print("urgency:", " ".join(f"{key}={value}" for key, value in urgency_counts.items()))
    print("basis:", " ".join(f"{key}={value}" for key, value in basis_counts.items()))
    print(
        f"cross-player evidence threshold={threshold:.2f} min_neighbors={min_neighbors} "
        f"history_days={history_days} covered={len(evidence_rows)} "
        f"consensus={len(supported)} usable_lift={len(usable_lift_rows)}"
    )
    for top_score, count, ratio, current, neighbors in sorted(
        evidence_rows, key=lambda value: value[0], reverse=True
    )[: max(0, int(args.show))]:
        names = ", ".join(str(row.get("player_name") or "") for row in neighbors[:3])
        print(
            f"{top_score:.4f} neighbors={count} agreement={ratio:.0%} · "
            f"{current.get('player_name')} · {_headline(current)} · prior={names}"
        )

    storage_complete = bool(
        rows
        and classified
        and embedded == len(rows)
        and len(vectors) == len(rows)
        and not invalid_vectors
        and not invalid_metadata
        and len(assessed) == len(classified)
        and not violations
    )
    evidence_ready, live_players = _readiness(
        eligible_live_assessed,
        usable_lift_rows,
        min_neighbors=min_neighbors,
    )
    if violations:
        print("violations:", ", ".join(violations[:20]))
    print(
        "archive storage/invariant audit:",
        "PASS" if storage_complete else "FAIL",
    )
    print(
        "live urgency evidence readiness:",
        "READY" if evidence_ready else "NOT READY",
        f"eligible_live={len(eligible_live_assessed)}/{MIN_LIVE_EVIDENCE_ROWS} "
        f"distinct_players={len(live_players)}/{MIN_LIVE_EVIDENCE_PLAYERS} "
        f"usable_lift={len(usable_lift_rows)}/1",
    )
    strict_ready = storage_complete and evidence_ready
    print("strict urgency readiness:", "PASS" if strict_ready else "FAIL")
    return 0 if strict_ready or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
