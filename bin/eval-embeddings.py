#!/usr/bin/env python3
"""Audit the complete saved-news archive and simulate production coalescing."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from notifier.config import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD,
)
from notifier.embeddings import (  # noqa: E402
    INPUT_VERSION,
    alert_from_row,
    canonical_embedding_text,
    cosine_similarity,
    embedding_input_hash,
    embedding_transition_guard,
    news_item_from_row,
    unpack_vector,
)
from notifier.event_store import EventStore  # noqa: E402


def _stamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _summary(counter: Counter[str]) -> str:
    return " ".join(f"{key or 'blank'}={value}" for key, value in counter.most_common())


def _headline(row: dict) -> str:
    return str(row.get("headline") or "").replace("\n", " ")[:72]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-hours", type=int, default=6)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--show", type=int, default=25)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero unless every archived report has one valid vector",
    )
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
            os.environ.get("EMBEDDING_SIMILARITY_THRESHOLD", "").strip()
            or DEFAULT_EMBEDDING_SIMILARITY_THRESHOLD
        )
    )
    if not math.isfinite(threshold) or threshold < 0.5 or threshold > 1.0:
        print("threshold must be a finite value between 0.5 and 1.0", file=sys.stderr)
        return 2
    state_dir = Path(os.environ.get("NOTIFIER_STATE_DIR", "").strip() or root / "state")
    store = EventStore(state_dir)
    try:
        rows = store.all_reports()
    finally:
        store.close()
    vectors: dict[str, tuple[float, ...]] = {}
    embedded_rows: list[dict] = []
    invalid_vectors: list[dict] = []
    invalid_metadata: list[dict] = []
    for row in rows:
        metadata_matches = bool(
            row.get("embedding")
            and row.get("embedding_model") == model
            and int(row.get("embedding_dimensions") or 0) == dimensions
            and row.get("embedding_input_version") == INPUT_VERSION
        )
        if not metadata_matches:
            continue
        embedded_rows.append(row)
        try:
            item = news_item_from_row(row)
            expected_hash = embedding_input_hash(canonical_embedding_text(item))
            if (
                not row.get("embedding_provider")
                or not row.get("embedding_at")
                or row.get("embedding_input_hash") != expected_hash
            ):
                invalid_metadata.append(row)
            vectors[str(row["report_id"])] = unpack_vector(
                bytes(row["embedding"]),
                dimensions=dimensions,
            )
        except (TypeError, ValueError):
            invalid_vectors.append(row)
    comparable = [
        row
        for row in embedded_rows
        if row.get("event_type")
        and row.get("player_name")
        and str(row.get("report_id")) in vectors
    ]

    window_hours = max(1, int(args.window_hours))
    pairs: list[tuple[float, dict, dict]] = []
    proxy_pairs: list[tuple[float, dict, dict]] = []
    for index, current in enumerate(comparable):
        current_time = _stamp(current["received_at"])
        current_vector = vectors[str(current["report_id"])]
        prior_candidates: list[dict] = []
        for previous in comparable[:index]:
            if str(previous["player_name"]).strip().casefold() != str(
                current["player_name"]
            ).strip().casefold():
                continue
            age_hours = (current_time - _stamp(previous["received_at"])).total_seconds() / 3600
            if age_hours < 0 or age_hours > window_hours:
                continue
            previous_vector = vectors[str(previous["report_id"])]
            score = cosine_similarity(current_vector, previous_vector)
            pairs.append((score, previous, current))
            if (
                previous.get("outcome") == "delivered"
                and previous.get("telegram_message_id")
                and previous.get("alert_token")
            ):
                prior_candidates.append(previous)

        # Production can edit only the latest active Telegram thread for this
        # player, not whichever historical report happens to score highest.
        if (
            current.get("outcome") == "delivered"
            and current.get("telegram_message_id")
            and prior_candidates
        ):
            previous = prior_candidates[-1]
            score = cosine_similarity(
                current_vector,
                vectors[str(previous["report_id"])],
            )
            proxy_pairs.append((score, previous, current))

    decisions: list[tuple[float, bool, str, dict, dict]] = []
    high_reasons: Counter[str] = Counter()
    for score, previous, current in proxy_pairs:
        allowed, reason = embedding_transition_guard(
            alert_from_row(current),
            previous,
            score=score,
            threshold=threshold,
        )
        decisions.append((score, allowed, reason, previous, current))
        if score >= threshold:
            high_reasons[reason] += 1
    high = [decision for decision in decisions if decision[0] >= threshold]
    safe = [decision for decision in high if decision[1]]
    blocked = [decision for decision in high if not decision[1]]

    source_counts = Counter(str(row.get("source") or "") for row in rows)
    event_counts = Counter(str(row.get("event_type") or "unlabeled") for row in rows)
    high_water = max((int(row.get("id") or 0) for row in rows), default=0)
    print(
        f"archive reports={len(rows)} high_water_id={high_water} "
        f"model={model} dimensions={dimensions} input={INPUT_VERSION}"
    )
    print(
        f"coverage embedded={len(embedded_rows)}/{len(rows)} "
        f"valid={len(vectors)} invalid={len(invalid_vectors)} "
        f"bad_metadata={len(invalid_metadata)} "
        f"labeled={len(comparable)} unlabeled={len(rows) - len(comparable)}"
    )
    print("sources:", _summary(source_counts))
    print("events:", _summary(event_counts))
    print(
        f"six-hour pairs all={len(pairs)} production_proxy={len(proxy_pairs)} "
        f"threshold={threshold:.3f} high={len(high)} safe={len(safe)} "
        f"blocked={len(blocked)}"
    )
    print("high-score guard reasons:", _summary(high_reasons) or "none")

    sweep_values = sorted({0.85, 0.88, 0.90, 0.92, 0.95, round(threshold, 4)})
    print("threshold sweep:")
    for candidate_threshold in sweep_values:
        allowed_count = 0
        blocked_count = 0
        for score, previous, current in proxy_pairs:
            if score < candidate_threshold:
                continue
            allowed, _reason = embedding_transition_guard(
                alert_from_row(current),
                previous,
                score=score,
                threshold=candidate_threshold,
            )
            allowed_count += int(allowed)
            blocked_count += int(not allowed)
        print(
            f"  {candidate_threshold:.2f}: edit={allowed_count} "
            f"blocked={blocked_count}"
        )

    shown = max(0, int(args.show))
    print("allowed production-proxy edits:")
    for score, _allowed, reason, previous, current in sorted(
        safe, key=lambda decision: decision[0], reverse=True
    )[:shown]:
        print(
            f"{score:.4f} COALESCE:{reason} · {current['player_name']} · "
            f"{_headline(previous)} || {_headline(current)}"
        )

    print("blocked high-score production-proxy edits:")
    for score, _allowed, reason, previous, current in sorted(
        blocked, key=lambda decision: decision[0], reverse=True
    )[:shown]:
        print(
            f"{score:.4f} BLOCK:{reason} · {current['player_name']} · "
            f"{_headline(previous)} || {_headline(current)}"
        )

    near = [
        decision
        for decision in decisions
        if threshold - 0.05 <= decision[0] < threshold
    ]
    print("production-proxy results just below threshold:")
    for score, _allowed, reason, previous, current in sorted(
        near, key=lambda decision: decision[0], reverse=True
    )[:shown]:
        print(
            f"{score:.4f} {reason} · {current['player_name']} · "
            f"{_headline(previous)} || {_headline(current)}"
        )

    print("top all-pair calibration results:")
    for score, previous, current in sorted(
        pairs, key=lambda pair: pair[0], reverse=True
    )[:shown]:
        allowed, reason = embedding_transition_guard(
            alert_from_row(current),
            previous,
            score=score,
            threshold=threshold,
        )
        decision = "COALESCE" if allowed else f"BLOCK:{reason}"
        print(
            f"{score:.4f} {decision} · {current['player_name']} · "
            f"{previous.get('event_type')}/{previous.get('severity')} -> "
            f"{current.get('event_type')}/{current.get('severity')} · "
            f"{_headline(previous)} || {_headline(current)}"
        )

    complete = (
        len(embedded_rows) == len(rows)
        and not invalid_vectors
        and not invalid_metadata
    )
    print("strict archive coverage:", "PASS" if complete else "FAIL")
    return 0 if complete or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
