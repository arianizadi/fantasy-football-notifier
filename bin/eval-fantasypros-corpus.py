#!/usr/bin/env python3
"""Evaluate FantasyPros corpus retrieval using provider categories as weak labels.

This command is observational: it never opens ``EventStore`` and makes no API
calls.  The event-store reader copies a stable database/WAL snapshot into a
private temporary directory before SQLite opens it.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import sqlite3
import statistics
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.event_store import read_fantasypros_corpus_snapshot  # noqa: E402


WEAK_LABELS = frozenset({"injury", "recap", "transaction", "rumor", "breaking"})
EXPECTED_API_DOCS_URL = "https://api.fantasypros.com/public/v2/docs"
# Ingestion commits one provider response atomically, so a 5,000-row target can
# legitimately finish with up to 99 extra rows. Audit the whole standard corpus
# by default; larger configured targets remain available through the CLI.
DEFAULT_CORPUS_LIMIT = 6_000
MAX_CORPUS_LIMIT = 50_000
DEFAULT_SAMPLE_LIMIT = 128
MAX_SAMPLE_LIMIT = 512
DEFAULT_CANDIDATE_LIMIT = 512
MAX_CANDIDATE_LIMIT = 2_048
DEFAULT_HOLDOUT_FRACTION = 0.20
# Pure-Python cosine is deliberately bounded.  At 512 dimensions the default
# 128 x 512 evaluation performs roughly 34 million multiply/add operations.
MAX_VECTOR_SCALAR_OPERATIONS = 40_000_000

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "was",
        "will",
        "with",
    }
)

_CORPUS_COLUMNS = frozenset(
    {
        "id",
        "provider_item_id",
        "sport",
        "player_id",
        "team_id",
        "title",
        "description",
        "impact",
        "categories_json",
        "author",
        "source_url",
        "provider_created_at",
        "first_seen_at",
        "last_seen_at",
        "first_query_key",
        "last_query_key",
        "canonical_text",
        "content_hash",
        "source_provider",
        "attribution",
        "usage_scope",
        "api_docs_url",
        "embedding_model",
        "embedding_provider",
        "embedding_dimensions",
        "embedding_input_version",
        "embedding_input_hash",
        "embedding_at",
        "embedding",
        "updated_at",
    }
)
_OBSERVATION_COLUMNS = frozenset(
    {"run_id", "query_key", "provider_item_id", "observed_at"}
)
_EMBEDDING_METADATA_FIELDS = (
    "embedding_model",
    "embedding_provider",
    "embedding_dimensions",
    "embedding_input_version",
    "embedding_input_hash",
    "embedding_at",
)


@dataclass(frozen=True)
class Example:
    item_id: str
    player_id: str
    when: datetime
    ordinal: int
    labels: frozenset[str]
    tokens: frozenset[str]
    vector_space: tuple[str, str, int, str] | None
    vector: tuple[float, ...] | None

    @property
    def chronological_key(self) -> tuple[datetime, int, str]:
        return (self.when, self.ordinal, self.item_id)


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def query_category(query_key: str) -> tuple[bool, str | None]:
    """Return ``(valid_query_key, weak_category_or_none)``."""

    fields: dict[str, str] = {}
    for segment in str(query_key or "").split(";"):
        if "=" not in segment:
            return False, None
        key, value = segment.split("=", 1)
        if not key or key in fields:
            return False, None
        fields[key] = value
    if set(fields) != {"order", "category", "fpid", "limit"}:
        return False, None
    if fields["order"] not in {"created", "updated"}:
        return False, None
    try:
        limit = int(fields["limit"])
    except ValueError:
        return False, None
    if not 1 <= limit <= 100:
        return False, None
    if fields["fpid"] != "all":
        try:
            if int(fields["fpid"]) <= 0:
                return False, None
        except ValueError:
            return False, None
    category = fields["category"].strip().lower()
    if category == "all":
        return True, None
    if category not in WEAK_LABELS:
        return False, None
    return True, category


def _retrieval_tokens(row: dict[str, Any]) -> frozenset[str]:
    # categories_json and canonical_text are intentionally excluded so the
    # lexical baseline cannot read the weak label directly.
    words = [
        token
        for token in _TOKEN.findall(
            " ".join(
                str(row.get(field) or "")
                for field in ("title", "description", "impact")
            ).casefold()
        )
        if len(token) > 1 and token not in _STOPWORDS
    ]
    features = set(words)
    features.update(f"{left}_{right}" for left, right in zip(words, words[1:]))
    return frozenset(features)


def _lexical_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _unpack_vector(blob: Any, dimensions: Any) -> tuple[float, ...]:
    if isinstance(dimensions, bool):
        raise ValueError("invalid dimensions")
    size = int(dimensions)
    raw = bytes(blob)
    if size <= 0 or size > 4_096 or len(raw) != size * 4:
        raise ValueError("embedding byte length does not match dimensions")
    values = struct.unpack(f"<{size}f", raw)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("embedding contains invalid values")
    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("embedding has invalid magnitude")
    return tuple(value / magnitude for value in values)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


def _record(violations: Counter[str], key: str, amount: int = 1) -> None:
    violations[key] += amount


def inspect_storage(
    snapshot: dict[str, Any],
) -> tuple[
    dict[str, frozenset[str]],
    dict[str, tuple[tuple[str, str, int, str], tuple[float, ...]]],
    dict[str, Any],
]:
    """Validate storage/provenance and decode locally comparable vectors."""

    violations: Counter[str] = Counter()
    rows = list(snapshot.get("rows") or [])
    observations = list(snapshot.get("observations") or [])
    if not snapshot.get("database_exists"):
        _record(violations, "database_missing")
    if not snapshot.get("corpus_table_exists"):
        _record(violations, "corpus_table_missing")
    if not snapshot.get("observations_table_exists"):
        _record(violations, "observations_table_missing")
    missing_columns = _CORPUS_COLUMNS - set(snapshot.get("corpus_columns") or [])
    if missing_columns:
        _record(violations, "corpus_columns_missing", len(missing_columns))
    missing_observation_columns = _OBSERVATION_COLUMNS - set(
        snapshot.get("observation_columns") or []
    )
    if missing_observation_columns:
        _record(
            violations,
            "observation_columns_missing",
            len(missing_observation_columns),
        )
    corpus_count = int(snapshot.get("corpus_count") or 0)
    if corpus_count <= 0:
        _record(violations, "corpus_empty")
    if corpus_count != len(rows):
        # Strict mode must not claim all-row invariants when the bounded reader
        # intentionally loaded only a window of a larger corpus.
        _record(violations, "strict_audit_truncated", abs(corpus_count - len(rows)))

    row_ids = {str(row.get("provider_item_id") or "") for row in rows}
    labels: dict[str, set[str]] = defaultdict(set)
    observed_ids: set[str] = set()
    for observation in observations:
        item_id = str(observation.get("provider_item_id") or "")
        if not item_id or item_id not in row_ids:
            _record(violations, "orphan_observation")
            continue
        observed_ids.add(item_id)
        valid, category = query_category(str(observation.get("query_key") or ""))
        if not valid:
            _record(violations, "invalid_observation_query")
        elif category:
            labels[item_id].add(category)
        if _parse_time(observation.get("observed_at")) is None:
            _record(violations, "invalid_observation_time")

    vectors: dict[
        str, tuple[tuple[str, str, int, str], tuple[float, ...]]
    ] = {}
    seen_ids: set[str] = set()
    for row in rows:
        item_id = str(row.get("provider_item_id") or "").strip()
        if not item_id or item_id in seen_ids:
            _record(violations, "invalid_or_duplicate_provider_id")
        seen_ids.add(item_id)
        if item_id not in observed_ids:
            _record(violations, "missing_query_provenance")
        if row.get("sport") != "NFL":
            _record(violations, "invalid_sport")
        if row.get("source_provider") != "FantasyPros":
            _record(violations, "invalid_source_provider")
        if row.get("attribution") != "FantasyPros":
            _record(violations, "invalid_attribution")
        if row.get("usage_scope") != "personal_reference":
            _record(violations, "invalid_usage_scope")
        if row.get("api_docs_url") != EXPECTED_API_DOCS_URL:
            _record(violations, "invalid_api_docs_url")

        canonical = str(row.get("canonical_text") or "")
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if not canonical or row.get("content_hash") != expected_hash:
            _record(violations, "invalid_content_hash")
        try:
            categories = json.loads(str(row.get("categories_json") or ""))
            if not isinstance(categories, list) or not all(
                isinstance(value, str) for value in categories
            ):
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError):
            _record(violations, "invalid_categories_json")

        provider_time = _parse_time(row.get("provider_created_at"))
        first_seen = _parse_time(row.get("first_seen_at"))
        last_seen = _parse_time(row.get("last_seen_at"))
        if provider_time is None or first_seen is None or last_seen is None:
            _record(violations, "invalid_corpus_time")
        elif last_seen < first_seen:
            _record(violations, "reversed_seen_window")
        for field in ("first_query_key", "last_query_key"):
            valid, _category = query_category(str(row.get(field) or ""))
            if not valid:
                _record(violations, "invalid_row_query_provenance")

        blob = row.get("embedding")
        metadata_present = [row.get(field) not in (None, "") for field in _EMBEDDING_METADATA_FIELDS]
        if blob is None:
            if any(metadata_present):
                _record(violations, "partial_embedding_metadata")
            continue
        if not all(metadata_present):
            _record(violations, "partial_embedding_metadata")
            continue
        if row.get("embedding_input_hash") != row.get("content_hash"):
            _record(violations, "stale_embedding_input_hash")
            continue
        try:
            dimensions = int(row["embedding_dimensions"])
            vector = _unpack_vector(blob, dimensions)
        except (KeyError, TypeError, ValueError, OverflowError):
            _record(violations, "invalid_embedding")
            continue
        space = (
            str(row.get("embedding_model") or ""),
            str(row.get("embedding_provider") or ""),
            dimensions,
            str(row.get("embedding_input_version") or ""),
        )
        vectors[item_id] = (space, vector)

    strict_pass = not violations
    return (
        {item_id: frozenset(values) for item_id, values in labels.items()},
        vectors,
        {
            "strict_pass": strict_pass,
            "violations": dict(sorted(violations.items())),
            "checked_rows": len(rows),
            "checked_observations": len(observations),
        },
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _run_retrieval(
    samples: list[Example],
    training: list[Example],
    *,
    candidate_limit: int,
    method: str,
) -> dict[str, Any]:
    latencies: list[float] = []
    evaluated = 0
    represented = 0
    top_1_hits = 0
    top_3_hits = 0
    unsafe = 0
    returned = 0
    scored_total = 0
    same_player_excluded = 0

    for query in samples:
        if method == "embedding" and query.vector is None:
            continue
        if method == "lexical" and not query.tokens:
            continue
        represented += 1
        started = time.perf_counter_ns()
        candidates: list[Example] = []
        for candidate in training:
            if candidate.chronological_key >= query.chronological_key:
                continue
            if query.player_id and candidate.player_id == query.player_id:
                same_player_excluded += 1
                continue
            if method == "embedding":
                if (
                    candidate.vector is None
                    or candidate.vector_space != query.vector_space
                ):
                    continue
            elif not candidate.tokens:
                continue
            candidates.append(candidate)
        candidates = candidates[-candidate_limit:]

        scored: list[tuple[float, Example]] = []
        if method == "embedding":
            assert query.vector is not None
            scored = [
                (_cosine(query.vector, candidate.vector or ()), candidate)
                for candidate in candidates
            ]
        else:
            scored = [
                (_lexical_similarity(query.tokens, candidate.tokens), candidate)
                for candidate in candidates
            ]
            scored = [entry for entry in scored if entry[0] > 0]
        top = heapq.nlargest(
            3,
            scored,
            key=lambda entry: (entry[0], entry[1].chronological_key),
        )
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        scored_total += len(scored)
        if not top:
            continue
        evaluated += 1
        top_1_hits += int(bool(query.labels & top[0][1].labels))
        top_3_hits += int(any(query.labels & candidate.labels for _, candidate in top))
        for _score, candidate in top:
            returned += 1
            unsafe += int(not bool(query.labels & candidate.labels))

    return {
        "method": method,
        "sample_count": len(samples),
        "represented_queries": represented,
        "evaluated_queries": evaluated,
        "coverage": _rate(evaluated, len(samples)) or 0.0,
        "top_1_label_consistency": _rate(top_1_hits, evaluated),
        "top_3_label_consistency": _rate(top_3_hits, evaluated),
        "unsafe_cross_label_rate": _rate(unsafe, returned),
        "p50_lookup_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_lookup_latency_ms": _percentile(latencies, 0.95),
        "lookup_count": len(latencies),
        "mean_scored_candidates": _rate(scored_total, len(latencies)),
        "same_player_candidates_excluded": same_player_excluded,
    }


def _dominant_space(
    vectors: dict[str, tuple[tuple[str, str, int, str], tuple[float, ...]]]
) -> tuple[tuple[str, str, int, str] | None, Counter[tuple[str, str, int, str]]]:
    counts: Counter[tuple[str, str, int, str]] = Counter(
        space for space, _vector in vectors.values()
    )
    if not counts:
        return None, counts
    selected = sorted(counts, key=lambda space: (-counts[space], space))[0]
    return selected, counts


def evaluate_snapshot(
    snapshot: dict[str, Any],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    """Evaluate chronological retrieval consistency without asserting accuracy."""

    labels, vectors, storage = inspect_storage(snapshot)
    selected_space, space_counts = _dominant_space(vectors)
    examples: list[Example] = []
    invalid_evaluation_times = 0
    for fallback_ordinal, row in enumerate(snapshot.get("rows") or []):
        item_id = str(row.get("provider_item_id") or "")
        weak_labels = labels.get(item_id, frozenset())
        if not weak_labels:
            continue
        when = _parse_time(row.get("provider_created_at"))
        if when is None:
            invalid_evaluation_times += 1
            continue
        vector_record = vectors.get(item_id)
        space = vector_record[0] if vector_record else None
        vector = vector_record[1] if vector_record and space == selected_space else None
        try:
            ordinal = int(row.get("id") or fallback_ordinal)
        except (TypeError, ValueError):
            ordinal = fallback_ordinal
        player_raw = row.get("player_id")
        player_id = str(player_raw) if player_raw not in (None, "") else ""
        examples.append(
            Example(
                item_id=item_id,
                player_id=player_id,
                when=when,
                ordinal=ordinal,
                labels=weak_labels,
                tokens=_retrieval_tokens(row),
                vector_space=space if space == selected_space else None,
                vector=vector,
            )
        )
    examples.sort(key=lambda example: example.chronological_key)

    if len(examples) >= 2:
        split = int(len(examples) * (1.0 - holdout_fraction))
        split = max(1, min(len(examples) - 1, split))
        training = examples[:split]
        heldout = examples[split:]
    else:
        training = examples[:]
        heldout = []
    samples = heldout[-sample_limit:]

    requested_candidate_limit = candidate_limit
    effective_candidate_limit = candidate_limit
    selected_dimensions = selected_space[2] if selected_space else 0
    if samples and selected_dimensions:
        budget_limit = max(
            1,
            MAX_VECTOR_SCALAR_OPERATIONS // (len(samples) * selected_dimensions),
        )
        effective_candidate_limit = min(candidate_limit, budget_limit)

    lexical = _run_retrieval(
        samples,
        training,
        candidate_limit=effective_candidate_limit,
        method="lexical",
    )
    embedding = _run_retrieval(
        samples,
        training,
        candidate_limit=effective_candidate_limit,
        method="embedding",
    )

    label_distribution: Counter[str] = Counter()
    for example in examples:
        label_distribution.update(example.labels)
    selected_vector_count = space_counts.get(selected_space, 0) if selected_space else 0
    selected_space_payload = None
    if selected_space:
        selected_space_payload = {
            "model": selected_space[0],
            "provider": selected_space[1],
            "dimensions": selected_space[2],
            "input_version": selected_space[3],
            "valid_vector_count": selected_vector_count,
        }

    return {
        "evaluation_kind": "fantasypros_query_category_weak_label_consistency",
        "is_human_ground_truth": False,
        "measures_urgency_accuracy": False,
        "warnings": [
            "FantasyPros query categories are weak provider-derived labels, not human ground truth.",
            "Consistency and cross-label rates do not measure urgency or recommendation accuracy.",
            "Retrieval inputs use headline, report, and impact text only; query categories remain evaluation-only.",
        ],
        "storage": {
            "database_exists": bool(snapshot.get("database_exists")),
            "database_size_bytes": int(snapshot.get("database_size_bytes") or 0),
            "database_sidecar_bytes": int(snapshot.get("database_sidecar_bytes") or 0),
            "database_storage_bytes": int(snapshot.get("database_storage_bytes") or 0),
            "corpus_count": int(snapshot.get("corpus_count") or 0),
            "loaded_corpus_count": len(snapshot.get("rows") or []),
            "vector_count": int(snapshot.get("vector_count") or 0),
            "valid_loaded_vector_count": len(vectors),
            "loaded_observation_count": len(snapshot.get("observations") or []),
            **storage,
        },
        "weak_labels": {
            "source": "fantasypros_corpus_observations.query_key category",
            "labeled_loaded_rows": len(examples),
            "invalid_evaluation_times": invalid_evaluation_times,
            "distribution": dict(sorted(label_distribution.items())),
        },
        "sampling": {
            "chronological_holdout": True,
            "holdout_fraction": holdout_fraction,
            "training_count": len(training),
            "heldout_count": len(heldout),
            "sample_count": len(samples),
            "sample_limit": sample_limit,
            "requested_candidate_limit": requested_candidate_limit,
            "effective_candidate_limit": effective_candidate_limit,
            "same_player_neighbors_excluded": True,
            "max_vector_scalar_operations": MAX_VECTOR_SCALAR_OPERATIONS,
        },
        "embedding_space": {
            "selected": selected_space_payload,
            "space_count": len(space_counts),
            "selected_vector_coverage": _rate(
                selected_vector_count, len(snapshot.get("rows") or [])
            )
            or 0.0,
        },
        "lexical_baseline": lexical,
        "embedding_cosine_knn": embedding,
    }


def _percentage(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _milliseconds(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}ms"


def _print_text(report: dict[str, Any]) -> None:
    storage = report["storage"]
    sampling = report["sampling"]
    print("FantasyPros corpus retrieval evaluation")
    print("WEAK LABELS ONLY — this does not measure human-judged urgency accuracy.")
    print(
        "storage "
        f"corpus={storage['corpus_count']} loaded={storage['loaded_corpus_count']} "
        f"vectors={storage['vector_count']} valid_loaded={storage['valid_loaded_vector_count']} "
        f"db={storage['database_storage_bytes']} bytes"
    )
    print(
        "split "
        f"train={sampling['training_count']} heldout={sampling['heldout_count']} "
        f"sampled={sampling['sample_count']} candidates={sampling['effective_candidate_limit']} "
        "same-player=excluded"
    )
    for key, label in (
        ("lexical_baseline", "lexical Jaccard"),
        ("embedding_cosine_knn", "embedding cosine kNN"),
    ):
        metrics = report[key]
        print(
            f"{label}: coverage={_percentage(metrics['coverage'])} "
            f"top1={_percentage(metrics['top_1_label_consistency'])} "
            f"top3={_percentage(metrics['top_3_label_consistency'])} "
            f"unsafe-cross-label={_percentage(metrics['unsafe_cross_label_rate'])} "
            f"p50={_milliseconds(metrics['p50_lookup_latency_ms'])} "
            f"p95={_milliseconds(metrics['p95_lookup_latency_ms'])}"
        )
    strict = storage["strict_pass"]
    print(
        "strict storage/provenance invariants: "
        f"{'PASS' if strict else 'FAIL'} (not an accuracy gate)"
    )
    if storage["violations"]:
        print(
            "violations: "
            + " ".join(
                f"{key}={value}" for key, value in storage["violations"].items()
            )
        )


def _bounded_int(name: str, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from None
        if not 1 <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between 1 and {maximum}"
            )
        return parsed

    return parse


def _holdout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("holdout must be a number") from None
    if not math.isfinite(parsed) or not 0.05 <= parsed <= 0.50:
        raise argparse.ArgumentTypeError("holdout must be between 0.05 and 0.50")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("NOTIFIER_STATE_DIR", "").strip() or root / "state"),
    )
    parser.add_argument(
        "--corpus-limit",
        type=_bounded_int("corpus-limit", MAX_CORPUS_LIMIT),
        default=DEFAULT_CORPUS_LIMIT,
        help=f"newest corpus rows to inspect (max {MAX_CORPUS_LIMIT})",
    )
    parser.add_argument(
        "--samples",
        type=_bounded_int("samples", MAX_SAMPLE_LIMIT),
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"latest held-out rows to evaluate (max {MAX_SAMPLE_LIMIT})",
    )
    parser.add_argument(
        "--candidates",
        type=_bounded_int("candidates", MAX_CANDIDATE_LIMIT),
        default=DEFAULT_CANDIDATE_LIMIT,
        help=f"prior candidates per lookup (max {MAX_CANDIDATE_LIMIT})",
    )
    parser.add_argument(
        "--holdout",
        type=_holdout,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="chronological held-out fraction, 0.05-0.50",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero only for storage/provenance invariant failures",
    )
    args = parser.parse_args(argv)

    try:
        snapshot = read_fantasypros_corpus_snapshot(
            args.state_dir,
            limit=args.corpus_limit,
        )
        report = evaluate_snapshot(
            snapshot,
            sample_limit=args.samples,
            candidate_limit=args.candidates,
            holdout_fraction=args.holdout,
        )
    except (OSError, ValueError, sqlite3.Error) as error:
        message = {
            "evaluation_kind": "fantasypros_query_category_weak_label_consistency",
            "error": type(error).__name__,
            "is_human_ground_truth": False,
            "measures_urgency_accuracy": False,
        }
        if args.json:
            print(json.dumps(message, sort_keys=True))
        else:
            print(f"evaluation failed: {type(error).__name__}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if not args.strict or report["storage"]["strict_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
