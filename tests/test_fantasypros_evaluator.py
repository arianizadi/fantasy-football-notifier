from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from notifier.event_store import EventStore, read_fantasypros_corpus_snapshot


SCRIPT = Path(__file__).resolve().parent.parent / "bin" / "eval-fantasypros-corpus.py"
SPEC = importlib.util.spec_from_file_location("eval_fantasypros_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)

BASE = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _query(category: str) -> str:
    return f"order=created;category={category};fpid=all;limit=100"


def _row(
    number: int,
    label: str,
    text: str,
    vector: tuple[float, float],
    *,
    player_id: int,
) -> dict:
    item_id = str(50_000 + number)
    created = (BASE + timedelta(hours=number)).isoformat()
    canonical = f"Headline: {text}\nReport: {text} details\nFantasyPros impact: Monitor."
    content_hash = hashlib.sha256(canonical.encode()).hexdigest()
    query = _query(label)
    return {
        "id": number,
        "provider_item_id": item_id,
        "sport": "NFL",
        "player_id": player_id,
        "team_id": "SF",
        "title": text,
        "description": f"{text} details",
        "impact": "Monitor.",
        "categories_json": '["News"]',
        "author": "FantasyPros Staff",
        "source_url": f"https://www.fantasypros.com/nfl/news/{item_id}/test.php",
        "provider_created_at": created,
        "first_seen_at": created,
        "last_seen_at": created,
        "first_query_key": query,
        "last_query_key": query,
        "canonical_text": canonical,
        "content_hash": content_hash,
        "source_provider": "FantasyPros",
        "attribution": "FantasyPros",
        "usage_scope": "personal_reference",
        "api_docs_url": evaluator.EXPECTED_API_DOCS_URL,
        "embedding_model": "provider/model",
        "embedding_provider": "openrouter",
        "embedding_dimensions": 2,
        "embedding_input_version": "fantasypros-news-v1",
        "embedding_input_hash": content_hash,
        "embedding_at": created,
        "embedding": struct.pack("<2f", *vector),
        "updated_at": created,
    }


def _snapshot(rows: list[dict]) -> dict:
    observations = [
        {
            "run_id": f"run-{row['id']}",
            "query_key": row["first_query_key"],
            "provider_item_id": row["provider_item_id"],
            "observed_at": row["first_seen_at"],
        }
        for row in rows
    ]
    return {
        "database_exists": True,
        "database_size_bytes": 4096,
        "database_sidecar_bytes": 0,
        "database_storage_bytes": 4096,
        "corpus_table_exists": True,
        "observations_table_exists": True,
        "corpus_columns": sorted(evaluator._CORPUS_COLUMNS),
        "observation_columns": sorted(evaluator._OBSERVATION_COLUMNS),
        "corpus_count": len(rows),
        "vector_count": len(rows),
        "rows": rows,
        "observations": observations,
    }


def _stored_record(number: int = 1) -> dict:
    canonical = "Headline: Player practices\nReport: Player was limited."
    return {
        "provider_item_id": str(number),
        "sport": "NFL",
        "player_id": number,
        "team_id": "SF",
        "title": "Player practices",
        "description": "Player was limited.",
        "impact": "",
        "categories": ["News"],
        "author": "FantasyPros Staff",
        "source_url": "https://www.fantasypros.com/nfl/news/1/test.php",
        "provider_created_at": BASE.isoformat(),
        "canonical_text": canonical,
        "content_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "source_provider": "FantasyPros",
        "attribution": "FantasyPros",
        "usage_scope": "personal_reference",
        "api_docs_url": evaluator.EXPECTED_API_DOCS_URL,
    }


def _source_signatures(state_dir: Path) -> dict[str, tuple[int, int, bytes]]:
    signatures = {}
    for path in state_dir.glob("news-events.sqlite3*"):
        stat = path.stat()
        signatures[path.name] = (stat.st_size, stat.st_mtime_ns, path.read_bytes())
    return signatures


def test_query_categories_are_explicit_weak_labels() -> None:
    assert evaluator.DEFAULT_CORPUS_LIMIT >= 5_100
    assert evaluator.MAX_CORPUS_LIMIT >= evaluator.DEFAULT_CORPUS_LIMIT
    assert evaluator.query_category(_query("injury")) == (True, "injury")
    assert evaluator.query_category(_query("all")) == (True, None)
    assert evaluator.query_category(_query("opinion")) == (False, None)


def test_chronological_holdout_excludes_same_player_and_does_not_gate_strict() -> None:
    # The held-out vectors intentionally point across weak labels.  Strict mode
    # must still pass because it covers storage/provenance, not retrieval quality.
    rows = [
        _row(1, "injury", "ankle practice limitation", (0.0, 1.0), player_id=1),
        _row(2, "transaction", "signed roster contract", (1.0, 0.0), player_id=2),
        _row(3, "transaction", "joined a new team", (1.0, 0.0), player_id=99),
        _row(4, "injury", "missed practice hurt", (0.0, 1.0), player_id=100),
        _row(5, "injury", "injury update", (1.0, 0.0), player_id=99),
        _row(6, "transaction", "transaction update", (0.0, 1.0), player_id=100),
    ]

    report = evaluator.evaluate_snapshot(
        _snapshot(rows),
        sample_limit=10,
        candidate_limit=10,
        holdout_fraction=1 / 3,
    )

    assert report["is_human_ground_truth"] is False
    assert report["measures_urgency_accuracy"] is False
    assert report["sampling"]["training_count"] == 4
    assert report["sampling"]["sample_count"] == 2
    assert report["embedding_cosine_knn"]["top_1_label_consistency"] == 0.0
    assert report["embedding_cosine_knn"]["unsafe_cross_label_rate"] > 0
    assert report["embedding_cosine_knn"]["same_player_candidates_excluded"] == 2
    assert report["storage"]["strict_pass"] is True
    assert "query categories remain evaluation-only" in report["warnings"][2]


def test_strict_storage_check_rejects_bad_content_hash() -> None:
    rows = [_row(1, "injury", "limited in practice", (1.0, 0.0), player_id=1)]
    rows[0]["content_hash"] = "not-the-canonical-hash"

    report = evaluator.evaluate_snapshot(_snapshot(rows))

    assert report["storage"]["strict_pass"] is False
    assert report["storage"]["violations"]["invalid_content_hash"] == 1
    assert report["storage"]["violations"]["stale_embedding_input_hash"] == 1


def test_json_cli_reads_private_snapshot_without_touching_source_database(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    assert read_fantasypros_corpus_snapshot(missing)["database_exists"] is False
    assert not missing.exists()

    store = EventStore(tmp_path)
    query = _query("injury")
    store.begin_fantasypros_corpus_run("read-only-eval", (query,))
    store.store_fantasypros_corpus_batch(
        "read-only-eval",
        query,
        [_stored_record()],
        fetched_at=BASE,
    )
    store.close()
    before = _source_signatures(tmp_path)

    output = io.StringIO()
    with redirect_stdout(output):
        result = evaluator.main(
            [
                "--state-dir",
                str(tmp_path),
                "--json",
                "--strict",
                "--samples",
                "2",
                "--candidates",
                "2",
            ]
        )

    payload = json.loads(output.getvalue())
    assert result == 0
    assert payload["storage"]["strict_pass"] is True
    assert payload["storage"]["corpus_count"] == 1
    assert payload["storage"]["vector_count"] == 0
    assert _source_signatures(tmp_path) == before
