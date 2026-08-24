from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from notifier.dedupe import semantic_event_type
from notifier.embeddings import (
    INPUT_VERSION,
    EmbeddingService,
    EmbeddingUnavailable,
    EmbeddingVector,
    OpenRouterEmbeddingClient,
    alert_from_row,
    canonical_embedding_text,
    cosine_similarity,
    embedding_input_hash,
    embedding_transition_guard,
    normalize_vector,
    pack_vector,
    unpack_vector,
)
from notifier.event_store import EventStore
from notifier.models import Alert, Classification, NewsItem
from notifier.telegram_state import TelegramState


def _item(
    guid: str,
    headline: str,
    *,
    body: str | None = None,
    player: str = "Example Player",
    source: str = "twitter",
) -> NewsItem:
    return NewsItem(
        source=source,
        guid=guid,
        player_name=player,
        headline=headline,
        body=body if body is not None else headline,
        url="https://x.com/reporter/status/123",
        published_at=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
    )


def _classification(
    event: str,
    severity: int = 3,
    direction: str = "neutral",
) -> Classification:
    return Classification(event, severity, "Impact", True, {"direction": direction})


def _delivered_row(
    item: NewsItem,
    classification: Classification,
    *,
    message_id: int = 42,
) -> dict:
    return {
        "source": item.source,
        "guid": item.guid,
        "player_name": item.player_name,
        "headline": item.headline,
        "body": item.body,
        "url": item.url,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "subject_confident": 1,
        "event_type": classification.event_type,
        "direction": classification.raw["direction"],
        "severity": classification.severity,
        "summary": classification.fantasy_impact,
        "is_actionable": 1,
        "tier": "league",
        "feedback": None,
        "telegram_message_id": message_id,
        "alert_token": "prior-token",
    }


def test_canonical_text_removes_source_noise_and_duplicate_body() -> None:
    item = _item(
        "twitter:1",
        "Example Player will start https://t.co/abc @Reporter",
        body="Example Player will start https://t.co/abc @Reporter",
    )

    text = canonical_embedding_text(item)

    assert text == "Headline: Example Player will start"
    assert "https" not in text
    assert "@Reporter" not in text


def test_vector_pack_round_trip_and_cosine_validation() -> None:
    normalized = normalize_vector([3.0, 4.0], dimensions=2)
    assert normalized == pytest.approx((0.6, 0.8))
    restored = unpack_vector(pack_vector(normalized), dimensions=2)
    assert restored == pytest.approx(normalized)
    assert cosine_similarity(normalized, restored) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="dimensions"):
        normalize_vector([1.0], dimensions=2)
    with pytest.raises(ValueError, match="non-finite"):
        normalize_vector([math.nan, 1.0], dimensions=2)
    with pytest.raises(ValueError, match="byte length"):
        unpack_vector(b"bad", dimensions=2)
    with pytest.raises(ValueError, match="zero"):
        unpack_vector(pack_vector((0.0, 0.0)), dimensions=2)


def test_openrouter_client_validates_payload_and_vector() -> None:
    captured = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [{"index": 0, "embedding": [3.0, 4.0]}],
                "usage": {"prompt_tokens": 7},
            }

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    client = OpenRouterEmbeddingClient(
        "secret",
        "provider/model",
        2,
        post=post,
    )
    vector = client.embed_one("hello")

    assert captured["json"]["dimensions"] == 2
    assert captured["json"]["encoding_format"] == "float"
    assert vector.values == pytest.approx((0.6, 0.8))
    assert vector.prompt_tokens == 7
    assert vector.input_hash == embedding_input_hash("hello")


def test_openrouter_client_fails_closed_on_malformed_vector() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"index": 0, "embedding": [1.0]}]}

    client = OpenRouterEmbeddingClient(
        "secret",
        "provider/model",
        2,
        post=lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(EmbeddingUnavailable):
        client.embed_one("hello")


def test_openrouter_client_rejects_duplicate_or_missing_indices() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 0, "embedding": [0.0, 1.0]},
                ]
            }

    client = OpenRouterEmbeddingClient(
        "secret",
        "provider/model",
        2,
        post=lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(EmbeddingUnavailable):
        client.embed_many(["one", "two"])


def test_dry_run_config_keeps_embedding_service_inert(tmp_path) -> None:
    store = EventStore(tmp_path, in_memory=True)
    service = EmbeddingService.from_config(
        store,
        SimpleNamespace(
            dry_run=True,
            openrouter_api_key="secret",
            embedding_mode="coalesce",
        ),
    )
    try:
        assert service.enabled is False
        assert service.status().mode == "off"
    finally:
        service.close()
        store.close()


def test_guard_allows_high_similarity_unspecified_paraphrase() -> None:
    prior_item = _item("twitter:1", "Example Player worked with the first team")
    current_item = _item("twitter:2", "First-team work continued for Example Player")
    classification = _classification("usage")
    current = Alert(current_item, classification, "league")
    previous = _delivered_row(prior_item, classification)

    assert embedding_transition_guard(
        current,
        previous,
        score=0.95,
        threshold=0.90,
    ) == (True, "safe_paraphrase")


@pytest.mark.parametrize(
    ("prior_item", "prior_classification", "current_item", "current_classification", "reason"),
    [
        (
            _item(
                "rotowire:trey",
                "Returns to practice",
                player="Trey Benson",
                source="rotowire",
            ),
            _classification("injury", 4, "positive"),
            _item(
                "rotowire:trey",
                "Cut by Cardinals",
                player="Trey Benson",
                source="rotowire",
            ),
            _classification("release", 5, "negative"),
            "event_transition",
        ),
        (
            _item("twitter:out1", "Example Player was ruled out"),
            _classification("inactive", 3, "negative"),
            _item("twitter:out2", "Example Player remains ruled out"),
            _classification("inactive", 5, "negative"),
            "severity_escalation",
        ),
        (
            _item("twitter:a", "Example Player has an ankle injury"),
            _classification("injury", 3, "negative"),
            _item("twitter:h", "Example Player has a hamstring injury"),
            _classification("injury", 3, "negative"),
            "fact_transition",
        ),
        (
            _item("twitter:q", "Example Player is questionable"),
            _classification("injury", 3, "negative"),
            _item("twitter:o", "Example Player was ruled out"),
            _classification("injury", 3, "negative"),
            "status_transition",
        ),
    ],
)
def test_guard_blocks_high_similarity_transitions(
    prior_item,
    prior_classification,
    current_item,
    current_classification,
    reason,
) -> None:
    allowed, actual = embedding_transition_guard(
        Alert(current_item, current_classification, "league"),
        _delivered_row(prior_item, prior_classification),
        score=0.99,
        threshold=0.90,
    )

    assert allowed is False
    assert actual == reason


def test_guard_blocks_information_regression_and_invalid_threshold() -> None:
    prior_item = _item(
        "twitter:full",
        "Example Player worked with the first team throughout practice and handled every two-minute rep",
    )
    current_item = _item("twitter:terse", "More work for Example Player")
    classification = _classification("usage", 3, "neutral")
    previous = _delivered_row(prior_item, classification)

    assert embedding_transition_guard(
        Alert(current_item, classification, "league"),
        previous,
        score=0.97,
        threshold=0.90,
    ) == (False, "information_regression")
    assert embedding_transition_guard(
        Alert(prior_item, classification, "league"),
        previous,
        score=0.97,
        threshold=math.nan,
    ) == (False, "invalid_similarity")


def test_waived_injured_label_normalizes_to_release() -> None:
    report = _item(
        "twitter:benson",
        "Cardinals also Waived/Injured running back Trey Benson.",
        player="Trey Benson",
    )

    assert semantic_event_type(report, "other") == "release"
    assert semantic_event_type(report, "injury") == "release"


@pytest.mark.parametrize(
    "headline",
    [
        "John Doe cut his hand during practice",
        "John Doe cut sharply during drills",
        "A piece of equipment cut John Doe on the hand",
        "John Doe was released from the hospital Monday",
        "John Doe has been released from concussion protocol",
    ],
)
def test_non_transaction_cut_does_not_normalize_to_release(headline) -> None:
    report = _item("twitter:cut", headline, player="John Doe")

    assert semantic_event_type(report, "injury") == "injury"


def test_event_store_embedding_metadata_and_candidate_query(tmp_path) -> None:
    store = EventStore(tmp_path)
    prior = _item("twitter:prior", "Example Player worked with the first team")
    current = _item("twitter:current", "First-team work continued for Example Player")
    classification = _classification("usage")
    store.record_classification(prior, classification, tier="league")
    store.mark_outcome(prior, "delivered", message_id=42)
    updated_at = store.get(prior)["updated_at"]
    store.record_received(current)
    vector = normalize_vector([3.0, 4.0], dimensions=2)
    assert store.store_embedding(
        prior,
        "provider/model",
        pack_vector(vector),
        provider="openrouter",
        dimensions=2,
        input_version=INPUT_VERSION,
        input_hash="hash",
    )

    rows = store.recent_embedded_for_player(
        "Example Player",
        model="provider/model",
        dimensions=2,
        input_version=INPUT_VERSION,
        exclude_report_id=store.get(current)["report_id"],
        active_message_id=42,
        active_alert_token=store.get(prior)["alert_token"],
        since_hours=6,
    )

    assert len(rows) == 1
    assert rows[0]["embedding_provider"] == "openrouter"
    assert rows[0]["embedding_dimensions"] == 2
    assert store.get(prior)["updated_at"] == updated_at
    assert store.embedding_count(model="provider/model", dimensions=2) == 1
    store.close()


def test_embedding_backlog_includes_legacy_vector_without_metadata(tmp_path) -> None:
    store = EventStore(tmp_path)
    item = _item("twitter:legacy", "Example Player practiced")
    store.record_received(item)
    vector = pack_vector(normalize_vector([3.0, 4.0], dimensions=2))
    store.store_embedding(
        item,
        "provider/model",
        vector,
        dimensions=None,
        input_version=None,
    )

    rows = store.embedding_backlog(
        model="provider/model",
        dimensions=2,
        input_version=INPUT_VERSION,
    )

    assert [row["report_id"] for row in rows] == [store.get(item)["report_id"]]
    store.close()


def test_embedding_candidates_never_move_backward_in_chronology(tmp_path) -> None:
    store = EventStore(tmp_path)
    classification = _classification("usage")
    older = replace(
        _item("twitter:older", "Earlier report about Example Player"),
        published_at=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
    )
    newer = replace(
        _item("twitter:newer", "Later report about Example Player"),
        published_at=datetime(2026, 8, 24, 12, 5, tzinfo=timezone.utc),
    )
    store.record_classification(older, classification, tier="league")
    store.record_classification(newer, classification, tier="league")
    store.mark_outcome(newer, "delivered", message_id=99)
    vector = pack_vector(normalize_vector([3.0, 4.0], dimensions=2))
    store.store_embedding(
        newer,
        "provider/model",
        vector,
        dimensions=2,
        input_version=INPUT_VERSION,
    )

    rows = store.recent_embedded_for_player(
        "Example Player",
        model="provider/model",
        dimensions=2,
        input_version=INPUT_VERSION,
        exclude_report_id=store.get(older)["report_id"],
        active_message_id=99,
        active_alert_token=store.get(newer)["alert_token"],
        since_hours=6,
    )

    assert rows == []
    store.close()


class _FixedClient:
    def __init__(self, model: str, dimensions: int) -> None:
        self.model = model
        self.dimensions = dimensions

    def embed_one(self, text: str) -> EmbeddingVector:
        values = normalize_vector([3.0, 4.0], dimensions=2)
        return EmbeddingVector(
            model=self.model,
            provider="openrouter",
            dimensions=self.dimensions,
            input_version=INPUT_VERSION,
            input_hash=embedding_input_hash(text),
            values=values,
            blob=pack_vector(values),
            prompt_tokens=5,
        )


def _service_with_prior(tmp_path, mode: str):
    store = EventStore(tmp_path)
    prior = _item("twitter:prior", "Example Player worked with the first team")
    classification = _classification("usage")
    store.record_classification(prior, classification, tier="league")
    store.mark_outcome(prior, "delivered", message_id=42)
    values = normalize_vector([3.0, 4.0], dimensions=2)
    store.store_embedding(
        prior,
        "provider/model",
        pack_vector(values),
        provider="openrouter",
        dimensions=2,
        input_version=INPUT_VERSION,
        input_hash=embedding_input_hash(canonical_embedding_text(prior)),
    )
    service = EmbeddingService(
        store,
        api_key="secret",
        mode=mode,
        model="provider/model",
        dimensions=2,
        threshold=0.90,
        wait_ms=1000,
        client=_FixedClient("provider/model", 2),
    )
    current = _item("twitter:current", "First-team work continued for Example Player")
    store.record_classification(current, classification, tier="league")
    alert = Alert(current, classification, "league")
    service.enqueue(current)
    return store, service, alert


def test_embedding_service_shadow_scores_but_does_not_hint(tmp_path) -> None:
    store, service, alert = _service_with_prior(tmp_path, "shadow")
    try:
        result = service.annotate(
            alert,
            active_message_id=42,
            active_alert_token=store.get("twitter:prior")["alert_token"],
        )
        assert result.embedding_match_message_id is None
        assert service.status().matches == 0
    finally:
        service.close()
        store.close()


def test_embedding_service_coalesce_adds_exact_message_hint(tmp_path) -> None:
    store, service, alert = _service_with_prior(tmp_path, "coalesce")
    try:
        result = service.annotate(
            alert,
            active_message_id=42,
            active_alert_token=store.get("twitter:prior")["alert_token"],
        )
        assert result.embedding_match_message_id == 42
        assert result.embedding_match_token == store.get("twitter:prior")["alert_token"]
        assert result.embedding_similarity == pytest.approx(1.0)
        assert result.embedding_model == "provider/model"
    finally:
        service.close()
        store.close()


def test_embedding_service_does_not_fall_back_below_unsafe_top_match(tmp_path) -> None:
    store = EventStore(tmp_path)
    classification = _classification("usage")
    safe = _item("twitter:safe", "Example Player worked with the first team")
    unsafe = _item("twitter:unsafe", "Example Player worked with the first team")
    unsafe_classification = _classification("usage", 2)
    values = normalize_vector([3.0, 4.0], dimensions=2)
    for item, item_classification, message_id in (
        (safe, classification, 42),
        (unsafe, unsafe_classification, 99),
    ):
        store.record_classification(item, item_classification, tier="league")
        store.mark_outcome(item, "delivered", message_id=message_id)
        store.store_embedding(
            item,
            "provider/model",
            pack_vector(values),
            provider="openrouter",
            dimensions=2,
            input_version=INPUT_VERSION,
            input_hash=embedding_input_hash(canonical_embedding_text(item)),
        )
    current = _item("twitter:current", "First-team work continued for Example Player")
    store.record_classification(current, classification, tier="league")
    alert = Alert(current, classification, "league")
    service = EmbeddingService(
        store,
        api_key="secret",
        mode="coalesce",
        model="provider/model",
        dimensions=2,
        threshold=0.90,
        wait_ms=1000,
        client=_FixedClient("provider/model", 2),
    )
    service.enqueue(current)
    try:
        assert service.annotate(
            alert,
            active_message_id=99,
            active_alert_token=store.get(unsafe)["alert_token"],
        ).embedding_match_message_id is None
    finally:
        service.close()
        store.close()


def test_embedding_service_cache_and_candidate_failures_are_fail_open(tmp_path) -> None:
    store, service, alert = _service_with_prior(tmp_path, "coalesce")
    store._connection.execute(
        "UPDATE news_events SET published_at = 'not-a-timestamp' WHERE guid = ?",
        ("twitter:prior",),
    )
    store._connection.commit()
    try:
        assert service.annotate(
            alert,
            active_message_id=42,
            active_alert_token=store.get("twitter:prior")["alert_token"],
        ) == alert
        store.close()
        service.enqueue(alert.item)
        assert service.annotate(
            alert,
            active_message_id=42,
            active_alert_token="prior-token",
        ) == alert
    finally:
        service.close()


def test_embedding_service_rejects_stale_candidate_hash(tmp_path) -> None:
    store, service, alert = _service_with_prior(tmp_path, "coalesce")
    store._connection.execute(
        "UPDATE news_events SET embedding_input_hash = 'stale' WHERE guid = ?",
        ("twitter:prior",),
    )
    store._connection.commit()
    try:
        assert service.annotate(
            alert,
            active_message_id=42,
            active_alert_token=store.get("twitter:prior")["alert_token"],
        ) == alert
    finally:
        service.close()
        store.close()


def test_telegram_state_accepts_only_matching_embedding_hint(tmp_path) -> None:
    state = TelegramState(tmp_path / "telegram.json")
    prior = Alert(
        _item("twitter:prior", "Example Player worked with the first team"),
        _classification("usage"),
        "league",
    )
    token = state.record_sent(prior, 42)
    assert state.active_edit_identity("Example Player") == (42, token)
    current = Alert(
        _item("twitter:current", "First-team work continued for Example Player"),
        _classification("usage"),
        "league",
        embedding_match_message_id=42,
        embedding_match_token=token,
        embedding_similarity=0.95,
        embedding_model="provider/model",
    )

    assert state.coalescing_target(current).message_id == 42
    assert state.coalescing_target(
        replace(current, embedding_match_token="wrong")
    ) is None


@pytest.mark.parametrize(
    ("prior_item", "prior_classification", "current_item", "current_classification"),
    [
        (
            _item("twitter:usage", "Example Player practiced"),
            _classification("usage"),
            _item("twitter:release", "Team waived Example Player"),
            _classification("release", direction="negative"),
        ),
        (
            _item("twitter:ankle", "Example Player has an ankle injury"),
            _classification("injury", direction="negative"),
            _item("twitter:hamstring", "Example Player has a hamstring injury"),
            _classification("injury", direction="negative"),
        ),
        (
            _item("twitter:q", "Example Player is questionable"),
            _classification("injury", direction="negative"),
            _item("twitter:out", "Example Player was ruled out"),
            _classification("injury", direction="negative"),
        ),
    ],
)
def test_telegram_state_revalidates_embedding_transition(
    tmp_path,
    prior_item,
    prior_classification,
    current_item,
    current_classification,
) -> None:
    state = TelegramState(tmp_path / "telegram.json")
    token = state.record_sent(Alert(prior_item, prior_classification, "league"), 42)
    current = Alert(
        current_item,
        current_classification,
        "league",
        embedding_match_message_id=42,
        embedding_match_token=token,
        embedding_similarity=0.99,
        embedding_model="provider/model",
    )

    assert state.coalescing_target(current) is None


def test_alert_from_row_preserves_direction_for_archive_replay() -> None:
    item = _item("twitter:1", "Example Player was ruled out")
    row = _delivered_row(item, _classification("inactive", 4, "negative"))

    alert = alert_from_row(row)

    assert alert.classification.raw["direction"] == "negative"
    assert alert.classification.severity == 4
