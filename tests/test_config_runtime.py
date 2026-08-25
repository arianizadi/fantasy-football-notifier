from __future__ import annotations

from types import SimpleNamespace

import pytest

from notifier.config import load_config, optional_float
from notifier.logging_utils import NotifierError
from notifier.models import RosterSnapshot
from notifier.pipeline import Notifier
from notifier.sources.sleeper import PlayerIndex


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_optional_float_rejects_non_finite_values(monkeypatch, value) -> None:
    monkeypatch.setenv("EXAMPLE_FLOAT", value)

    with pytest.raises(NotifierError, match="between"):
        optional_float("EXAMPLE_FLOAT", 0.9, 0.5, 1.0)


def test_dry_run_config_does_not_create_state_directory(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "does-not-exist"
    values = {
        "TELEGRAM_BOT_TOKEN": "fake-token",
        "TELEGRAM_CHAT_ID": "123",
        "OPENROUTER_API_KEY": "fake-key",
        "SLEEPER_USERNAME": "example",
        "ESPN_LEAGUE_ID": "",
        "ESPN_DEBUG": "false",
        "NOTIFIER_STATE_DIR": str(state_dir),
        "DRY_RUN": "true",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TELEGRAM_CONTROLS_ENABLED", raising=False)

    config = load_config()

    assert config.dry_run is True
    assert config.telegram_controls_enabled is False
    assert config.fantasypros_refresh_hours == 2
    assert config.daily_digest_hour == 8
    assert config.waiver_report_enabled is True
    assert config.waiver_report_lead_hours == 8
    assert config.embedding_mode == "off"
    assert config.embedding_model == "qwen/qwen3-embedding-8b"
    assert config.embedding_dimensions == 512
    assert config.embedding_similarity_threshold == pytest.approx(0.90)
    assert config.urgency_embedding_threshold == pytest.approx(0.70)
    assert config.urgency_embedding_min_neighbors == 2
    assert config.urgency_embedding_history_days == 365
    assert config.urgency_embedding_lift_enabled is False
    assert config.fantasypros_corpus_enabled is False
    assert config.fantasypros_corpus_target == 5000
    assert config.fantasypros_corpus_max_requests == 300
    assert config.fantasypros_corpus_live_reserve == 75
    assert config.fantasypros_corpus_player_limit == 250
    assert config.fantasypros_corpus_embedding_budget_usd == pytest.approx(0.25)
    assert config.fantasypros_corpus_embedding_price_per_million_usd == pytest.approx(
        0.01
    )
    assert config.fantasypros_corpus_embedding_timeout_seconds == 30
    assert not state_dir.exists()


def test_fantasypros_corpus_request_budget_must_leave_live_reserve(
    tmp_path, monkeypatch
) -> None:
    values = {
        "TELEGRAM_BOT_TOKEN": "fake-token",
        "TELEGRAM_CHAT_ID": "123",
        "OPENROUTER_API_KEY": "fake-key",
        "SLEEPER_USERNAME": "example",
        "ESPN_LEAGUE_ID": "",
        "ESPN_DEBUG": "false",
        "NOTIFIER_STATE_DIR": str(tmp_path / "state"),
        "DRY_RUN": "false",
        "FANTASYPROS_API_KEY": "fake-fp-key",
        "FANTASYPROS_CORPUS_ENABLED": "true",
        "FANTASYPROS_REQUEST_LIMIT": "100",
        "FANTASYPROS_CORPUS_MAX_REQUESTS": "50",
        "FANTASYPROS_CORPUS_LIVE_RESERVE": "75",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(NotifierError, match="must not exceed"):
        load_config()


def test_state_directory_lock_rejects_second_notifier_process(tmp_path, monkeypatch) -> None:
    config = SimpleNamespace(
        dry_run=False,
        state_dir=tmp_path,
        twitter_bearer_token="",
        player_thread_hours=168,
        telegram_bot_token="fake-token",
        telegram_chat_id="123",
        telegram_controls_enabled=False,
        daily_digest_enabled=False,
        daily_digest_hour=18,
        daily_digest_timezone="America/Los_Angeles",
    )
    monkeypatch.setattr(
        "notifier.pipeline.load_snapshot",
        lambda _config: RosterSnapshot(generated_at=None),
    )
    monkeypatch.setattr("notifier.pipeline.snapshot_mtime", lambda _config: 0.0)
    monkeypatch.setattr(
        "notifier.pipeline.sleeper.load_player_index",
        lambda *_args, **_kwargs: PlayerIndex(),
    )

    first = Notifier(config)
    try:
        with pytest.raises(NotifierError, match="Another notifier process"):
            Notifier(config)
    finally:
        first.close()

    replacement = Notifier(config)
    replacement.close()
