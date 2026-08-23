from __future__ import annotations

from types import SimpleNamespace

import pytest

from notifier.config import load_config
from notifier.logging_utils import NotifierError
from notifier.models import RosterSnapshot
from notifier.pipeline import Notifier
from notifier.sources.sleeper import PlayerIndex


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
    assert not state_dir.exists()


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
