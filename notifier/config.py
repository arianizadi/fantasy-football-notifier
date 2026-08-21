"""Environment-driven configuration with fail-fast validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .logging_utils import NotifierError

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_POLL_SECONDS = 15
DEFAULT_POLL_SECONDS_IDLE = 60
DEFAULT_MIN_SEVERITY = 2
DEFAULT_MIN_SEVERITY_OTHER = 3


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    telegram_chat_id: str
    openrouter_api_key: str
    openrouter_model: str
    espn_enabled: bool
    espn_league_id: int
    espn_year: int
    espn_swid: str
    espn_s2: str
    espn_team_id: int | None
    sleeper_username: str
    sleeper_league_ids: tuple[str, ...]
    twitter_bearer_token: str
    poll_seconds: int
    poll_seconds_idle: int
    min_severity: int
    min_severity_other: int
    adaptive_polling: bool
    watch_trending: bool
    trending_limit: int
    state_dir: Path
    message_ttl_hours: int
    dry_run: bool


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise NotifierError(f"Missing required environment variable: {name}")
    return value


def optional_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise NotifierError(f"{name} must be an integer") from error
    if value < minimum or value > maximum:
        raise NotifierError(f"{name} must be between {minimum} and {maximum}")
    return value


def optional_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise NotifierError(f"{name} must be a boolean value")


def validate_model(value: str) -> str:
    # A bare "latest" alias can silently change classification behaviour
    # mid-season, so require an explicit provider/model slug.
    if "/" not in value:
        raise NotifierError(
            "OPENROUTER_MODEL must be a full OpenRouter slug, e.g. "
            "deepseek/deepseek-v4-flash-0731"
        )
    return value


def load_config() -> Config:
    state_dir = Path(
        os.environ.get("NOTIFIER_STATE_DIR", "").strip()
        or Path(__file__).resolve().parent.parent / "state"
    )
    state_dir.mkdir(parents=True, exist_ok=True)

    espn_team_raw = os.environ.get("ESPN_TEAM_ID", "").strip()

    config = Config(
        telegram_bot_token=required("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=required("TELEGRAM_CHAT_ID"),
        openrouter_api_key=required("OPENROUTER_API_KEY"),
        openrouter_model=validate_model(
            os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
        ),
        espn_enabled=bool(os.environ.get("ESPN_LEAGUE_ID", "").strip()),
        espn_league_id=int(os.environ.get("ESPN_LEAGUE_ID", "0").strip() or 0),
        espn_year=optional_int("ESPN_YEAR", 2026, 2015, 2100),
        espn_swid=os.environ.get("ESPN_SWID", "").strip(),
        espn_s2=os.environ.get("ESPN_S2", "").strip(),
        espn_team_id=int(espn_team_raw) if espn_team_raw else None,
        sleeper_username=os.environ.get("SLEEPER_USERNAME", "").strip(),
        sleeper_league_ids=tuple(
            entry.strip()
            for entry in os.environ.get("SLEEPER_LEAGUE_IDS", "").split(",")
            if entry.strip()
        ),
        twitter_bearer_token=os.environ.get("TWITTER_BEARER_TOKEN", "").strip(),
        poll_seconds=optional_int("POLL_SECONDS", DEFAULT_POLL_SECONDS, 10, 900),
        poll_seconds_idle=optional_int(
            "POLL_SECONDS_IDLE", DEFAULT_POLL_SECONDS_IDLE, 10, 3600
        ),
        min_severity=optional_int("MIN_SEVERITY", DEFAULT_MIN_SEVERITY, 1, 5),
        min_severity_other=optional_int(
            "MIN_SEVERITY_OTHER", DEFAULT_MIN_SEVERITY_OTHER, 1, 5
        ),
        adaptive_polling=optional_bool("ADAPTIVE_POLLING", True),
        watch_trending=optional_bool("WATCH_TRENDING", True),
        trending_limit=optional_int("TRENDING_LIMIT", 50, 5, 200),
        state_dir=state_dir,
        # 0 disables expiry. Capped at 47h: Telegram refuses to let a bot
        # delete anything older than 48h.
        message_ttl_hours=optional_int("MESSAGE_TTL_HOURS", 24, 0, 47),
        dry_run=optional_bool("DRY_RUN", False),
    )

    if not config.espn_enabled and not config.sleeper_username:
        raise NotifierError(
            "Configure at least one league: ESPN_LEAGUE_ID and/or SLEEPER_USERNAME."
        )
    if config.espn_enabled and not (config.espn_swid and config.espn_s2):
        raise NotifierError("ESPN_LEAGUE_ID is set, so ESPN_SWID and ESPN_S2 are required.")

    if os.environ.get("ESPN_DEBUG", "false").strip().lower() == "true":
        # Matches sync.py: raw ESPN payloads carry private member data.
        raise NotifierError("ESPN_DEBUG must be false; raw ESPN responses contain private data.")

    return config


def roster_path(config: Config) -> Path:
    return config.state_dir / "roster-snapshot.json"


def seen_path(config: Config) -> Path:
    return config.state_dir / "seen-items.json"


def history_path(config: Config) -> Path:
    return config.state_dir / "sent-messages.json"
