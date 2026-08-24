"""Environment-driven configuration with fail-fast validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .logging_utils import NotifierError

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_POLL_SECONDS = 15
DEFAULT_POLL_SECONDS_IDLE = 60
DEFAULT_MIN_SEVERITY = 2
DEFAULT_MIN_SEVERITY_OTHER = 3
DEFAULT_FANTASYPROS_REQUEST_LIMIT = 425
DEFAULT_FANTASYPROS_REFRESH_HOURS = 2
DEFAULT_FANTASYPROS_MAX_AGE_HOURS = 12


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
    fantasypros_api_key: str
    fantasypros_request_limit: int
    fantasypros_refresh_hours: int
    fantasypros_max_age_hours: int
    poll_seconds: int
    poll_seconds_idle: int
    min_severity: int
    min_severity_other: int
    adaptive_polling: bool
    state_dir: Path
    telegram_controls_enabled: bool
    player_thread_hours: int
    daily_digest_enabled: bool
    daily_digest_hour: int
    daily_digest_timezone: str
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
    dry_run = optional_bool("DRY_RUN", False)
    state_dir = Path(
        os.environ.get("NOTIFIER_STATE_DIR", "").strip()
        or Path(__file__).resolve().parent.parent / "state"
    )
    if not dry_run:
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
        # FantasyPros is optional cached context. It is never required for the
        # breaking-news or live league-availability paths.
        fantasypros_api_key=os.environ.get("FANTASYPROS_API_KEY", "").strip(),
        fantasypros_request_limit=optional_int(
            "FANTASYPROS_REQUEST_LIMIT",
            DEFAULT_FANTASYPROS_REQUEST_LIMIT,
            1,
            450,
        ),
        fantasypros_refresh_hours=optional_int(
            "FANTASYPROS_REFRESH_HOURS",
            DEFAULT_FANTASYPROS_REFRESH_HOURS,
            1,
            24,
        ),
        fantasypros_max_age_hours=optional_int(
            "FANTASYPROS_MAX_AGE_HOURS",
            DEFAULT_FANTASYPROS_MAX_AGE_HOURS,
            1,
            72,
        ),
        poll_seconds=optional_int("POLL_SECONDS", DEFAULT_POLL_SECONDS, 10, 900),
        poll_seconds_idle=optional_int(
            "POLL_SECONDS_IDLE", DEFAULT_POLL_SECONDS_IDLE, 10, 3600
        ),
        min_severity=optional_int("MIN_SEVERITY", DEFAULT_MIN_SEVERITY, 1, 5),
        min_severity_other=optional_int(
            "MIN_SEVERITY_OTHER", DEFAULT_MIN_SEVERITY_OTHER, 1, 5
        ),
        adaptive_polling=optional_bool("ADAPTIVE_POLLING", True),
        state_dir=state_dir,
        # getUpdates permits only one consumer. Keep controls opt-in so an
        # alert token already used by OpenClaw or another bot process is not
        # silently hijacked. A dedicated notifier bot may enable this.
        telegram_controls_enabled=optional_bool("TELEGRAM_CONTROLS_ENABLED", False),
        # Alerts for the same player reply to the previous alert while it is
        # still inside the chat's retention window.
        player_thread_hours=optional_int("PLAYER_THREAD_HOURS", 168, 1, 24 * 30),
        daily_digest_enabled=optional_bool("DAILY_DIGEST_ENABLED", True),
        daily_digest_hour=optional_int("DAILY_DIGEST_HOUR", 18, 0, 23),
        daily_digest_timezone=(
            os.environ.get("DAILY_DIGEST_TIMEZONE", "").strip()
            or "America/Los_Angeles"
        ),
        dry_run=dry_run,
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

    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(config.daily_digest_timezone)
    except (KeyError, ValueError) as error:
        raise NotifierError(
            "DAILY_DIGEST_TIMEZONE must be a valid IANA timezone, "
            "e.g. America/Los_Angeles"
        ) from error

    return config


def roster_path(config: Config) -> Path:
    return config.state_dir / "roster-snapshot.json"


def seen_path(config: Config) -> Path:
    return config.state_dir / "seen-items.json"


def telegram_state_path(config: Config) -> Path:
    return config.state_dir / "telegram-state.json"
