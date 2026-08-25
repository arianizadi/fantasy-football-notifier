"""Build and read the cached multi-league roster snapshot.

The notifier normally reads the atomic local snapshot. Scheduled jobs refresh
it twice daily. Any alert eligible to recommend a waiver addition refreshes
only leagues that have drafted, so a still-pre-draft provider cannot block a
live ownership check for an active league.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests

from .config import Config, roster_path
from .logging_utils import NotifierError, structured_log
from .models import (
    LeagueRef,
    RosterCapacity,
    RosterPlayer,
    RosterSnapshot,
    lineup_slot_is_starter,
)

SNAPSHOT_VERSION = 2

ESPN_LEAGUE_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/"
    "{year}/segments/0/leagues/{league_id}"
)
ESPN_REQUEST_TIMEOUT = 25

# ESPN uses one set of ids for a player's default position and another for
# fantasy lineup slots. Keep the provider-specific translation at the adapter
# boundary so the rest of the notifier sees the same labels as Sleeper.
ESPN_DEFAULT_POSITIONS = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    16: "D/ST",
}
ESPN_LINEUP_SLOTS = {
    0: "QB",
    1: "TQB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",
    8: "DT",
    9: "DE",
    10: "LB",
    11: "DL",
    12: "CB",
    13: "S",
    14: "DB",
    15: "DP",
    16: "D/ST",
    17: "K",
    18: "P",
    19: "HC",
    20: "BE",
    21: "IR",
    22: "",
    23: "FLEX",
    24: "ER",
    25: "ROOKIE",
}
ESPN_PRO_TEAMS = {
    0: "",
    1: "ATL",
    2: "BUF",
    3: "CHI",
    4: "CIN",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GB",
    10: "TEN",
    11: "IND",
    12: "KC",
    13: "LV",
    14: "LAR",
    15: "MIA",
    16: "MIN",
    17: "NE",
    18: "NO",
    19: "NYG",
    20: "NYJ",
    21: "PHI",
    22: "ARI",
    23: "PIT",
    24: "LAC",
    25: "SF",
    26: "SEA",
    27: "TB",
    28: "WSH",
    29: "CAR",
    30: "JAX",
    33: "BAL",
    34: "HOU",
}


def _normalized_swid(value: str) -> str:
    return unquote(value).strip().strip("{}").casefold()


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_nonnegative_int(value: Any) -> int | None:
    parsed = _as_int(value)
    return parsed if parsed >= 0 else None


def _scoring_format_from_receptions(value: Any) -> str:
    """Map reception points to the closest FantasyPros scoring enum."""
    try:
        points = float(value)
    except (TypeError, ValueError):
        return ""
    if points >= 0.75:
        return "PPR"
    if points >= 0.25:
        return "HALF"
    return "STD"


def _espn_scoring_format(settings: dict[str, Any]) -> str:
    scoring = settings.get("scoringSettings")
    if not isinstance(scoring, dict):
        return ""
    items = scoring.get("scoringItems")
    if not isinstance(items, list):
        return ""
    # ESPN stat id 53 is a reception. Its absence in an otherwise complete
    # scoring list means standard/non-PPR scoring.
    for item in items:
        if isinstance(item, dict) and _as_int(item.get("statId")) == 53:
            return _scoring_format_from_receptions(item.get("points"))
    return "STD"


def _espn_capacity(
    settings: dict[str, Any], my_team: dict[str, Any]
) -> RosterCapacity:
    """Read bench/IR limits and occupancy from ESPN's existing mSettings view."""
    roster_settings = settings.get("rosterSettings") or {}
    if not isinstance(roster_settings, dict):
        roster_settings = {}
    counts = roster_settings.get("lineupSlotCounts") or {}
    if not isinstance(counts, dict):
        counts = {}

    def limit(slot_id: int) -> int | None:
        return _optional_nonnegative_int(counts.get(str(slot_id), counts.get(slot_id)))

    roster = my_team.get("roster") or {}
    entries = (roster.get("entries") or []) if isinstance(roster, dict) else []
    slot_ids = [
        _as_int(entry.get("lineupSlotId"))
        for entry in entries
        if isinstance(entry, dict)
    ]
    return RosterCapacity(
        bench_used=sum(slot_id == 20 for slot_id in slot_ids),
        bench_limit=limit(20),
        ir_used=sum(slot_id == 21 for slot_id in slot_ids),
        ir_limit=limit(21),
    )


def _find_my_espn_team(teams: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    if config.espn_team_id is not None:
        for team in teams:
            if _as_int(team.get("id")) == config.espn_team_id:
                return team
        raise NotifierError(f"ESPN_TEAM_ID={config.espn_team_id} not found in league")

    target = _normalized_swid(config.espn_swid)
    for team in teams:
        owners = team.get("owners")
        if not isinstance(owners, list):
            owners = []
        primary_owner = team.get("primaryOwner")
        if primary_owner:
            owners = [*owners, primary_owner]
        for owner in owners:
            owner_id = owner.get("id") if isinstance(owner, dict) else owner
            if _normalized_swid(str(owner_id or "")) == target:
                return team

    raise NotifierError(
        "Could not identify your ESPN team from ESPN_SWID. "
        "Set ESPN_TEAM_ID explicitly to the team id shown in your league URL."
    )


def _espn_team_name(team: dict[str, Any]) -> str:
    explicit = str(team.get("name") or "").strip()
    if explicit:
        return explicit
    location = str(team.get("location") or "").strip()
    nickname = str(team.get("nickname") or "").strip()
    combined = " ".join(part for part in (location, nickname) if part)
    return (
        combined
        or str(team.get("abbrev") or "").strip()
        or f"Team {_as_int(team.get('id'))}"
    )


def _espn_player(entry: dict[str, Any]) -> dict[str, Any]:
    pool_entry = entry.get("playerPoolEntry") or {}
    if not isinstance(pool_entry, dict):
        pool_entry = {}
    player = pool_entry.get("player") or entry.get("player") or {}
    return player if isinstance(player, dict) else {}


def _espn_position(player: dict[str, Any]) -> str:
    default_position = _as_int(player.get("defaultPositionId"))
    if default_position in ESPN_DEFAULT_POSITIONS:
        return ESPN_DEFAULT_POSITIONS[default_position]

    for raw_slot in player.get("eligibleSlots") or []:
        slot = ESPN_LINEUP_SLOTS.get(_as_int(raw_slot), "")
        if (
            slot
            and slot not in {"BE", "IR", "ER", "ROOKIE", "FLEX", "OP"}
            and "/" not in slot
        ):
            return slot
    return ""


def _espn_lineup_slot(entry: dict[str, Any]) -> str:
    slot_id = _as_int(entry.get("lineupSlotId"))
    return ESPN_LINEUP_SLOTS.get(slot_id, f"SLOT_{slot_id}" if slot_id >= 0 else "")


def _espn_payload(config: Config, session: requests.Session) -> dict[str, Any]:
    url = ESPN_LEAGUE_URL.format(year=config.espn_year, league_id=config.espn_league_id)
    try:
        response = session.get(
            url,
            params={"view": ["mTeam", "mRoster", "mSettings"]},
            cookies={"SWID": config.espn_swid, "espn_s2": config.espn_s2},
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; fantasy-football-notifier/1.0)",
            },
            timeout=ESPN_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        hint = " Check ESPN_SWID and ESPN_S2." if status in {401, 403} else ""
        raise NotifierError(
            f"ESPN roster request failed (HTTP {status or 'error'}).{hint}"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise NotifierError("ESPN roster response was not valid JSON") from error

    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise NotifierError("ESPN roster response had an unexpected shape")
    return payload


def _load_espn(
    config: Config, session: requests.Session
) -> tuple[LeagueRef, list[RosterPlayer], RosterCapacity, str]:
    payload = _espn_payload(config, session)
    teams = payload.get("teams") or []
    if not isinstance(teams, list):
        raise NotifierError("ESPN roster response did not include a team list")
    typed_teams = [team for team in teams if isinstance(team, dict)]
    my_team = _find_my_espn_team(typed_teams, config)
    my_team_id = _as_int(my_team.get("id"))
    settings = payload.get("settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    capacity = _espn_capacity(settings, my_team)

    ref = LeagueRef(
        provider="espn",
        league_id=str(config.espn_league_id),
        name=str(settings.get("name") or "ESPN League"),
        my_team_name=_espn_team_name(my_team),
    )

    players: list[RosterPlayer] = []
    for team in typed_teams:
        team_id = _as_int(team.get("id"))
        team_name = _espn_team_name(team)
        roster = team.get("roster") or {}
        entries = (roster.get("entries") or []) if isinstance(roster, dict) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            player = _espn_player(entry)
            name = str(player.get("fullName") or "").strip()
            if not name:
                continue
            lineup_slot = _espn_lineup_slot(entry)
            players.append(
                RosterPlayer(
                    name=name,
                    position=_espn_position(player),
                    pro_team=ESPN_PRO_TEAMS.get(_as_int(player.get("proTeamId")), ""),
                    lineup_slot=lineup_slot,
                    on_my_team=team_id == my_team_id,
                    fantasy_team=team_name,
                    league_key=ref.key,
                    fantasy_starter=lineup_slot_is_starter(lineup_slot),
                )
            )

    structured_log(
        logging.INFO,
        "espn.league_loaded",
        leagueName=ref.name,
        myTeamName=ref.my_team_name,
        myPlayerCount=sum(1 for p in players if p.on_my_team),
        leaguePlayerCount=len(players),
    )
    return ref, players, capacity, _espn_scoring_format(settings)


def _load_sleeper(
    config: Config,
    session: requests.Session,
    *,
    league_ids: set[str] | None = None,
) -> list[tuple[LeagueRef, list[RosterPlayer], RosterCapacity, str]]:
    from .sources import sleeper, sleeper_league

    player_index = sleeper.load_player_index(config.state_dir, session)
    user_id = sleeper_league.resolve_user_id(session, config.sleeper_username)
    leagues = sleeper_league.list_leagues(session, user_id, config.espn_year)

    if config.sleeper_league_ids:
        allowed = set(config.sleeper_league_ids)
        leagues = [lg for lg in leagues if str(lg.get("league_id")) in allowed]
    if league_ids is not None:
        leagues = [
            league
            for league in leagues
            if str(league.get("league_id")) in league_ids
        ]

    results = []
    for league in leagues:
        results.append(
            sleeper_league.fetch_league_rosters(session, league, user_id, player_index)
        )
    return results


def refresh_snapshot(config: Config) -> RosterSnapshot:
    """Fetch rosters from every configured league and cache them to disk."""
    session = requests.Session()
    leagues: list[LeagueRef] = []
    players: list[RosterPlayer] = []
    capacities: dict[str, RosterCapacity] = {}
    scoring_formats: dict[str, str] = {}
    try:
        if config.espn_enabled:
            ref, espn_players, capacity, scoring_format = _load_espn(config, session)
            leagues.append(ref)
            players.extend(espn_players)
            capacities[ref.key] = capacity
            if scoring_format:
                scoring_formats[ref.key] = scoring_format

        if config.sleeper_username:
            for ref, sleeper_players, capacity, scoring_format in _load_sleeper(
                config, session
            ):
                leagues.append(ref)
                players.extend(sleeper_players)
                capacities[ref.key] = capacity
                if scoring_format:
                    scoring_formats[ref.key] = scoring_format
    finally:
        session.close()

    if not leagues:
        raise NotifierError(
            "No leagues configured. Set ESPN_LEAGUE_ID and/or SLEEPER_USERNAME."
        )

    snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=leagues,
        players=players,
        capacities=capacities,
        scoring_formats=scoring_formats,
    )
    _write_snapshot(roster_path(config), snapshot)

    structured_log(
        logging.INFO,
        "roster.snapshot_refreshed",
        leagueCount=len(leagues),
        totalPlayerCount=len(players),
        myPlayerCount=len(snapshot.mine()),
    )
    return snapshot


def refresh_drafted_snapshot(
    config: Config,
    previous: RosterSnapshot,
    league_keys: set[str] | None = None,
) -> tuple[RosterSnapshot, int]:
    """Refresh ownership only for leagues already active in ``previous``.

    A pre-draft league has no trustworthy free-agent pool yet. It therefore
    must not participate in a just-in-time ownership refresh, and an outage at
    that provider must not suppress a valid pickup in a league that has
    drafted. Metadata for skipped leagues is carried forward unchanged; the
    scheduled full refresh and hourly draft discovery remain responsible for
    noticing their eventual draft completion.
    """
    drafted = previous.drafted_leagues()
    if league_keys is not None:
        drafted = [league for league in drafted if league.key in league_keys]
    target_keys = {league.key for league in drafted}
    if not target_keys:
        return previous, snapshot_mtime(config)

    refreshed: dict[
        str,
        tuple[LeagueRef, list[RosterPlayer], RosterCapacity, str],
    ] = {}
    session = requests.Session()
    try:
        espn_keys = {
            league.key for league in drafted if league.provider == "espn"
        }
        if espn_keys:
            result = _load_espn(config, session)
            refreshed[result[0].key] = result

        sleeper_ids = {
            league.league_id
            for league in drafted
            if league.provider == "sleeper"
        }
        if sleeper_ids:
            for result in _load_sleeper(
                config,
                session,
                league_ids=sleeper_ids,
            ):
                refreshed[result[0].key] = result
    finally:
        session.close()

    missing = target_keys - set(refreshed)
    if missing:
        raise NotifierError(
            "Drafted league ownership refresh omitted: "
            + ", ".join(sorted(missing))
        )

    for league_key in target_keys:
        _, players, _, _ = refreshed[league_key]
        if not any(player.on_my_team for player in players):
            raise NotifierError(
                f"Drafted league ownership refresh returned an empty roster: {league_key}"
            )

    path = roster_path(config)
    with _snapshot_write_lock(path):
        # A scheduled/full refresh may have completed while the drafted-only
        # network calls were in flight. Merge against that atomic result so a
        # newly drafted league is never overwritten with the older pre-draft
        # copy captured by the daemon worker.
        base = load_snapshot(config) if path.exists() else previous
        snapshot = _merge_drafted_refresh(base, target_keys, refreshed)
        _write_snapshot_unlocked(path, snapshot)
        # Capture the exact version while no other writer can replace it.
        # The daemon must not pair this snapshot with the mtime of a newer full
        # refresh that lands after the lock is released.
        written_version = path.stat().st_mtime_ns

    structured_log(
        logging.INFO,
        "roster.drafted_snapshot_refreshed",
        refreshedLeagueKeys=sorted(target_keys),
        preservedLeagueKeys=sorted(
            league.key for league in base.leagues if league.key not in target_keys
        ),
        playerCount=len(snapshot.players),
    )
    return snapshot, written_version


def _merge_drafted_refresh(
    base: RosterSnapshot,
    target_keys: set[str],
    refreshed: dict[
        str,
        tuple[LeagueRef, list[RosterPlayer], RosterCapacity, str],
    ],
) -> RosterSnapshot:
    """Overlay refreshed active leagues while retaining every other league."""
    leagues = [
        refreshed[league.key][0] if league.key in refreshed else league
        for league in base.leagues
    ]
    players = [
        player for player in base.players if player.league_key not in target_keys
    ]
    capacities = dict(base.capacities)
    scoring_formats = dict(base.scoring_formats)
    for league_key in target_keys:
        _, fresh_players, capacity, scoring_format = refreshed[league_key]
        players.extend(fresh_players)
        capacities[league_key] = capacity
        if scoring_format:
            scoring_formats[league_key] = scoring_format
        else:
            scoring_formats.pop(league_key, None)

    return RosterSnapshot(
        generated_at=datetime.now(timezone.utc),
        leagues=leagues,
        players=players,
        capacities=capacities,
        scoring_formats=scoring_formats,
    )


@contextmanager
def _snapshot_write_lock(path: Path):
    """Serialize cross-process roster writes and JIT read/merge/write cycles."""
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_snapshot(path: Path, snapshot: RosterSnapshot) -> None:
    with _snapshot_write_lock(path):
        _write_snapshot_unlocked(path, snapshot)


def _write_snapshot_unlocked(path: Path, snapshot: RosterSnapshot) -> None:
    payload = {
        "version": SNAPSHOT_VERSION,
        "generatedAt": snapshot.generated_at.isoformat() if snapshot.generated_at else None,
        "leagues": [
            {
                "provider": ref.provider,
                "leagueId": ref.league_id,
                "name": ref.name,
                "myTeamName": ref.my_team_name,
            }
            for ref in snapshot.leagues
        ],
        # Additive to snapshot version 2 so a pre-feature production snapshot
        # remains loadable during deployment and an older rollback safely
        # ignores the extra root field.
        "capacities": {
            league_key: {
                "benchUsed": capacity.bench_used,
                "benchLimit": capacity.bench_limit,
                "irUsed": capacity.ir_used,
                "irLimit": capacity.ir_limit,
            }
            for league_key, capacity in snapshot.capacities.items()
        },
        "scoringFormats": snapshot.scoring_formats,
        "players": [
            {
                "name": p.name,
                "position": p.position,
                "proTeam": p.pro_team,
                "lineupSlot": p.lineup_slot,
                "onMyTeam": p.on_my_team,
                "fantasyTeam": p.fantasy_team,
                "leagueKey": p.league_key,
                "fantasyStarter": p.fantasy_starter,
            }
            for p in snapshot.players
        ],
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary, path)  # atomic; a reader never sees a partial file
    finally:
        temporary.unlink(missing_ok=True)


def load_snapshot(config: Config) -> RosterSnapshot:
    path = roster_path(config)
    if not path.exists():
        raise NotifierError(f"No roster snapshot at {path}. Run bin/refresh-roster.py first.")

    payload = json.loads(path.read_text())
    if payload.get("version") != SNAPSHOT_VERSION:
        raise NotifierError("Roster snapshot version mismatch; re-run bin/refresh-roster.py.")

    generated_raw = payload.get("generatedAt")
    raw_capacities = payload.get("capacities") or {}
    if not isinstance(raw_capacities, dict):
        raw_capacities = {}
    raw_scoring_formats = payload.get("scoringFormats") or {}
    if not isinstance(raw_scoring_formats, dict):
        raw_scoring_formats = {}
    return RosterSnapshot(
        generated_at=datetime.fromisoformat(generated_raw) if generated_raw else None,
        leagues=[
            LeagueRef(
                provider=str(entry.get("provider") or ""),
                league_id=str(entry.get("leagueId") or ""),
                name=str(entry.get("name") or ""),
                my_team_name=str(entry.get("myTeamName") or "My Team"),
            )
            for entry in payload.get("leagues", [])
        ],
        players=[
            RosterPlayer(
                name=str(e.get("name") or ""),
                position=str(e.get("position") or ""),
                pro_team=str(e.get("proTeam") or ""),
                lineup_slot=str(e.get("lineupSlot") or ""),
                on_my_team=bool(e.get("onMyTeam")),
                fantasy_team=str(e.get("fantasyTeam") or ""),
                league_key=str(e.get("leagueKey") or ""),
                fantasy_starter=(
                    bool(e.get("fantasyStarter"))
                    if e.get("fantasyStarter") is not None
                    else None
                ),
            )
            for e in payload.get("players", [])
        ],
        capacities={
            str(league_key): RosterCapacity(
                bench_used=_optional_nonnegative_int(entry.get("benchUsed")),
                bench_limit=_optional_nonnegative_int(entry.get("benchLimit")),
                ir_used=_optional_nonnegative_int(entry.get("irUsed")),
                ir_limit=_optional_nonnegative_int(entry.get("irLimit")),
            )
            for league_key, entry in raw_capacities.items()
            if isinstance(entry, dict)
        },
        scoring_formats={
            str(league_key): str(scoring_format).upper()
            for league_key, scoring_format in raw_scoring_formats.items()
            if str(scoring_format).upper() in {"STD", "HALF", "PPR"}
        },
    )


def snapshot_mtime(config: Config) -> int:
    path = roster_path(config)
    return path.stat().st_mtime_ns if path.exists() else 0
