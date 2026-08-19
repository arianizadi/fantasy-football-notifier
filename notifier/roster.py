"""Build and read the cached multi-league roster snapshot.

The notifier polls news every ~15s but rosters change a few times a week, so
providers are contacted only by refresh_snapshot() on a slow cron. The hot
path reads the JSON snapshot from disk. This keeps ESPN request volume in
line with the existing espn-sync worker, which deliberately runs twice a day.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import Config, roster_path
from .logging_utils import NotifierError, structured_log
from .models import LeagueRef, RosterPlayer, RosterSnapshot

SNAPSHOT_VERSION = 2


def _normalized_swid(value: str) -> str:
    return value.strip().strip("{}").lower()


def _find_my_espn_team(league: Any, config: Config) -> Any:
    if config.espn_team_id is not None:
        for team in league.teams:
            if int(getattr(team, "team_id", -1)) == config.espn_team_id:
                return team
        raise NotifierError(f"ESPN_TEAM_ID={config.espn_team_id} not found in league")

    target = _normalized_swid(config.espn_swid)
    for team in league.teams:
        owners = getattr(team, "owners", None)
        if not isinstance(owners, list):
            continue
        for owner in owners:
            if isinstance(owner, dict) and _normalized_swid(str(owner.get("id") or "")) == target:
                return team

    raise NotifierError(
        "Could not identify your ESPN team from ESPN_SWID. "
        "Set ESPN_TEAM_ID explicitly to the team id shown in your league URL."
    )


def _load_espn(config: Config) -> tuple[LeagueRef, list[RosterPlayer]]:
    from espn_api.football import League

    league = League(
        league_id=config.espn_league_id,
        year=config.espn_year,
        espn_s2=config.espn_s2,
        swid=config.espn_swid,
        debug=False,
    )
    my_team = _find_my_espn_team(league, config)
    my_team_id = int(getattr(my_team, "team_id", -1))

    ref = LeagueRef(
        provider="espn",
        league_id=str(config.espn_league_id),
        name=str(getattr(league, "league_name", "") or "ESPN League"),
        my_team_name=str(getattr(my_team, "team_name", "") or "My Team"),
    )

    players: list[RosterPlayer] = []
    for team in league.teams:
        team_id = int(getattr(team, "team_id", -1))
        team_name = str(getattr(team, "team_name", "") or f"Team {team_id}")
        for player in getattr(team, "roster", []) or []:
            name = str(getattr(player, "name", "") or "").strip()
            if not name:
                continue
            players.append(
                RosterPlayer(
                    name=name,
                    position=str(getattr(player, "position", "") or ""),
                    pro_team=str(getattr(player, "proTeam", "") or ""),
                    lineup_slot=str(getattr(player, "lineupSlot", "") or ""),
                    on_my_team=team_id == my_team_id,
                    fantasy_team=team_name,
                    league_key=ref.key,
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
    return ref, players


def _load_sleeper(
    config: Config, session: requests.Session
) -> list[tuple[LeagueRef, list[RosterPlayer]]]:
    from .sources import sleeper, sleeper_league

    player_index = sleeper.load_player_index(config.state_dir, session)
    user_id = sleeper_league.resolve_user_id(session, config.sleeper_username)
    leagues = sleeper_league.list_leagues(session, user_id, config.espn_year)

    if config.sleeper_league_ids:
        allowed = set(config.sleeper_league_ids)
        leagues = [lg for lg in leagues if str(lg.get("league_id")) in allowed]

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

    if config.espn_enabled:
        ref, espn_players = _load_espn(config)
        leagues.append(ref)
        players.extend(espn_players)

    if config.sleeper_username:
        for ref, sleeper_players in _load_sleeper(config, session):
            leagues.append(ref)
            players.extend(sleeper_players)

    if not leagues:
        raise NotifierError(
            "No leagues configured. Set ESPN_LEAGUE_ID and/or SLEEPER_USERNAME."
        )

    snapshot = RosterSnapshot(
        generated_at=datetime.now(timezone.utc), leagues=leagues, players=players
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


def _write_snapshot(path: Path, snapshot: RosterSnapshot) -> None:
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
        "players": [
            {
                "name": p.name,
                "position": p.position,
                "proTeam": p.pro_team,
                "lineupSlot": p.lineup_slot,
                "onMyTeam": p.on_my_team,
                "fantasyTeam": p.fantasy_team,
                "leagueKey": p.league_key,
            }
            for p in snapshot.players
        ],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, path)  # atomic; a reader never sees a partial file


def load_snapshot(config: Config) -> RosterSnapshot:
    path = roster_path(config)
    if not path.exists():
        raise NotifierError(f"No roster snapshot at {path}. Run bin/refresh-roster.py first.")

    payload = json.loads(path.read_text())
    if payload.get("version") != SNAPSHOT_VERSION:
        raise NotifierError("Roster snapshot version mismatch; re-run bin/refresh-roster.py.")

    generated_raw = payload.get("generatedAt")
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
            )
            for e in payload.get("players", [])
        ],
    )


def snapshot_mtime(config: Config) -> float:
    path = roster_path(config)
    return path.stat().st_mtime if path.exists() else 0.0
