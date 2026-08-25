"""Read a user's Sleeper leagues and rosters.

Public, unauthenticated, and free - only a username is needed. Sleeper stores
rosters as player-id lists, so the player index resolves them to names.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..logging_utils import NotifierError, structured_log
from ..models import LeagueRef, RosterCapacity, RosterPlayer

BASE_URL = "https://api.sleeper.app/v1"
REQUEST_TIMEOUT = 20

# Sleeper returns starters as an ordered list; index maps to a lineup slot only
# loosely. Reserve/taxi/NFL-inactive state is encoded in the eligibility field;
# raw fantasy starter membership is persisted separately in the additive
# provider-neutral snapshot field.
STARTER_SLOT = "ST"
BENCH_SLOT = "BE"
RESERVE_SLOT = "RESERVE"
TAXI_SLOT = "TAXI"
NFL_INACTIVE_SLOT = "NFL_INACTIVE"


def _player_ids(values: Any) -> set[str]:
    """Normalize Sleeper roster ids while discarding empty placeholders."""
    if not isinstance(values, list):
        return set()
    return {
        str(value)
        for value in values
        if value is not None and str(value).strip() not in {"", "0", "None"}
    }


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _scoring_format_from_receptions(value: Any) -> str:
    try:
        points = float(value)
    except (TypeError, ValueError):
        return ""
    if points >= 0.75:
        return "PPR"
    if points >= 0.25:
        return "HALF"
    return "STD"


def roster_slot(
    player_id: str,
    *,
    starters: set[str],
    reserve: set[str],
    taxi: set[str],
    nfl_status: str,
) -> str:
    """Return an availability-safe synthetic lineup slot for Sleeper."""
    if player_id in reserve:
        return RESERVE_SLOT
    if player_id in taxi:
        return TAXI_SLOT
    if nfl_status and nfl_status.casefold() != "active":
        return NFL_INACTIVE_SLOT
    return STARTER_SLOT if player_id in starters else BENCH_SLOT


def resolve_user_id(session: requests.Session, username: str) -> str:
    response = session.get(f"{BASE_URL}/user/{username}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if not payload or not payload.get("user_id"):
        raise NotifierError(f"Sleeper user '{username}' not found")
    return str(payload["user_id"])


def list_leagues(session: requests.Session, user_id: str, season: int) -> list[dict[str, Any]]:
    response = session.get(
        f"{BASE_URL}/user/{user_id}/leagues/nfl/{season}", timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json() or []


def fetch_league_rosters(
    session: requests.Session,
    league: dict[str, Any],
    user_id: str,
    player_index: dict[str, Any],
) -> tuple[LeagueRef, list[RosterPlayer], RosterCapacity, str]:
    league_id = str(league.get("league_id"))
    league_name = str(league.get("name") or f"Sleeper {league_id}")

    rosters = session.get(
        f"{BASE_URL}/league/{league_id}/rosters", timeout=REQUEST_TIMEOUT
    )
    rosters.raise_for_status()
    users = session.get(f"{BASE_URL}/league/{league_id}/users", timeout=REQUEST_TIMEOUT)
    users.raise_for_status()

    team_names: dict[str, str] = {}
    for user in users.json() or []:
        uid = str(user.get("user_id"))
        metadata = user.get("metadata") or {}
        team_names[uid] = str(
            metadata.get("team_name") or user.get("display_name") or f"Team {uid}"
        )

    ref = LeagueRef(
        provider="sleeper",
        league_id=league_id,
        name=league_name,
        my_team_name=team_names.get(user_id, "My Team"),
    )
    players: list[RosterPlayer] = []
    raw_rosters = rosters.json() or []
    my_roster: dict[str, Any] | None = None
    for roster in raw_rosters:
        if not isinstance(roster, dict):
            continue
        owner_id = str(roster.get("owner_id") or "")
        is_mine = owner_id == user_id
        if is_mine:
            my_roster = roster
        fantasy_team = team_names.get(owner_id, f"Team {roster.get('roster_id')}")
        starters = _player_ids(roster.get("starters"))
        reserve = _player_ids(roster.get("reserve"))
        taxi = _player_ids(roster.get("taxi"))
        for player_id in roster.get("players") or []:
            player_key = str(player_id)
            record = player_index.get(player_key)
            if not record or not record.get("full_name"):
                continue
            players.append(
                RosterPlayer(
                    name=record["full_name"],
                    position=record.get("position") or "",
                    pro_team=record.get("team") or "",
                    lineup_slot=roster_slot(
                        player_key,
                        starters=starters,
                        reserve=reserve,
                        taxi=taxi,
                        nfl_status=str(record.get("status") or ""),
                    ),
                    on_my_team=is_mine,
                    fantasy_team=fantasy_team,
                    league_key=ref.key,
                    # Reserve/taxi membership wins inconsistent provider data,
                    # but daily-cached NFL status does not erase the raw fantasy
                    # lineup fact used to assess replacement urgency.
                    fantasy_starter=(
                        player_key in starters
                        and player_key not in reserve
                        and player_key not in taxi
                    ),
                )
            )

    positions = league.get("roster_positions")
    bench_limit = (
        sum(
            str(position).strip().upper() in {"BN", "BE", "BENCH"}
            for position in positions
        )
        if isinstance(positions, list)
        else None
    )
    league_settings = league.get("settings") or {}
    if not isinstance(league_settings, dict):
        league_settings = {}
    ir_limit = _nonnegative_int(league_settings.get("reserve_slots"))
    if ir_limit is None and isinstance(positions, list):
        ir_limit = sum(
            str(position).strip().upper() in {"IR", "RESERVE"}
            for position in positions
        )

    bench_used: int | None = None
    ir_used: int | None = None
    if my_roster is not None:
        rostered = _player_ids(my_roster.get("players"))
        starters = _player_ids(my_roster.get("starters"))
        reserve = _player_ids(my_roster.get("reserve"))
        taxi = _player_ids(my_roster.get("taxi"))
        # Count raw roster membership, not the normalized lineup_slot. An NFL
        # inactive player still consumes an ordinary bench slot even though
        # lineup safety labels that player NFL_INACTIVE elsewhere.
        bench_used = len(rostered - starters - reserve - taxi)
        ir_used = len(reserve)

    capacity = RosterCapacity(
        bench_used=bench_used,
        bench_limit=bench_limit,
        ir_used=ir_used,
        ir_limit=ir_limit,
    )

    scoring_settings = league.get("scoring_settings")
    scoring_format = ""
    if isinstance(scoring_settings, dict):
        scoring_format = _scoring_format_from_receptions(
            scoring_settings.get("rec", 0)
        )

    structured_log(
        logging.INFO,
        "sleeper.league_loaded",
        leagueName=league_name,
        myPlayerCount=sum(1 for p in players if p.on_my_team),
        leaguePlayerCount=len(players),
    )
    return ref, players, capacity, scoring_format
