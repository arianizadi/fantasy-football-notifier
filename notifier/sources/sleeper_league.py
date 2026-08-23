"""Read a user's Sleeper leagues and rosters.

Public, unauthenticated, and free - only a username is needed. Sleeper stores
rosters as player-id lists, so the player index resolves them to names.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..logging_utils import NotifierError, structured_log
from ..models import LeagueRef, RosterPlayer

BASE_URL = "https://api.sleeper.app/v1"
REQUEST_TIMEOUT = 20

# Sleeper returns starters as an ordered list; index maps to a lineup slot only
# loosely. Reserve/taxi/NFL-inactive state is encoded in the same field so it
# survives the provider-neutral roster snapshot without a schema migration.
STARTER_SLOT = "ST"
BENCH_SLOT = "BE"
RESERVE_SLOT = "RESERVE"
TAXI_SLOT = "TAXI"
NFL_INACTIVE_SLOT = "NFL_INACTIVE"


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
) -> tuple[LeagueRef, list[RosterPlayer]]:
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
    for roster in rosters.json() or []:
        owner_id = str(roster.get("owner_id") or "")
        is_mine = owner_id == user_id
        fantasy_team = team_names.get(owner_id, f"Team {roster.get('roster_id')}")
        starters = {str(p) for p in (roster.get("starters") or []) if p}
        reserve = {str(p) for p in (roster.get("reserve") or []) if p}
        taxi = {str(p) for p in (roster.get("taxi") or []) if p}
        for player_id in roster.get("players") or []:
            record = player_index.get(str(player_id))
            if not record or not record.get("full_name"):
                continue
            players.append(
                RosterPlayer(
                    name=record["full_name"],
                    position=record.get("position") or "",
                    pro_team=record.get("team") or "",
                    lineup_slot=roster_slot(
                        str(player_id),
                        starters=starters,
                        reserve=reserve,
                        taxi=taxi,
                        nfl_status=str(record.get("status") or ""),
                    ),
                    on_my_team=is_mine,
                    fantasy_team=fantasy_team,
                    league_key=ref.key,
                )
            )

    structured_log(
        logging.INFO,
        "sleeper.league_loaded",
        leagueName=league_name,
        myPlayerCount=sum(1 for p in players if p.on_my_team),
        leaguePlayerCount=len(players),
    )
    return ref, players
