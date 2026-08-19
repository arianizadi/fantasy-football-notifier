"""Helpers for tracking draft completion independently per fantasy league."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from .models import RosterSnapshot

DraftState = tuple[str, datetime | None]


def completed_unsynced_league_keys(
    snapshot: RosterSnapshot, states: Mapping[str, DraftState]
) -> list[str]:
    """Return completed leagues that still have no user roster in the snapshot."""
    return [
        league_key
        for league_key, (status, _) in states.items()
        if status == "complete" and not snapshot.is_drafted(league_key)
    ]
