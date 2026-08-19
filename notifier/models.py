"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class NewsItem:
    """A single piece of raw news from an upstream source."""

    source: str
    guid: str
    player_name: str
    headline: str
    body: str
    url: str
    published_at: datetime | None

    def fingerprint_text(self) -> str:
        return f"{self.player_name}|{self.headline}"


@dataclass(frozen=True)
class LeagueRef:
    """One fantasy league the user plays in."""

    provider: str  # "espn" | "sleeper"
    league_id: str
    name: str
    my_team_name: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.league_id}"

    @property
    def label(self) -> str:
        return self.name or self.provider.upper()

    @property
    def short_label(self) -> str:
        """Compact, scannable league tag for alert lines.

        The provider name reads more consistently than an arbitrary league
        name, whose first word may not be a useful notification label.
        """
        return (self.provider or "LEAGUE").upper()


@dataclass(frozen=True)
class RosterPlayer:
    name: str
    position: str
    pro_team: str
    lineup_slot: str
    on_my_team: bool
    fantasy_team: str
    league_key: str


@dataclass
class RosterSnapshot:
    generated_at: datetime | None
    leagues: list[LeagueRef] = field(default_factory=list)
    players: list[RosterPlayer] = field(default_factory=list)

    def mine(self, league_key: str | None = None) -> list[RosterPlayer]:
        return [
            player
            for player in self.players
            if player.on_my_team and (league_key is None or player.league_key == league_key)
        ]

    def drafted_leagues(self) -> list[LeagueRef]:
        """Leagues whose snapshot contains at least one player owned by the user."""
        drafted_keys = {player.league_key for player in self.players if player.on_my_team}
        return [league for league in self.leagues if league.key in drafted_keys]

    def is_drafted(self, league_key: str) -> bool:
        return any(player.on_my_team and player.league_key == league_key for player in self.players)

    def league(self, league_key: str) -> LeagueRef | None:
        for ref in self.leagues:
            if ref.key == league_key:
                return ref
        return None


@dataclass(frozen=True)
class Classification:
    event_type: str
    severity: int
    fantasy_impact: str
    is_actionable: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class Alert:
    item: NewsItem
    classification: Classification
    tier: str  # "mine" | "claimable" | "rival" | "league" | "preseason"
    per_league: list[Any] = field(default_factory=list)  # list[LeaguePlays]
    context: Any = None  # TeamContext, for the investigation block
    # Every league, not just ones with an action, so ownership tags in the
    # depth chart stay complete.
    all_leagues: list[LeagueRef] = field(default_factory=list)
