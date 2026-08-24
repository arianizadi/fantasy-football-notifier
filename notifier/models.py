"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

import hashlib
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
    # X posts can name an injured starter and several possible replacements.
    # Mechanical roster moves are allowed only when deterministic source
    # parsing confidently attributes the report to this subject.
    subject_confident: bool = True

    def fingerprint_text(self) -> str:
        return f"{self.player_name}|{self.headline}"


def report_revision_identity(item: NewsItem) -> str:
    """Return a stable identity for one exact upstream report revision.

    Some feeds reuse a source GUID while changing the headline or body.  A
    GUID therefore identifies the upstream object, not necessarily the exact
    text the user saw.  Length-prefixing the raw identity fields avoids
    separator ambiguity while keeping true byte-for-byte duplicates stable.
    """
    digest = hashlib.sha256()
    for value in (item.source, item.guid, item.headline, item.body):
        encoded = (value or "").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


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
        """A human label that does not collapse every league to its provider.

        Uniqueness across a set of leagues is handled by the formatter, which
        can see sibling leagues and append a provider/id only when necessary.
        Keeping the real league name here prevents two Sleeper leagues from
        both appearing as the indistinguishable ``SLEEPER``.
        """
        return self.name.strip() or (self.provider or "LEAGUE").upper()


@dataclass(frozen=True)
class RosterPlayer:
    name: str
    position: str
    pro_team: str
    lineup_slot: str
    on_my_team: bool
    fantasy_team: str
    league_key: str

    @property
    def can_be_started_from_bench(self) -> bool:
        """Whether this roster entry is a real, active bench alternative.

        Provider adapters encode reserve/taxi/NFL-inactive state in
        ``lineup_slot`` so it survives the existing roster snapshot format.
        ESPN's IR slot and Sleeper reserve/taxi players must never be offered
        as a player the manager can start now.
        """
        return self.lineup_slot.strip().upper() in {"BE", "BN", "BENCH"}


@dataclass(frozen=True)
class RosterCapacity:
    """Current occupancy and configured limits for one fantasy roster.

    The values are provider facts, not inferred from player injury status.
    ``None`` means the provider did not return enough information to make a
    trustworthy claim, so formatters should omit that value.
    """

    bench_used: int | None = None
    bench_limit: int | None = None
    ir_used: int | None = None
    ir_limit: int | None = None


@dataclass
class RosterSnapshot:
    generated_at: datetime | None
    leagues: list[LeagueRef] = field(default_factory=list)
    players: list[RosterPlayer] = field(default_factory=list)
    capacities: dict[str, RosterCapacity] = field(default_factory=dict)
    # FantasyPros publishes scoring-specific rankings. Keep the provider's
    # actual reception scoring beside each league rather than guessing one
    # global format for both ESPN and Sleeper.
    scoring_formats: dict[str, str] = field(default_factory=dict)

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
    # Every waiver candidate is rechecked against live league rosters. If that
    # provider refresh fails, the alert remains useful as news but all ADD/FA
    # claims are removed and the formatter explains the limitation.
    availability_refresh_failed: bool = False
    # Persisted retries are labelled so an older report is never presented as
    # if it just broke. Claimable actions are revalidated before this is set.
    delivery_delayed: bool = False
