"""Turn "player X is hurt" into "here is the move you can make, per league".

The value of fast news is not knowing sooner, it is claiming the backup before
the other managers do. That requires three things joined together:

  1. NFL depth charts    -> who is next up behind the affected player
  2. every team's roster -> whether that backup is actually claimable
  3. your own roster     -> whether you already cover the hole internally

(2) is why the snapshot stores every team's roster and not just yours.
Availability is computed per league: the same backup can be a free agent in
one league and rostered in another, so each league gets its own verdict.

Depth charts are computed in plain code, never asked of the model. A
hallucinated depth chart would produce confidently wrong waiver advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .matcher import compact_key
from .models import LeagueRef, RosterSnapshot

SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
# Sleeper overall rank. Applied only to DEEPER candidates: the immediate next
# man up is always surfaced regardless of rank, because a backup QB ranks badly
# precisely because he is a backup. Filtering him out would remove exactly the
# handcuff this feature exists to surface (QB1 out -> claim QB2).
MAX_USEFUL_SEARCH_RANK = 400
MAX_BENEFICIARIES = 3
# How much of the NFL depth chart to show for investigation. The recommendation
# is only ever the mechanical next-man-up; the real beneficiary is often the
# slot receiver or the TE absorbing targets, so the full chain plus the other
# skill positions on that team go in the message for you to judge.
MAX_DEPTH_SHOWN = 5
ADJACENT_POSITIONS = ("WR", "TE", "RB", "QB")
BENCH_SLOTS = frozenset({"BE", "BN", "BENCH", "IR"})


@dataclass(frozen=True)
class Beneficiary:
    name: str
    position: str
    depth_order: int | None
    state: str  # "free_agent" | "mine" | "rostered"
    fantasy_team: str = ""


@dataclass(frozen=True)
class DepthEntry:
    """One player on the affected NFL team, with per-league ownership."""

    name: str
    position: str
    depth_order: int | None
    search_rank: int | None
    is_subject: bool
    # league_key -> ("free_agent" | "mine" | "rostered", fantasy_team)
    ownership: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class TeamContext:
    """Everything on the injured player's NFL team worth eyeballing."""

    team: str
    subject_position: str
    same_position: list[DepthEntry] = field(default_factory=list)
    adjacent: list[DepthEntry] = field(default_factory=list)


@dataclass
class LeaguePlays:
    """What this news means inside one specific league."""

    league: LeagueRef
    subject_state: str  # "mine" | "rostered" | "free_agent"
    subject_owner: str
    beneficiaries: list[Beneficiary] = field(default_factory=list)
    bench_options: list[str] = field(default_factory=list)

    @property
    def claimable(self) -> list[Beneficiary]:
        return [b for b in self.beneficiaries if b.state == "free_agent"]

    @property
    def has_action(self) -> bool:
        return bool(self.claimable or self.bench_options)


class DepthCharts:
    """NFL depth charts joined against per-league roster ownership."""

    def __init__(self, player_index: dict[str, Any], snapshot: RosterSnapshot) -> None:
        self._by_key: dict[str, dict[str, Any]] = {}
        self._by_team_position: dict[tuple[str, str], list[dict[str, Any]]] = {}

        for record in player_index.values():
            name = record.get("full_name") or ""
            if not name:
                continue
            key = compact_key(name)
            existing = self._by_key.get(key)
            # Sleeper carries retired duplicates; prefer the active entry.
            if existing is None or (
                record.get("status") == "Active" and existing.get("status") != "Active"
            ):
                self._by_key[key] = record
            team, position = record.get("team") or "", record.get("position") or ""
            if team and position in SKILL_POSITIONS and record.get("status") == "Active":
                self._by_team_position.setdefault((team, position), []).append(record)

        for entries in self._by_team_position.values():
            # Unknown depth order sorts last rather than posing as a starter.
            entries.sort(
                key=lambda r: (r.get("depth_chart_order") is None, r.get("depth_chart_order") or 99)
            )

        # league_key -> compact name -> (state, fantasy_team)
        self._ownership: dict[str, dict[str, tuple[str, str]]] = {}
        for player in snapshot.players:
            self._ownership.setdefault(player.league_key, {})[compact_key(player.name)] = (
                "mine" if player.on_my_team else "rostered",
                player.fantasy_team,
            )

    def lookup(self, *names: str) -> dict[str, Any] | None:
        for name in names:
            record = self._by_key.get(compact_key(name))
            if record:
                return record
        return None

    def _state(self, league_key: str, name: str) -> tuple[str, str]:
        return self._ownership.get(league_key, {}).get(compact_key(name), ("free_agent", ""))

    def _entry(
        self,
        record: dict[str, Any],
        snapshot: RosterSnapshot,
        *,
        is_subject: bool = False,
    ) -> DepthEntry:
        name = record.get("full_name") or ""
        return DepthEntry(
            name=name,
            position=record.get("position") or "",
            depth_order=record.get("depth_chart_order"),
            search_rank=record.get("search_rank"),
            is_subject=is_subject,
            ownership={
                league.key: self._state(league.key, name) for league in snapshot.leagues
            },
        )

    def team_context(
        self, record: dict[str, Any], snapshot: RosterSnapshot
    ) -> TeamContext | None:
        """Full depth chart at the subject's position, plus other skill spots."""
        team = record.get("team") or ""
        position = record.get("position") or ""
        if not team or position not in SKILL_POSITIONS:
            return None

        subject_name = record.get("full_name") or ""
        context = TeamContext(team=team, subject_position=position)

        for candidate in self._by_team_position.get((team, position), [])[:MAX_DEPTH_SHOWN]:
            context.same_position.append(
                self._entry(
                    candidate,
                    snapshot,
                    is_subject=candidate.get("full_name") == subject_name,
                )
            )

        # The starter at each other skill position: when a WR goes down the
        # targets often land on the TE or the pass-catching back, which the
        # same-position chain cannot show.
        for other in ADJACENT_POSITIONS:
            if other == position:
                continue
            group = self._by_team_position.get((team, other), [])
            if group:
                context.adjacent.append(self._entry(group[0], snapshot))
        return context

    def build(
        self, *, subject_names: tuple[str, ...], snapshot: RosterSnapshot
    ) -> tuple[dict[str, Any] | None, list[LeaguePlays]]:
        record = self.lookup(*subject_names)
        if record is None:
            return None, []

        subject_name = record.get("full_name") or ""
        team = record.get("team") or ""
        position = record.get("position") or ""
        order = record.get("depth_chart_order")

        # Candidates behind the subject on the NFL depth chart. Grouped by
        # `position` not `depth_chart_position`: a RWR going down promotes the
        # next WR regardless of which side he lines up on.
        candidates: list[dict[str, Any]] = []
        if team and position in SKILL_POSITIONS:
            for candidate in self._by_team_position.get((team, position), []):
                if candidate.get("full_name") == subject_name:
                    continue
                candidate_order = candidate.get("depth_chart_order")
                if order is not None and (candidate_order is None or candidate_order <= order):
                    continue
                is_next_man_up = (
                    order is not None and candidate_order == order + 1
                ) or (order is None and not candidates)
                rank = candidate.get("search_rank")
                if (
                    not is_next_man_up
                    and rank is not None
                    and rank > MAX_USEFUL_SEARCH_RANK
                ):
                    continue
                candidates.append(candidate)
                if len(candidates) >= MAX_BENEFICIARIES:
                    break

        per_league: list[LeaguePlays] = []
        for league in snapshot.leagues:
            state, owner = self._state(league.key, subject_name)
            plays = LeaguePlays(league=league, subject_state=state, subject_owner=owner)

            for candidate in candidates:
                name = candidate.get("full_name") or ""
                cand_state, cand_owner = self._state(league.key, name)
                plays.beneficiaries.append(
                    Beneficiary(
                        name=name,
                        position=candidate.get("position") or "",
                        depth_order=candidate.get("depth_chart_order"),
                        state=cand_state,
                        fantasy_team=cand_owner,
                    )
                )

            if state == "mine" and position:
                for player in snapshot.mine(league.key):
                    if player.position != position:
                        continue
                    if compact_key(player.name) == compact_key(subject_name):
                        continue
                    if player.lineup_slot.upper() in BENCH_SLOTS:
                        plays.bench_options.append(player.name)
                plays.bench_options = plays.bench_options[:4]

            per_league.append(plays)

        return record, per_league


def plays_context_for_model(per_league: list[LeaguePlays]) -> str:
    """Compact factual grounding handed to the classifier."""
    if not per_league or not per_league[0].beneficiaries:
        return ""
    names = [f"{b.name} (depth {b.depth_order})" for b in per_league[0].beneficiaries]
    return "Next up on the NFL depth chart: " + ", ".join(names)
