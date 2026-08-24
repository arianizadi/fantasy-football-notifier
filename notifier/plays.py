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

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .matcher import compact_key, player_name_in_text
from .models import LeagueRef, RosterCapacity, RosterSnapshot

SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
# Sleeper overall rank. Applied only to DEEPER candidates: the immediate next
# man up is always surfaced regardless of rank, because a backup QB ranks badly
# precisely because he is a backup. Filtering him out would remove exactly the
# handcuff this feature exists to surface (QB1 out -> claim QB2).
MAX_USEFUL_SEARCH_RANK = 400
MAX_BENEFICIARIES = 3
# Keep the investigation block to the subject and nearby useful players. Five
# deep names plus unrelated starters made the source data look more precise
# than it is and buried the actual news subject.
MAX_DEPTH_SHOWN = 3
ADJACENT_POSITIONS = ("WR", "TE", "RB", "QB")

# Mechanical next-man-up recommendations are safe only when the event removes
# or materially threatens a player. A return/signing/trade may affect a depth
# chart, but direction cannot be inferred from event type alone and must never
# turn into an automatic ADD of the old backup.
BACKUP_MOVE_MIN_SEVERITY = {
    "injury": 3,
    "inactive": 3,
    "suspension": 4,
    "release": 4,
}
LINEUP_SUB_MIN_SEVERITY = {
    "injury": 4,
    "inactive": 3,
    "suspension": 4,
    "release": 4,
}
def normalized_event_type(event_type: str) -> str:
    return (event_type or "other").strip().lower().replace("-", "_").replace(" ", "_")


def event_allows_backup_moves(event_type: str, severity: int) -> bool:
    """Whether next-man-up waiver suggestions are valid for this event."""
    event = normalized_event_type(event_type)
    minimum = BACKUP_MOVE_MIN_SEVERITY.get(event)
    return minimum is not None and severity >= minimum


def event_allows_lineup_substitution(event_type: str, severity: int) -> bool:
    """Whether an owned subject is unavailable enough to suggest a bench sub."""
    event = normalized_event_type(event_type)
    minimum = LINEUP_SUB_MIN_SEVERITY.get(event)
    return minimum is not None and severity >= minimum


@dataclass(frozen=True)
class Beneficiary:
    name: str
    position: str
    depth_order: int | None
    state: str  # "free_agent" | "mine" | "rostered"
    fantasy_team: str = ""
    named_in_report: bool = False
    pro_team: str = ""
    # Cached, scoring-specific FantasyPros consensus context. These fields can
    # rank/corroborate Sleeper-generated options but never establish a role or
    # live league availability.
    fantasypros_waiver_rank: int | None = None
    fantasypros_waiver_pos_rank: str = ""
    fantasypros_ros_rank: int | None = None
    fantasypros_ros_pos_rank: str = ""
    fantasypros_scoring: str = ""
    fantasypros_updated_at: str = ""


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
    # Both fields are explicitly attributed to Sleeper by the formatter. They
    # are not conclusions drawn from the news item or the language model.
    sleeper_injury_status: str = ""
    sleeper_status: str = ""


@dataclass
class TeamContext:
    """Sleeper depth context around the news subject."""

    team: str
    subject_position: str
    same_position: list[DepthEntry] = field(default_factory=list)
    adjacent: list[DepthEntry] = field(default_factory=list)
    player_index_refreshed_at: datetime | None = None


@dataclass
class LeaguePlays:
    """What this news means inside one specific league."""

    league: LeagueRef
    subject_state: str  # "mine" | "rostered" | "free_agent"
    subject_owner: str
    beneficiaries: list[Beneficiary] = field(default_factory=list)
    bench_options: list[str] = field(default_factory=list)
    capacity: RosterCapacity | None = None
    scoring_format: str = ""

    @property
    def claimable(self) -> list[Beneficiary]:
        return [b for b in self.beneficiaries if b.state == "free_agent"]

    @property
    def has_action(self) -> bool:
        return bool(self.claimable or self.bench_options)

    def for_event(self, event_type: str, severity: int) -> LeaguePlays:
        """Copy containing only recommendations valid for this classified event."""
        return replace(
            self,
            beneficiaries=(
                self.beneficiaries
                if event_allows_backup_moves(event_type, severity)
                else []
            ),
            bench_options=(
                self.bench_options
                if event_allows_lineup_substitution(event_type, severity)
                else []
            ),
        )

    def has_action_for(self, event_type: str, severity: int) -> bool:
        return self.for_event(event_type, severity).has_action


def plays_for_event(
    per_league: list[LeaguePlays], event_type: str, severity: int
) -> list[LeaguePlays]:
    """Apply the deterministic action policy to all league-specific plays.

    The pipeline should call this immediately after classification and before
    tier selection. That prevents a positive return from being mislabeled as
    a waiver opportunity merely because the raw depth chart has a free backup.
    """
    return [plays.for_event(event_type, severity) for plays in per_league]


class DepthCharts:
    """NFL depth charts joined against per-league roster ownership."""

    def __init__(
        self,
        player_index: dict[str, Any],
        snapshot: RosterSnapshot,
        *,
        player_index_refreshed_at: datetime | None = None,
    ) -> None:
        self._by_key: dict[str, dict[str, Any]] = {}
        self._by_team_position: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.player_index_refreshed_at = (
            player_index_refreshed_at
            if player_index_refreshed_at is not None
            else getattr(player_index, "refreshed_at", None)
        )

        for record in player_index.values():
            if not isinstance(record, dict):
                continue
            name = record.get("full_name") or ""
            if not name:
                continue
            key = compact_key(name)
            existing = self._by_key.get(key)
            # Sleeper carries retired duplicates; prefer the active entry.
            if existing is None or (
                self._is_active(record) and not self._is_active(existing)
            ):
                self._by_key[key] = record
            team, position = record.get("team") or "", record.get("position") or ""
            if team and position in SKILL_POSITIONS and self._is_active(record):
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

    @staticmethod
    def _is_active(record: dict[str, Any]) -> bool:
        return str(record.get("status") or "").strip().casefold() == "active"

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
                league.key: self._state(league.key, name)
                for league in snapshot.drafted_leagues()
            },
            sleeper_injury_status=str(record.get("injury_status") or ""),
            sleeper_status=str(record.get("status") or ""),
        )

    def _nearby_same_position(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Subject plus the nearest useful players in Sleeper's depth order."""
        team = record.get("team") or ""
        position = record.get("position") or ""
        subject_key = compact_key(record.get("full_name") or "")
        group = list(self._by_team_position.get((team, position), []))

        # PUP/IR players are absent from Sleeper's active depth group. The
        # subject still belongs in its own alert, with Sleeper's actual status
        # displayed rather than a notifier-invented injury label.
        if not any(compact_key(candidate.get("full_name") or "") == subject_key for candidate in group):
            group.append(record)
            group.sort(
                key=lambda candidate: (
                    candidate.get("depth_chart_order") is None,
                    candidate.get("depth_chart_order") or 99,
                )
            )

        subject_index = next(
            (
                index
                for index, candidate in enumerate(group)
                if compact_key(candidate.get("full_name") or "") == subject_key
            ),
            0,
        )
        selected: list[dict[str, Any]] = []
        if subject_index > 0:
            selected.append(group[subject_index - 1])
        selected.append(group[subject_index])

        following = group[subject_index + 1 :]
        for index, candidate in enumerate(following):
            rank = candidate.get("search_rank")
            # Always retain the next two depth entries; deeper long shots are
            # only useful when Sleeper's overall search rank supports them.
            if index > 1 and rank is not None and rank > MAX_USEFUL_SEARCH_RANK:
                continue
            selected.append(candidate)
            if len(selected) >= MAX_DEPTH_SHOWN:
                break
        return selected[:MAX_DEPTH_SHOWN]

    def team_context(
        self, record: dict[str, Any], snapshot: RosterSnapshot
    ) -> TeamContext | None:
        """Subject-centered depth context plus top players at adjacent spots."""
        team = record.get("team") or ""
        position = record.get("position") or ""
        if not team or position not in SKILL_POSITIONS:
            return None

        subject_name = record.get("full_name") or ""
        context = TeamContext(
            team=team,
            subject_position=position,
            player_index_refreshed_at=self.player_index_refreshed_at,
        )

        for candidate in self._nearby_same_position(record):
            context.same_position.append(
                self._entry(
                    candidate,
                    snapshot,
                    is_subject=(
                        compact_key(candidate.get("full_name") or "")
                        == compact_key(subject_name)
                    ),
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
        self,
        *,
        subject_names: tuple[str, ...],
        snapshot: RosterSnapshot,
        report_text: str = "",
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
        if team and position in SKILL_POSITIONS and order is not None:
            successors: list[dict[str, Any]] = []
            for candidate in self._by_team_position.get((team, position), []):
                if candidate.get("full_name") == subject_name:
                    continue
                candidate_order = candidate.get("depth_chart_order")
                if candidate_order is None or candidate_order <= order:
                    continue
                successors.append(candidate)

            for successor_index, candidate in enumerate(successors):
                rank = candidate.get("search_rank")
                # Sleeper order values sometimes contain gaps, and search_rank
                # describes search relevance rather than expected workload.
                # Always retain the nearest two sorted successors so an
                # unsettled backfield can surface both plausible options.
                if (
                    successor_index >= 2
                    and rank is not None
                    and rank > MAX_USEFUL_SEARCH_RANK
                ):
                    continue
                candidates.append(candidate)
                if len(candidates) >= MAX_BENEFICIARIES:
                    break

        per_league: list[LeaguePlays] = []
        # An empty, pre-draft league must not make every NFL player appear to be
        # a free agent. Activate each league only after its own roster is present.
        for league in snapshot.drafted_leagues():
            state, owner = self._state(league.key, subject_name)
            plays = LeaguePlays(
                league=league,
                subject_state=state,
                subject_owner=owner,
                capacity=snapshot.capacities.get(league.key),
                scoring_format=snapshot.scoring_formats.get(league.key, ""),
            )

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
                        named_in_report=(
                            bool(report_text)
                            and player_name_in_text(name, report_text)
                        ),
                        pro_team=str(candidate.get("team") or ""),
                    )
                )

            if state == "mine" and position:
                for player in snapshot.mine(league.key):
                    if player.position != position:
                        continue
                    if compact_key(player.name) == compact_key(subject_name):
                        continue
                    if player.can_be_started_from_bench:
                        plays.bench_options.append(player.name)
                plays.bench_options = plays.bench_options[:4]

            per_league.append(plays)

        return record, per_league


def plays_context_for_model(per_league: list[LeaguePlays]) -> str:
    """Compact factual grounding handed to the classifier."""
    if not per_league or not per_league[0].beneficiaries:
        return ""
    names = [f"{b.name} (depth {b.depth_order})" for b in per_league[0].beneficiaries]
    return (
        "Sleeper lists these players after the subject: "
        + ", ".join(names)
        + ". This order does not confirm a starter, workload, or touch share."
    )
