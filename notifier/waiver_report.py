"""Pure, deterministic waiver-report ranking and HTML formatting.

This module deliberately owns no I/O.  Callers provide current league
availability, roster values, cached ranking percentiles, normalized news
facts, and market movement.  The result is suitable for a scheduled report,
but this module never fetches data, calls a model, or sends a message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from typing import Iterable

from .matcher import compact_key

MAX_TELEGRAM_TEXT = 4096
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
BENCH_SLOTS = frozenset({"BE", "BN", "BENCH"})
OPPORTUNITY_ROLES = frozenset(
    {"committee_leader", "immediate_replacement", "uncertain_replacement"}
)
ROLE_POINTS = {
    "confirmed_starter": 30,
    "committee_leader": 24,
    "immediate_replacement": 22,
    "uncertain_replacement": 15,
    "depth_two": 8,
    "ordinary_backup": 5,
    "unknown": 0,
}
ROLE_REASONS = {
    "confirmed_starter": "Confirmed starting role creates the clearest path to volume.",
    "committee_leader": "Leads the current committee, although the workload is unsettled.",
    "immediate_replacement": "Next in line for the newly available role.",
    "uncertain_replacement": "Has a plausible path to the vacated role, but it is not confirmed.",
    "depth_two": "Currently sits second in the relevant depth order.",
    "ordinary_backup": "Remains a backup without a confirmed workload change.",
    "unknown": "No confirmed role increase is available yet.",
}


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _visible_units(markup: str) -> int:
    parser = _VisibleTextParser()
    parser.feed(markup)
    parser.close()
    return len("".join(parser.parts).encode("utf-16-le")) // 2


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _bounded(value: float | int | None, low: float = 0.0, high: float = 100.0) -> float:
    if value is None:
        return low
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(parsed):
        return low
    return min(high, max(low, parsed))


def _normalized_role(value: str) -> str:
    role = (value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return role if role in ROLE_POINTS else "unknown"


def _normalized_event(value: str) -> str:
    return (value or "other").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class NewsFact:
    """One normalized fact that may support or close a waiver opportunity.

    ``signature`` is the semantic fact identity, not the source post id.  Two
    reporters confirming the same injury should carry the same signature and
    distinct ``source`` values.  A later return for ``subject_name`` cancels
    older opportunity-creating facts for that same subject.
    """

    signature: str
    event_type: str
    severity: int
    published_at: datetime
    source: str
    subject_name: str = ""
    status: str = ""
    directly_names_candidate: bool = False
    creates_opportunity: bool = False
    resolves_opportunity: bool = False
    supports_candidate: bool = True
    description: str = ""


@dataclass(frozen=True)
class CorroboratedFact:
    """A semantic fact after source-post dedupe and corroboration counting."""

    signature: str
    event_type: str
    severity: int
    latest_at: datetime
    sources: tuple[str, ...]
    subject_name: str = ""
    status: str = ""
    directly_names_candidate: bool = False
    creates_opportunity: bool = False
    resolves_opportunity: bool = False
    supports_candidate: bool = True
    description: str = ""

    @property
    def corroboration_count(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class RosterAsset:
    """One player on the user's roster, valued on the report's 0-100 scale."""

    name: str
    position: str
    lineup_slot: str
    waiver_value: float | None
    pro_team: str = ""
    protected: bool = False
    elite: bool = False

    @property
    def is_bench(self) -> bool:
        return self.lineup_slot.strip().upper() in BENCH_SLOTS


@dataclass(frozen=True)
class CandidateEvidence:
    """All deterministic evidence needed to score one league-available player."""

    name: str
    position: str
    pro_team: str
    available: bool = True
    role: str = "unknown"
    role_subject: str = ""
    depth_order: int | None = None
    fantasypros_waiver_percentile: float | None = None
    fantasypros_ros_percentile: float | None = None
    fantasypros_updated_at: datetime | None = None
    # A league-provider projection/rank is a deliberately lower-weight
    # fallback while FantasyPros has not published the requested dataset.
    # It must be labeled separately and is never presented as FantasyPros.
    platform_quality_percentile: float | None = None
    platform_quality_source: str = ""
    sleeper_adds_6h: int = 0
    sleeper_adds_24h: int = 0
    sleeper_data_age_hours: float | None = None
    injury_status: str = ""
    committee_unresolved: bool = False
    contradictory_reports: bool = False
    speculative: bool = False
    # Direct injury/provider availability facts can leave a player worth
    # monitoring without making an immediate claim safe.
    recommendation_blocked: bool = False
    blocked_reason: str = ""
    facts: tuple[NewsFact, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WaiverReportContext:
    """League and roster facts shared by every candidate in one report."""

    league_name: str
    scoring_format: str
    generated_at: datetime
    expected_waiver_at: datetime | None = None
    bench_used: int | None = None
    bench_limit: int | None = None
    ir_used: int | None = None
    ir_limit: int | None = None
    roster: tuple[RosterAsset, ...] = field(default_factory=tuple)
    swap_threshold: int = 10
    fact_lookback_hours: int = 168
    fantasypros_max_age_hours: int = 12
    starting_qb_unavailable: bool = False
    top_limit: int = 5
    watch_limit: int = 5
    waiver_method: str = ""
    waiver_priority: int | None = None

    @property
    def bench_capacity_known(self) -> bool:
        return bool(
            self.bench_limit is not None
            and self.bench_limit > 0
            and self.bench_used is not None
        )

    @property
    def bench_full(self) -> bool:
        return bool(
            self.bench_capacity_known
            and self.bench_used is not None
            and self.bench_limit is not None
            and self.bench_used >= self.bench_limit
        )


@dataclass(frozen=True)
class ScoreBreakdown:
    role_opportunity: int
    fantasy_quality: int
    news_confidence: int
    roster_fit: int
    market_movement: int
    risk_penalty: int
    quarterback_penalty: int

    @property
    def total(self) -> int:
        return max(
            0,
            min(
                100,
                self.role_opportunity
                + self.fantasy_quality
                + self.news_confidence
                + self.roster_fit
                + self.market_movement
                - self.risk_penalty
                - self.quarterback_penalty,
            ),
        )


@dataclass(frozen=True)
class RankedCandidate:
    evidence: CandidateEvidence
    score: ScoreBreakdown
    tier: str
    confidence: str
    reasons: tuple[str, ...]
    risks: tuple[str, ...]
    fantasypros_used: bool
    quality_source: str = ""
    suggested_drop: RosterAsset | None = None
    swap_delta: int | None = None
    swap_allowed: bool = True
    return_cancelled: bool = False


@dataclass(frozen=True)
class LeagueWaiverReport:
    context: WaiverReportContext
    candidates: tuple[RankedCandidate, ...]
    evidence_label: str


def dedupe_recent_facts(
    facts: Iterable[NewsFact],
    *,
    now: datetime,
    max_age_hours: int = 168,
) -> tuple[CorroboratedFact, ...]:
    """Collapse source posts into semantic facts while retaining corroboration."""

    current = _utc(now)
    grouped: dict[tuple[str, str], list[NewsFact]] = {}
    for fact in facts:
        published = _utc(fact.published_at)
        age_hours = (current - published).total_seconds() / 3600
        if age_hours < -1 or age_hours > max_age_hours:
            continue
        subject = compact_key(fact.subject_name)
        signature = (fact.signature or "").strip().casefold()
        if not signature:
            signature = "|".join(
                (
                    _normalized_event(fact.event_type),
                    subject,
                    (fact.status or "unspecified").strip().casefold(),
                    compact_key(fact.description)[:80],
                )
            )
        grouped.setdefault((subject, signature), []).append(fact)

    collapsed: list[CorroboratedFact] = []
    for (_, signature), entries in grouped.items():
        newest = max(entries, key=lambda entry: _utc(entry.published_at))
        sources = tuple(
            sorted(
                {
                    (entry.source or "unknown").strip()
                    for entry in entries
                    if (entry.source or "unknown").strip()
                },
                key=str.casefold,
            )
        )
        collapsed.append(
            CorroboratedFact(
                signature=signature,
                event_type=_normalized_event(newest.event_type),
                severity=max(1, min(5, max(int(entry.severity) for entry in entries))),
                latest_at=max(_utc(entry.published_at) for entry in entries),
                sources=sources,
                subject_name=newest.subject_name,
                status=newest.status,
                directly_names_candidate=any(
                    entry.directly_names_candidate for entry in entries
                ),
                creates_opportunity=any(entry.creates_opportunity for entry in entries),
                resolves_opportunity=any(
                    entry.resolves_opportunity
                    or _normalized_event(entry.event_type) == "return"
                    for entry in entries
                ),
                supports_candidate=any(entry.supports_candidate for entry in entries),
                description=max(
                    (entry.description for entry in entries),
                    key=len,
                    default="",
                ),
            )
        )
    return tuple(
        sorted(collapsed, key=lambda entry: entry.latest_at, reverse=True)
    )


def _active_facts(
    evidence: CandidateEvidence,
    context: WaiverReportContext,
) -> tuple[tuple[CorroboratedFact, ...], bool]:
    facts = dedupe_recent_facts(
        evidence.facts,
        now=context.generated_at,
        max_age_hours=context.fact_lookback_hours,
    )
    latest_resolution: dict[str, datetime] = {}
    for fact in facts:
        subject = compact_key(fact.subject_name)
        if subject and fact.resolves_opportunity:
            latest_resolution[subject] = max(
                fact.latest_at,
                latest_resolution.get(subject, datetime.min.replace(tzinfo=timezone.utc)),
            )

    active: list[CorroboratedFact] = []
    cancelled = False
    for fact in facts:
        subject = compact_key(fact.subject_name)
        if (
            fact.creates_opportunity
            and subject
            and latest_resolution.get(subject, datetime.min.replace(tzinfo=timezone.utc))
            >= fact.latest_at
        ):
            cancelled = True
            continue
        if fact.supports_candidate and not fact.resolves_opportunity:
            active.append(fact)
    return tuple(active), cancelled


def _fantasy_quality_score(
    evidence: CandidateEvidence,
    context: WaiverReportContext,
) -> tuple[int, bool, str]:
    if evidence.fantasypros_updated_at is None:
        fantasypros_fresh = False
    else:
        age = (
            _utc(context.generated_at) - _utc(evidence.fantasypros_updated_at)
        ).total_seconds() / 3600
        fantasypros_fresh = -1 <= age <= context.fantasypros_max_age_hours
    if fantasypros_fresh and (
        evidence.fantasypros_waiver_percentile is not None
        or evidence.fantasypros_ros_percentile is not None
    ):
        waiver = round(_bounded(evidence.fantasypros_waiver_percentile) * 0.15)
        ros = round(_bounded(evidence.fantasypros_ros_percentile) * 0.10)
        return min(25, waiver + ros), True, "FantasyPros"

    fallback = evidence.platform_quality_percentile
    source = (evidence.platform_quality_source or "league-provider rank").strip()
    if fallback is None:
        return 0, False, ""
    # Keep provider projections useful but subordinate to fresh consensus.
    return min(20, round(_bounded(fallback) * 0.20)), False, source


def _news_score(
    facts: tuple[CorroboratedFact, ...],
    context: WaiverReportContext,
) -> int:
    if not facts:
        return 0
    newest = max(facts, key=lambda fact: fact.latest_at)
    age = (_utc(context.generated_at) - newest.latest_at).total_seconds() / 3600
    recency = 6 if age <= 12 else 5 if age <= 24 else 3 if age <= 72 else 1
    severity = round(max(fact.severity for fact in facts) / 5 * 4)
    direct = 4 if any(fact.directly_names_candidate for fact in facts) else 0
    corroboration = max(fact.corroboration_count for fact in facts)
    corroborated = 6 if corroboration >= 3 else 4 if corroboration == 2 else 1
    return min(20, recency + severity + direct + corroborated)


def _market_score(evidence: CandidateEvidence, max_adds_6h: int) -> int:
    adds_6h = max(0, int(evidence.sleeper_adds_6h))
    adds_24h = max(adds_6h, int(evidence.sleeper_adds_24h))
    movement = 0
    if adds_6h and max_adds_6h:
        movement = round(7 * math.log1p(adds_6h) / math.log1p(max_adds_6h))

    prior_18h = max(0, adds_24h - adds_6h)
    acceleration = 0
    if adds_6h:
        if prior_18h == 0:
            acceleration = 3
        else:
            ratio = (adds_6h / 6) / (prior_18h / 18)
            acceleration = 3 if ratio >= 2 else 2 if ratio >= 1.25 else 1
    return min(10, movement + acceleration)


def _risk_penalty(evidence: CandidateEvidence, role: str) -> tuple[int, tuple[str, ...]]:
    penalty = 0
    risks: list[str] = []
    if evidence.committee_unresolved or role == "committee_leader":
        penalty += 4
        risks.append("The workload may be split in an unresolved committee.")
    injury = evidence.injury_status.strip()
    if injury and injury.casefold() not in {"healthy", "none"}:
        normalized_injury = injury.casefold()
        if any(
            marker in normalized_injury
            for marker in ("out", "ir", "pup", "suspend", "recent injury", "recent inactive")
        ):
            penalty += 15
        elif "doubtful" in normalized_injury or "questionable" in normalized_injury:
            penalty += 8
        else:
            penalty += 5
        risks.append(f"Candidate injury status: {injury}.")
    if evidence.depth_order is None and role != "confirmed_starter":
        penalty += 6
        risks.append("No current depth order confirms the path to snaps.")
    if evidence.contradictory_reports:
        penalty += 8
        risks.append("Recent reports conflict about the player's role or availability.")
    if (
        evidence.sleeper_data_age_hours is not None
        and evidence.sleeper_data_age_hours > 24
    ):
        penalty += 4
        risks.append("Sleeper depth data is more than 24 hours old.")
    if evidence.speculative:
        penalty += 5
        risks.append("The opportunity is still speculative.")
    return min(20, penalty), tuple(risks)


def _has_elite_qb(context: WaiverReportContext) -> bool:
    for asset in context.roster:
        if asset.position.upper() != "QB" or asset.is_bench:
            continue
        if asset.elite or compact_key(asset.name) == "lamarjackson":
            return True
    return False


def _position_need(evidence: CandidateEvidence, context: WaiverReportContext) -> int:
    position = evidence.position.upper()
    bench_counts = {
        wanted: sum(
            asset.is_bench and asset.position.upper() == wanted
            for asset in context.roster
        )
        for wanted in SKILL_POSITIONS
    }
    if position in {"RB", "WR"}:
        return 3 if bench_counts[position] < 2 else 1
    if position in {"QB", "TE"}:
        has_starter = any(
            asset.position.upper() == position and not asset.is_bench
            for asset in context.roster
        )
        return 0 if has_starter else 3
    return 0


def _candidate_roster_value(
    evidence: CandidateEvidence,
    context: WaiverReportContext,
    *,
    role: str,
    facts: tuple[CorroboratedFact, ...],
) -> float | None:
    """Estimate candidate hold value on the roster asset's 0-100 scale."""
    value: float | None = None
    if evidence.fantasypros_updated_at is not None:
        age = (
            _utc(context.generated_at) - _utc(evidence.fantasypros_updated_at)
        ).total_seconds() / 3600
        if -1 <= age <= context.fantasypros_max_age_hours:
            if evidence.fantasypros_ros_percentile is not None:
                value = _bounded(evidence.fantasypros_ros_percentile)
    if value is None and evidence.platform_quality_percentile is not None:
        value = _bounded(evidence.platform_quality_percentile)
    if facts:
        opportunity_floor = {
            "confirmed_starter": 72,
            "committee_leader": 60,
            "immediate_replacement": 65,
            "uncertain_replacement": 50,
        }.get(role)
        if opportunity_floor is not None:
            value = max(value or 0, opportunity_floor)
    return value


def _roster_fit(
    evidence: CandidateEvidence,
    context: WaiverReportContext,
    candidate_value: float | None,
) -> tuple[int, RosterAsset | None, int | None, bool]:
    need = _position_need(evidence, context)
    if not context.bench_capacity_known:
        return 0, None, None, False
    if not context.bench_full:
        return min(15, 12 + need), None, None, True

    expendable = sorted(
        (
            asset
            for asset in context.roster
            if asset.is_bench
            and not asset.protected
            and asset.waiver_value is not None
        ),
        key=lambda asset: (asset.waiver_value, compact_key(asset.name)),
    )
    if not expendable:
        return 0, None, None, False
    drop = expendable[0]
    if candidate_value is None or drop.waiver_value is None:
        return 0, drop, None, False
    delta = round(candidate_value - _bounded(drop.waiver_value))
    if delta < context.swap_threshold:
        return 0, drop, delta, False
    base = 12 if delta >= 25 else 10 if delta >= 15 else 8
    return min(15, base + need), drop, delta, True


def _tier(total: int, *, swap_allowed: bool) -> str:
    if swap_allowed and total >= 75:
        return "HIGH PRIORITY"
    if swap_allowed and total >= 62:
        return "CLAIM"
    if total >= 48:
        return "WATCH"
    return "ALTERNATIVE"


def _confidence(
    *,
    role: str,
    facts: tuple[CorroboratedFact, ...],
    evidence: CandidateEvidence,
    return_cancelled: bool,
) -> str:
    corroboration = max((fact.corroboration_count for fact in facts), default=0)
    direct = any(fact.directly_names_candidate for fact in facts)
    if (
        role in {"confirmed_starter", "immediate_replacement"}
        and corroboration >= 2
        and direct
        and not evidence.contradictory_reports
        and not return_cancelled
    ):
        return "High"
    if role in OPPORTUNITY_ROLES or facts:
        return "Medium"
    return "Low"


def score_candidate(
    evidence: CandidateEvidence,
    context: WaiverReportContext,
    *,
    max_adds_6h: int | None = None,
) -> RankedCandidate:
    """Score one available candidate against one current league roster."""

    facts, return_cancelled = _active_facts(evidence, context)
    role = _normalized_role(evidence.role)
    role_subject = compact_key(evidence.role_subject)
    if return_cancelled and role in OPPORTUNITY_ROLES:
        # Only a return for the player whose absence created this role should
        # erase the mechanical next-man-up boost.
        all_facts = dedupe_recent_facts(
            evidence.facts,
            now=context.generated_at,
            max_age_hours=context.fact_lookback_hours,
        )
        returned_subjects = {
            compact_key(fact.subject_name)
            for fact in all_facts
            if fact.resolves_opportunity
        }
        if not role_subject or role_subject in returned_subjects:
            role = "ordinary_backup"
    elif role == "unknown" and any(fact.creates_opportunity for fact in facts):
        role = "uncertain_replacement"

    role_score = ROLE_POINTS[role]
    fantasy_score, fantasypros_used, quality_source = _fantasy_quality_score(
        evidence, context
    )
    news_score = _news_score(facts, context)
    market_score = _market_score(
        evidence,
        max(1, max_adds_6h if max_adds_6h is not None else evidence.sleeper_adds_6h),
    )
    risk, risk_reasons = _risk_penalty(evidence, role)

    qb_penalty = 0
    reasons: list[str] = [ROLE_REASONS[role]]
    risks = list(risk_reasons)
    if (
        evidence.position.upper() == "QB"
        and _has_elite_qb(context)
        and not context.starting_qb_unavailable
    ):
        qb_penalty = 25
        risks.append("Lamar Jackson already fills the only weekly QB need.")

    if fantasypros_used:
        reasons.append(
            "Fresh FantasyPros scoring-specific WAIVER/ROS context supports the ranking."
        )
    elif quality_source and fantasy_score:
        reasons.append(f"{quality_source} supplies the fallback player-quality signal.")
    if facts:
        source_count = max(fact.corroboration_count for fact in facts)
        if source_count >= 2:
            reasons.append(
                f"{source_count} independent sources corroborate the same recent fact."
            )
        elif any(fact.directly_names_candidate for fact in facts):
            reasons.append("The candidate is directly named in the recent report.")
    if market_score >= 7:
        reasons.append("Sleeper adds show strong recent market movement.")
    if return_cancelled:
        risks.append("Later return news canceled the earlier backup opportunity.")

    candidate_value = _candidate_roster_value(
        evidence,
        context,
        role=role,
        facts=facts,
    )
    roster_fit, drop, swap_delta, swap_allowed = _roster_fit(
        evidence,
        context,
        candidate_value,
    )
    if evidence.recommendation_blocked:
        swap_allowed = False
        blocked_reason = (
            evidence.blocked_reason.strip()
            or "Current availability is unresolved; monitor, but do not claim yet."
        )
        risks.insert(0, blocked_reason)
    if not context.bench_capacity_known:
        risks.append(
            "Bench capacity is unavailable, so no claim is authorized."
        )
    elif context.bench_full:
        if swap_allowed and drop is not None:
            reasons.append(
                f"Clears the {context.swap_threshold}-point swap threshold over {drop.name}."
            )
        elif drop is not None:
            risks.append(
                f"Does not clear the {context.swap_threshold}-point swap threshold over {drop.name}."
            )
        else:
            risks.append("The full bench has no expendable player.")

    score = ScoreBreakdown(
        role_opportunity=role_score,
        fantasy_quality=fantasy_score,
        news_confidence=news_score,
        roster_fit=roster_fit,
        market_movement=market_score,
        risk_penalty=risk,
        quarterback_penalty=qb_penalty,
    )
    return RankedCandidate(
        evidence=evidence,
        score=score,
        tier=_tier(score.total, swap_allowed=swap_allowed),
        confidence=_confidence(
            role=role,
            facts=facts,
            evidence=evidence,
            return_cancelled=return_cancelled,
        ),
        reasons=tuple(dict.fromkeys(reasons)),
        risks=tuple(dict.fromkeys(risks)),
        fantasypros_used=fantasypros_used,
        quality_source=quality_source,
        suggested_drop=drop,
        swap_delta=swap_delta,
        swap_allowed=swap_allowed,
        return_cancelled=return_cancelled,
    )


def rank_candidates(
    candidates: Iterable[CandidateEvidence],
    context: WaiverReportContext,
) -> tuple[RankedCandidate, ...]:
    """Rank only players that are currently unowned in this league."""

    available = [candidate for candidate in candidates if candidate.available]
    max_adds = max((max(0, candidate.sleeper_adds_6h) for candidate in available), default=1)
    ranked = [
        score_candidate(candidate, context, max_adds_6h=max_adds)
        for candidate in available
    ]
    tier_order = {"HIGH PRIORITY": 0, "CLAIM": 1, "WATCH": 2, "ALTERNATIVE": 3}
    ranked.sort(
        key=lambda candidate: (
            tier_order[candidate.tier],
            -candidate.score.total,
            SKILL_POSITIONS.index(candidate.evidence.position.upper())
            if candidate.evidence.position.upper() in SKILL_POSITIONS
            else len(SKILL_POSITIONS),
            compact_key(candidate.evidence.name),
        )
    )
    return tuple(ranked)


def build_waiver_report(
    context: WaiverReportContext,
    candidates: Iterable[CandidateEvidence],
) -> LeagueWaiverReport:
    ranked = rank_candidates(candidates, context)
    fantasypros_used = any(candidate.fantasypros_used for candidate in ranked)
    fallback_sources = sorted(
        {
            candidate.quality_source
            for candidate in ranked
            if candidate.quality_source and not candidate.fantasypros_used
        },
        key=str.casefold,
    )
    label = (
        f"Fresh FantasyPros {context.scoring_format.upper()} context plus live news, "
        "Sleeper depth, and market movement."
        if fantasypros_used
        else (
            "FantasyPros unavailable; ranked from live news, Sleeper depth, "
            + (
                f"{', '.join(fallback_sources)} fallback quality, "
                if fallback_sources
                else ""
            )
            + "roster fit, and market movement."
        )
    )
    return LeagueWaiverReport(context=context, candidates=ranked, evidence_label=label)


def _clean_text(value: str, limit: int) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 1)].rstrip() + "…"


def _html(value: str, limit: int = 220) -> str:
    return escape(_clean_text(value, limit), quote=True)


def _capacity_line(context: WaiverReportContext) -> str:
    parts: list[str] = []
    if context.bench_used is not None and context.bench_limit is not None:
        status = "full" if context.bench_full else "open"
        parts.append(f"Bench {context.bench_used}/{context.bench_limit} {status}")
    if context.ir_used is not None and context.ir_limit is not None:
        status = "full" if context.ir_used >= context.ir_limit else "open"
        parts.append(f"IR {context.ir_used}/{context.ir_limit} {status}")
    return " · ".join(parts)


def _waiver_time(context: WaiverReportContext) -> str:
    if context.expected_waiver_at is None:
        return "Waiver time unavailable"
    value = context.expected_waiver_at
    zone = value.tzname() or ""
    return value.strftime("%a %b %d · %I:%M %p").replace(" 0", " ") + (f" {zone}" if zone else "")


def _candidate_block(candidate: RankedCandidate, index: int) -> str:
    evidence = candidate.evidence
    lines = [
        f"<b>{index}. {_html(evidence.name, 80)} · {_html(candidate.tier, 30)}</b>",
        f"{_html(evidence.position, 8)} · {_html(evidence.pro_team or 'FA', 12)} · {_html(candidate.confidence, 12)} confidence",
    ]
    if candidate.reasons:
        lines.append(f"<b>Why</b> · {_html(candidate.reasons[0], 260)}")
    if len(candidate.reasons) > 1:
        lines.append(f"<b>Evidence</b> · {_html(candidate.reasons[1], 260)}")
    if candidate.suggested_drop is not None and candidate.swap_allowed:
        lines.append(f"<b>Move</b> · Add over {_html(candidate.suggested_drop.name, 80)}.")
    if candidate.risks:
        lines.append(f"<b>Risk</b> · {_html(candidate.risks[0], 260)}")
    return "\n".join(lines)


def _watch_block(candidate: RankedCandidate) -> str:
    reason = candidate.reasons[0] if candidate.reasons else "Monitor for a confirmed role change."
    risk = candidate.risks[0] if candidate.risks else "Workload remains uncertain."
    return (
        f"<b>{_html(candidate.evidence.name, 80)}</b> · "
        f"{_html(candidate.evidence.position, 8)} · {_html(candidate.confidence, 12)} confidence\n"
        f"↳ {_html(reason, 220)}\n"
        f"↳ Risk: {_html(risk, 180)}"
    )


def _pack_html_blocks(blocks: list[str], max_chars: int) -> tuple[str, ...]:
    if max_chars < 256 or max_chars > MAX_TELEGRAM_TEXT:
        raise ValueError(f"max_chars must be between 256 and {MAX_TELEGRAM_TEXT}")
    continuation = "<b>🏈 WAIVER PLAN · continued</b>"
    parts: list[str] = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if _visible_units(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = f"{continuation}\n\n{block}"
        if _visible_units(current) > max_chars:
            # Every caller-controlled field is already bounded, so this is a
            # defensive invariant rather than a lossy HTML string split.
            raise ValueError("A formatted waiver-report block exceeds max_chars")
    if current:
        parts.append(current)
    return tuple(parts)


def format_waiver_report_html(
    report: LeagueWaiverReport,
    *,
    max_chars: int = MAX_TELEGRAM_TEXT,
) -> tuple[str, ...]:
    """Render scan-first HTML parts, each safe for Telegram's text limit."""

    context = report.context
    header = [
        "<b>🏈 WAIVER PLAN</b>",
        f"<b>{_html(context.league_name, 120)}</b>",
        f"Claims expected · {_html(_waiver_time(context), 80)}",
    ]
    if context.waiver_method:
        method = _html(context.waiver_method, 100)
        if context.waiver_priority is not None:
            method += f" · Priority {int(context.waiver_priority)}"
        header.append(method)
    capacity = _capacity_line(context)
    if capacity:
        header.append(_html(capacity, 120))
    header.extend(
        (
            f"Scoring · {_html(context.scoring_format.upper(), 12)}",
            f"<i>{_html(report.evidence_label, 300)}</i>",
        )
    )
    blocks = ["\n".join(header), "<b>🎯 TOP CLAIMS</b>"]

    top = [
        candidate
        for candidate in report.candidates
        if candidate.tier in {"HIGH PRIORITY", "CLAIM"}
    ][: max(0, context.top_limit)]
    if top:
        blocks.extend(
            _candidate_block(candidate, index)
            for index, candidate in enumerate(top, start=1)
        )
    else:
        blocks.append("No available player clears the claim and roster-swap thresholds.")

    shown = {compact_key(candidate.evidence.name) for candidate in top}
    watch = [
        candidate
        for candidate in report.candidates
        if candidate.tier == "WATCH" and compact_key(candidate.evidence.name) not in shown
    ][: max(0, context.watch_limit)]
    blocks.append("<b>👀 WATCHLIST</b>")
    if watch:
        blocks.extend(_watch_block(candidate) for candidate in watch)
        shown.update(compact_key(candidate.evidence.name) for candidate in watch)
    else:
        blocks.append("No conditional watch candidates currently qualify.")

    alternatives: list[RankedCandidate] = []
    for position in SKILL_POSITIONS:
        candidate = next(
            (
                entry
                for entry in report.candidates
                if entry.evidence.position.upper() == position
                and compact_key(entry.evidence.name) not in shown
                and entry.swap_allowed
                and not entry.evidence.recommendation_blocked
            ),
            None,
        )
        if candidate is not None:
            alternatives.append(candidate)
            shown.add(compact_key(candidate.evidence.name))

    blocks.append("<b>📋 POSITIONAL ALTERNATIVES</b>")
    if alternatives:
        blocks.append(
            "\n".join(
                f"<b>{_html(candidate.evidence.position, 8)}</b> · "
                f"{_html(candidate.evidence.name, 80)} · {_html(candidate.confidence, 12)} confidence"
                for candidate in alternatives
            )
        )
    else:
        blocks.append("No additional position-specific alternative meets the evidence floor.")

    if context.bench_full:
        blocks.append(
            "<i>Bench is full. Every claim requires a cut; an open IR spot does not confirm IR eligibility.</i>"
        )
    elif not context.bench_capacity_known:
        blocks.append(
            "<i>Bench capacity is unavailable. Claims are held until the league roster refresh confirms room or a valid swap.</i>"
        )
    return _pack_html_blocks(blocks, max_chars)


__all__ = [
    "CandidateEvidence",
    "CorroboratedFact",
    "LeagueWaiverReport",
    "NewsFact",
    "RankedCandidate",
    "RosterAsset",
    "ScoreBreakdown",
    "WaiverReportContext",
    "build_waiver_report",
    "dedupe_recent_facts",
    "format_waiver_report_html",
    "rank_candidates",
    "score_candidate",
]
