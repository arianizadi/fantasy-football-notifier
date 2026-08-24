from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from notifier.waiver_report import (
    CandidateEvidence,
    NewsFact,
    RosterAsset,
    WaiverReportContext,
    build_waiver_report,
    dedupe_recent_facts,
    format_waiver_report_html,
    rank_candidates,
    score_candidate,
    _visible_units,
)


NOW = datetime(2026, 8, 25, 23, 0, tzinfo=timezone.utc)


def _roster(*, weakest_value: int = 25, include_lamar: bool = True):
    starters = (
        RosterAsset("Lamar Jackson", "QB", "QB", 95, elite=True),
    ) if include_lamar else ()
    return starters + (
        RosterAsset("Bench One", "RB", "BE", weakest_value),
        RosterAsset("Bench Two", "RB", "BE", 42),
        RosterAsset("Bench Three", "RB", "BE", 45),
        RosterAsset("Bench Four", "WR", "BE", 48),
        RosterAsset("Protected Handcuff", "WR", "BE", 15, protected=True),
    )


def _context(
    *,
    weakest_value: int = 25,
    include_lamar: bool = True,
    top_limit: int = 5,
    watch_limit: int = 5,
) -> WaiverReportContext:
    return WaiverReportContext(
        league_name="The Certified Sped League 2.0",
        scoring_format="PPR",
        generated_at=NOW,
        expected_waiver_at=NOW + timedelta(hours=8),
        bench_used=5,
        bench_limit=5,
        ir_used=0,
        ir_limit=1,
        roster=_roster(weakest_value=weakest_value, include_lamar=include_lamar),
        top_limit=top_limit,
        watch_limit=watch_limit,
    )


def _fact(
    *,
    source: str,
    signature: str = "starter:right-knee:out",
    subject: str = "Injured Starter",
    event_type: str = "injury",
    hours_ago: int = 2,
    creates: bool = True,
    resolves: bool = False,
    direct: bool = True,
) -> NewsFact:
    return NewsFact(
        signature=signature,
        event_type=event_type,
        severity=4,
        published_at=NOW - timedelta(hours=hours_ago),
        source=source,
        subject_name=subject,
        status="out" if not resolves else "cleared",
        directly_names_candidate=direct,
        creates_opportunity=creates,
        resolves_opportunity=resolves,
        supports_candidate=not resolves,
        description="Candidate is expected to handle the available work.",
    )


def _strong_candidate(name: str = "Fast Backup", position: str = "RB") -> CandidateEvidence:
    return CandidateEvidence(
        name=name,
        position=position,
        pro_team="NE",
        role="immediate_replacement",
        role_subject="Injured Starter",
        depth_order=2,
        fantasypros_waiver_percentile=90,
        fantasypros_ros_percentile=80,
        fantasypros_updated_at=NOW - timedelta(hours=1),
        sleeper_adds_6h=500,
        sleeper_adds_24h=700,
        facts=(_fact(source="AdamSchefter"), _fact(source="RapSheet")),
    )


def test_dedupe_retains_independent_corroboration_not_repeat_posts() -> None:
    facts = (
        _fact(source="AdamSchefter"),
        _fact(source="AdamSchefter", hours_ago=1),
        _fact(source="RapSheet"),
    )

    collapsed = dedupe_recent_facts(facts, now=NOW)

    assert len(collapsed) == 1
    assert collapsed[0].sources == ("AdamSchefter", "RapSheet")
    assert collapsed[0].corroboration_count == 2
    assert collapsed[0].latest_at == NOW - timedelta(hours=1)


def test_same_source_duplicates_do_not_receive_corroboration_boost() -> None:
    one_source = CandidateEvidence(
        **{
            **_strong_candidate().__dict__,
            "facts": (_fact(source="AdamSchefter"), _fact(source="AdamSchefter", hours_ago=1)),
        }
    )
    two_sources = _strong_candidate()

    first = score_candidate(one_source, _context())
    second = score_candidate(two_sources, _context())

    assert first.score.news_confidence < second.score.news_confidence


def test_later_return_cancels_the_backup_opportunity() -> None:
    injury = _fact(source="AdamSchefter", hours_ago=8)
    returned = _fact(
        source="Team",
        signature="starter:return:cleared",
        event_type="return",
        hours_ago=1,
        creates=False,
        resolves=True,
        direct=False,
    )
    candidate = CandidateEvidence(
        name="Fast Backup",
        position="RB",
        pro_team="NE",
        role="immediate_replacement",
        role_subject="Injured Starter",
        depth_order=2,
        facts=(injury, returned),
    )

    ranked = score_candidate(candidate, _context(weakest_value=0))

    assert ranked.return_cancelled is True
    assert ranked.score.role_opportunity == 5
    assert ranked.score.news_confidence == 0
    assert ranked.tier == "ALTERNATIVE"
    assert any("canceled" in risk for risk in ranked.risks)


def test_return_for_different_player_does_not_cancel_opportunity() -> None:
    candidate = _strong_candidate()
    unrelated_return = _fact(
        source="Team",
        subject="Different Starter",
        signature="different:return",
        event_type="return",
        hours_ago=1,
        creates=False,
        resolves=True,
        direct=False,
    )
    candidate = CandidateEvidence(**{**candidate.__dict__, "facts": (*candidate.facts, unrelated_return)})

    ranked = score_candidate(candidate, _context())

    assert ranked.return_cancelled is False
    assert ranked.score.role_opportunity == 22


def test_fantasypros_is_optional_and_stale_data_gets_fallback_label() -> None:
    candidate = CandidateEvidence(
        **{
            **_strong_candidate().__dict__,
            "fantasypros_updated_at": NOW - timedelta(hours=13),
        }
    )

    report = build_waiver_report(_context(), (candidate,))

    assert report.candidates[0].fantasypros_used is False
    assert report.candidates[0].score.fantasy_quality == 0
    assert report.evidence_label.startswith("FantasyPros unavailable")


def test_fresh_fantasypros_context_is_scoring_specific_support() -> None:
    report = build_waiver_report(_context(), (_strong_candidate(),))

    assert report.candidates[0].fantasypros_used is True
    assert report.candidates[0].score.fantasy_quality == 22
    assert "FantasyPros PPR" in report.evidence_label


def test_lamar_suppresses_a_marginal_backup_quarterback() -> None:
    quarterback = CandidateEvidence(
        name="Marginal Quarterback",
        position="QB",
        pro_team="CLE",
        role="uncertain_replacement",
        depth_order=2,
        fantasypros_waiver_percentile=70,
        fantasypros_ros_percentile=70,
        fantasypros_updated_at=NOW,
        sleeper_adds_6h=400,
        sleeper_adds_24h=500,
        facts=(_fact(source="AdamSchefter"), _fact(source="RapSheet")),
    )

    with_lamar = score_candidate(quarterback, _context(include_lamar=True))
    without_lamar = score_candidate(quarterback, _context(include_lamar=False))

    assert with_lamar.score.quarterback_penalty == 25
    assert without_lamar.score.quarterback_penalty == 0
    assert with_lamar.score.total < without_lamar.score.total
    assert with_lamar.tier == "ALTERNATIVE"
    assert any("Lamar Jackson" in risk for risk in with_lamar.risks)


def test_full_five_player_bench_requires_ten_point_swap_improvement() -> None:
    candidate = CandidateEvidence(
        name="Small Upgrade",
        position="WR",
        pro_team="NYJ",
        role="confirmed_starter",
        depth_order=1,
        fantasypros_waiver_percentile=80,
        fantasypros_ros_percentile=70,
        fantasypros_updated_at=NOW,
    )

    blocked_context = replace(
        _context(weakest_value=48),
        roster=(
            RosterAsset("Lamar Jackson", "QB", "QB", 95, elite=True),
            RosterAsset("Bench One", "RB", "BE", 65),
            RosterAsset("Bench Two", "RB", "BE", 66),
            RosterAsset("Bench Three", "RB", "BE", 67),
            RosterAsset("Bench Four", "WR", "BE", 68),
            RosterAsset("Protected Handcuff", "WR", "BE", 15, protected=True),
        ),
    )
    blocked = score_candidate(candidate, blocked_context)
    allowed = score_candidate(candidate, _context(weakest_value=20))

    assert blocked.swap_allowed is False
    assert blocked.tier == "WATCH"
    assert blocked.suggested_drop is not None
    assert blocked.suggested_drop.name == "Bench One"
    assert allowed.swap_allowed is True
    assert allowed.tier in {"CLAIM", "HIGH PRIORITY"}
    assert allowed.suggested_drop is not None
    assert allowed.suggested_drop.name == "Bench One"


def test_unresolved_candidate_injury_can_never_become_a_claim() -> None:
    candidate = CandidateEvidence(
        name="Injured Breakout",
        position="RB",
        pro_team="ARI",
        role="confirmed_starter",
        depth_order=1,
        fantasypros_waiver_percentile=100,
        fantasypros_ros_percentile=100,
        fantasypros_updated_at=NOW,
        sleeper_adds_6h=1000,
        sleeper_adds_24h=1200,
        recommendation_blocked=True,
        blocked_reason=(
            "Unresolved injury or availability status; monitor, but do not claim yet."
        ),
    )

    ranked = score_candidate(candidate, _context(weakest_value=0))

    assert ranked.score.total >= 75
    assert ranked.swap_allowed is False
    assert ranked.tier == "WATCH"
    assert ranked.risks[0].startswith("Unresolved injury")


def test_unknown_bench_capacity_fails_closed() -> None:
    context = replace(
        _context(weakest_value=0),
        bench_used=None,
        bench_limit=None,
    )
    ranked = score_candidate(_strong_candidate(), context)

    assert context.bench_capacity_known is False
    assert ranked.swap_allowed is False
    assert ranked.tier == "WATCH"
    assert any("Bench capacity is unavailable" in risk for risk in ranked.risks)

    text = "\n".join(
        format_waiver_report_html(build_waiver_report(context, (_strong_candidate(),)))
    )
    assert "Claims are held" in text


def test_unavailable_players_are_filtered_and_ties_are_deterministic() -> None:
    alpha = CandidateEvidence(
        name="Alpha Back",
        position="RB",
        pro_team="ARI",
        role="depth_two",
        depth_order=2,
    )
    beta = CandidateEvidence(
        name="Beta Back",
        position="RB",
        pro_team="BUF",
        role="depth_two",
        depth_order=2,
    )
    rostered = CandidateEvidence(
        name="Rostered Star",
        position="RB",
        pro_team="DAL",
        available=False,
        role="confirmed_starter",
        depth_order=1,
    )

    ranked = rank_candidates((beta, rostered, alpha), _context())

    assert [candidate.evidence.name for candidate in ranked] == ["Alpha Back", "Beta Back"]


def test_html_is_readable_escaped_and_split_without_exceeding_limit() -> None:
    candidates = tuple(
        CandidateEvidence(
            **{
                **_strong_candidate(
                    name=f"Candidate <{index}>",
                    position=("RB", "WR", "TE", "QB")[index % 4],
                ).__dict__,
                "sleeper_adds_6h": 500 - index,
            }
        )
        for index in range(16)
    )
    context = _context(top_limit=12, watch_limit=12)
    report = build_waiver_report(context, candidates)

    parts = format_waiver_report_html(report, max_chars=600)

    assert len(parts) > 1
    assert all(_visible_units(part) <= 600 for part in parts)
    combined = "\n".join(parts)
    assert "<b>🎯 TOP CLAIMS</b>" in combined
    assert "<b>👀 WATCHLIST</b>" in combined
    assert "<b>📋 POSITIONAL ALTERNATIVES</b>" in combined
    assert "Candidate &lt;0&gt;" in combined
    assert "Candidate <0>" not in combined
    assert "open IR spot does not confirm IR eligibility" in combined


def test_positional_alternatives_include_best_unshown_player_per_position() -> None:
    candidates = tuple(
        CandidateEvidence(
            name=f"{position} Alternative",
            position=position,
            pro_team="FA",
            role="ordinary_backup",
            depth_order=2,
        )
        for position in ("QB", "RB", "WR", "TE")
    )
    context = replace(
        _context(top_limit=0, watch_limit=0),
        bench_used=4,
    )
    report = build_waiver_report(context, candidates)

    text = "\n".join(format_waiver_report_html(report))

    for position in ("QB", "RB", "WR", "TE"):
        assert f"<b>{position}</b> · {position} Alternative" in text


def test_positional_alternatives_never_hide_a_claim_blocker() -> None:
    injured = CandidateEvidence(
        name="Injured Alternative",
        position="RB",
        pro_team="ARI",
        role="confirmed_starter",
        depth_order=1,
        recommendation_blocked=True,
        blocked_reason="Unresolved injury; do not claim.",
    )
    no_swap = CandidateEvidence(
        name="No Swap Alternative",
        position="WR",
        pro_team="NYJ",
        role="ordinary_backup",
        depth_order=3,
        platform_quality_percentile=20,
    )
    report = build_waiver_report(
        _context(top_limit=0, watch_limit=0),
        (injured, no_swap),
    )

    text = "\n".join(format_waiver_report_html(report))

    assert "Injured Alternative" not in text
    assert "No Swap Alternative" not in text
    assert "No additional position-specific alternative" in text
