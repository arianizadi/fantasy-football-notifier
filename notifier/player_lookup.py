"""Read-only player summaries for the Telegram ``/player`` command."""

from __future__ import annotations

import html
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .matcher import compact_key
from .models import LeagueRef, RosterSnapshot


def _escape(value: object) -> str:
    return html.escape(str(value or ""), quote=False)


def _league_labels(leagues: list[LeagueRef]) -> dict[str, str]:
    names = [league.short_label for league in leagues]
    counts = Counter(name.casefold() for name in names)
    labels: dict[str, str] = {}
    for league, name in zip(leagues, names):
        if counts[name.casefold()] == 1:
            labels[league.key] = name
        else:
            labels[league.key] = f"{name} ({league.provider.upper()} {league.league_id[-4:]})"
    return labels


def _candidates(query: str, player_index: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = compact_key(query)
    if not wanted:
        return []
    matches: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for record in player_index.values():
        if not isinstance(record, dict):
            continue
        name = str(record.get("full_name") or "").strip()
        key = compact_key(name)
        if not key or wanted not in key:
            continue
        surname = compact_key(name.split()[-1])
        rank = record.get("search_rank")
        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            rank_value = 999999
        score = (
            0 if key == wanted else 1 if surname == wanted else 2 if key.startswith(wanted) else 3,
            rank_value,
            len(key),
        )
        matches.append((score, record))
    matches.sort(key=lambda pair: pair[0])
    return [record for _score, record in matches]


def _ownership_lines(record: dict[str, Any], snapshot: RosterSnapshot) -> list[str]:
    player_key = compact_key(record.get("full_name") or "")
    leagues = snapshot.drafted_leagues()
    labels = _league_labels(leagues)
    lines: list[str] = []
    for league in leagues:
        player = next(
            (
                entry
                for entry in snapshot.players
                if entry.league_key == league.key and compact_key(entry.name) == player_key
            ),
            None,
        )
        if player is None:
            state = "FREE AGENT"
        elif player.on_my_team:
            state = f"YOU · {player.lineup_slot or 'roster'}"
        else:
            state = player.fantasy_team or "rostered"
        lines.append(f"  {_escape(labels[league.key])}: {_escape(state)}")
    return lines


def _depth_lines(record: dict[str, Any], player_index: dict[str, Any]) -> list[str]:
    team = str(record.get("team") or "")
    position = str(record.get("position") or "")
    if not team or not position:
        return []
    group = [
        candidate
        for candidate in player_index.values()
        if isinstance(candidate, dict)
        and candidate.get("team") == team
        and candidate.get("position") == position
        and (
            str(candidate.get("status") or "").casefold() == "active"
            or compact_key(candidate.get("full_name") or "")
            == compact_key(record.get("full_name") or "")
        )
    ]
    group.sort(
        key=lambda candidate: (
            candidate.get("depth_chart_order") is None,
            candidate.get("depth_chart_order") or 99,
            candidate.get("search_rank") or 999999,
        )
    )
    subject_key = compact_key(record.get("full_name") or "")
    index = next(
        (
            position
            for position, candidate in enumerate(group)
            if compact_key(candidate.get("full_name") or "") == subject_key
        ),
        0,
    )
    start = max(0, index - 1)
    selected = group[start : max(index + 3, start + 3)]
    lines = [f"<b>Sleeper {_escape(team)} {_escape(position)} depth</b>"]
    for candidate in selected:
        order = candidate.get("depth_chart_order")
        slot = f"{position}{order}" if order is not None else position
        marker = " · SUBJECT" if compact_key(candidate.get("full_name") or "") == subject_key else ""
        rank = candidate.get("search_rank")
        rank_text = f" · rank #{rank}" if rank not in {None, 999999} else ""
        lines.append(
            f"  {_escape(slot)} {_escape(candidate.get('full_name'))}{rank_text}{marker}"
        )
    return lines


def format_player_lookup(
    query: str,
    player_index: dict[str, Any],
    snapshot: RosterSnapshot,
    *,
    refreshed_at: datetime | None = None,
) -> str:
    """Return a concise, source-attributed HTML player summary."""
    matches = _candidates(query, player_index)
    if not matches:
        return f"No Sleeper player matched <b>{_escape(query)}</b>."

    wanted = compact_key(query)
    exact_names = [
        record
        for record in matches
        if compact_key(record.get("full_name") or "") == wanted
    ]
    surname_matches = [
        record
        for record in matches
        if compact_key(str(record.get("full_name") or "").split()[-1]) == wanted
    ]
    if len(exact_names) == 1:
        best = exact_names[0]
    elif len(surname_matches) == 1:
        best = surname_matches[0]
    elif len(matches) == 1:
        best = matches[0]
    else:
        options = "\n".join(
            f"  {_escape(record.get('full_name'))} · {_escape(record.get('team'))} {_escape(record.get('position'))}"
            for record in matches[:6]
        )
        return f"Multiple players matched <b>{_escape(query)}</b>:\n{options}"

    name = str(best.get("full_name") or query)
    team = str(best.get("team") or "FA")
    position = str(best.get("position") or "-")
    depth = best.get("depth_chart_order")
    rank = best.get("search_rank")
    status = str(best.get("status") or "unknown")
    injury = str(best.get("injury_status") or "")
    lines = [f"<b>{_escape(name)}</b> · {_escape(team)} {_escape(position)}"]
    facts = [f"Sleeper status: {_escape(status)}"]
    if injury:
        facts.append(f"injury: {_escape(injury)}")
    if depth is not None:
        facts.append(f"depth: {_escape(position)}{_escape(depth)}")
    if rank is not None and rank != 999999:
        facts.append(f"overall/search rank: #{_escape(rank)}")
    lines.append(" · ".join(facts))

    stamp = refreshed_at or getattr(player_index, "refreshed_at", None)
    if stamp is not None:
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        lines.append(f"Sleeper refreshed: {stamp.astimezone(timezone.utc):%Y-%m-%d %H:%M UTC}")
    if snapshot.generated_at is not None:
        roster_stamp = snapshot.generated_at
        if roster_stamp.tzinfo is None:
            roster_stamp = roster_stamp.replace(tzinfo=timezone.utc)
        lines.append(
            f"League ownership refreshed: "
            f"{roster_stamp.astimezone(timezone.utc):%Y-%m-%d %H:%M UTC}"
        )

    ownership = _ownership_lines(best, snapshot)
    if ownership:
        lines += ["", "<b>League ownership</b>", *ownership]
    depth_lines = _depth_lines(best, player_index)
    if depth_lines:
        lines += ["", *depth_lines]
    return "\n".join(lines)
