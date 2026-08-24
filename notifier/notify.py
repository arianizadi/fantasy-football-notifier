"""Telegram delivery and message formatting."""

from __future__ import annotations

import html
import logging
import re
import threading
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

import requests

from .config import Config, telegram_state_path
from .health import HEALTH
from .logging_utils import structured_log
from .models import Alert
from .plays import LeaguePlays, normalized_event_type
from .telegram_state import (
    TelegramState,
    alert_token,
    feedback_markup,
    feedback_markup_for_token,
)

API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT = 15
TELEGRAM_TEXT_LIMIT = 4096
_SEND_CONTEXT = threading.local()

# Severity always remains textual for accessibility. The colored dot is only
# a visual anchor; Telegram does not support arbitrary text colors.
TIER_LABEL = {
    "mine": "YOUR ROSTER",
    "claimable": "WAIVER WATCH",
    "rival": "RIVAL ROSTER",
    "league": "LEAGUE NEWS",
    "preseason": "PRESEASON",
}

SEVERITY_ICON = {
    1: "⚪",
    2: "🔵",
    3: "🟡",
    4: "🟠",
    5: "🔴",
}

EVENT_LABEL = {
    "injury": "INJURY",
    "practice_report": "PRACTICE",
    "inactive": "OUT / INACTIVE",
    "return": "RETURN",
    "trade": "TRADE",
    "signing": "SIGNING",
    "release": "RELEASE",
    "suspension": "SUSPENSION",
    "depth_chart": "DEPTH CHART",
    "usage": "ROLE / USAGE",
    "other": "UPDATE",
}

def _escape(value: str) -> str:
    # Some feeds occasionally send already-escaped HTML entities (for
    # example, ``&amp;``). Normalize once before escaping so Telegram renders
    # one ampersand instead of the literal characters "&amp;".
    entity_values = {
        "amp": "&",
        "lt": "<",
        "gt": ">",
        "quot": '"',
        "apos": "'",
        "#39": "'",
    }
    normalized = re.sub(
        r"&(?:amp|lt|gt|quot|apos|#39);",
        lambda match: entity_values[match.group(0)[1:-1].casefold()],
        value or "",
        flags=re.IGNORECASE,
    )
    # Dynamic feed/provider fields are inline inside a layout we control.
    # Collapsing their whitespace prevents upstream newlines from breaking a
    # section boundary or opening an oversized partial HTML block.
    normalized = " ".join(normalized.split())
    return html.escape(normalized, quote=False)


class _VisibleTextParser(HTMLParser):
    """Extract the text Telegram counts after parsing HTML entities."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_html_text(markup: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(markup)
    parser.close()
    return "".join(parser.parts)


def _utf16_units(value: str) -> int:
    """Telegram measures message length in UTF-16 code units."""
    return len(value.encode("utf-16-le")) // 2


def _visible_units(markup: str) -> int:
    return _utf16_units(_plain_html_text(markup))


def _clip_utf16(value: str, limit: int) -> str:
    """Clip plain text without splitting a supplementary-plane character."""
    if limit <= 0:
        return ""
    ellipsis_units = _utf16_units("…")
    budget = max(0, limit - ellipsis_units)
    used = 0
    kept: list[str] = []
    for character in value:
        units = _utf16_units(character)
        if used + units > budget:
            break
        kept.append(character)
        used += units
    clipped = "".join(kept).rstrip()
    return clipped + ("…" if clipped != value else "")


def _fit_telegram_limit(markup: str) -> str:
    """Keep complete sections when possible and never send invalid HTML.

    Telegram rejects sendMessage and editMessageText payloads over 4096
    visible UTF-16 units. Alerts are ordered by usefulness, so retain complete
    leading sections and replace an oversized tail with an explicit marker.
    A pathological single block falls back to safely escaped clipped text.
    """
    if _visible_units(markup) <= TELEGRAM_TEXT_LIMIT:
        return markup

    suffix = "<i>… Some details omitted to fit Telegram.</i>"
    separator = "\n\n"
    content_budget = TELEGRAM_TEXT_LIMIT - _visible_units(separator + suffix)
    kept: list[str] = []
    for block in markup.split(separator):
        candidate = separator.join([*kept, block])
        if _visible_units(candidate) <= content_budget:
            kept.append(block)
            continue

        prefix = separator.join(kept)
        remaining = content_budget - _visible_units(prefix)
        if kept:
            remaining -= _visible_units(separator)
        # Preserve a useful fragment only when there is room for more than a
        # tiny orphaned label. It is plain escaped text, so tags stay balanced.
        if remaining >= 80:
            fragment = _clip_utf16(_plain_html_text(block), remaining)
            if fragment:
                kept.append(_escape(fragment))
        elif not kept:
            kept.append(_escape(_clip_utf16(_plain_html_text(block), content_budget)))
        break

    result = separator.join(kept) + separator + suffix
    # Defensive assertion for future changes to the suffix or counter.
    if _visible_units(result) > TELEGRAM_TEXT_LIMIT:
        fallback = _clip_utf16(_plain_html_text(markup), content_budget)
        result = _escape(fallback) + separator + suffix
    return result


MIN_SEVERITY_FOR_PLAYS = 3
# Below this the full depth chart is noise; above it you want to judge for
# yourself rather than trust the mechanical next-man-up.
MIN_SEVERITY_FOR_CONTEXT = 3
DISPLAY_TIMEZONE = ZoneInfo("America/Los_Angeles")

# The current model field is named fantasy_impact and its old prompt explicitly
# asks what the manager should do. Only retain it when it reads as a short
# descriptive summary; lineup and waiver instructions come from code below.
MODEL_DIRECTIVE = re.compile(
    r"\b(add|activate|avoid|bench|buy|check|claim|draft|drop|fade|hold|lineups?|"
    r"monitor|move|pick\s*up|plug|replace|roster|sell|sit|start(?:ing)?|stash|"
    r"stream|target|trade\s+for|waiver|you\s+should|managers?\s+should)\b",
    re.IGNORECASE,
)


def _league_labels(leagues) -> dict[str, str]:
    """Human, unique labels for every league represented in an alert."""
    bases = [league.short_label for league in leagues]
    counts = Counter(base.casefold() for base in bases)
    labels: dict[str, str] = {}
    for league, base in zip(leagues, bases):
        if counts[base.casefold()] == 1:
            labels[league.key] = base
            continue
        provider = (league.provider or "league").upper()
        suffix = league.league_id[-4:] if league.league_id else ""
        discriminator = f"{provider} {suffix}".strip()
        labels[league.key] = f"{base} ({discriminator})"
    return labels


def _ownership_labels(entry, leagues, labels: dict[str, str]) -> list[str]:
    """Readable ownership lines, split by league when states differ."""
    states = []
    for league in leagues:
        state, team = entry.ownership.get(league.key, ("free_agent", ""))
        if state == "free_agent":
            label = "Available"
        elif state == "mine":
            label = "Your roster"
        else:
            owner = team or "another team"
            if len(owner) > 16:
                owner = owner[:15] + "…"
            label = f"Owned by {owner}"
        states.append((labels.get(league.key, league.short_label), label))

    if not states:
        return []
    if len({label for _, label in states}) == 1:
        only = states[0][1]
        if len(states) > 1:
            if only == "Available":
                return ["Available in all leagues"]
            if only == "Your roster":
                return ["Your roster in all leagues"]
        return [only]
    return [f"{league_label}: {label}" for league_label, label in states]


def _ownership_icon(entry, leagues) -> str:
    """Color anchor for ownership; the adjacent text carries the meaning."""
    states = [
        entry.ownership.get(league.key, ("free_agent", ""))[0]
        for league in leagues
    ]
    if not states:
        return "•"
    if "mine" in states:
        return "⭐"
    if all(state == "free_agent" for state in states):
        return "🟢"
    if all(state == "rostered" for state in states):
        return "🔒"
    return "◐"


def _freshness_label(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    local = value.astimezone(DISPLAY_TIMEZONE)
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{local:%b} {local.day} · {clock} PT"


def _event_label(event_type: str) -> str:
    event = normalized_event_type(event_type)
    return EVENT_LABEL.get(event, event.replace("_", " ").upper() or "UPDATE")


def _safe_model_summary(value: str) -> str:
    """Keep concise model description, never model-authored roster advice."""
    summary = " ".join((value or "").split()).strip()
    if not summary or MODEL_DIRECTIVE.search(summary):
        return ""
    return summary[:140]


def _context_block(
    context,
    leagues,
    severity: int,
    *,
    show_ownership: bool = True,
) -> list[str]:
    """A readable Sleeper view of only the affected position."""
    if context is None or severity < MIN_SEVERITY_FOR_CONTEXT:
        return []

    labels = _league_labels(leagues)
    lines = [
        f"📋 <b>{_escape(context.team)} {_escape(context.subject_position)} DEPTH</b>"
    ]
    refreshed = _freshness_label(context.player_index_refreshed_at)
    provenance = "Sleeper"
    if refreshed:
        provenance += f" · updated {refreshed}"
    lines.append(f"<i>{_escape(provenance)}</i>")
    if not show_ownership:
        lines.append("⚠️ Live league ownership unavailable.")

    for index, entry in enumerate(context.same_position):
        depth_slot = (
            f"{entry.position}{entry.depth_order}"
            if entry.depth_order is not None
            else entry.position
        )
        icon = (
            "➡️"
            if entry.is_subject
            else (_ownership_icon(entry, leagues) if show_ownership else "•")
        )
        first = f"{icon} <b>{_escape(depth_slot)} {_escape(entry.name)}</b>"
        if entry.is_subject:
            first += " · report subject"
        lines.append(first)

        ownership = (
            _ownership_labels(entry, leagues, labels) if show_ownership else []
        )
        lines += [f"↳ {_escape(owner)}" for owner in ownership]

        metadata: list[str] = []
        # Sleeper uses 999 as an unranked placeholder; displaying it looks
        # precise while adding no useful signal.
        if entry.search_rank and entry.search_rank < 999:
            metadata.append(f"Sleeper #{entry.search_rank}")
        if entry.sleeper_injury_status:
            metadata.append(f"Injury: {_escape(entry.sleeper_injury_status)}")
        if entry.sleeper_status and entry.sleeper_status.casefold() != "active":
            metadata.append(f"Status: {_escape(entry.sleeper_status)}")
        if metadata:
            lines.append("↳ " + " · ".join(metadata))
        if index < len(context.same_position) - 1:
            lines.append("")
    return lines


def _fantasypros_rank_detail(beneficiary) -> str:
    """Compact, explicitly attributed secondary ranking context."""
    if not beneficiary.fantasypros_scoring:
        return ""
    ranks: list[str] = []
    if beneficiary.fantasypros_waiver_rank is not None:
        value = (
            beneficiary.fantasypros_waiver_pos_rank
            or f"#{beneficiary.fantasypros_waiver_rank}"
        )
        ranks.append(f"waiver {value}")
    if beneficiary.fantasypros_ros_rank is not None:
        value = (
            beneficiary.fantasypros_ros_pos_rank
            or f"#{beneficiary.fantasypros_ros_rank}"
        )
        ranks.append(f"ROS {value}")
    if not ranks:
        return ""
    return (
        f"FantasyPros {_escape(beneficiary.fantasypros_scoring)} "
        + " · ".join(_escape(value) for value in ranks)
    )


def _fantasypros_lean(claimable) -> str:
    """A conservative ranking lean that never changes candidate order."""
    ranked = [
        candidate
        for candidate in claimable
        if candidate.fantasypros_waiver_rank is not None
    ]
    if len(ranked) < 2:
        return ""
    ranked.sort(key=lambda candidate: candidate.fantasypros_waiver_rank)
    best, runner_up = ranked[:2]
    if runner_up.fantasypros_waiver_rank - best.fantasypros_waiver_rank < 5:
        return ""
    # Do not call it a lean when rest-of-season consensus points the other way.
    if (
        best.fantasypros_ros_rank is not None
        and runner_up.fantasypros_ros_rank is not None
        and best.fantasypros_ros_rank > runner_up.fantasypros_ros_rank
    ):
        return ""
    return best.name


def _fantasypros_freshness(claimable) -> str:
    enriched = [
        candidate
        for candidate in claimable
        if candidate.fantasypros_scoring
        and (
            candidate.fantasypros_waiver_rank is not None
            or candidate.fantasypros_ros_rank is not None
        )
    ]
    if not enriched:
        return ""
    scoring = "/".join(
        dict.fromkeys(candidate.fantasypros_scoring for candidate in enriched)
    )
    timestamps: list[datetime] = []
    for candidate in enriched:
        if not candidate.fantasypros_updated_at:
            continue
        try:
            parsed = datetime.fromisoformat(candidate.fantasypros_updated_at)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamps.append(parsed)
    as_of = f" · updated {_freshness_label(min(timestamps))}" if timestamps else ""
    return (
        f"FantasyPros {scoring} rankings{as_of}; may lag this report and do "
        "not confirm the role."
    )


def _league_block(
    plays: LeaguePlays,
    severity: int,
    event_type: str,
    league_label: str,
    *,
    allow_adds: bool = True,
) -> list[str]:
    """One readable action block per league. Availability differs by league."""
    if severity < MIN_SEVERITY_FOR_PLAYS:
        return []
    # Defense in depth: callers should filter before tier selection with
    # plays_for_event(), but formatting can never leak a raw positive-event
    # backup suggestion even if an older caller forgets.
    plays = plays.for_event(event_type, severity)
    claimable = plays.claimable if allow_adds else []
    if not claimable and not plays.bench_options:
        return []

    name = _escape(league_label)
    lines = [f"<b>{name}</b>"]
    for beneficiary in claimable[:2]:
        slot = f"{beneficiary.position}{beneficiary.depth_order or ''}"
        lines.append(
            f"🟢 <b>{_escape(beneficiary.name)}</b> · Sleeper {_escape(slot)}"
        )
        details: list[str] = []
        if beneficiary.named_in_report:
            details.append("named in report")
        fantasypros = _fantasypros_rank_detail(beneficiary)
        if fantasypros:
            details.append(fantasypros)
        if details:
            lines.append("↳ " + " · ".join(details))
    if claimable:
        option_label = "Pickup options · choose one" if len(claimable) > 1 else "Pickup option"
        lines.insert(1, option_label)
    if plays.bench_options:
        lines.append(f"🔁 Start instead: <b>{_escape(plays.bench_options[0])}</b>")

    # Occupancy only: an open IR spot is not a claim that the injured player
    # is eligible for it, and the notifier never chooses a drop candidate.
    if claimable and allow_adds and plays.capacity is not None:
        capacity_parts: list[str] = []
        if (
            plays.capacity.bench_used is not None
            and plays.capacity.bench_limit is not None
        ):
            bench = f"Bench {plays.capacity.bench_used}/{plays.capacity.bench_limit}"
            if plays.capacity.bench_limit > 0:
                bench += (
                    " full"
                    if plays.capacity.bench_used >= plays.capacity.bench_limit
                    else " open"
                )
            capacity_parts.append(bench)
        if plays.capacity.ir_used is not None and plays.capacity.ir_limit is not None:
            reserve = f"IR {plays.capacity.ir_used}/{plays.capacity.ir_limit}"
            if plays.capacity.ir_limit > 0:
                reserve += (
                    " full"
                    if plays.capacity.ir_used >= plays.capacity.ir_limit
                    else " open"
                )
            capacity_parts.append(reserve)
        if capacity_parts:
            lines.append("📦 Roster space · " + " · ".join(capacity_parts))
            if (
                plays.capacity.ir_used is not None
                and plays.capacity.ir_limit is not None
                and plays.capacity.ir_limit > 0
                and plays.capacity.ir_used < plays.capacity.ir_limit
            ):
                lines.append("↳ IR eligibility is not checked.")
    lean = _fantasypros_lean(claimable[:2])
    if lean:
        lines.append(
            f"📊 FantasyPros lean · <b>{_escape(lean)}</b> "
            "<i>(role unconfirmed)</i>"
        )
    freshness = _fantasypros_freshness(claimable[:2])
    if freshness:
        lines.append(f"<i>{_escape(freshness)}</i>")
    return lines


def _deterministic_note(alert: Alert) -> str:
    """Conservative event guidance; never sourced from model prose."""
    event = normalized_event_type(alert.classification.event_type)
    if alert.tier == "preseason":
        if event == "return":
            return (
                "Return news improves availability. Confirm full practice and Week 1 "
                "status before changing draft value."
            )
        if event == "signing":
            return "Recheck the official role and depth chart before changing draft value."
        if event in {"injury", "inactive", "suspension"}:
            return "Recheck official availability before drafting this player."
        if event in {"trade", "release", "depth_chart", "usage"}:
            return "Reassess the role and team context before changing draft value."
        if event == "practice_report":
            return "Wait for the next official practice or game-status update."
        return "Verify the report before changing draft value."

    if event == "return":
        return "Confirm active status and expected workload before lineup lock."
    if event == "signing":
        return "No automatic waiver move. Confirm the player's official role first."
    if event in {"trade", "depth_chart", "usage"}:
        return "Recheck the official role before making a lineup or waiver move."
    if event == "practice_report" and alert.classification.severity < 4:
        return "Wait for the final game-status report before making a change."
    return ""


def _backup_watch_note(alert: Alert) -> str:
    """Explain why positive return news shows backups without recommending one."""
    if (
        normalized_event_type(alert.classification.event_type) != "return"
        or alert.classification.severity < MIN_SEVERITY_FOR_CONTEXT
        or alert.context is None
    ):
        return ""
    backups = [entry for entry in alert.context.same_position if not entry.is_subject]
    if not backups:
        return ""
    immediate = backups[0]
    position = alert.context.subject_position or immediate.position or "position"
    return (
        f"{immediate.name} is next in Sleeper's {position} depth order. No pickup "
        "is recommended from return news; an injury or inactive report will "
        "recheck availability."
    )


def _source_line(item) -> str:
    """Linked provenance with a compact mobile-friendly timestamp."""
    if not item.url:
        return ""
    source_name = {
        "twitter": "X source",
        "rotowire": "RotoWire source",
    }.get(item.source.casefold(), f"{item.source or 'news'} source")
    raw_url = item.url.strip()
    parsed = urlsplit(raw_url)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        source = (
            f'<a href="{html.escape(raw_url, quote=True)}">'
            f"{_escape(source_name)}</a>"
        )
    else:
        # Telegram accepts only safe URL schemes in HTML links. Keep the
        # provenance label even when an upstream URL is malformed.
        source = _escape(source_name)
    if item.published_at is not None:
        published = item.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        source += f" · {_escape(_freshness_label(published))}"
    return source


def format_alert(alert: Alert) -> str:
    """Render a scan-first mobile alert with decision-relevant sections."""
    item, classification = alert.item, alert.classification
    tier = TIER_LABEL.get(alert.tier, "NEWS")
    event = _event_label(classification.event_type)
    severity_icon = SEVERITY_ICON.get(classification.severity, "⚪")

    head = f"{severity_icon} <b>[{classification.severity}/5] {tier}"
    if item.player_name:
        head += f" · {_escape(item.player_name)}"
    head += "</b>"

    lines = [head, f"<b>{_escape(event)}</b> · {_escape(item.headline)}"]
    warning_lines: list[str] = []
    if alert.delivery_delayed:
        warning_lines.append(
            "Delayed after a Telegram retry; league availability was rechecked."
        )

    if not item.subject_confident:
        warning_lines.append(
            "Player attribution is unclear in this report; automatic "
            "pickup and lineup moves are withheld."
        )

    if alert.availability_refresh_failed:
        warning_lines.append(
            "League availability refresh failed; no ADD or free-agent "
            "recommendation is shown."
        )
    if warning_lines:
        lines += ["", "⚠️ <b>CHECK FIRST</b>"]
        lines += [f"• {_escape(warning)}" for warning in warning_lines]

    context_successors = (
        [entry for entry in alert.context.same_position if not entry.is_subject]
        if alert.context is not None
        else []
    )
    has_multiple_successors = len(context_successors) > 1 or any(
        len(plays.beneficiaries) > 1 for plays in alert.per_league
    )
    # A prose model summary can accidentally turn a depth-listing signal into
    # a workload prediction. When several successors exist, the deterministic
    # option block and source report are safer and more useful.
    model_summary = (
        ""
        if has_multiple_successors or not item.subject_confident
        else _safe_model_summary(classification.fantasy_impact)
    )

    action_lines: list[str] = []
    # Build labels against every drafted league, not only leagues with an
    # action. Otherwise one of two same-named leagues becomes ambiguous when
    # only one happens to have a claimable player.
    label_scope = list(alert.all_leagues)
    scoped_keys = {league.key for league in label_scope}
    label_scope.extend(
        plays.league for plays in alert.per_league if plays.league.key not in scoped_keys
    )
    labels = _league_labels(label_scope)
    for plays in alert.per_league:
        block = _league_block(
            plays,
            classification.severity,
            classification.event_type,
            labels.get(plays.league.key, plays.league.short_label),
            allow_adds=not alert.availability_refresh_failed,
        )
        if not block:
            continue
        if action_lines:
            action_lines.append("")
        action_lines += block
    has_multiple_claimable = not alert.availability_refresh_failed and any(
        len(plays.claimable) > 1 for plays in alert.per_league
    )
    has_any_claimable = not alert.availability_refresh_failed and any(
        plays.claimable for plays in alert.per_league
    )
    if action_lines:
        lines += ["", "🎯 <b>YOUR OPTIONS</b>"] + action_lines
        if has_multiple_claimable:
            lines += [
                "",
                "<i>Choose one option. Sleeper depth order does not confirm "
                "workload or touches.</i>",
            ]
        elif has_any_claimable and has_multiple_successors:
            lines += [
                "",
                "<i>Sleeper depth order does not confirm workload or touches.</i>",
            ]

    deterministic_note = _deterministic_note(alert)
    if deterministic_note:
        lines += ["", "⚠️ <b>NEXT STEP</b>", _escape(deterministic_note)]

    if model_summary:
        lines += ["", "💡 <b>WHY IT MATTERS</b>", _escape(model_summary)]

    backup_watch_note = _backup_watch_note(alert)
    if backup_watch_note:
        lines += ["", "👀 <b>BACKUP WATCH</b>", _escape(backup_watch_note)]

    # For tweets, headline is normally the full post, so printing both would
    # repeat it. RotoWire-style headline/body pairs get a visually separate
    # source report before the mechanical depth chart.
    body = (item.body or "").strip()
    headline = (item.headline or "").strip()
    redundant = (
        not body
        or body == headline
        or body.startswith(headline[:120])
        or headline.startswith(body[:120])
    )
    source_line = _source_line(item)
    source_rendered = False
    if not redundant:
        lines += ["", "📰 <b>REPORT</b>", f"<blockquote>{_escape(body[:280])}</blockquote>"]
        if source_line:
            lines.append(source_line)
            source_rendered = True

    context_lines = _context_block(
        alert.context,
        alert.all_leagues,
        classification.severity,
        show_ownership=not alert.availability_refresh_failed,
    )
    if context_lines:
        lines += [""] + context_lines

    if source_line and not source_rendered:
        lines += ["", f"🔗 {source_line}"]

    return _fit_telegram_limit("\n".join(lines).strip())


_TELEGRAM_STATES: dict[str, TelegramState] = {}


def telegram_state(config: Config) -> TelegramState:
    """Return the process-wide state shared by sender and command listener."""
    path = telegram_state_path(config)
    key = str(path.resolve())
    state = _TELEGRAM_STATES.get(key)
    if state is None:
        state = TelegramState(
            path,
            thread_hours=int(getattr(config, "player_thread_hours", 168)),
        )
        _TELEGRAM_STATES[key] = state
    return state


def retry_after_seconds() -> int:
    """Telegram's rate-limit hint for the current delivery thread."""
    return int(getattr(_SEND_CONTEXT, "retry_after", 0) or 0)


def _safe_error(error: object, token: str) -> str:
    """Never let a requests exception echo the bot token into logs or health."""
    rendered = str(error)
    return rendered.replace(token, "[redacted]") if token else rendered


def _post(session: requests.Session, config: Config, payload: dict) -> int | None:
    """POST sendMessage; return the created message_id.

    On failure this logs Telegram's own `description` and the offending text.
    Without them a 400 is undiagnosable after the fact - a literal "<--" in a
    message once broke every alert for a day and the logs only said
    "400 Bad Request".
    """
    _SEND_CONTEXT.retry_after = 0
    try:
        response = session.post(
            f"{API_BASE}/bot{config.telegram_bot_token}/sendMessage",
            timeout=REQUEST_TIMEOUT,
            json={"chat_id": config.telegram_chat_id, "parse_mode": "HTML", **payload},
        )
        if not response.ok:
            description = ""
            response_payload = {}
            try:
                response_payload = response.json()
                description = str(response_payload.get("description") or "")
            except ValueError:
                description = response.text[:200]
            description = _safe_error(description, config.telegram_bot_token)
            if response.status_code == 429:
                try:
                    _SEND_CONTEXT.retry_after = max(
                        0,
                        int((response_payload.get("parameters") or {}).get("retry_after") or 0),
                    )
                except (TypeError, ValueError):
                    _SEND_CONTEXT.retry_after = 0
            text = str(payload.get("text") or "")
            structured_log(
                logging.ERROR,
                "notify.rejected",
                httpStatus=response.status_code,
                telegramError=description,
                textLength=len(text),
                textPreview=text[:400],
            )
        response.raise_for_status()
        message_id = int(response.json()["result"]["message_id"])
        HEALTH.mark("telegram", ok=True, detail=f"message {message_id}")
        return message_id
    except (requests.RequestException, KeyError, ValueError, TypeError) as error:
        safe_error = _safe_error(error, config.telegram_bot_token)
        HEALTH.mark("telegram", ok=False, detail=safe_error)
        structured_log(logging.ERROR, "notify.send_failed", error=safe_error)
        return None


def _edit(
    session: requests.Session,
    config: Config,
    message_id: int,
    payload: dict,
) -> bool:
    """Edit an existing Telegram message, returning False for safe fallback."""
    try:
        response = session.post(
            f"{API_BASE}/bot{config.telegram_bot_token}/editMessageText",
            timeout=REQUEST_TIMEOUT,
            json={
                "chat_id": config.telegram_chat_id,
                "message_id": int(message_id),
                "parse_mode": "HTML",
                **payload,
            },
        )
        if not response.ok:
            try:
                description = str(response.json().get("description") or "")
            except ValueError:
                description = response.text[:200]
            # The prior attempt may have reached Telegram and then failed to
            # persist local state. Repeating that exact edit is idempotent;
            # commit the requested message state instead of falling back to a
            # duplicate sendMessage.
            if (
                response.status_code == 400
                and "message is not modified" in description.casefold()
            ):
                HEALTH.mark("telegram", ok=True, detail=f"message {message_id} unchanged")
                structured_log(
                    logging.INFO,
                    "notify.edit_already_applied",
                    messageId=message_id,
                )
                return True
            structured_log(
                logging.WARNING,
                "notify.edit_rejected",
                httpStatus=response.status_code,
                telegramError=_safe_error(description, config.telegram_bot_token),
            )
        response.raise_for_status()
        result = response.json().get("result")
        if not isinstance(result, dict) or int(result.get("message_id")) != int(message_id):
            raise ValueError("Telegram edit response did not identify the requested message")
        HEALTH.mark("telegram", ok=True, detail=f"message {message_id} edited")
        return True
    except (requests.RequestException, KeyError, ValueError, TypeError) as error:
        safe_error = _safe_error(error, config.telegram_bot_token)
        HEALTH.mark("telegram", ok=False, detail=safe_error)
        structured_log(logging.WARNING, "notify.edit_failed", error=safe_error)
        return False


def send_alert(session: requests.Session, config: Config, alert: Alert) -> int | None:
    text = format_alert(alert)

    if config.dry_run:
        structured_log(
            logging.INFO,
            "notify.dry_run",
            player=alert.item.player_name,
            tier=alert.tier,
            severity=alert.classification.severity,
        )
        print(text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
        print("-" * 60)
        return -1

    state = telegram_state(config)
    payload: dict = {
        "text": text,
        "link_preview_options": {"is_disabled": True},
        # Severity 1-2 lands silently; 3+ buzzes the phone.
        "disable_notification": alert.classification.severity <= 2,
    }
    controls_enabled = bool(getattr(config, "telegram_controls_enabled", False))
    if controls_enabled:
        payload["reply_markup"] = feedback_markup(alert.item)

    edit_target = state.coalescing_target(alert)
    if edit_target is not None:
        # A lower-urgency corroboration may improve the facts but must never
        # visually downgrade the original alert's severity.
        display_alert = alert
        if alert.classification.severity < edit_target.severity:
            display_alert = replace(
                alert,
                classification=replace(
                    alert.classification,
                    severity=edit_target.severity,
                ),
            )
        edit_payload = {
            "text": format_alert(display_alert),
            "link_preview_options": {"is_disabled": True},
        }
        current_token = alert_token(alert.item)
        if controls_enabled:
            edit_payload["reply_markup"] = feedback_markup_for_token(
                current_token,
                selected=state.feedback_verdict(current_token),
            )
        if _edit(session, config, edit_target.message_id, edit_payload):
            try:
                state_recorded = state.record_edited(
                    display_alert,
                    edit_target.message_id,
                    edit_target.token,
                )
            except Exception as error:  # noqa: BLE001 - retain outbox and retry
                state_recorded = False
                structured_log(
                    logging.ERROR,
                    "notify.edit_state_failed",
                    errorType=type(error).__name__,
                )
            if not state_recorded:
                structured_log(
                    logging.ERROR,
                    "notify.edit_not_committed",
                    player=alert.item.player_name,
                    eventType=alert.classification.event_type,
                    messageId=edit_target.message_id,
                )
                return None
            structured_log(
                logging.INFO,
                "notify.edited",
                player=alert.item.player_name,
                tier=alert.tier,
                severity=alert.classification.severity,
                eventType=alert.classification.event_type,
            )
            return edit_target.message_id

    previous = state.previous_message_id(alert.item.player_name)
    if previous is not None:
        # Reply to the most recent alert, not the oldest root. This rolling
        # chain remains readable with Telegram's per-message seven-day TTL.
        payload["reply_parameters"] = {
            "message_id": previous,
            "allow_sending_without_reply": True,
        }

    message_id = _post(session, config, payload)
    if message_id is not None:
        state.record_sent(alert, message_id)
        structured_log(
            logging.INFO,
            "notify.sent",
            player=alert.item.player_name,
            tier=alert.tier,
            severity=alert.classification.severity,
            eventType=alert.classification.event_type,
        )
    return message_id


def send_plain(
    session: requests.Session,
    config: Config,
    text: str,
    *,
    silent: bool = True,
    reply_to: int | None = None,
) -> int | None:
    if config.dry_run:
        structured_log(logging.INFO, "notify.dry_run_plain", preview=text[:160])
        return -1
    payload: dict = {
        "text": text,
        "disable_notification": bool(silent),
        "link_preview_options": {"is_disabled": True},
    }
    if reply_to is not None:
        payload["reply_parameters"] = {
            "message_id": int(reply_to),
            "allow_sending_without_reply": True,
        }
    return _post(session, config, payload)
