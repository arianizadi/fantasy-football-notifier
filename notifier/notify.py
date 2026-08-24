"""Telegram delivery and message formatting."""

from __future__ import annotations

import html
import logging
import re
import threading
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
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
_SEND_CONTEXT = threading.local()
PACIFIC = ZoneInfo("America/Los_Angeles")

# Plain text only. Severity leads as an explicit n/5 so the notification
# preview is scannable without decoding symbols.
TIER_LABEL = {
    "mine": "YOUR ROSTER",
    "claimable": "WAIVER OPPORTUNITY",
    "rival": "RIVAL ROSTER",
    "league": "LEAGUE NEWS",
    "preseason": "PRESEASON",
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
    return html.escape(value or "", quote=False)


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


def _own_tag(entry, leagues, labels: dict[str, str]) -> str:
    """Per-league ownership without collapsing same-provider leagues."""
    states = []
    for league in leagues:
        state, team = entry.ownership.get(league.key, ("free_agent", ""))
        if state == "free_agent":
            label = "FA"
        elif state == "mine":
            label = "YOU"
        else:
            label = (team or "taken")[:11]
        states.append((labels.get(league.key, league.short_label), label))

    if len({label for _, label in states}) == 1:
        only = states[0][1]
        return "FA all leagues" if only == "FA" and len(states) > 1 else only
    return " ".join(f"{league_label}:{label}" for league_label, label in states)


def _freshness_label(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M PT")


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
    event_type: str,
    *,
    show_ownership: bool = True,
) -> list[str]:
    """A source-attributed Sleeper view centered on the news subject."""
    if context is None or severity < MIN_SEVERITY_FOR_CONTEXT:
        return []

    labels = _league_labels(leagues)
    lines = [
        f"<b>{_escape(context.team)} {_escape(context.subject_position)} "
        "DEPTH / BACKUP WATCH · SLEEPER</b>"
    ]
    refreshed = _freshness_label(context.player_index_refreshed_at)
    if refreshed:
        lines.append(f"  refreshed {_escape(refreshed)}")
    if not show_ownership:
        lines.append("  league ownership hidden: live refresh failed")

    for entry in context.same_position:
        depth_slot = (
            f"{entry.position}{entry.depth_order}"
            if entry.depth_order is not None
            else entry.position
        )
        segments = [f"{_escape(depth_slot)} {_escape(entry.name)}"]
        owner = _own_tag(entry, leagues, labels) if show_ownership else ""
        if owner:
            segments.append(_escape(owner))
        if entry.search_rank and entry.search_rank < 99999:
            segments.append(f"Sleeper rank #{entry.search_rank}")
        if entry.sleeper_injury_status:
            segments.append(f"Sleeper injury: {_escape(entry.sleeper_injury_status)}")
        if entry.sleeper_status and entry.sleeper_status.casefold() != "active":
            segments.append(f"Sleeper status: {_escape(entry.sleeper_status)}")
        if entry.is_subject:
            # This marks why the row is highlighted without claiming an injury
            # state Sleeper did not report.
            segments.append(f"SUBJECT · {_escape(_event_label(event_type))}")
        lines.append("  " + " · ".join(segments))

    if context.adjacent:
        lines.append(f"<b>{_escape(context.team)} OTHER SLEEPER DEPTH LEADERS</b>")
        for entry in context.adjacent:
            owner = _own_tag(entry, leagues, labels) if show_ownership else ""
            depth_slot = (
                f"{entry.position}{entry.depth_order}"
                if entry.depth_order is not None
                else entry.position
            )
            suffix = f" · {_escape(owner)}" if owner else ""
            lines.append(f"  {_escape(depth_slot)} {_escape(entry.name)}{suffix}")
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
    as_of = f" · provider updated {_freshness_label(min(timestamps))}" if timestamps else ""
    return (
        f"FantasyPros cached {scoring} rankings{as_of}; may lag this breaking "
        "report and do not confirm role or workload."
    )


def _league_block(
    plays: LeaguePlays,
    severity: int,
    event_type: str,
    league_label: str,
    *,
    allow_adds: bool = True,
) -> list[str]:
    """One tight line per league. Availability differs between leagues."""
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
    parts: list[str] = []
    options: list[str] = []
    for beneficiary in claimable[:2]:
        slot = f"{beneficiary.position}{beneficiary.depth_order or ''}"
        detail = f"Sleeper depth {_escape(slot)}"
        if beneficiary.named_in_report:
            detail += " · named in report"
        fantasypros = _fantasypros_rank_detail(beneficiary)
        if fantasypros:
            detail += f" · {fantasypros}"
        options.append(f"<b>{_escape(beneficiary.name)}</b> ({detail})")
    if options:
        option_label = "PICKUP OPTIONS" if len(options) > 1 else "ADD OPTION"
        parts.append(f"{option_label} — " + " | ".join(options))
    if plays.bench_options:
        parts.append(f"START <b>{_escape(plays.bench_options[0])}</b>")
    lines = [f"{name}: " + " | ".join(parts)]

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
                if plays.capacity.ir_used < plays.capacity.ir_limit:
                    reserve += " (eligibility not checked)"
            capacity_parts.append(reserve)
        if capacity_parts:
            lines.append("  Roster occupancy: " + " · ".join(capacity_parts))
    lean = _fantasypros_lean(claimable[:2])
    if lean:
        lines.append(
            f"  FantasyPros rank lean: <b>{_escape(lean)}</b> "
            "(ranking context only; role unconfirmed)"
        )
    freshness = _fantasypros_freshness(claimable[:2])
    if freshness:
        lines.append("  " + _escape(freshness))
    return lines


def _deterministic_note(alert: Alert) -> str:
    """Conservative event guidance; never sourced from model prose."""
    event = normalized_event_type(alert.classification.event_type)
    if alert.tier == "preseason":
        if event == "return":
            return (
                "Draft note: Return news improves availability; confirm full practice "
                "and Week 1 status before adjusting draft value."
            )
        if event == "signing":
            return "Draft note: Recheck the official role and depth chart before adjusting draft value."
        if event in {"injury", "inactive", "suspension"}:
            return "Draft note: Recheck official availability before drafting this player."
        if event in {"trade", "release", "depth_chart", "usage"}:
            return "Draft note: Reassess role and team context before changing your draft value."
        if event == "practice_report":
            return "Draft note: Wait for the next official practice or game-status update."
        return "Draft note: Verify the report before changing your draft value."

    if event == "return":
        return "Lineup note: Confirm active status and expected workload before lineup lock."
    if event == "signing":
        return "Roster note: No automatic waiver move; confirm the player's official role first."
    if event in {"trade", "depth_chart", "usage"}:
        return "Roster note: Recheck the official role before making a lineup or waiver move."
    if event == "practice_report" and alert.classification.severity < 4:
        return "Lineup note: Wait for the final game-status report before making a change."
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
        f"Backup watch: {immediate.name} is next in Sleeper's {position} depth order. "
        "No pickup is recommended from return news; injury or inactive alerts "
        "recheck league availability."
    )


def format_alert(alert: Alert) -> str:
    """Scannable header, then everything needed to investigate.

    Line one carries severity + tier + player because Telegram previews it.
    Below that: an optional non-prescriptive model summary, deterministic
    moves per league, and source-attributed Sleeper depth context.
    """
    item, classification = alert.item, alert.classification
    tier = TIER_LABEL.get(alert.tier, "NEWS")
    event = _event_label(classification.event_type)

    head = f"<b>[{classification.severity}/5] {tier} — {event}</b>"
    if item.player_name:
        head += f" - <b>{_escape(item.player_name)}</b>"

    lines = [head, _escape(item.headline)]
    if alert.delivery_delayed:
        lines += ["", "Delivery note: delayed after a Telegram retry; availability was rechecked."]

    if not item.subject_confident:
        lines += [
            "",
            "Player attribution is unclear in this report; automatic "
            "pickup and lineup moves are withheld.",
        ]

    if alert.availability_refresh_failed:
        lines += [
            "",
            "League availability refresh failed; no ADD or free-agent recommendation is shown.",
        ]

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
    if model_summary:
        lines += ["", f"Model summary: {_escape(model_summary)}"]

    deterministic_note = _deterministic_note(alert)
    if deterministic_note:
        lines += ["", _escape(deterministic_note)]

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
        action_lines += _league_block(
            plays,
            classification.severity,
            classification.event_type,
            labels.get(plays.league.key, plays.league.short_label),
            allow_adds=not alert.availability_refresh_failed,
        )
    has_multiple_claimable = not alert.availability_refresh_failed and any(
        len(plays.claimable) > 1 for plays in alert.per_league
    )
    has_any_claimable = not alert.availability_refresh_failed and any(
        plays.claimable for plays in alert.per_league
    )
    if action_lines:
        lines += ["", "<b>LEAGUE-SPECIFIC MOVES</b>"] + action_lines
        if has_multiple_claimable:
            lines += [
                "",
                "Backup note: Pickup options are alternatives, not instructions to add both. "
                "Sleeper depth order does not confirm workload or touch share.",
            ]
        elif has_any_claimable and has_multiple_successors:
            lines += [
                "",
                "Backup note: Sleeper depth order does not confirm workload or touch share.",
            ]

    backup_watch_note = _backup_watch_note(alert)
    if backup_watch_note:
        lines += ["", _escape(backup_watch_note)]

    context_lines = _context_block(
        alert.context,
        alert.all_leagues,
        classification.severity,
        classification.event_type,
        show_ownership=not alert.availability_refresh_failed,
    )
    if context_lines:
        lines += [""] + context_lines

    # For tweets, headline is just body truncated, so printing both repeats the
    # whole tweet. Only show the body when it actually adds something.
    body = (item.body or "").strip()
    headline = (item.headline or "").strip()
    redundant = (
        not body
        or body == headline
        or body.startswith(headline[:120])
        or headline.startswith(body[:120])
    )
    if not redundant:
        lines += ["", _escape(body[:280])]

    if item.url:
        source_name = {
            "twitter": "X source",
            "rotowire": "RotoWire source",
        }.get(item.source.casefold(), f"{item.source or 'news'} source")
        raw_url = item.url.strip()
        parsed = urlsplit(raw_url)
        if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
            source_line = (
                f'<a href="{html.escape(raw_url, quote=True)}">'
                f"{_escape(source_name)}</a>"
            )
        else:
            # Telegram accepts only safe URL schemes in HTML links. Keep the
            # provenance label even when an upstream URL is malformed.
            source_line = _escape(source_name)
        if item.published_at is not None:
            published = item.published_at
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            source_line += f" · reported {published.astimezone(PACIFIC):%Y-%m-%d %H:%M PT}"
        lines += ["", source_line]

    return "\n".join(lines)


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


def send_plain(session: requests.Session, config: Config, text: str) -> int | None:
    if config.dry_run:
        structured_log(logging.INFO, "notify.dry_run_plain", preview=text[:160])
        return -1
    return _post(session, config, {"text": text, "disable_notification": True})
