"""Telegram delivery and message formatting."""

from __future__ import annotations

import html
import logging

import requests

from .config import Config, history_path
from .logging_utils import structured_log
from .models import Alert
from .history import MessageHistory
from .plays import LeaguePlays

API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT = 15

# Plain text only. Severity leads as an explicit n/5 so the notification
# preview is scannable without decoding symbols.
TIER_LABEL = {
    "mine": "YOUR ROSTER",
    "claimable": "WAIVER OPPORTUNITY",
    "rival": "RIVAL ROSTER",
    "league": "LEAGUE NEWS",
    "preseason": "PRESEASON — DRAFT IMPACT",
}

STATE_LABEL = {
    "free_agent": "FREE AGENT",
    "mine": "already yours",
    "rostered": "rostered",
}


def _escape(value: str) -> str:
    return html.escape(value or "", quote=False)


MIN_SEVERITY_FOR_PLAYS = 3
# Below this the full depth chart is noise; above it you want to judge for
# yourself rather than trust the mechanical next-man-up.
MIN_SEVERITY_FOR_CONTEXT = 3
LEAGUE_ABBREV = {"espn": "ESP", "sleeper": "SLP"}


def _own_tag(entry, leagues) -> str:
    """Per-league ownership, compact: 'FA both' or 'ESP:FA SLP:Gusty'."""
    states = []
    for league in leagues:
        state, team = entry.ownership.get(league.key, ("free_agent", ""))
        if state == "free_agent":
            label = "FA"
        elif state == "mine":
            label = "YOU"
        else:
            label = (team or "taken")[:11]
        states.append((LEAGUE_ABBREV.get(league.provider, league.provider[:3].upper()), label))

    if len({label for _, label in states}) == 1:
        only = states[0][1]
        return "FA both" if only == "FA" and len(states) > 1 else only
    return " ".join(f"{abbrev}:{label}" for abbrev, label in states)


def _context_block(context, leagues, severity: int) -> list[str]:
    """The investigation view: full depth chart + other skill positions."""
    if context is None or severity < MIN_SEVERITY_FOR_CONTEXT:
        return []

    lines = [f"<b>{_escape(context.team)} {_escape(context.subject_position)} DEPTH</b>"]
    for entry in context.same_position:
        order = entry.depth_order if entry.depth_order is not None else "-"
        # Pre-draft there are no drafted leagues, so ownership is empty. Build
        # the line from present segments only, or it renders "Name ·  · #23".
        segments = [f"{order} {_escape(entry.name)}"]
        owner = _own_tag(entry, leagues)
        if owner:
            segments.append(owner)
        if entry.search_rank and entry.search_rank < 99999:
            segments.append(f"#{entry.search_rank}")
        # Must not start with "<": Telegram's HTML parser reads "<--" as an
        # opening tag and rejects the whole message with a 400.
        marker = "   [INJURED]" if entry.is_subject else ""
        lines.append("  " + " · ".join(segments) + marker)

    if context.adjacent:
        lines.append(f"<b>{_escape(context.team)} other starters</b>")
        for entry in context.adjacent:
            owner = _own_tag(entry, leagues)
            suffix = f" · {owner}" if owner else ""
            lines.append(f"  {_escape(entry.position)} {_escape(entry.name)}{suffix}")
    return lines


def _league_block(plays: LeaguePlays, severity: int) -> list[str]:
    """One tight line per league. Availability differs between leagues."""
    if severity < MIN_SEVERITY_FOR_PLAYS:
        return []
    claimable = plays.claimable
    if not claimable and not plays.bench_options:
        return []

    name = _escape(plays.league.short_label)
    parts: list[str] = []
    for beneficiary in claimable[:2]:
        slot = f"{beneficiary.position}{beneficiary.depth_order or ''}"
        parts.append(f"ADD <b>{_escape(beneficiary.name)}</b> ({_escape(slot)})")
    if plays.bench_options:
        parts.append(f"start {_escape(plays.bench_options[0])}")
    return [f"{name}: " + " | ".join(parts)]


def format_alert(alert: Alert) -> str:
    """Scannable header, then everything needed to investigate.

    Line one carries severity + tier + player because Telegram previews it.
    Below that: the model's read, the recommended move per league, and the
    full NFL depth chart with per-league ownership so you can disagree with
    the mechanical pick - the real beneficiary of a WR injury is often the
    slot man or the TE, not the literal WR3.
    """
    item, classification = alert.item, alert.classification
    tier = TIER_LABEL.get(alert.tier, "NEWS")

    head = f"<b>[{classification.severity}/5] {tier}</b>"
    if item.player_name:
        head += f" - <b>{_escape(item.player_name)}</b>"

    lines = [head, _escape(item.headline)]

    if classification.fantasy_impact:
        lines += ["", _escape(classification.fantasy_impact)]

    action_lines: list[str] = []
    for plays in alert.per_league:
        action_lines += _league_block(plays, classification.severity)
    if action_lines:
        lines += [""] + action_lines

    context_lines = _context_block(
        alert.context, alert.all_leagues, classification.severity
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
        lines += ["", f'<a href="{_escape(item.url)}">source</a>']

    return "\n".join(lines)


_HISTORY: MessageHistory | None = None


def history(config: Config) -> MessageHistory:
    global _HISTORY
    if _HISTORY is None:
        _HISTORY = MessageHistory(history_path(config))
    return _HISTORY


def delete_message(session: requests.Session, config: Config, message_id: int) -> bool:
    try:
        response = session.post(
            f"{API_BASE}/bot{config.telegram_bot_token}/deleteMessage",
            timeout=REQUEST_TIMEOUT,
            json={"chat_id": config.telegram_chat_id, "message_id": message_id},
        )
        return response.ok
    except requests.RequestException:
        return False


def _post(session: requests.Session, config: Config, payload: dict) -> int | None:
    """POST sendMessage; return the created message_id.

    On failure this logs Telegram's own `description` and the offending text.
    Without them a 400 is undiagnosable after the fact - a literal "<--" in a
    message once broke every alert for a day and the logs only said
    "400 Bad Request".
    """
    try:
        response = session.post(
            f"{API_BASE}/bot{config.telegram_bot_token}/sendMessage",
            timeout=REQUEST_TIMEOUT,
            json={"chat_id": config.telegram_chat_id, "parse_mode": "HTML", **payload},
        )
        if not response.ok:
            description = ""
            try:
                description = str(response.json().get("description") or "")
            except ValueError:
                description = response.text[:200]
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
        # Track it so expiry only ever deletes messages this bot created.
        store = history(config)
        store.record(message_id)
        store.save()
        return message_id
    except (requests.RequestException, KeyError, ValueError, TypeError) as error:
        structured_log(logging.ERROR, "notify.send_failed", error=str(error))
        return None


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

    message_id = _post(
        session,
        config,
        {
            "text": text,
            "link_preview_options": {"is_disabled": True},
            # Severity 1-2 lands silently; 3+ buzzes the phone.
            "disable_notification": alert.classification.severity <= 2,
        },
    )
    if message_id is not None:
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
