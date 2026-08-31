"""League-aware action urgency with conservative embedding corroboration.

Severity answers how important an NFL report is. Urgency answers how quickly
this manager should review or act on it. Live roster position and verified
pickup options therefore remain authoritative. Historical vectors corroborate
the rule result. A separately gated future mode can promote MONITOR to ACT
TODAY only after enough context-compatible live evidence exists.

Embeddings never lower urgency, never suppress an alert, and never create
ACT NOW. Provider/database failures return the byte-for-byte rule assessment.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from dataclasses import dataclass, replace
from typing import Any

from .dedupe import semantic_event_status, semantic_event_type
from .embeddings import (
    INPUT_VERSION,
    EmbeddingService,
    EmbeddingVector,
    canonical_embedding_text,
    cosine_similarity,
    embedding_input_hash,
    news_item_from_row,
    unpack_vector,
)
from .event_store import EventStore, classification_direction
from .logging_utils import structured_log
from .matcher import compact_key
from .models import ActionUrgency, Alert, Classification
from .plays import normalized_event_type

POLICY_VERSION = "urgency-v2"
LEVELS = ("fyi", "monitor", "act_today", "act_now")
LEVEL_RANK = {level: index for index, level in enumerate(LEVELS)}
MAX_HISTORY_CANDIDATES = 200

REMOVAL_EVENTS = frozenset({"injury", "inactive", "release", "suspension"})
ROLE_EVENTS = frozenset({"trade", "signing", "depth_chart", "usage"})

REASON_TEXT = {
    "uncertain_subject": "Confirm which player the report is about before acting.",
    "availability_unverified": "League availability could not be refreshed; verify before acting.",
    "starter_unavailable": "A starting-slot player may require an immediate replacement.",
    "starter_major_risk": "A starting-slot player has a major availability risk.",
    "claimable_replacement": "A possible next-man-up option was confirmed available.",
    "claimable_watch": "Review the available next-man-up options today.",
    "roster_contingency": "Build a contingency for a player on your roster.",
    "roster_role_change": "Review how this role change affects your roster today.",
    "await_final_status": "Wait for the next official status or workload update.",
    "return_monitor": "Confirm active status and workload before changing plans.",
    "draft_monitor": "Track the draft-value effect; no immediate roster move exists.",
    "major_league_watch": "Important league news, but no verified immediate move exists.",
    "historical_context_incomplete": "Saved history lacks the live roster context needed for action timing.",
    "embedding_history_lift": "Independent similar saved reports support reviewing this today.",
    "embedding_history_support": "Independent similar saved reports support this urgency level.",
    "informational": "No immediate roster action is supported.",
}

_DIRECT_RETURN_CUE = re.compile(
    r"\b(?:activated\b.{0,90}\b(?:from|off)\b.{0,30}"
    r"\b(?:active[ /-]*)?(?:pup|physically\s+unable\s+to\s+perform|"
    r"ir|injured\s+reserve)\b|return(?:s|ed)?\s+to\s+(?:full\s+)?practice|"
    r"resum(?:e|es|ed|ing)\s+(?:full\s+)?practice|"
    r"cleared\s+to\s+(?:practice|play|return)|removed\s+from\s+the\s+"
    r"(?:active[ /-]*)?(?:pup|physically\s+unable\s+to\s+perform)\s+list)\b",
    re.I,
)
_RENEWED_UNAVAILABLE_CUE = re.compile(
    r"\b(?:not\s+activated|re[- ]?injur(?:ed|y)|ruled\s+out|will\s+not\s+play|"
    r"inactive|placed\s+(?:back\s+)?on\s+(?:ir|injured\s+reserve)|"
    r"season[- ]ending|out\s+for\s+the\s+season|torn\s+(?:acl|achilles))\b",
    re.I,
)
_RETURN_STATUS_CUE = re.compile(
    r"\b(?:activated|cleared|full\s+participant|"
    r"return(?:s|ed)?\s+to\s+(?:full\s+)?practice|"
    r"resum(?:e|es|ed|ing)\s+(?:full\s+)?practice)\b",
    re.I,
)
_GAME_CONTEXT = (
    r"(?:week\s+\d{1,2}|today|tonight|tomorrow|this\s+week|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"the\s+(?:game|opener)|(?:season|home)\s+opener)"
)
_GAME_AVAILABLE_CUE = re.compile(
    r"\b(?:"
    r"(?:is|are|will\s+be)\s+(?:officially\s+)?active\s+(?:for\s+)?"
    + _GAME_CONTEXT
    + r"|will\s+play\s+(?:(?:in|on|for|against)\s+)?"
    + _GAME_CONTEXT
    + r"|(?:is|are|will\s+be)\s+(?:fully\s+)?available\s+"
    r"(?:for|against)\s+"
    + _GAME_CONTEXT
    + r")\b",
    re.I,
)
_EXPECTED_GAME_AVAILABLE_CUE = re.compile(
    r"\b(?:"
    r"(?:is|are|was|were|remains?|remain|should\s+be|will\s+be)\s+"
    r"(?:expected|likely|projected|hopeful)\s+(?:to\s+(?:be\s+)?)?"
    r"(?:active|available|play|suit\s+up|[\"'\u201c\u201d]?good\s+to\s+go[\"'\u201c\u201d]?)"
    r"|(?:is|are|was|were|remains?|remain)\s+on\s+track\s+"
    r"(?:to\s+(?:be\s+)?(?:play|active|available|suit\s+up)|for\s+"
    + _GAME_CONTEXT
    + r")"
    r"|(?:expected|likely|projected|hopeful|on\s+track)\s+"
    r"(?:to\s+(?:be\s+)?(?:play|active|available|suit\s+up|"
    r"[\"'\u201c\u201d]?good\s+to\s+go[\"'\u201c\u201d]?)|for\s+"
    + _GAME_CONTEXT
    + r")"
    r"|(?:should|will)\s+be\s+[\"'\u201c\u201d]?good\s+to\s+go"
    r"[\"'\u201c\u201d]?(?:\s+for\s+"
    + _GAME_CONTEXT
    + r")?"
    r"|(?:is|are|was|were|will\s+be)\s+(?:being\s+)?counted\s+on\s+"
    r"(?:to\s+play|for\s+"
    + _GAME_CONTEXT
    + r")"
    r")\b",
    re.I,
)
_RETURN_NEGATION_PREFIX = re.compile(
    # Keep this bounded to the current clause. It covers ordinary NFL wording
    # such as "has not yet been activated" and "is not expected to be
    # activated" without letting a negation in an earlier sentence cancel a
    # later, affirmative return update. "Not only" is a positive construction.
    r"\b(?:"
    r"(?:not(?!\s+only\b)|never|"
    r"(?:has|have|had|is|are|was|were|can|could|will|would|do|does|did)n['’]t)\b"
    r"(?:(?![.!?;\n]|\bbut\b|\bhowever\b).){0,64}\b|"
    r"(?:has|have|had)\s+yet\s+to\s+(?:be\s+)?|"
    r"(?:is|are|was|were)\s+unable\s+to\s+(?:be\s+)?|"
    r"fail(?:s|ed)?\s+to\s+(?:be\s+)?"
    r")"
    r"\s*$",
    re.I,
)
_HISTORICAL_UNAVAILABLE_PREFIX = re.compile(
    r"\b(?:after|following|despite)\s+(?:(?:he|she)\s+was\s+|"
    r"(?:having\s+)?been\s+|being\s+)?$",
    re.I,
)
_SUSPENSION_RESOLUTION_CUE = re.compile(
    r"\b(?:lifted|ended|reinstated)\b",
    re.I,
)
_SUSPENSION_CONTEXT_CUE = re.compile(r"\b(?:suspension|suspended|ban)\b", re.I)
_SUSPENSION_RENEWED_CUE = re.compile(
    r"\b(?:re[- ]?suspended|suspended\s+again|again\s+suspended|"
    r"suspension\s+(?:was\s+)?reinstated)\b",
    re.I,
)
_CONFIRMED_SUSPENSION_CUE = re.compile(
    r"\b(?:suspended|inactive|"
    r"commissioner(?:['\u2019]s)?[-\s]+exempt\s+list)\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    r"[-\s]+(?:game|week)[-\s]+suspension\b|"
    r"\bsuspension\b(?:(?![.!?;\n]).){0,48}\b(?:issued|imposed|announced|"
    r"upheld|reinstated|begins?|starts?|takes\s+effect|remains?\s+in\s+effect)\b|"
    r"\b(?:issued|imposed|announced|upheld|reinstated|serving)\b"
    r"(?:(?![.!?;\n]).){0,48}\bsuspension\b",
    re.I,
)
_SUSPENSION_SPECULATION_PREFIX = re.compile(
    r"\b(?:could|may|might|should|possibly|potentially|likely|expected|"
    r"possible|potential|risk|possibility|if|face|faces|facing|seek|seeks|"
    r"seeking|recommend|recommends|recommended)\b"
    r"(?:(?![.!?;\n]).){0,64}$",
    re.I,
)
_SUSPENSION_STILL_ACTIVE_CUE = re.compile(
    r"\b(?:not\s+reinstated\b(?:(?![.!?;\n]).){0,64}\bsuspension\b|"
    r"suspension\b(?:(?![.!?;\n]).){0,48}\b(?:not\s+lifted|not\s+ended|"
    r"remains?\s+in\s+effect))",
    re.I,
)
_RELEASE_TRANSACTION_CUE = re.compile(
    r"\b(?:releas(?:e|es|ed|ing)|waiv(?:e|es|ed|ing)(?:\s*/\s*injured)?|"
    r"cut(?:s|ting)?)\b",
    re.I,
)
_ROSTER_RETURN_CUE = re.compile(
    r"\b(?:re[- ]sign(?:s|ed|ing)|"
    r"claim(?:s|ed)\b(?:(?![.!?;\n]).){0,48}\bback)\b",
    re.I,
)


@dataclass(frozen=True)
class UrgencyServiceStatus:
    assessed: int
    corroborated: int
    lifted: int
    abstained: int
    failures: int


def urgency_rank(level: str) -> int:
    return LEVEL_RANK.get((level or "").strip().lower(), 0)


def urgency_reason(assessment: ActionUrgency | None) -> str:
    if assessment is None:
        return ""
    for code in assessment.reason_codes:
        if code in REASON_TEXT:
            return REASON_TEXT[code]
    return REASON_TEXT["informational"]


def _return_cue_is_negated(text: str, cue_start: int) -> bool:
    """Whether a return cue has a nearby, same-clause negative prefix."""
    prefix = text[max(0, cue_start - 96) : cue_start]
    return _RETURN_NEGATION_PREFIX.search(prefix) is not None


def _unavailable_cue_is_background(text: str, cue_start: int) -> bool:
    """Whether a later-written absence cue is background for a return."""
    prefix = text[max(0, cue_start - 64) : cue_start]
    return _HISTORICAL_UNAVAILABLE_PREFIX.search(prefix) is not None


def _subject_reference_pattern(player_name: str) -> re.Pattern[str] | None:
    parts = [part for part in player_name.split() if part]
    if not parts:
        return None
    aliases = [r"\s+".join(re.escape(part) for part in parts)]
    if len(parts) > 1 and len(parts[-1]) >= 3:
        aliases.append(re.escape(parts[-1]))
    return re.compile(r"\b(?:" + "|".join(aliases) + r")\b", re.I)


def _clause_bounds(text: str, cue: re.Match[str]) -> tuple[int, int]:
    clause_start = max(
        text.rfind(".", 0, cue.start()),
        text.rfind("!", 0, cue.start()),
        text.rfind("?", 0, cue.start()),
        text.rfind(";", 0, cue.start()),
        text.rfind("\n", 0, cue.start()),
    ) + 1
    clause_end_candidates = [
        index
        for marker in (".", "!", "?", ";", "\n")
        if (index := text.find(marker, cue.end())) >= 0
    ]
    clause_end = min(clause_end_candidates) if clause_end_candidates else len(text)
    return clause_start, clause_end


def _cue_refers_to_subject(text: str, cue: re.Match[str], item: Any) -> bool:
    """Conservatively bind a status/transaction cue to the saved subject."""
    player = _subject_reference_pattern(str(item.player_name or ""))
    if player is None:
        return False
    clause_start, clause_end = _clause_bounds(text, cue)
    before = text[clause_start : cue.start()]
    after = text[cue.start() : clause_end]
    after_match = player.search(after[:80])
    if after_match is not None:
        lead = after[: after_match.start()]
        if not re.search(r"\b(?:but|while|whereas|although|except)\b", lead, re.I):
            return True
    if re.search(
        player.pattern
        + r"(?:['’]s?)?\s+(?:(?:was|is|has|had|will|now|officially|"
        + r"reportedly|fully|also|been|being|be|remains?|not|never|yet|"
        + r"expected|unable|to|hasn['’]t|isn['’]t|wasn['’]t)\s+){0,8}$",
        before,
        re.I,
    ):
        return True
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", after[:80]):
        return False
    subject_precedes = player.search(before) is not None
    if subject_precedes and re.search(r"\b(?:him|her)\b", after[:64], re.I):
        return True
    if subject_precedes and re.search(
        r"\b(?:he|she)\s+(?:(?:was|is|has|had|will|now|been|being)\s+){0,4}$",
        before,
        re.I,
    ):
        return True
    if re.search(
        r"\b(?:he|she)\s+(?:(?:was|is|has|had|will|now|been|being)\s+){0,4}$",
        before,
        re.I,
    ) and player.search(text[max(0, clause_start - 120) : clause_start]):
        return True
    if subject_precedes and re.search(
        r"\b(?:but|and|then)\s+(?:(?:he|she)\s+)?"
        r"(?:(?:remains?|was|is|has|had|will|now|again|later)\s+){0,4}$",
        before,
        re.I,
    ):
        return True
    # RotoWire's dedicated article URL/title supplies the subject even when a
    # compact headline starts with only "Activated" or "Ruled out".
    return bool(
        str(item.source or "").casefold() == "rotowire"
        and bool(getattr(item, "subject_confident", True))
        and cue.start() < len(str(item.headline or ""))
        and cue.start() <= 12
    )


def _nearby_status_cue_refers_to_subject(
    text: str,
    cue: re.Match[str],
    item: Any,
) -> bool:
    """Bind compact status headlines without borrowing a teammate's status."""
    if _cue_refers_to_subject(text, cue, item):
        return True
    player = _subject_reference_pattern(str(item.player_name or ""))
    if player is None:
        return False
    clause_start, clause_end = _clause_bounds(text, cue)
    before = text[clause_start : cue.start()]
    after = text[cue.end() : clause_end]
    before_matches = list(player.finditer(before))
    if before_matches:
        between = before[before_matches[-1].end() :]
        if len(between) <= 80 and re.search(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            between,
        ) is None:
            return True
    after_match = player.search(after[:80])
    if after_match is not None:
        lead = after[: after_match.start()]
        if re.search(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            lead,
        ) is None:
            return True
    # RotoWire and FantasyPros establish the subject structurally in their
    # dedicated title/article even when a compact headline omits the name.
    return bool(
        str(item.source or "").casefold() in {"rotowire", "fantasypros"}
        and bool(getattr(item, "subject_confident", True))
        and cue.start() < len(str(item.headline or ""))
    )


def _game_available_cue_refers_to_subject(
    text: str,
    cue: re.Match[str],
    item: Any,
) -> bool:
    """Bind definitive game availability without looking past the cue.

    A player named after ``Chris Godwin will play Sunday`` is commonly a
    teammate discussed as context, not the subject of that status.  Unlike an
    activation (``activated Kittle from PUP``), these cues are grammatically
    preceded by their subject, so only that position or an unambiguous pronoun
    is accepted.
    """
    player = _subject_reference_pattern(str(item.player_name or ""))
    if player is None:
        return False
    clause_start, _clause_end = _clause_bounds(text, cue)
    before = text[clause_start : cue.start()]
    if re.search(player.pattern + r"(?:['’]s)?\s*$", before, re.I):
        return True
    if re.search(r"\b(?:he|she)\s*$", before, re.I):
        previous = text[max(0, clause_start - 120) : clause_start]
        if player.search(previous) is not None:
            return True
    return bool(
        str(item.source or "").casefold() == "rotowire"
        and bool(getattr(item, "subject_confident", True))
        and cue.start() < len(str(item.headline or ""))
        and cue.start() <= 12
    )


def _suspension_is_confirmed(text: str, item: Any) -> bool:
    """Require an actual NFL availability restriction, not legal speculation."""
    for pattern in (_SUSPENSION_RENEWED_CUE, _SUSPENSION_STILL_ACTIVE_CUE):
        for match in pattern.finditer(text):
            if _nearby_status_cue_refers_to_subject(text, match, item):
                return True
    for match in _CONFIRMED_SUSPENSION_CUE.finditer(text):
        prefix = text[max(0, match.start() - 96) : match.start()]
        if (
            _RETURN_NEGATION_PREFIX.search(prefix) is not None
            or _SUSPENSION_SPECULATION_PREFIX.search(prefix) is not None
        ):
            continue
        if _nearby_status_cue_refers_to_subject(text, match, item):
            return True
    return False


def _suspension_cue_refers_to_subject(
    text: str,
    cue: re.Match[str],
    item: Any,
) -> bool:
    if _cue_refers_to_subject(text, cue, item):
        return True
    player = _subject_reference_pattern(str(item.player_name or ""))
    if player is None:
        return False
    clause_start, _clause_end = _clause_bounds(text, cue)
    before = text[clause_start : cue.start()]
    return bool(
        re.search(
            player.pattern
            + r"(?:['’]s?)?\s+suspension\s+"
            + r"(?:(?:has|had|was|is|has\s+been)\s+){0,3}$",
            before,
            re.I,
        )
    )


def _suspension_was_resolved(text: str, item: Any) -> bool:
    renewed_matches = [
        match
        for match in _SUSPENSION_RENEWED_CUE.finditer(text)
        if _suspension_cue_refers_to_subject(text, match, item)
    ]
    positive = []
    for match in _SUSPENSION_RESOLUTION_CUE.finditer(text):
        if any(
            renewed.start() <= match.start() < renewed.end()
            for renewed in renewed_matches
        ):
            continue
        context = text[max(0, match.start() - 80) : match.end() + 80]
        if (
            match.group(0).casefold() != "reinstated"
            and _SUSPENSION_CONTEXT_CUE.search(context) is None
        ):
            continue
        if (
            not _return_cue_is_negated(text, match.start())
            and _suspension_cue_refers_to_subject(text, match, item)
        ):
            positive.append(match.start())
    if not positive:
        return False
    renewed = [match.start() for match in renewed_matches]
    return not renewed or max(positive) > max(renewed)


def _release_was_reversed(text: str, item: Any) -> bool:
    positive = [
        match.start()
        for match in _ROSTER_RETURN_CUE.finditer(text)
        if not _return_cue_is_negated(text, match.start())
        and _cue_refers_to_subject(text, match, item)
    ]
    if not positive:
        return False
    releases = []
    for match in _RELEASE_TRANSACTION_CUE.finditer(text):
        if not _cue_refers_to_subject(text, match, item):
            continue
        position = match.start()
        if (
            any(resolution < position for resolution in positive)
            and _unavailable_cue_is_background(text, position)
        ):
            continue
        releases.append(position)
    return not releases or max(positive) > max(releases)


def canonical_urgency_event(item: Any, classification: Any) -> str:
    """Canonical event used by both action filtering and urgency history."""
    event = semantic_event_type(
        item,
        classification.event_type,
        str(classification.raw.get("event_type") or ""),
    )
    text = f"{item.headline}\n{item.body}"
    if event == "suspension" and _suspension_was_resolved(text, item):
        return "return"
    if event == "suspension" and not _suspension_is_confirmed(text, item):
        # Arrests, charges, allegations, and investigations can eventually
        # produce discipline, but they do not remove a player today. Treating
        # that possibility as an absence creates a false next-man-up alert.
        return "other"
    if event == "release" and _release_was_reversed(text, item):
        return "signing"
    # A provider/model can label an activation as an injury because the report
    # also mentions PUP/IR or a limited first practice. A narrow direct return
    # cue wins unless the same report explicitly states a renewed absence.
    raw_status_matches = list(_RETURN_STATUS_CUE.finditer(text))
    raw_game_available_matches = list(_GAME_AVAILABLE_CUE.finditer(text))
    raw_expected_available_matches = list(
        _EXPECTED_GAME_AVAILABLE_CUE.finditer(text)
    )
    direct_matches = [
        match
        for match in _DIRECT_RETURN_CUE.finditer(text)
        if _cue_refers_to_subject(text, match, item)
    ]
    status_matches = [
        match
        for match in raw_status_matches
        if _cue_refers_to_subject(text, match, item)
    ]
    game_available_matches = [
        match
        for match in raw_game_available_matches
        if _game_available_cue_refers_to_subject(text, match, item)
    ]
    expected_available_matches = [
        match
        for match in raw_expected_available_matches
        if _nearby_status_cue_refers_to_subject(text, match, item)
    ]
    positive_status_positions = [
        match.start()
        for match in (
            *status_matches,
            *game_available_matches,
            *expected_available_matches,
        )
        if not _return_cue_is_negated(text, match.start())
    ]
    positive_status = bool(positive_status_positions)
    negated_return = bool(status_matches or game_available_matches) and not positive_status
    unavailable_positions = []
    for match in _RENEWED_UNAVAILABLE_CUE.finditer(text):
        if not _cue_refers_to_subject(text, match, item):
            continue
        position = match.start()
        if (
            any(positive < position for positive in positive_status_positions)
            and _unavailable_cue_is_background(text, position)
        ):
            continue
        unavailable_positions.append(position)
    # Reports commonly restate the old absence before the new development:
    # "ruled out last week, returned today." Let the latest explicit state in
    # the report win. The reverse order still fails closed when a player
    # returns and is then re-injured or ruled out.
    renewed_unavailable = bool(
        unavailable_positions
        and (
            not positive_status_positions
            or max(unavailable_positions) > max(positive_status_positions)
        )
    )
    direct_return = (
        (
            any(
                not _return_cue_is_negated(text, match.start())
                for match in direct_matches
            )
            or bool(
                (game_available_matches or expected_available_matches)
                and positive_status
            )
        )
        and not renewed_unavailable
    )
    if event == "return" and negated_return:
        return "injury"
    if (
        event == "return"
        and (
            raw_status_matches
            or raw_game_available_matches
            or raw_expected_available_matches
        )
        and not (
            status_matches
            or game_available_matches
            or expected_available_matches
        )
    ):
        return "other"
    if event in {"other", "injury", "inactive", "practice_report", "return"} and (
        direct_return
        or (
            positive_status
            and not renewed_unavailable
            and semantic_event_status(item, event) == "cleared"
        )
    ):
        return "return"
    return normalized_event_type(event)


def urgency_event_type(alert: Alert) -> str:
    return canonical_urgency_event(alert.item, alert.classification)


def urgency_event_status(alert: Alert) -> str:
    event = urgency_event_type(alert)
    if event == "return":
        return "cleared"
    return semantic_event_status(alert.item, event) or "unspecified"


def urgency_direction(alert: Alert) -> str:
    event = urgency_event_type(alert)
    if event == "return":
        return "positive"
    if (
        event == "signing"
        and normalized_event_type(alert.classification.event_type) == "release"
    ):
        return "positive"
    if event in REMOVAL_EVENTS:
        return "negative"
    return classification_direction(alert.classification)


def _assessment(
    level: str,
    reason: str,
    *,
    action_available: bool,
    roster_relevant: bool,
    availability_verified: bool,
    basis: str = "rules",
    canonical_event_type: str = "",
    direction: str = "unknown",
    event_status: str = "",
    action_context: str = "",
    subject_is_starter: bool = False,
) -> ActionUrgency:
    bounded = level if level in LEVEL_RANK else "fyi"
    return ActionUrgency(
        rule_level=bounded,
        level=bounded,
        reason_codes=(reason,),
        basis=basis,
        policy_version=POLICY_VERSION,
        action_available=action_available,
        roster_relevant=roster_relevant,
        availability_verified=availability_verified,
        canonical_event_type=canonical_event_type,
        direction=direction,
        event_status=event_status,
        action_context=action_context,
        subject_is_starter=subject_is_starter,
    )


def assess_rule_urgency(alert: Alert) -> ActionUrgency:
    """Return the authoritative rule result from current league facts."""
    event = urgency_event_type(alert)
    reported_event = normalized_event_type(
        str(alert.classification.raw.get("model_event_type") or "")
        or alert.classification.event_type
    )
    unconfirmed_suspension = bool(
        reported_event == "suspension" and event == "other"
    )
    direction = urgency_direction(alert)
    status = urgency_event_status(alert)
    severity = max(1, min(5, int(alert.classification.severity)))
    mine = [plays for plays in alert.per_league if plays.subject_state == "mine"]
    starter = any(plays.subject_is_starter for plays in mine)
    mine_starter_actions = bool(
        not alert.availability_refresh_failed
        and any(
            plays.subject_is_starter and (plays.claimable or plays.bench_options)
            for plays in mine
        )
    )
    external_claimable = [
        plays
        for plays in alert.per_league
        if plays.subject_state != "mine" and plays.claimable
    ]
    claimable = bool(not alert.availability_refresh_failed and external_claimable)
    action_available = mine_starter_actions or claimable
    roster_relevant = bool(mine)
    availability_verified = not alert.availability_refresh_failed
    action_context = (
        "uncertain"
        if not alert.item.subject_confident
        else "preseason"
        if alert.tier == "preseason"
        else "mine_starter"
        if starter
        else "mine_bench"
        if mine
        else "claimable"
        if claimable
        else "rival"
        if alert.tier == "rival"
        else "league"
    )

    def result(
        level: str,
        reason: str,
        *,
        has_action: bool = action_available,
        relevant: bool = roster_relevant,
        verified: bool = availability_verified,
    ) -> ActionUrgency:
        return _assessment(
            level,
            reason,
            action_available=has_action,
            roster_relevant=relevant,
            availability_verified=verified,
            canonical_event_type=event,
            direction=direction,
            event_status=status,
            action_context=action_context,
            subject_is_starter=starter,
        )

    if not alert.item.subject_confident:
        return result(
            "monitor" if severity >= 4 else "fyi",
            "uncertain_subject",
            has_action=False,
            relevant=False,
        )

    if alert.tier == "preseason":
        return result(
            "monitor" if severity >= 3 else "fyi",
            "draft_monitor" if severity >= 3 else "informational",
            has_action=False,
            relevant=False,
        )

    if unconfirmed_suspension:
        return result(
            "monitor" if (roster_relevant or severity >= 4) else "fyi",
            "await_final_status" if roster_relevant else "informational",
            has_action=False,
        )

    if not availability_verified:
        return result(
            "monitor" if (roster_relevant or severity >= 4) else "fyi",
            "availability_unverified",
            has_action=False,
            verified=False,
        )

    if event in REMOVAL_EVENTS:
        definitive_absence = bool(
            event in {"inactive", "release", "suspension"}
            or status in {"season_out", "injured_reserve", "inactive"}
        )
        if starter:
            if definitive_absence and mine_starter_actions:
                return result("act_now", "starter_unavailable")
            if definitive_absence:
                return result(
                    "act_today",
                    "starter_unavailable",
                    has_action=False,
                )
            if severity >= 3:
                return result(
                    "act_today",
                    "starter_major_risk" if severity >= 4 else "roster_contingency",
                    has_action=mine_starter_actions,
                )
            return result(
                "monitor",
                "await_final_status",
                has_action=mine_starter_actions,
            )

        # A player can be on this manager's bench in one league while an
        # actionable successor is free in another. Only non-mine league plays
        # participate in this branch; a successor to a bench stash never turns
        # that stash into an immediate action.
        if claimable and event == "release" and severity >= 4:
            return result(
                "act_today",
                "claimable_watch",
                relevant=roster_relevant,
            )
        urgent_claimable = any(
            plays.subject_depth_order == 1 for plays in external_claimable
        )
        uncertain_participation = status in {"questionable", "dnp", "limited"}
        if (
            claimable
            and severity >= 4
            and urgent_claimable
            and not uncertain_participation
        ):
            return result(
                "act_now",
                "claimable_replacement",
                relevant=roster_relevant,
            )
        if claimable and severity >= 3:
            return result(
                "act_today",
                "claimable_watch",
                relevant=roster_relevant,
            )
        if roster_relevant or severity >= 5:
            return result(
                "monitor",
                "await_final_status" if event == "injury" else "major_league_watch",
                has_action=False,
            )
        return result("fyi", "informational", has_action=False, relevant=False)

    if event == "practice_report":
        if starter and severity >= 3:
            return result(
                "act_today",
                "roster_contingency",
                has_action=mine_starter_actions,
            )
        return result(
            "monitor" if (roster_relevant or severity >= 2) else "fyi",
            "await_final_status" if (roster_relevant or severity >= 2) else "informational",
            has_action=mine_starter_actions,
        )

    if event == "return":
        return result(
            "monitor" if (roster_relevant or severity >= 2) else "fyi",
            "return_monitor" if (roster_relevant or severity >= 2) else "informational",
            has_action=False,
        )

    if event in ROLE_EVENTS:
        if roster_relevant and severity >= 3:
            return result(
                "act_today",
                "roster_role_change",
            )
        return result(
            "monitor" if severity >= 4 else "fyi",
            "major_league_watch" if severity >= 4 else "informational",
            has_action=False,
        )

    if roster_relevant and severity >= 4:
        return result(
            "act_today",
            "roster_contingency",
        )
    return result(
        "monitor" if severity >= 5 else "fyi",
        "major_league_watch" if severity >= 5 else "informational",
        has_action=False,
    )


def archive_rule_urgency(row: dict[str, Any]) -> ActionUrgency | None:
    """Conservatively label old rows whose live lineup/action facts are gone."""
    try:
        severity = int(row.get("severity") or 0)
    except (TypeError, ValueError):
        return None
    if severity < 1 or not row.get("event_type"):
        return None
    tier = str(row.get("tier") or "league")
    roster_relevant = tier == "mine"
    level = "monitor" if (severity >= 3 and (roster_relevant or severity >= 4)) else "fyi"
    item = news_item_from_row(row)
    classification = Classification(
        event_type=str(row.get("event_type") or "other"),
        severity=severity,
        fantasy_impact=str(row.get("summary") or ""),
        is_actionable=bool(row.get("is_actionable")),
        raw={"direction": str(row.get("direction") or "unknown")},
    )
    historical = Alert(item=item, classification=classification, tier=tier)
    return ActionUrgency(
        rule_level=level,
        level=level,
        reason_codes=("historical_context_incomplete",),
        basis="archive_replay",
        policy_version=POLICY_VERSION,
        action_available=False,
        roster_relevant=roster_relevant,
        availability_verified=False,
        canonical_event_type=urgency_event_type(historical),
        direction=urgency_direction(historical),
        event_status=urgency_event_status(historical),
        action_context=(
            "preseason"
            if tier == "preseason"
            else "mine_unknown"
            if tier == "mine"
            else tier
        ),
        subject_is_starter=False,
    )


def urgency_from_row(row: dict[str, Any]) -> ActionUrgency | None:
    """Restore a persisted assessment for evaluation and audit tooling."""
    rule_level = str(row.get("urgency_rule_level") or "").strip().lower()
    level = str(row.get("urgency_level") or "").strip().lower()
    if rule_level not in LEVEL_RANK or level not in LEVEL_RANK:
        return None

    def string_tuple(value: Any) -> tuple[str, ...]:
        if not value:
            return ()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return ()
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(entry) for entry in value)

    try:
        embedding_delta = int(row.get("urgency_embedding_delta") or 0)
        support_count = int(row.get("urgency_embedding_support_count") or 0)
        score = (
            float(row["urgency_embedding_score"])
            if row.get("urgency_embedding_score") is not None
            else None
        )
    except (TypeError, ValueError):
        return None
    return ActionUrgency(
        rule_level=rule_level,
        level=level,
        reason_codes=string_tuple(row.get("urgency_reason_codes")),
        basis=str(row.get("urgency_basis") or "rules"),
        embedding_delta=embedding_delta,
        embedding_score=score,
        embedding_support_count=support_count,
        embedding_report_ids=string_tuple(row.get("urgency_embedding_report_ids")),
        policy_version=str(row.get("urgency_policy_version") or ""),
        action_available=bool(row.get("urgency_action_available")),
        roster_relevant=bool(row.get("urgency_roster_relevant")),
        availability_verified=bool(row.get("urgency_availability_verified")),
        canonical_event_type=str(row.get("urgency_event_type") or ""),
        direction=str(row.get("urgency_direction") or "unknown"),
        event_status=str(row.get("urgency_event_status") or ""),
        action_context=str(row.get("urgency_action_context") or ""),
        subject_is_starter=bool(row.get("urgency_subject_is_starter")),
    )


def _proxy_level(row: dict[str, Any]) -> str:
    level = str(row.get("urgency_rule_level") or "").strip().lower()
    if level in LEVEL_RANK:
        return level
    try:
        severity = int(row.get("severity") or 0)
    except (TypeError, ValueError):
        severity = 0
    return "monitor" if severity >= 3 else "fyi"


def apply_embedding_support(
    alert: Alert,
    base: ActionUrgency,
    current: EmbeddingVector,
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    min_neighbors: int,
    allow_lift: bool = False,
) -> ActionUrgency:
    """Corroborate or raise one safe band from independent historical rows."""
    if (
        not alert.item.subject_confident
        or not base.availability_verified
        or not math.isfinite(threshold)
        or threshold < 0.5
        or threshold > 1.0
    ):
        return base
    current_event = urgency_event_type(alert)
    current_direction = urgency_direction(alert)
    if not current_event or current_direction == "unknown":
        return base

    current_player = compact_key(alert.item.player_name)
    best_by_player: dict[str, tuple[float, dict[str, Any]]] = {}
    for row in rows:
        try:
            candidate_player = compact_key(str(row.get("player_name") or ""))
            if not candidate_player or candidate_player == current_player:
                continue
            if not bool(row.get("subject_confident", True)):
                continue
            if row.get("feedback") in {"wrong", "noisy"}:
                continue
            if str(row.get("urgency_event_type") or "") != current_event:
                continue
            if (
                str(row.get("urgency_direction") or "unknown").strip().lower()
                != current_direction
            ):
                continue
            if str(row.get("tier") or "league") != alert.tier:
                continue
            if str(row.get("urgency_action_context") or "") != base.action_context:
                continue
            candidate_status = str(row.get("urgency_event_status") or "")
            if base.event_status and candidate_status != base.event_status:
                continue
            if (
                row.get("embedding_model") != current.model
                or row.get("embedding_provider") != current.provider
                or int(row.get("embedding_dimensions") or 0) != current.dimensions
                or row.get("embedding_input_version") != current.input_version
                or not row.get("embedding_at")
            ):
                continue
            previous_item = news_item_from_row(row)
            if row.get("embedding_input_hash") != embedding_input_hash(
                canonical_embedding_text(previous_item)
            ):
                continue
            previous_vector = unpack_vector(
                bytes(row.get("embedding") or b""),
                dimensions=current.dimensions,
            )
            score = cosine_similarity(current.values, previous_vector)
            if not math.isfinite(score) or score < threshold:
                continue
        except Exception:  # noqa: BLE001 - one corrupt historical row abstains
            continue
        prior = best_by_player.get(candidate_player)
        if prior is None or score > prior[0]:
            best_by_player[candidate_player] = (score, row)

    candidates = sorted(best_by_player.values(), key=lambda value: value[0], reverse=True)
    minimum = max(2, int(min_neighbors))
    if len(candidates) < minimum:
        return base

    same_level = [value for value in candidates if _proxy_level(value[1]) == base.rule_level]
    support_ratio = len(same_level) / len(candidates)
    support_pool = same_level if len(same_level) >= minimum and support_ratio >= 0.75 else []

    # Only rows assessed from live roster facts under this exact policy may
    # vote for a lift. Archive replays deliberately cannot invent historical
    # claimability or starting slots.
    lift_pool = [
        value
        for value in candidates
        if str(value[1].get("urgency_policy_version") or "") == POLICY_VERSION
        and str(value[1].get("urgency_basis") or "") != "archive_replay"
        and bool(value[1].get("urgency_availability_verified"))
        and urgency_rank(str(value[1].get("urgency_rule_level") or ""))
        >= urgency_rank("act_today")
    ]
    live_rows = [
        value
        for value in candidates
        if str(value[1].get("urgency_policy_version") or "") == POLICY_VERSION
        and str(value[1].get("urgency_basis") or "") != "archive_replay"
    ]
    lift_minimum = max(3, minimum)
    can_lift = bool(
        allow_lift
        and base.rule_level == "monitor"
        and base.action_available
        and alert.tier in {"mine", "claimable"}
        and len(lift_pool) >= lift_minimum
        and live_rows
        and len(lift_pool) / len(live_rows) >= 0.75
    )

    selected = lift_pool if can_lift else support_pool
    if not selected:
        return base
    selected = selected[:5]
    max_score = max(score for score, _row in selected)
    report_ids = tuple(str(row.get("report_id") or "") for _score, row in selected)
    if can_lift:
        return replace(
            base,
            level="act_today",
            reason_codes=("embedding_history_lift", *base.reason_codes),
            basis="rules+embedding_lift",
            embedding_delta=1,
            embedding_score=max_score,
            embedding_support_count=len(selected),
            embedding_report_ids=report_ids,
        )
    return replace(
        base,
        reason_codes=(*base.reason_codes, "embedding_history_support"),
        basis="rules+embedding_support",
        embedding_score=max_score,
        embedding_support_count=len(selected),
        embedding_report_ids=report_ids,
    )


class UrgencyService:
    """Thread-safe coordinator reusing the existing embedding vector/cache."""

    def __init__(
        self,
        store: EventStore,
        embeddings: EmbeddingService,
        *,
        threshold: float = 0.70,
        min_neighbors: int = 2,
        history_days: int = 365,
        allow_lift: bool = False,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.threshold = float(threshold)
        self.min_neighbors = max(2, int(min_neighbors))
        self.history_days = max(7, int(history_days))
        self.allow_lift = bool(allow_lift)
        self._lock = threading.Lock()
        self._assessed = 0
        self._corroborated = 0
        self._lifted = 0
        self._abstained = 0
        self._failures = 0

    @classmethod
    def from_config(
        cls,
        store: EventStore,
        embeddings: EmbeddingService,
        config: Any,
    ) -> "UrgencyService":
        return cls(
            store,
            embeddings,
            threshold=float(
                getattr(config, "urgency_embedding_threshold", 0.70)
            ),
            min_neighbors=int(
                getattr(config, "urgency_embedding_min_neighbors", 2)
            ),
            history_days=int(
                getattr(config, "urgency_embedding_history_days", 365)
            ),
            allow_lift=bool(
                getattr(config, "urgency_embedding_lift_enabled", False)
            ),
        )

    def assess(self, alert: Alert) -> Alert:
        base = assess_rule_urgency(alert)
        assessment = base
        failed = False
        if self.embeddings.enabled and alert.item.subject_confident:
            try:
                current = self.embeddings.current_vector(alert.item)
                if current is not None:
                    rows = self.store.recent_urgency_candidates(
                        alert.item,
                        event_type=urgency_event_type(alert),
                        direction=urgency_direction(alert),
                        event_status=base.event_status,
                        action_context=base.action_context,
                        tier=alert.tier,
                        model=current.model,
                        provider=current.provider,
                        dimensions=current.dimensions,
                        input_version=INPUT_VERSION,
                        since_days=self.history_days,
                        limit=MAX_HISTORY_CANDIDATES,
                    )
                    assessment = apply_embedding_support(
                        alert,
                        base,
                        current,
                        rows,
                        threshold=self.threshold,
                        min_neighbors=self.min_neighbors,
                        allow_lift=self.allow_lift,
                    )
            except Exception as error:  # noqa: BLE001 - urgency always falls back
                failed = True
                structured_log(
                    logging.WARNING,
                    "urgency.embedding_failed",
                    errorType=type(error).__name__,
                )

        with self._lock:
            self._assessed += 1
            if failed:
                self._failures += 1
            if assessment.embedding_delta > 0:
                self._lifted += 1
            elif assessment.embedding_support_count > 0:
                self._corroborated += 1
            else:
                self._abstained += 1
        if assessment.embedding_support_count > 0:
            structured_log(
                logging.INFO,
                "urgency.embedding_assessed",
                player=alert.item.player_name,
                ruleLevel=base.rule_level,
                finalLevel=assessment.level,
                supportCount=assessment.embedding_support_count,
                maxScore=round(float(assessment.embedding_score or 0.0), 4),
            )
        return replace(alert, urgency=assessment)

    def status(self) -> UrgencyServiceStatus:
        with self._lock:
            return UrgencyServiceStatus(
                assessed=self._assessed,
                corroborated=self._corroborated,
                lifted=self._lifted,
                abstained=self._abstained,
                failures=self._failures,
            )
