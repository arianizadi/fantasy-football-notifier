"""Suppress repeat alerts.

Two layers:
  1. GUID - the same feed item seen on a later poll.
  2. Content fingerprint - the same event reported again with different
     wording within a time window (a practice report followed by a beat
     writer's confirmation of the same thing).

State persists to disk so a restart mid-Sunday does not replay old news.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from .logging_utils import structured_log
from .models import NewsItem, report_revision_identity

DEFAULT_WINDOW_SECONDS = 30 * 60
# A tweet and the RotoWire write-up of the same event are worded completely
# differently, so text similarity cannot catch the duplicate. After
# classification we know (player, event_type), which is stable across sources.
SEMANTIC_WINDOW_SECONDS = 90 * 60
# Concrete medical facts and explicit role decisions are commonly rehashed all
# day. Keep those identities longer than generic usage chatter; real severity,
# status, condition, and timetable changes still pass the checks below.
STABLE_FACT_WINDOW_SECONDS = 24 * 60 * 60
MAX_TRACKED_ENTRIES = 5000
STOPWORDS = frozenset({"the", "a", "an", "is", "was", "for", "of", "to", "in", "on", "at", "with"})

# Ordered from weak/uncertain participation notes to definitive absences.
# Within the 90-minute semantic window a strictly worse status is a meaningful
# update and should alert even when the model chooses the same event type.
STATUS_PATTERNS = (
    (
        "season_out",
        100,
        re.compile(
            r"\b(season[-\s]ending|out\s+for\s+the\s+season|"
            r"(?:torn|tore)\s+(acl|achilles))\b",
            re.I,
        ),
    ),
    (
        "injured_reserve",
        90,
        re.compile(r"\b(injured\s+reserve|reserve/injured|placed\s+on\s+ir)\b", re.I),
    ),
    (
        "inactive",
        80,
        re.compile(r"\b(inactive|ruled\s+out|will\s+not\s+play)\b", re.I),
    ),
    ("doubtful", 60, re.compile(r"\bdoubtful\b", re.I)),
    ("questionable", 50, re.compile(r"\bquestionable\b", re.I)),
    ("dnp", 40, re.compile(r"\b(dnp|did\s+not\s+practice)\b", re.I)),
    ("limited", 30, re.compile(r"\b(limited|limited\s+participant)\b", re.I)),
    (
        "cleared",
        20,
        re.compile(
            r"\b(activated|cleared|full\s+participant|returned\s+to\s+practice)\b",
            re.I,
        ),
    ),
)

# Condition details are materially different facts even when a classifier gives
# them the same broad event, severity, and availability status.  Keep this list
# deliberately mechanical: it is used only to prove that two reports describe
# the same fact, never to diagnose an injury.
_CONDITION_PATTERNS = (
    ("concussion", re.compile(r"\bconcussion(?: protocol)?\b", re.I)),
    ("head", re.compile(r"\bhead (?:injury|issue|trauma)\b", re.I)),
    ("illness", re.compile(r"\b(illness|ill|sick|flu|virus|covid(?:-19)?)\b", re.I)),
    ("personal", re.compile(r"\b(personal matter|personal reasons?|bereavement)\b", re.I)),
    ("acl", re.compile(r"\b(?:acl|anterior cruciate ligament)\b", re.I)),
    ("mcl", re.compile(r"\b(?:mcl|medial collateral ligament)\b", re.I)),
    ("meniscus", re.compile(r"\bmenisc(?:us|al)\b", re.I)),
    ("achilles", re.compile(r"\bachilles\b", re.I)),
    ("hamstring", re.compile(r"\bhamstrings?\b", re.I)),
    ("quadriceps", re.compile(r"\b(?:quad|quadriceps)\b", re.I)),
    ("groin", re.compile(r"\bgroin\b", re.I)),
    ("calf", re.compile(r"\bcalf\b", re.I)),
    ("ankle", re.compile(r"\bankle\b", re.I)),
    ("knee", re.compile(r"\bknee\b", re.I)),
    ("foot", re.compile(r"\bfoot\b", re.I)),
    ("toe", re.compile(r"\btoe\b", re.I)),
    ("hip", re.compile(r"\bhip\b", re.I)),
    (
        "back",
        re.compile(
            r"\b(?:lower |upper )?back (?:injury|issue|pain|tightness|soreness|spasms?)\b",
            re.I,
        ),
    ),
    ("neck", re.compile(r"\bneck\b", re.I)),
    ("shoulder", re.compile(r"\bshoulder\b", re.I)),
    ("pectoral", re.compile(r"\b(?:pec|pectoral)\b", re.I)),
    ("biceps", re.compile(r"\bbiceps?\b", re.I)),
    ("triceps", re.compile(r"\btriceps?\b", re.I)),
    ("elbow", re.compile(r"\belbow\b", re.I)),
    ("forearm", re.compile(r"\bforearm\b", re.I)),
    ("wrist", re.compile(r"\bwrist\b", re.I)),
    ("hand", re.compile(r"\bhand\b", re.I)),
    ("finger", re.compile(r"\bfinger\b", re.I)),
    ("thumb", re.compile(r"\bthumb\b", re.I)),
    ("ribs", re.compile(r"\bribs?\b", re.I)),
    ("chest", re.compile(r"\bchest\b", re.I)),
    ("abdomen", re.compile(r"\b(?:abdomen|abdominal)\b", re.I)),
    ("oblique", re.compile(r"\boblique\b", re.I)),
    ("hernia", re.compile(r"\bhernia\b", re.I)),
    ("kidney", re.compile(r"\bkidney\b", re.I)),
    ("fracture", re.compile(r"\b(fractur(?:e|ed)|broken)\b", re.I)),
    ("dislocation", re.compile(r"\bdislocat(?:e|ed|ion)\b", re.I)),
    ("sprain", re.compile(r"\bsprain(?:ed)?\b", re.I)),
    ("strain", re.compile(r"\bstrain(?:ed)?\b", re.I)),
    ("tear", re.compile(r"\b(torn|tore|tear|ruptur(?:e|ed))\b", re.I)),
    ("surgery", re.compile(r"\b(surgery|surgical|operation|procedure)\b", re.I)),
    ("day_to_day", re.compile(r"\bday[- ]to[- ]day\b", re.I)),
    ("week_to_week", re.compile(r"\bweek[- ]to[- ]week\b", re.I)),
    ("multiple_weeks", re.compile(r"\b(?:multiple|several) weeks\b", re.I)),
    ("indefinite", re.compile(r"\b(?:indefinitely|no timetable)\b", re.I)),
    (
        "expected_absence",
        re.compile(r"\b(?:expected|set|likely) to miss\b|\bwill miss\b", re.I),
    ),
    ("imaging", re.compile(r"\b(?:mri|x-ray|x ray|imaging|scan)\b", re.I)),
)

_DYNAMIC_FACT_PATTERNS = (
    (
        "duration",
        re.compile(r"\b(\d{1,2}(?:\s*(?:-|to)\s*\d{1,2})?)\s*(days?|weeks?|months?)\b", re.I),
    ),
    ("grade", re.compile(r"\bgrade\s+([123])\b", re.I)),
    ("season_week", re.compile(r"\bweek\s+(\d{1,2})\b", re.I)),
    (
        "side",
        re.compile(
            r"\b(left|right)\s+(?:achilles|ankle|knee|foot|toe|hip|hamstring|"
            r"quadriceps|quad|groin|calf|shoulder|pectoral|pec|biceps|triceps|"
            r"elbow|forearm|wrist|hand|finger|thumb|rib)\b",
            re.I,
        ),
    ),
)

# These statuses are concrete enough that two otherwise detail-free reports can
# safely be treated as corroboration.  A generic same-severity "injury" cannot:
# the second report may be a new condition that used different wording.
_DETAIL_FREE_CORROBORATION_STATUSES = frozenset(
    {
        "season_out",
        "injured_reserve",
        "inactive",
        "doubtful",
        "questionable",
        "dnp",
        "limited",
        "cleared",
    }
)

# Models reasonably disagree over whether an explicit starter announcement is
# ``depth_chart`` or a generic ``other`` update. Treat the deterministic role
# decision as the durable identity instead of letting that presentation label
# create several alerts. ``usage`` remains separate because a snap/rotation
# restriction can be actionable even when its report repeats who will start.
_ROLE_EVENT_TYPES = frozenset({"depth_chart", "other"})
_MEDICAL_EVENT_TYPES = frozenset({"injury", "inactive", "practice_report"})
_MEDICAL_EVENT_PATTERN = re.compile(
    r"\b(?:injur(?:y|ed)|hurt|concussion|illness|sick|"
    r"sprain(?:ed)?|strain(?:ed)?|fractur(?:e|ed)|broken|"
    r"tear|tore|torn|ruptur(?:e|ed)|surgery|"
    r"ankle|knee|hamstring|quadriceps|quad|groin|calf|foot|toe|hip|"
    r"back|neck|shoulder|pectoral|pec|biceps|triceps|elbow|forearm|"
    r"wrist|hand|finger|thumb|ribs?|chest|abdomen|abdominal|oblique|"
    r"hernia|achilles|acl|mcl|meniscus)\b",
    re.I,
)

# Trade reports arrive as a burst: a breaking-news sentence, full terms, and
# several source confirmations can all describe the same completed move.  A
# text fingerprint cannot join those wordings, while treating every generic
# ``trade`` label as equivalent would hide corrections and unrelated rumors.
# The destination is the smallest stable identity shared by those reports.
_NFL_TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "ARI": ("Arizona Cardinals", "Cardinals"),
    "ATL": ("Atlanta Falcons", "Falcons"),
    "BAL": ("Baltimore Ravens", "Ravens"),
    "BUF": ("Buffalo Bills", "Bills"),
    "CAR": ("Carolina Panthers", "Panthers"),
    "CHI": ("Chicago Bears", "Bears"),
    "CIN": ("Cincinnati Bengals", "Bengals"),
    "CLE": ("Cleveland Browns", "Browns"),
    "DAL": ("Dallas Cowboys", "Cowboys"),
    "DEN": ("Denver Broncos", "Broncos"),
    "DET": ("Detroit Lions", "Lions"),
    "GB": ("Green Bay Packers", "Green Bay", "Packers"),
    "HOU": ("Houston Texans", "Houston", "Texans"),
    "IND": ("Indianapolis Colts", "Indianapolis", "Colts"),
    "JAX": ("Jacksonville Jaguars", "Jacksonville", "Jaguars", "Jags"),
    "KC": ("Kansas City Chiefs", "Kansas City", "Chiefs"),
    "LV": ("Las Vegas Raiders", "Las Vegas", "Raiders"),
    "LAC": ("Los Angeles Chargers", "LA Chargers", "Chargers"),
    "LAR": ("Los Angeles Rams", "LA Rams", "Rams"),
    "MIA": ("Miami Dolphins", "Dolphins"),
    "MIN": ("Minnesota Vikings", "Vikings"),
    "NE": ("New England Patriots", "New England", "Patriots", "Pats"),
    "NO": ("New Orleans Saints", "New Orleans", "Saints"),
    "NYG": ("New York Giants", "NY Giants", "Giants"),
    "NYJ": ("New York Jets", "NY Jets", "Jets"),
    "PHI": ("Philadelphia Eagles", "Philadelphia", "Eagles"),
    "PIT": ("Pittsburgh Steelers", "Pittsburgh", "Steelers"),
    "SEA": ("Seattle Seahawks", "Seahawks"),
    "SF": ("San Francisco 49ers", "San Francisco", "49ers", "Niners"),
    "TB": ("Tampa Bay Buccaneers", "Tampa Bay", "Buccaneers", "Bucs"),
    "TEN": ("Tennessee Titans", "Tennessee", "Titans"),
    "WSH": (
        "Washington Commanders",
        "Washington",
        "Commanders",
    ),
}


def _trade_alias_pattern(alias: str) -> str:
    return re.escape(alias).replace(r"\ ", r"\s+")


_TRADE_TEAM_PATTERN = re.compile(
    r"\b(?P<team>"
    + "|".join(
        sorted(
            (
                _trade_alias_pattern(alias)
                for aliases in _NFL_TEAM_ALIASES.values()
                for alias in aliases
            ),
            key=len,
            reverse=True,
        )
    )
    + r")\b",
    re.I,
)
_TRADE_TEAM_LOOKUP = {
    re.sub(r"\s+", " ", alias).casefold(): team
    for team, aliases in _NFL_TEAM_ALIASES.items()
    for alias in aliases
}
_TEAM_ABBREVIATION_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:ARI|ATL|BAL|BUF|CAR|CHI|CIN|CLE|DAL|DEN|DET|"
    r"GB|HOU|IND|JAX|JAC|KC|LV|LAC|LAR|MIA|MIN|NE|NO|NYG|NYJ|PHI|"
    r"PIT|SEA|SF|TB|TEN|WAS|WSH)(?![A-Za-z])"
)
_TRADE_CANCELLATION_PATTERN = re.compile(
    r"\b(?:"
    r"(?:trade|deal)[^.\n]{0,100}?"
    r"(?:is\s+|was\s+|has\s+been\s+)?"
    r"(?:off|dead|cancel(?:l)?ed|voided|rescinded|vetoed|nixed)|"
    r"(?:cancel(?:l)?ed|voided|rescinded|vetoed|nixed)\s+(?:the\s+)?"
    r"(?:trade|deal)|"
    r"trade\s+(?:fell|falls|has\s+fallen)\s+through|"
    r"deal\s+(?:fell|falls|has\s+fallen)\s+through|"
    r"failed\s+(?:his\s+|the\s+)?physical[^.\n]{0,80}"
    r"(?:trade|deal)|"
    r"(?:trade|deal)[^.\n]{0,100}failed\s+(?:his\s+|the\s+)?physical)\b",
    re.I,
)
_TRADE_CANCELLATION_DENIAL_PATTERN = re.compile(
    r"\b(?:"
    r"(?:trade|deal)[^.\n]{0,60}\b(?:is|was|has\s+been)\s+not\s+"
    r"(?:off|dead|cancel(?:l)?ed|voided|rescinded|vetoed|nixed)|"
    r"(?:off|dead|cancel(?:l)?ed|voided|rescinded|vetoed|nixed)\s+"
    r"(?:(?:reports?|rumou?rs?|claims?)\s+)?"
    r"(?:is|are|was|were)\s+(?:false|incorrect|wrong|denied))\b",
    re.I,
)
_TRADE_CORRECTION_PATTERN = re.compile(
    r"\b(?:"
    r"correction|retraction|retracted|corrected\s+report|"
    r"(?:was|is|has)\s+not\s+(?:(?:been|being)\s+)?"
    r"(?:traded|dealt|sent)|"
    r"(?:wasn|isn|hasn)['’]t\s+(?:(?:been|being)\s+)?"
    r"(?:traded|dealt|sent)|"
    r"incorrectly\s+reported"
    r")\b",
    re.I,
)
_TRADE_RUMOR_PATTERN = re.compile(
    r"\b(?:"
    r"trade\s+(?:talks?|rumou?rs?|interest|candidate)|"
    r"(?:on|placed\s+on)\s+the\s+trade\s+block|"
    r"(?:could|may|might|would|expected|likely)\s+(?:to\s+)?"
    r"(?:be\s+)?traded|"
    r"(?:could|may|might|would)\s+trade\s+for|"
    r"(?:could|may|might|would)\s+(?:to\s+)?"
    r"(?:get|acquire|land)\b|"
    r"(?:exploring|discussing|considering|seeking|working\s+on)\s+"
    r"(?:a\s+)?trade|"
    r"interested\s+in\s+trading\s+for)\b",
    re.I,
)
_TRADE_COMPLETION_PATTERN = re.compile(
    r"\b(?:"
    r"traded|dealt|acquir(?:e|es|ed|ing)|receiv(?:e|es|ed|ing)|"
    r"get(?:s|ting)?|"
    r"sent\s+to|sends\s+[^.\n]{0,100}\s+to|"
    r"(?:is|are|was|were)\s+(?:being\s+)?(?:traded|dealt|sent)|"
    r"(?:is|are|was|were)\s+sending|"
    r"(?:heads?|headed|goes?)\s+(?:(?:from|out\s+of)\s+"
    r"[^,.\n]{1,45}\s+)?to|lands?\s+(?:with|in)|"
    r"trade[sd]?\s+for|deal\s+(?:agreed|complete|completed|done)|"
    r"full\s+(?:trade\s+)?terms|trade\s*[!:—-])(?=\s|$|[.,;:!?—-])",
    re.I,
)
_TRADE_SIGNATURE_PATTERN = re.compile(
    r"^trade:(?P<state>completed|cancelled|correction|rumor):to:"
    r"(?P<team>[A-Z?]+)$"
)


def _normalized_event_type(event_type: str) -> str:
    return (event_type or "other").strip().lower().replace("-", "_").replace(" ", "_")


def _report_text(item: NewsItem) -> str:
    """Join headline/body without repeating a tweet's truncated full text."""
    headline = (item.headline or "").strip()
    body = (item.body or "").strip()
    if not headline:
        return body
    if not body:
        return headline
    if body == headline or body.startswith(headline) or headline.startswith(body):
        return body if len(body) >= len(headline) else headline
    return f"{headline}. {body}"


def _canonical_trade_team(value: str) -> str:
    return _TRADE_TEAM_LOOKUP.get(re.sub(r"\s+", " ", value).casefold(), "")


def _team_mentions(fragment: str) -> list[tuple[int, int, str]]:
    mentions = [
        (match.start(), match.end(), _canonical_trade_team(match.group("team")))
        for match in _TRADE_TEAM_PATTERN.finditer(fragment)
    ]
    occupied = [(start, end) for start, end, _team in mentions]
    for match in _TEAM_ABBREVIATION_PATTERN.finditer(fragment):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        team = {"JAC": "JAX", "WAS": "WSH"}.get(match.group(0), match.group(0))
        mentions.append((match.start(), match.end(), team))
    return sorted(
        (entry for entry in mentions if entry[2]),
        key=lambda entry: entry[0],
    )


def _first_trade_team(fragment: str) -> str:
    match = _TRADE_TEAM_PATTERN.search(fragment)
    if match is not None:
        return _canonical_trade_team(match.group("team"))
    abbreviation = _TEAM_ABBREVIATION_PATTERN.search(fragment)
    if abbreviation is None:
        return ""
    value = abbreviation.group(0)
    return {"JAC": "JAX", "WAS": "WSH"}.get(value, value)


def _trade_subject_patterns(item: NewsItem) -> tuple[re.Pattern[str], ...]:
    parts = [part for part in item.player_name.split() if part]
    if not parts:
        return ()
    names = [item.player_name]
    # Dedicated player items and confidently attributed tweets frequently use
    # only the surname in their prose. Keep the full name first, and require a
    # non-trivial surname to avoid matching initials or position labels.
    if len(parts) > 1 and len(parts[-1]) >= 4:
        names.append(parts[-1])
    return tuple(
        re.compile(
            r"\b" + r"\s+".join(re.escape(part) for part in name.split()) + r"\b",
            re.I,
        )
        for name in names
    )


def trade_destination(item: NewsItem) -> str:
    """Return a destination team only when the report states direction.

    A transaction report often names two clubs, so selecting the first team
    mention would confuse the player's former club or the team receiving the
    return compensation for the destination. These narrow subject-relative
    patterns handle both ``Player to Texans`` and ``Texans acquire Player``.
    """
    if not item.subject_confident or not item.player_name:
        return ""
    text = _report_text(item)
    after_cue = re.compile(
        r"\b(?:"
        r"to|joins?|(?:heads?|headed|goes?|bound)\s+(?:to|for)|"
        r"lands?\s+(?:with|in)|"
        r"(?:is|are|was|were|has\s+been|have\s+been)?\s*"
        r"(?:traded|dealt|sent|shipped|moved|moving)\s+to"
        r")\b\s+(?:to\s+|for\s+|the\s+)?(?P<destination>[^.,;\n]{0,45})",
        re.I,
    )
    subject_after_cue = re.compile(
        r"^\s*(?:"
        r"to|"
        r"(?:(?:is|was|will\s+be)\s+)?joins?|"
        r"(?:(?:is|was|will\s+be)\s+)?(?:heads?|headed|goes?|bound)\s+"
        r"(?:(?:from|out\s+of)\s+[^,.;\n]{1,45}\s+)?(?:to|for)|"
        r"(?:(?:is|was|will\s+be)\s+)?lands?\s+(?:with|in)|"
        r"(?:(?:is|are|was|were|has\s+been|have\s+been|"
        r"is\s+being|was\s+being)\s+)?"
        r"(?:traded|dealt|sent|shipped|moved|moving)"
        r"(?:(?:\s+from)\s+[^,.;\n]{1,45})?\s+to"
        r")\b\s+(?:the\s+)?(?P<destination>[^,.;\n]{0,45})",
        re.I,
    )
    subjects: list[re.Match[str]] = []
    for pattern in _trade_subject_patterns(item):
        subjects = list(pattern.finditer(text))
        if subjects:
            break
    if not subjects:
        # RotoWire's player-specific article URL/title already establishes the
        # structured subject. Its terse headline can therefore omit the name
        # entirely (for example, ``Dealt to Texans``) while still proving the
        # destination. Do not grant this fallback to free-form social posts.
        if item.source == "rotowire":
            destination = after_cue.search(text)
            if destination is not None:
                return _first_trade_team(destination.group("destination"))
        return ""
    acquiring_cue = re.compile(
        r"^\s*(?:(?:have|has)\s+agreed\s+to\s+|agreed\s+to\s+|"
        r"(?:have|has|had|is|are|was|were|will|will\s+be|"
        r"could|may|might|would)\s+)?"
        r"(?:get(?:s|ting)?|acquir(?:e|es|ed|ing)|receiv(?:e|es|ed|ing)|"
        r"add(?:s|ed|ing)?|land(?:s|ed|ing)?|"
        r"trad(?:e|es|ed|ing)\s+for)\b",
        re.I,
    )

    for subject in subjects:
        # Work backwards through nearby club mentions; the closest club with
        # an acquisition verb is the one receiving the report subject.
        prefix_start = max(0, subject.start() - 140)
        prefix = text[prefix_start : subject.start()]
        team_mentions = list(_TRADE_TEAM_PATTERN.finditer(prefix))
        for team_match in reversed(team_mentions):
            between = prefix[team_match.end() :]
            if acquiring_cue.search(between):
                team = _canonical_trade_team(team_match.group("team"))
                if team:
                    return team

        # Only accept direction immediately attached to the subject. Searching
        # the whole remaining sentence can accidentally assign the club that
        # receives a different player in the return package.
        tail = text[subject.end() : subject.end() + 140]
        destination = subject_after_cue.search(tail)
        if destination is not None:
            team = _first_trade_team(destination.group("destination"))
            if team:
                return team
    return ""


def transaction_teams(item: NewsItem, event_type: str) -> tuple[str, ...]:
    """Return teams explicitly tied to the subject's roster transaction.

    This is intentionally narrower than returning every team mentioned in a
    report. Draft-pick provenance and return-package details can name unrelated
    clubs; only subject-relative release/signing/trade language is accepted.
    """
    normalized_event = event_type.strip().casefold().replace("-", "_")
    if (
        normalized_event not in {"trade", "release", "signing"}
        or not item.subject_confident
        or not item.player_name
    ):
        return ()

    text = _report_text(item)
    subjects: list[re.Match[str]] = []
    for pattern in _trade_subject_patterns(item):
        subjects = list(pattern.finditer(text))
        if subjects:
            break

    teams: list[str] = []

    def add(team: str) -> None:
        if team and team not in teams:
            teams.append(team)

    if normalized_event == "trade":
        add(trade_destination(item))

    prefix_aux = (
        r"^\s*(?:,\s*)?(?:who\s+)?"
        r"(?:(?:have|has|had|are|is|was|were|will|would|"
        r"now|officially|reportedly)\s+)*"
    )
    seller_cue = re.compile(
        prefix_aux
        + r"(?:sends?|sent|sending|ships?|shipped|shipping|"
        r"deals?|dealt|dealing|moves?|moved|moving|"
        r"trad(?:e|es|ed|ing))(?!\s+for)\b",
        re.I,
    )
    acquiring_cue = re.compile(
        prefix_aux
        + r"(?:gets?|getting|acquir(?:e|es|ed|ing)|receiv(?:e|es|ed|ing)|"
        r"adds?|added|adding|lands?|landed|landing|"
        r"trad(?:e|es|ed|ing)\s+for)\b",
        re.I,
    )
    release_cue = re.compile(
        prefix_aux
        + r"(?:releas(?:e|es|ed|ing)|waiv(?:e|es|ed|ing)|"
        r"cuts?|cutting)\b",
        re.I,
    )
    signing_cue = re.compile(
        prefix_aux
        + r"(?:sign(?:s|ed|ing)?|claim(?:s|ed|ing)?|"
        r"adds?|added|adding|acquir(?:e|es|ed|ing))\b",
        re.I,
    )
    trade_tail_open = re.compile(
        r"^\s*(?:(?:is|was|were|has\s+been|have\s+been|will\s+be|"
        r"reportedly)\s+)*(?:being\s+)?(?:"
        r"traded|sent|dealt|shipped|moved|"
        r"heads?|headed|goes?|went|bound|joins?|joined|lands?|landed|"
        r"to|from)\b",
        re.I,
    )
    direct_team_cue = re.compile(
        r"(?:\b(?:to|from|with|by)\s+(?:the\s+)?|"
        r"\b(?:joins?|joined)\s+(?:the\s+)?|"
        r"\b(?:lands?|landed)\s+(?:with|in)\s+(?:the\s+)?)$",
        re.I,
    )
    chained_team_cue = re.compile(
        r"^\s*(?:to|from)\s+(?:the\s+)?$",
        re.I,
    )
    compensation_cue = re.compile(
        r"\b(?:for|picks?|selection|compensation|draft\s+choice|"
        r"in\s+exchange)\b",
        re.I,
    )

    for subject in subjects:
        prefix_start = max(0, subject.start() - 160)
        prefix = text[prefix_start : subject.start()]
        # A transaction involving a draft pick in the previous sentence is
        # not evidence about this player merely because it is nearby.
        prefix = re.split(r"[.;\n]", prefix)[-1]
        prefix_mentions = _team_mentions(prefix)
        if prefix_mentions:
            _start, end, team = prefix_mentions[-1]
            between = prefix[end:]
            if normalized_event == "trade" and seller_cue.search(between):
                add(team)
            elif normalized_event == "trade" and acquiring_cue.search(between):
                add(team)
            elif normalized_event == "release" and release_cue.search(between):
                add(team)
            elif normalized_event == "signing" and signing_cue.search(between):
                add(team)

        tail = text[subject.end() : subject.end() + 160]
        tail_clause = re.split(r"[.;\n]", tail, maxsplit=1)[0]
        if normalized_event == "trade":
            # Stay inside the subject's immediate transaction clause. After
            # the first directly attached team, only a bare "to/from TEAM"
            # chain is accepted; compensation provenance such as "a pick
            # acquired from Saints" therefore cannot leak into player context.
            mentions = _team_mentions(tail_clause)
            if trade_tail_open.search(tail_clause) and mentions:
                start, end, team = mentions[0]
                before_first_team = tail_clause[:start]
                if (
                    direct_team_cue.search(before_first_team)
                    and not compensation_cue.search(before_first_team)
                ):
                    add(team)
                    previous_end = end
                    for start, end, team in mentions[1:]:
                        if not chained_team_cue.fullmatch(
                            tail_clause[previous_end:start]
                        ):
                            break
                        add(team)
                        previous_end = end
            continue

        for start, _end, team in _team_mentions(tail_clause):
            before_team = tail_clause[:start]
            if normalized_event == "release" and re.search(
                r"\b(?:released|waived|cut)\s+(?:by|from)\s+(?:the\s+)?$",
                before_team,
                re.I,
            ):
                add(team)
                break
            elif normalized_event == "signing" and re.search(
                r"\b(?:sign(?:s|ed|ing)?\s+with|claimed\s+by)\s+(?:the\s+)?$",
                before_team,
                re.I,
            ):
                add(team)
                break

    # Player-specific RotoWire headlines often omit the subject ("Released by
    # Raiders" / "Signs with Texans"). The URL has already established the
    # subject, so these narrow whole-report fallbacks remain deterministic.
    if item.source.casefold() == "rotowire" and not subjects:
        fallback_patterns = {
            "release": r"\b(?:released|waived|cut)\s+(?:by|from)\s+(?:the\s+)?",
            "signing": (
                r"\b(?:sign(?:s|ed|ing)?\s+with|claimed\s+by)\s+(?:the\s+)?"
            ),
        }
        cue = fallback_patterns.get(normalized_event)
        if cue:
            match = re.search(cue, text, re.I)
            if match is not None:
                add(_first_trade_team(text[match.end() : match.end() + 60]))

    return tuple(teams)


def _trade_is_cancelled(text: str) -> bool:
    if _TRADE_CANCELLATION_DENIAL_PATTERN.search(text):
        return False
    return _TRADE_CANCELLATION_PATTERN.search(text) is not None


def trade_event_status(item: NewsItem) -> str:
    """Deterministic phase for a trade-labelled report.

    Completed trades retain the legacy status string ``trade`` so a deployed
    ``trade / unspecified`` thread can be enriched instead of replayed once
    during this schema upgrade.
    """
    text = _report_text(item)
    if _trade_is_cancelled(text):
        return "trade_cancelled"
    if _TRADE_RUMOR_PATTERN.search(text):
        return "trade_rumor"
    return "trade"


def trade_fact_signature(item: NewsItem) -> str:
    """Stable transaction identity, or ``unspecified`` when not provable."""
    text = _report_text(item)
    destination = trade_destination(item) or "?"
    if _trade_is_cancelled(text):
        return f"trade:cancelled:to:{destination}"
    if _TRADE_CORRECTION_PATTERN.search(text):
        return f"trade:correction:to:{destination}"
    if _TRADE_RUMOR_PATTERN.search(text):
        return f"trade:rumor:to:{destination}"
    if _TRADE_COMPLETION_PATTERN.search(text):
        return f"trade:completed:to:{destination}"
    return "unspecified"


def role_decision_status(item: NewsItem) -> str:
    """Return a deterministic role outcome only when the report states one.

    This intentionally ignores ordinary snap-count or first-team-rep usage.
    Those can be separate actionable facts.  It is narrowly for announcements
    such as ``named QB1`` / ``will start`` and their reversals.
    """
    if not item.subject_confident:
        return ""
    text = _report_text(item)
    player = r"\s+".join(re.escape(part) for part in item.player_name.split())
    if not player:
        return ""
    if not re.search(rf"\b{player}\b", text, re.I):
        # A RotoWire item has already attributed its structured player from a
        # dedicated article title/URL. Its prose often switches to surname
        # only. X does not get this fallback because a surname can refer to a
        # different player in a multi-player post.
        surname = item.player_name.split()[-1]
        if item.source != "rotowire" or len(surname) < 4:
            return ""
        player = re.escape(surname)
        if not re.search(rf"\b{player}\b", text, re.I):
            return ""

    after_player = rf"\b{player}\b[^.\n]{{0,80}}"
    before_player = rf"[^.\n]{{0,100}}\b{player}\b"
    role_one = (
        r"(?:starter|starting\s+(?:quarterback|running\s+back|wide\s+receiver|"
        r"tight\s+end|qb|rb|wr|te)|(?:qb|rb|wr|te)1)"
    )
    role_backup = r"(?:backup|(?:qb|rb|wr|te)[2-9])"

    open_competition = (
        r"(?:(?:open|ongoing|unsettled)\s+(?:competition|battle)|"
        r"(?:competition|battle)[^.\n]{0,100}"
        r"(?:remains?|is\s+still|still)\s+open|no\s+decision|undecided)"
    )
    predecision_selection = (
        r"(?:expected|plans?|set|will)\s+to\s+name[^.\n]{0,80}"
        r"(?:between|from)"
    )
    conditional_start = (
        rf"(?:\b(?:if|whether)\b[^.\n]{{0,45}}\b{player}\b[^.\n]{{0,35}}"
        rf"\bstart(?:s|ing)?\b|\bunclear\b[^.\n]{{0,70}}\b{player}\b"
        rf"[^.\n]{{0,35}}\bstart(?:s|ing)?\b)"
    )
    if (
        re.search(rf"(?:{after_player}\b{open_competition}\b|"
                  rf"\b{open_competition}\b{before_player})", text, re.I)
        or re.search(rf"\b{predecision_selection}\b{before_player}", text, re.I)
        or re.search(conditional_start, text, re.I)
    ):
        return "role_uncertain"

    projected_negative_start = (
        r"(?:may\s+not\s+start|might\s+not\s+start|could\s+not\s+start|"
        r"should\s+not\s+start|"
        r"(?:is|was)\s+not\s+(?:expected|likely|projected)\s+to\s+start|"
        r"(?:isn|wasn)['’]t\s+(?:expected|likely|projected)\s+to\s+start|"
        r"(?:is|was)\s+unlikely\s+to\s+start|unlikely\s+to\s+start|"
        r"(?:expected|projected)\s+not\s+to\s+start|"
        r"not\s+(?:expected|likely|projected)\s+to\s+start|"
        r"no\s+longer\s+(?:(?:expected|likely|projected)\s+to\s+)?start)"
    )
    if re.search(rf"{after_player}\b{projected_negative_start}\b", text, re.I):
        return "role_expected_not_starter"

    definitive_negative_start = (
        r"(?:will\s+not\s+start|won['’]t\s+start|"
        r"(?:is\s+not|isn['’]t)\s+starting|"
        r"(?:does\s+not|doesn['’]t|will\s+not|won['’]t)\s+"
        r"get(?:\s+(?:the|week\s+\d{1,2}))?\s+start)"
    )
    negative_selection_after = (
        r"(?:not|(?:is|was|has\s+been|had\s+been)\s+not|"
        r"(?:isn|wasn|hasn|hadn)['’]t|will\s+not\s+be|won['’]t\s+be)\s+"
        r"(?:named|picked|selected)\b(?:\s+as)?[^.\n]{0,45}"
    )
    negative_selection_before = (
        r"(?:(?:did|do|does|have|has|will)\s+not|"
        r"(?:didn|don|doesn|haven|hasn|won)['’]t)\s+"
        r"(?:name|named|pick|picked|select|selected)\b"
    )
    if re.search(
        rf"(?:{after_player}\b(?:{definitive_negative_start}|"
        rf"{negative_selection_after}"
        rf"{role_one}|not\s+(?:the\s+)?{role_one}|"
        rf"(?:is\s+)?no\s+longer\s+(?:the\s+)?{role_one}|"
        rf"(?:(?:has\s+)?lost|loses?)\s+(?:(?:the|his|their)\s+)?"
        rf"(?:starting\s+(?:job|role)|starter\s+(?:job|role))|"
        rf"bench(?:ed|ing)?|demot(?:ed|ion))\b|"
        rf"\b{negative_selection_before}{before_player}[^.\n]{{0,45}}"
        rf"\b{role_one}\b|"
        rf"\b(?:bench(?:ed|ing)?|demot(?:ed|ion))\b{before_player}|"
        rf"\b{role_one}\b[^.\n]{{0,50}}\bover\b{before_player})",
        text,
        re.I,
    ):
        return "role_not_starter"

    expected_start = (
        r"(?:(?:is|was)\s+(?:expected|likely|projected)\s+to\s+start|"
        r"(?:expected|likely|projected)\s+to\s+start|"
        r"(?:could|may)\s+start)"
    )
    expected_before = (
        r"(?:expects?|expected|projects?|projected)[^.\n]{0,55}"
    )
    projected_selection = (
        r"(?:expected|likely|projected|plans?)\s+to\s+"
        r"(?:name|pick|select)"
    )
    if re.search(rf"{after_player}\b{expected_start}\b", text, re.I) or re.search(
        rf"\b{expected_before}{before_player}[^.\n]{{0,30}}\bto\s+start\b",
        text,
        re.I,
    ) or re.search(
        rf"\b{projected_selection}\b{before_player}[^.\n]{{0,50}}"
        rf"(?:\b{role_one}\b|\bto\s+start\b)",
        text,
        re.I,
    ):
        return "role_expected_starter"

    game_start_context = (
        r"(?:week\s+\d{1,2}|this\s+week|sunday|monday|thursday|tonight|"
        r"against\b|at\b|vs\.?\b)"
    )
    direct_start = (
        rf"(?:(?:will\s+start|starts|is\s+starting)"
        rf"(?=\s+{game_start_context}|\s*(?:$|[.,;:!?]))|"
        rf"starting\s+{game_start_context}|"
        r"gets?(?:\s+(?:the|week\s+\d{1,2}))?\s+start|"
        r"is\s+set\s+to\s+start|is\s+slated\s+to\s+start)"
    )
    direct_role = (
        rf"(?:(?:is|was|remains?|will\s+be)\s+(?:(?:the|their)\s+)?{role_one}|"
        rf"(?:(?:is|was|has\s+been|had\s+been)\s+)?"
        rf"(?:named|picked|selected)[^.\n]{{0,35}}{role_one})"
    )
    selected_before = (
        r"(?:name|named|names|naming|pick|picked|picks|select|selected|selects|"
        r"confirm|confirmed|confirms|confirming)"
    )
    explicit_decision = r"(?:decision|decided|chooses?|chose)\s+to\s+start"
    if (
        re.search(rf"{after_player}\b(?:{direct_start}|{direct_role})\b", text, re.I)
        or re.search(
            rf"\b{selected_before}\b{before_player}[^.\n]{{0,50}}\b{role_one}\b",
            text,
            re.I,
        )
        or re.search(
            rf"\b{selected_before}\b{before_player}[^.\n]{{0,25}}\bto\s+start\b",
            text,
            re.I,
        )
        or re.search(rf"\b{explicit_decision}\b{before_player}", text, re.I)
        or re.search(
            rf"\b{role_one}\b\s+(?:is|will\s+be|was)\s+{before_player}",
            text,
            re.I,
        )
    ):
        return "role_starter"

    if re.search(
        rf"{after_player}\b(?:(?:is|was|remains?|named)\b[^.\n]{{0,30}})?"
        rf"\b{role_backup}\b",
        text,
        re.I,
    ):
        return "role_not_starter"
    return ""


def _direct_subject_medical_report(item: NewsItem) -> bool:
    """Whether medical language is deterministically about this item subject.

    The attribution helper rejects beneficiary wording such as ``Washington
    got extra work after Jeanty injured his knee``.  This lets a model's
    generic ``other`` label meet a corroborating ``injury`` label without
    assigning Jeanty's condition to Washington.
    """
    if not item.subject_confident or not item.player_name:
        return False
    text = _report_text(item)
    if not _MEDICAL_EVENT_PATTERN.search(text):
        return False
    # Imported lazily to keep source ingestion independent from state loading.
    from .sources.twitter import attributed_absence_subject

    return attributed_absence_subject(text, [item.player_name]) == item.player_name


def semantic_event_type(
    item: NewsItem,
    event_type: str,
    event_hint: str = "",
) -> str:
    """Canonical event family used only for dedupe/update decisions."""
    normalized = _normalized_event_type(event_type)
    if normalized in _ROLE_EVENT_TYPES and role_decision_status(item):
        return "depth_chart"
    if normalized == "other" and _direct_subject_medical_report(item):
        return "injury"
    hint = _normalized_event_type(event_hint)
    if (
        normalized == "other"
        and not item.subject_confident
        and hint in _MEDICAL_EVENT_TYPES
        and event_fact_signature(item) != "unspecified"
    ):
        # The pipeline deliberately displays ambiguous multi-player reports as
        # ``other`` and withholds actions. Preserve the model's original
        # medical family only for dedupe, and only when deterministic concrete
        # medical facts exist. Beneficiary reports normally carry a ``usage``
        # hint and therefore cannot assign the starter's injury to the backup.
        return hint
    return normalized


def semantic_event_status(item: NewsItem, event_type: str) -> str:
    """Status that preserves role reversals while coalescing corroboration."""
    normalized = semantic_event_type(item, event_type)
    if normalized == "depth_chart":
        role_status = role_decision_status(item)
        if role_status:
            return role_status
    if normalized == "trade":
        return trade_event_status(item)
    return event_status(item, normalized)


def semantic_event_fact_signature(item: NewsItem, event_type: str) -> str:
    """Facts relevant to the canonical event family.

    A generic ``Week 1`` marker is meaningful for injury return/timetable
    news, but not for five reports repeating the same Week 1 starter choice.
    Role decisions therefore use their deterministic outcome as the complete
    semantic fact.  Other event families retain the conservative injury fact
    behavior unchanged.
    """
    normalized = semantic_event_type(item, event_type)
    if normalized == "depth_chart":
        role_status = role_decision_status(item)
        if role_status:
            return role_status.replace("role_", "role:", 1)
    if normalized == "trade":
        return trade_fact_signature(item)
    return event_fact_signature(item)


def _semantic_window_seconds(
    event_type: str,
    status: str,
    fact_signature: str,
) -> int:
    normalized = _normalized_event_type(event_type)
    if normalized == "depth_chart" and fact_signature.startswith("role:"):
        return STABLE_FACT_WINDOW_SECONDS
    if normalized == "trade" and fact_signature.startswith("trade:"):
        return STABLE_FACT_WINDOW_SECONDS
    if normalized in {"injury", "inactive", "practice_report"} and (
        fact_signature not in {"", "unspecified"}
        or status in _DETAIL_FREE_CORROBORATION_STATUSES
    ):
        return STABLE_FACT_WINDOW_SECONDS
    return SEMANTIC_WINDOW_SECONDS


def event_status(item: NewsItem, event_type: str) -> str:
    text = f"{item.headline} {item.body}"
    for label, _rank, pattern in STATUS_PATTERNS:
        if pattern.search(text):
            return label
    return event_type


def event_fact_signature(item: NewsItem) -> str:
    """Return deterministic condition markers, or ``unspecified``.

    This is intentionally not a similarity hash.  Two sources may phrase a
    true corroboration very differently (``ruled out`` versus ``will not
    play``), while one changed word such as ``ankle`` versus ``concussion`` is
    a new urgent fact that must not be edited away or semantically suppressed.
    """
    text = f"{item.headline} {item.body}"
    markers = [label for label, pattern in _CONDITION_PATTERNS if pattern.search(text)]
    for label, pattern in _DYNAMIC_FACT_PATTERNS:
        for match in pattern.finditer(text):
            value = "-".join(
                re.sub(r"\s+", "", group.casefold())
                for group in match.groups()
                if group
            )
            markers.append(f"{label}:{value}")
    return "|".join(sorted(set(markers))) if markers else "unspecified"


def _trade_fact_is_less_informative(previous: str, current: str) -> bool:
    """True when current repeats a known trade phase but omits destination."""
    previous_trade = _TRADE_SIGNATURE_PATTERN.fullmatch(previous)
    current_trade = _TRADE_SIGNATURE_PATTERN.fullmatch(current)
    return bool(
        previous_trade is not None
        and current_trade is not None
        and previous_trade.group("state") == current_trade.group("state")
        and previous_trade.group("team") != "?"
        and current_trade.group("team") == "?"
    )


def event_facts_equivalent(
    previous_signature: str,
    current_signature: str,
    *,
    status: str,
) -> bool:
    """Whether condition metadata proves two reports are corroboration."""
    if not previous_signature or not current_signature:
        return False

    previous_trade = _TRADE_SIGNATURE_PATTERN.fullmatch(previous_signature)
    current_trade = _TRADE_SIGNATURE_PATTERN.fullmatch(current_signature)
    if previous_trade is not None or current_trade is not None:
        # A legacy deployed trade thread has status=trade with no fact
        # signature. Let the first destination-aware completed report upgrade
        # that exact state in place. Do not extend this exception to generic
        # unspecified events or to cancellation/rumor transitions.
        if (
            status == "trade"
            and previous_signature == "unspecified"
            and current_trade is not None
            and current_trade.group("state") == "completed"
            and current_trade.group("team") != "?"
        ):
            return True
        if previous_trade is None or current_trade is None:
            return False
        previous_state = previous_trade.group("state")
        current_state = current_trade.group("state")
        previous_team = previous_trade.group("team")
        current_team = current_trade.group("team")
        # Matching known destinations prove corroboration. A first breaking
        # report may establish the completed move before naming the receiving
        # club; a later known destination is a safe one-way refinement. The
        # reverse is deliberately false so terse follow-ups cannot overwrite
        # richer state.
        return (
            previous_state == current_state
            and current_team != "?"
            and (previous_team == "?" or previous_team == current_team)
        )
    if previous_signature != current_signature:
        return False
    if current_signature != "unspecified":
        return True
    return status in _DETAIL_FREE_CORROBORATION_STATUSES


def _fact_signature_is_meaningfully_new(previous: str, current: str) -> bool:
    """Detect changed facts only when legacy state recorded a prior fact.

    A missing signature means the seen file predates fact tracking; it does
    not prove that the currently parsed condition is new. Treating it as new
    replays every unchanged injury report once during an upgrade. Modern
    revision-aware entries bypass this fallback, while legacy entries can
    still pass through on a recorded status escalation.
    """
    if not previous:
        return False
    return previous != current


def _status_rank(status: str) -> int:
    for label, rank, _pattern in STATUS_PATTERNS:
        if status == label:
            return rank
    return 0


def _status_is_meaningfully_new(previous: str, current: str) -> bool:
    """Allow definitive worsening and a later clearance through early dedupe."""
    # An absent previous value is legacy schema uncertainty, not evidence of
    # a status transition. This prevents a one-time replay of exact reports
    # from state files that predate status tracking entirely.
    if not previous or not current or current == previous:
        return False
    if current == "cleared" and _status_rank(previous) >= _status_rank("limited"):
        return True
    return _status_rank(current) > _status_rank(previous)


def fingerprint(item: NewsItem) -> str:
    """Hash the meaningful words of a headline so rewordings collide."""
    text = re.sub(r"[^a-z0-9 ]", " ", item.fingerprint_text().lower())
    tokens = sorted({token for token in text.split() if token and token not in STOPWORDS})
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:32]


class SeenStore:
    def __init__(self, path: Path, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> None:
        self._path = path
        self._window = window_seconds
        self._guids: dict[str, float] = {}
        self._fingerprints: dict[str, float] = {}
        self._guid_statuses: dict[str, str] = {}
        self._fingerprint_statuses: dict[str, str] = {}
        self._guid_fact_signatures: dict[str, str] = {}
        self._fingerprint_fact_signatures: dict[str, str] = {}
        # A source GUID identifies an upstream object, but several feeds edit
        # that object's headline/body in place. Track exact raw revisions so
        # only a literal replay is stopped here; changed text must reach the
        # classifier and the richer semantic coalescing policy.
        self._report_revisions: dict[str, float] = {}
        self._revision_aware_guids: dict[str, float] = {}
        self._revision_aware_fingerprints: dict[str, float] = {}
        self._semantic: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text())
            self._guids = {str(k): float(v) for k, v in payload.get("guids", {}).items()}
            self._fingerprints = {
                str(k): float(v) for k, v in payload.get("fingerprints", {}).items()
            }
            self._guid_statuses = {
                str(key): str(value)
                for key, value in payload.get("guidStatuses", {}).items()
            }
            self._fingerprint_statuses = {
                str(key): str(value)
                for key, value in payload.get("fingerprintStatuses", {}).items()
            }
            self._guid_fact_signatures = {
                str(key): str(value)
                for key, value in payload.get("guidFactSignatures", {}).items()
            }
            self._fingerprint_fact_signatures = {
                str(key): str(value)
                for key, value in payload.get("fingerprintFactSignatures", {}).items()
            }
            self._report_revisions = {
                str(key): float(value)
                for key, value in payload.get("reportRevisions", {}).items()
            }
            self._revision_aware_guids = {
                str(key): float(value)
                for key, value in payload.get("revisionAwareGuids", {}).items()
            }
            self._revision_aware_fingerprints = {
                str(key): float(value)
                for key, value in payload.get("revisionAwareFingerprints", {}).items()
            }
            semantic: dict[str, dict[str, object]] = {}
            for key, value in payload.get("semantic", {}).items():
                if isinstance(value, dict):
                    semantic[str(key)] = {
                        "seen_at": float(value.get("seen_at", 0)),
                        "severity": int(value["severity"])
                        if value.get("severity") is not None
                        else None,
                        "status": str(value.get("status") or ""),
                        "fact_signature": str(value.get("fact_signature") or ""),
                    }
                else:
                    # Backward compatibility: old stores only had timestamps.
                    semantic[str(key)] = {
                        "seen_at": float(value),
                        "severity": None,
                        "status": "",
                        "fact_signature": "",
                    }
            self._semantic = semantic
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            structured_log(logging.WARNING, "dedupe.state_unreadable", error=str(error))

    def save(self) -> bool:
        with self._lock:
            self._prune()
            payload = {
                "guids": self._guids,
                "fingerprints": self._fingerprints,
                "guidStatuses": self._guid_statuses,
                "fingerprintStatuses": self._fingerprint_statuses,
                "guidFactSignatures": self._guid_fact_signatures,
                "fingerprintFactSignatures": self._fingerprint_fact_signatures,
                "reportRevisions": self._report_revisions,
                "revisionAwareGuids": self._revision_aware_guids,
                "revisionAwareFingerprints": self._revision_aware_fingerprints,
                "semantic": self._semantic,
            }
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{self._path.name}.",
                    suffix=".tmp",
                    dir=self._path.parent,
                    text=True,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "w") as handle:
                        json.dump(payload, handle, separators=(",", ":"))
                    os.replace(temporary, self._path)
                finally:
                    temporary.unlink(missing_ok=True)
                return True
            except OSError as error:
                structured_log(logging.WARNING, "dedupe.state_write_failed", error=str(error))
                return False

    def _prune(self) -> None:
        now = time.time()
        # GUIDs are kept far longer than the fingerprint window: a feed item
        # can reappear hours later and must not re-alert.
        guid_ttl = max(self._window, 24 * 60 * 60)
        self._guids = {k: v for k, v in self._guids.items() if now - v < guid_ttl}
        self._fingerprints = {k: v for k, v in self._fingerprints.items() if now - v < self._window}
        self._guid_statuses = {
            key: value for key, value in self._guid_statuses.items() if key in self._guids
        }
        self._fingerprint_statuses = {
            key: value
            for key, value in self._fingerprint_statuses.items()
            if key in self._fingerprints
        }
        self._guid_fact_signatures = {
            key: value
            for key, value in self._guid_fact_signatures.items()
            if key in self._guids
        }
        self._fingerprint_fact_signatures = {
            key: value
            for key, value in self._fingerprint_fact_signatures.items()
            if key in self._fingerprints
        }
        self._report_revisions = {
            key: value
            for key, value in self._report_revisions.items()
            if now - value < guid_ttl
        }
        self._revision_aware_guids = {
            key: value
            for key, value in self._revision_aware_guids.items()
            if key in self._guids
        }
        self._revision_aware_fingerprints = {
            key: value
            for key, value in self._revision_aware_fingerprints.items()
            if key in self._fingerprints
        }
        self._semantic = {
            key: value
            for key, value in self._semantic.items()
            if now - float(value.get("seen_at", 0))
            < _semantic_window_seconds(
                key.rsplit("|", 1)[-1],
                str(value.get("status") or ""),
                str(value.get("fact_signature") or ""),
            )
        }

        for store in (
            self._guids,
            self._fingerprints,
            self._report_revisions,
            self._semantic,
        ):
            if len(store) > MAX_TRACKED_ENTRIES:
                if store is self._semantic:
                    ordering = lambda key: float(store[key].get("seen_at", 0))
                else:
                    ordering = store.get
                for key in sorted(store, key=ordering)[: len(store) - MAX_TRACKED_ENTRIES]:
                    del store[key]

    def is_new(self, item: NewsItem) -> bool:
        # Deliberately pure: previews/dry-runs must not advance state, and a
        # duplicate check must never make a failed delivery unretryable.
        with self._lock:
            now = time.time()
            revision = report_revision_identity(item)
            if revision in self._report_revisions:
                return False
            status = event_status(item, "")
            fact_signature = event_fact_signature(item)
            if item.guid in self._guids:
                if item.guid in self._revision_aware_guids:
                    return True
                # Legacy seen.json files do not contain raw report revisions.
                # Preserve their prior behavior instead of replaying every
                # currently visible feed item once during an upgrade.
                previous_signature = self._guid_fact_signatures.get(item.guid, "")
                if _fact_signature_is_meaningfully_new(
                    previous_signature, fact_signature
                ):
                    return True
                return _status_is_meaningfully_new(
                    self._guid_statuses.get(item.guid, ""),
                    status,
                )
            digest = fingerprint(item)
            recent = self._fingerprints.get(digest)
            if recent is None or (now - recent) >= self._window:
                return True
            if digest in self._revision_aware_fingerprints:
                return True
            previous_signature = self._fingerprint_fact_signatures.get(digest, "")
            if _fact_signature_is_meaningfully_new(
                previous_signature, fact_signature
            ):
                return True
            return _status_is_meaningfully_new(
                self._fingerprint_statuses.get(digest, ""),
                status,
            )

    def record(self, item: NewsItem) -> None:
        with self._lock:
            now = time.time()
            digest = fingerprint(item)
            revision = report_revision_identity(item)
            status = event_status(item, "")
            fact_signature = event_fact_signature(item)
            self._guids[item.guid] = now
            self._fingerprints[digest] = now
            self._report_revisions[revision] = now
            self._revision_aware_guids[item.guid] = now
            self._revision_aware_fingerprints[digest] = now
            self._guid_statuses[item.guid] = status
            self._fingerprint_statuses[digest] = status
            self._guid_fact_signatures[item.guid] = fact_signature
            self._fingerprint_fact_signatures[digest] = fact_signature

    @staticmethod
    def semantic_key(player_name: str, event_type: str) -> str:
        from .matcher import compact_key

        return f"{compact_key(player_name)}|{event_type}"

    def is_semantically_new(
        self,
        player_name: str,
        event_type: str,
        severity: int | None = None,
        status: str = "",
        fact_signature: str = "",
    ) -> bool:
        """False for a repeat, True for a new event or meaningful escalation."""
        if not player_name:
            return True
        with self._lock:
            previous = self._semantic.get(self.semantic_key(player_name, event_type))
            if previous is None:
                return True
            previous_window = _semantic_window_seconds(
                event_type,
                str(previous.get("status") or ""),
                str(previous.get("fact_signature") or ""),
            )
            if (time.time() - float(previous.get("seen_at", 0))) >= previous_window:
                return True

            old_severity = previous.get("severity")
            old_status = str(previous.get("status") or "")
            previous_signature = str(previous.get("fact_signature") or "")
            same_role_fact = (
                _normalized_event_type(event_type) == "depth_chart"
                and status.startswith("role_")
                and old_status == status
                and fact_signature.startswith("role:")
                and previous_signature == fact_signature
            )
            same_trade_fact = (
                _normalized_event_type(event_type) == "trade"
                and old_status == status
                and event_facts_equivalent(
                    previous_signature,
                    fact_signature,
                    status=status,
                )
            )
            less_informative_trade_fact = (
                _normalized_event_type(event_type) == "trade"
                and old_status == status
                and _trade_fact_is_less_informative(
                    previous_signature,
                    fact_signature,
                )
            )
            if (
                severity is not None
                and old_severity is not None
                and severity > old_severity
                and not same_role_fact
                and not same_trade_fact
                and not less_informative_trade_fact
            ):
                return True
            if fact_signature and status and old_status and status != old_status:
                return True
            if status and old_status and _status_rank(status) > _status_rank(old_status):
                return True
            if fact_signature:
                if not event_facts_equivalent(
                    previous_signature,
                    fact_signature,
                    status=status,
                ) and not less_informative_trade_fact:
                    return True
            return False

    def record_semantic(
        self,
        player_name: str,
        event_type: str,
        severity: int | None = None,
        status: str = "",
        fact_signature: str = "",
    ) -> None:
        if player_name:
            with self._lock:
                now = time.time()
                key = self.semantic_key(player_name, event_type)
                previous = self._semantic.get(key)
                stored_severity = severity
                previous_window = (
                    _semantic_window_seconds(
                        event_type,
                        str(previous.get("status") or ""),
                        str(previous.get("fact_signature") or ""),
                    )
                    if previous is not None
                    else SEMANTIC_WINDOW_SECONDS
                )
                if (
                    previous is not None
                    and now - float(previous.get("seen_at", 0))
                    < previous_window
                    and str(previous.get("status") or "") == status
                    and fact_signature
                    and event_facts_equivalent(
                        str(previous.get("fact_signature") or ""),
                        fact_signature,
                        status=status,
                    )
                ):
                    previous_severity = previous.get("severity")
                    if previous_severity is not None and severity is not None:
                        stored_severity = max(int(previous_severity), int(severity))
                self._semantic[key] = {
                    "seen_at": now,
                    "severity": stored_severity,
                    "status": status,
                    "fact_signature": fact_signature,
                }

    def prime(self, items: list[NewsItem]) -> None:
        """Mark existing items as seen without alerting (first run)."""
        for item in items:
            self.record(item)
        self.save()
