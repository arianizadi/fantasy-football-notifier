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
    return event_fact_signature(item)


def _semantic_window_seconds(
    event_type: str,
    status: str,
    fact_signature: str,
) -> int:
    normalized = _normalized_event_type(event_type)
    if normalized == "depth_chart" and fact_signature.startswith("role:"):
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


def event_facts_equivalent(
    previous_signature: str,
    current_signature: str,
    *,
    status: str,
) -> bool:
    """Whether condition metadata proves two reports are corroboration."""
    if not previous_signature or not current_signature:
        return False
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
            same_role_fact = (
                _normalized_event_type(event_type) == "depth_chart"
                and status.startswith("role_")
                and str(previous.get("status") or "") == status
                and fact_signature.startswith("role:")
                and str(previous.get("fact_signature") or "") == fact_signature
            )
            if (
                severity is not None
                and old_severity is not None
                and severity > old_severity
                and not same_role_fact
            ):
                return True
            old_status = str(previous.get("status") or "")
            if fact_signature and status and old_status and status != old_status:
                return True
            if status and old_status and _status_rank(status) > _status_rank(old_status):
                return True
            if fact_signature:
                previous_signature = str(previous.get("fact_signature") or "")
                if not event_facts_equivalent(
                    previous_signature,
                    fact_signature,
                    status=status,
                ):
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
