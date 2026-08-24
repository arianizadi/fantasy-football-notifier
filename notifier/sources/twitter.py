"""X/Twitter filtered stream ingest.

Runs on a background thread and pushes NewsItems into a queue that the main
poll loop drains, so dedupe/classify/notify stay in one place regardless of
which source an item came from.

X's filtered stream is a long-lived HTTP chunked connection delivering
newline-delimited JSON, not a WebSocket. Pay-per-use allows exactly one
concurrent connection, so this must be a singleton.

Reads are billed per post returned, which is why the server-side rules filter
to breaking-news accounts and exclude retweets and replies.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from ..logging_utils import structured_log
from ..matcher import player_name_spans
from ..models import NewsItem
from .reporters import PlayerNameIndex, build_stream_rules

RULES_URL = "https://api.x.com/2/tweets/search/stream/rules"
STREAM_URL = "https://api.x.com/2/tweets/search/stream"
REQUEST_TIMEOUT = 30
STREAM_READ_TIMEOUT = 90
BACKOFF_START = 5
BACKOFF_MAX = 320
QUEUE_MAXSIZE = 500

ABSENCE_CUE = re.compile(
    r"\b(?:ruled\s+out|out|inactive|injur(?:y|ed)|hurt|sidelined|"
    r"(?:will|expected\s+to|likely\s+to|set\s+to)\s+miss|"
    r"(?:won't|will\s+not|not\s+expected\s+to|unlikely\s+to)\s+play|"
    r"doubtful|questionable|game[-\s]time\s+decision|"
    r"did\s+not\s+practice|dnp|limited(?:\s+in\s+practice)?|"
    r"placed\s+on\s+(?:ir|injured\s+reserve|pup|nfi|"
    r"non[-\s]football\s+injury(?:\s+list)?)|"
    r"(?:entered|has\s+entered|had\s+entered|is\s+in|remains\s+in|placed\s+in|"
    r"was\s+placed\s+in|"
    r"has\s+been\s+placed\s+in)\s+(?:the\s+)?concussion\s+protocol|"
    r"carted\s+off|exited|left\s+(?:(?:the|today['’]?s|"
    r"monday['’]?s|tuesday['’]?s|wednesday['’]?s|thursday['’]?s|"
    r"friday['’]?s|saturday['’]?s|sunday['’]?s)\s+)?(?:game|practice)|"
    r"sprain(?:ed)?|fractur(?:e|ed)|concussion|tear|tore|torn|ruptured|"
    r"suspended|released|waived|cut)\b",
    re.IGNORECASE,
)
HARD_CLAUSE_BOUNDARY = re.compile(r"[;.!?\n]")
DIAGNOSIS_CUES = frozenset(
    {
        "injury",
        "sprain",
        "sprained",
        "fracture",
        "fractured",
        "concussion",
        "tear",
        "tore",
        "torn",
        "ruptured",
    }
)
DIRECT_STATUS_BRIDGES = frozenset(
    {
        "",
        "is",
        "is a",
        "is still",
        "is now",
        "is officially",
        "is reportedly",
        "was",
        "was a",
        "was still",
        "was just",
        "was officially",
        "was reportedly",
        "remains",
        "remained",
        "got",
        "gets",
        "has been",
        "has now been",
        "has officially been",
        "had been",
        "will be",
        "may be",
        "might be",
        "could be",
        "expected to be",
        "is expected to be",
        "was expected to be",
        "is likely",
        "was likely",
        "appears",
        "appeared",
        "reportedly",
        "officially",
        "now",
        "who is",
        "who is still",
        "who was",
        "who was just",
        "who remains",
        "who has been",
        "who had been",
        "who will be",
    }
)
ACTION_BRIDGES = frozenset(
    {
        "",
        "has",
        "had",
        "reportedly",
        "apparently",
        "may have",
        "might have",
        "could have",
        "is believed to have",
        "was believed to have",
    }
)
INJURY_LEADS = (
    "has been diagnosed with",
    "had been diagnosed with",
    "was diagnosed with",
    "is diagnosed with",
    "is dealing with",
    "was dealing with",
    "has suffered",
    "had suffered",
    "has sustained",
    "had sustained",
    "diagnosed with",
    "dealing with",
    "suffered",
    "sustained",
    "has",
    "had",
    "with",
)
INJURY_DETAIL_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "apparent",
        "possible",
        "suspected",
        "potential",
        "mild",
        "moderate",
        "severe",
        "significant",
        "minor",
        "major",
        "high",
        "low",
        "upper",
        "lower",
        "left",
        "right",
        "grade",
        "one",
        "two",
        "three",
        "i",
        "ii",
        "iii",
        "1",
        "2",
        "3",
        "season-ending",
        "non-contact",
        "partial",
        "complete",
        "acl",
        "mcl",
        "pcl",
        "achilles",
        "ankle",
        "knee",
        "hamstring",
        "quad",
        "quadriceps",
        "calf",
        "groin",
        "hip",
        "foot",
        "toe",
        "leg",
        "back",
        "neck",
        "head",
        "shoulder",
        "arm",
        "elbow",
        "wrist",
        "hand",
        "finger",
        "thumb",
        "rib",
        "ribs",
        "chest",
        "pectoral",
        "pec",
        "clavicle",
        "collarbone",
        "tibia",
        "fibula",
        "patella",
        "meniscus",
        "labrum",
        "lisfranc",
        "oblique",
        "core",
        "abdomen",
        "abdominal",
        "illness",
        "personal",
        "dnp",
    }
)
TRANSACTION_CUES = frozenset({"released", "waived", "suspended", "cut"})
PASSIVE_TRANSACTION_BRIDGES = frozenset(
    {
        "is",
        "is being",
        "was",
        "was just",
        "has been",
        "has now been",
        "had been",
        "will be",
        "who is",
        "who was",
        "who has been",
        "who had been",
        "who will be",
    }
)
NFL_ROSTER_MARKERS = frozenset(
    {
        "nfl",
        "team",
        "roster",
        "club",
        "organization",
        "league",
        "squad",
        "cardinals",
        "falcons",
        "ravens",
        "bills",
        "panthers",
        "bears",
        "bengals",
        "browns",
        "cowboys",
        "broncos",
        "lions",
        "packers",
        "texans",
        "colts",
        "jaguars",
        "chiefs",
        "raiders",
        "chargers",
        "rams",
        "dolphins",
        "vikings",
        "patriots",
        "saints",
        "giants",
        "jets",
        "eagles",
        "steelers",
        "seahawks",
        "49ers",
        "buccaneers",
        "titans",
        "commanders",
    }
)
NFL_TEAM_ABBREVIATIONS = frozenset(
    {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LV",
        "LAC",
        "LAR",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    }
)
NON_ROSTER_RELEASE_WORDS = frozenset(
    {
        "hospital",
        "custody",
        "jail",
        "statement",
        "footage",
        "video",
        "album",
        "update",
        "news",
        "report",
        "highlight",
        "protocol",
        "medical",
        "staff",
        "cleared",
        "clearance",
        "doctor",
        "doctors",
        "pup",
        "ir",
        "reserve",
        "injured",
    }
)


def _bridge_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)?", value.casefold())


def _valid_injury_details(words: list[str], *, maximum: int = 5) -> bool:
    return len(words) <= maximum and all(word in INJURY_DETAIL_WORDS for word in words)


def _strip_injury_parentheticals(value: str) -> str:
    def replace_tag(match: re.Match[str]) -> str:
        words = _bridge_words(match.group(1))
        return " " if words and _valid_injury_details(words, maximum=4) else match.group(0)

    return re.sub(r"\(([^()]*)\)", replace_tag, value)


def _has_roster_context(value: str) -> bool:
    words = set(_bridge_words(value))
    abbreviations = set(re.findall(r"(?<![A-Za-z])[A-Z]{2,3}(?![A-Za-z])", value))
    has_marker = bool(words & NFL_ROSTER_MARKERS) or bool(
        abbreviations & NFL_TEAM_ABBREVIATIONS
    )
    return has_marker and not bool(words & NON_ROSTER_RELEASE_WORDS)


def _direct_suffix(cue_text: str, between: str, after_cue: str) -> bool:
    """Whether a cue is a direct predicate of the preceding player name."""
    bridge = " ".join(_bridge_words(_strip_injury_parentheticals(between)))
    if cue_text in TRANSACTION_CUES:
        if bridge not in PASSIVE_TRANSACTION_BRIDGES:
            return False
        following = after_cue.lstrip(" \t,:—–-")
        direct_context = re.match(r"^(?:by|from|off)\b", following, re.IGNORECASE)
        suspension_context = cue_text == "suspended" and re.match(
            r"^(?:for\s+)?(?:\d+|one|two|three|four|five|six|eight|"
            r"indefinitely)\s+(?:games?|weeks?)\s+by\b",
            following,
            re.IGNORECASE,
        )
        if not direct_context and not suspension_context:
            return False
        return _has_roster_context(following)
    bridge_words = bridge.split()
    if bridge_words and _valid_injury_details(bridge_words, maximum=4):
        return True
    if cue_text in DIAGNOSIS_CUES:
        if bridge in DIRECT_STATUS_BRIDGES or bridge in ACTION_BRIDGES:
            return True
        words = bridge.split()
        if _valid_injury_details(words, maximum=4):
            return True
        for lead in INJURY_LEADS:
            if bridge == lead:
                return True
            if bridge.startswith(lead + " "):
                return _valid_injury_details(bridge[len(lead) + 1 :].split())
        return False
    if cue_text in {"sprained", "fractured", "tore", "ruptured", "exited"}:
        return bridge in ACTION_BRIDGES
    if cue_text.startswith("left "):
        return bridge in ACTION_BRIDGES
    return bridge in DIRECT_STATUS_BRIDGES


def _direct_prefix(
    cue_text: str,
    between: str,
    before_cue: str,
    after_player: str,
) -> bool:
    """Whether a headline-style cue directly labels the following player."""
    words = _bridge_words(between)
    if cue_text in TRANSACTION_CUES:
        tail_words = set(_bridge_words(after_player[:80]))
        return (
            not words
            and _has_roster_context(before_cue[-80:])
            and not bool(tail_words & NON_ROSTER_RELEASE_WORDS)
        )
    if cue_text in DIAGNOSIS_CUES:
        if not words:
            return True
        if words[-1] in {"for", "to"}:
            return _valid_injury_details(words[:-1], maximum=3)
        # ``Torn ACL: Jordan Mason`` is safe; arbitrary prose before a name is
        # not. A visible label separator is required when ``for`` is absent.
        return bool(re.search(r"[:—–-]", between)) and _valid_injury_details(
            words, maximum=3
        )
    if not words:
        return True
    if len(words) == 1 and words[0] in {"qb", "rb", "wr", "te", "k"}:
        return cue_text in {"injured", "hurt", "sidelined", "inactive"}
    return False


def attributed_absence_subject(text: str, players: list[str]) -> str:
    """Return one confidently associated unavailable player, or empty.

    Reporter posts often name an injured starter and two possible replacements.
    Distance alone is unsafe (the first replacement may sit right after
    ``ruled out``), so attribution stays within one punctuation-delimited
    clause and rejects paths that cross another player mention.
    """
    mentions = {
        player: player_name_spans(player, text)
        for player in players
    }
    all_mentions = [
        (start, end, player)
        for player, spans in mentions.items()
        for start, end in spans
    ]
    cues = list(ABSENCE_CUE.finditer(text))
    # ``Team released Jordan Mason injury update`` can mean the team published
    # an update, not that it cut Mason. Once that transaction-looking prefix is
    # rejected by its post-name label, a later ``injury`` word must not rescue
    # the same mention and turn it back into a confident roster transaction.
    blocked_mentions: set[tuple[int, int, str]] = set()
    for start, end, player in all_mentions:
        for cue in cues:
            cue_text = " ".join(cue.group(0).casefold().split())
            if cue_text not in TRANSACTION_CUES or cue.end() > start:
                continue
            between = text[cue.end() : start]
            if HARD_CLAUSE_BOUNDARY.search(between) or _bridge_words(between):
                continue
            clause_start = max(
                [
                    match.end()
                    for match in HARD_CLAUSE_BOUNDARY.finditer(text[: cue.start()])
                ]
                or [0]
            )
            next_boundary = HARD_CLAUSE_BOUNDARY.search(text, end)
            clause_end = next_boundary.start() if next_boundary is not None else len(text)
            tail_words = set(_bridge_words(text[end:clause_end]))
            if (
                _has_roster_context(text[clause_start : cue.start()][-80:])
                and tail_words & NON_ROSTER_RELEASE_WORDS
            ):
                blocked_mentions.add((start, end, player))

    scored: list[tuple[int, str, int, int, bool]] = []
    for player, spans in mentions.items():
        for start, end in spans:
            if (start, end, player) in blocked_mentions:
                continue
            for cue in cues:
                cue_start, cue_end = cue.span()
                if end <= cue_start:
                    between_start, between_end = end, cue_start
                    prefix = False
                elif cue_end <= start:
                    between_start, between_end = cue_end, start
                    prefix = True
                else:
                    between_start = between_end = start
                    prefix = False

                between = text[between_start:between_end]
                if HARD_CLAUSE_BOUNDARY.search(between):
                    continue
                if len(re.findall(r"\w+", between)) > 6:
                    continue
                if any(
                    other_player != player
                    and other_start >= between_start
                    and other_end <= between_end
                    for other_start, other_end, other_player in all_mentions
                ):
                    continue
                cue_text = " ".join(cue.group(0).casefold().split())
                clause_start = max(
                    [
                        match.end()
                        for match in HARD_CLAUSE_BOUNDARY.finditer(text[:cue_start])
                    ]
                    or [0]
                )
                next_boundary = HARD_CLAUSE_BOUNDARY.search(text, cue_end)
                clause_end = (
                    next_boundary.start() if next_boundary is not None else len(text)
                )
                if prefix:
                    if not _direct_prefix(
                        cue_text,
                        between,
                        text[clause_start:cue_start],
                        text[end:clause_end],
                    ):
                        continue
                elif not _direct_suffix(
                    cue_text,
                    between,
                    text[cue_end:clause_end],
                ):
                    continue
                scored.append(
                    (
                        len(between) + (4 if prefix else 0),
                        player,
                        cue_start,
                        cue_end,
                        prefix,
                    )
                )

    if not scored:
        return ""
    scored.sort(key=lambda value: value[0])
    best_score, best_player, cue_start, cue_end, prefix = scored[0]
    if any(
        score == best_score and player != best_player
        for score, player, _, _, _ in scored[1:]
    ):
        return ""

    clause_start = max(
        [match.end() for match in HARD_CLAUSE_BOUNDARY.finditer(text[:cue_start])]
        or [0]
    )
    next_boundary = HARD_CLAUSE_BOUNDARY.search(text, cue_end)
    clause_end = next_boundary.start() if next_boundary is not None else len(text)
    if prefix:
        same_side = [
            mention
            for mention in all_mentions
            if mention[0] >= cue_end and mention[1] <= clause_end
        ]
        side_text = text[cue_end : max((end for _, end, _ in same_side), default=cue_end)]
    else:
        same_side = [
            mention
            for mention in all_mentions
            if mention[0] >= clause_start and mention[1] <= cue_start
        ]
        side_text = text[
            min((start for start, _, _ in same_side), default=cue_start) : cue_start
        ]
    if len({player for _, _, player in same_side}) > 1 and re.search(
        r"\b(?:both|were|are|have|two|and)\b|[&/]", side_text, re.IGNORECASE
    ):
        return ""
    return best_player


class TwitterStream:
    def __init__(self, bearer_token: str, out_queue: queue.Queue) -> None:
        self._token = bearer_token
        self._queue = out_queue
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {bearer_token}", "User-Agent": "fantasy-notifier/1.0"}
        )
        self._names: PlayerNameIndex | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_connected_at = 0.0
        self._last_item_at = 0.0
        self._last_error_at = 0.0
        self._last_error = ""
        self._connected = False

    def set_player_index(self, player_index: dict[str, Any]) -> None:
        self._names = PlayerNameIndex(player_index)

    # ---------- rules ----------

    def sync_rules(self) -> bool:
        """Replace server-side rules with the current reporter list."""
        try:
            existing = self._session.get(RULES_URL, timeout=REQUEST_TIMEOUT)
            existing.raise_for_status()
            current = existing.json().get("data") or []

            if current:
                self._session.post(
                    RULES_URL,
                    timeout=REQUEST_TIMEOUT,
                    json={"delete": {"ids": [rule["id"] for rule in current]}},
                ).raise_for_status()

            wanted = build_stream_rules()
            response = self._session.post(
                RULES_URL, timeout=REQUEST_TIMEOUT, json={"add": wanted}
            )
            response.raise_for_status()
            structured_log(logging.INFO, "twitter.rules_synced", ruleCount=len(wanted))
            return True
        except requests.RequestException as error:
            structured_log(logging.ERROR, "twitter.rules_failed", error=str(error))
            return False

    # ---------- stream ----------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="twitter-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._session.close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def health_snapshot(self) -> dict[str, object]:
        """Thread-safe enough scalar snapshot for the Telegram /status command."""
        return {
            "alive": self.is_alive,
            "connected": self._connected,
            "last_connected_at": self._last_connected_at,
            "last_item_at": self._last_item_at,
            "last_error_at": self._last_error_at,
            "last_error": self._last_error,
        }

    def _run(self) -> None:
        backoff = BACKOFF_START
        while not self._stop.is_set():
            try:
                delivered = self._consume()
                # X drops long-lived connections routinely ("Response ended
                # prematurely"). If tweets actually arrived, the connection was
                # healthy, so reset the backoff instead of escalating toward
                # 320s and going deaf during a news burst.
                if delivered:
                    backoff = BACKOFF_START
                    continue
                backoff = BACKOFF_START
            except requests.RequestException as error:
                self._connected = False
                self._last_error_at = time.time()
                self._last_error = str(error)[:160]
                status = getattr(getattr(error, "response", None), "status_code", None)
                if status == 402:
                    # Out of credits: retrying just burns rate limit.
                    structured_log(logging.ERROR, "twitter.credits_depleted")
                    self._stop.set()
                    return
                structured_log(
                    logging.WARNING,
                    "twitter.stream_error",
                    error=str(error),
                    status=status,
                    backoffSeconds=backoff,
                )
            # X requires exponential backoff on reconnect or it will ban the app.
            self._stop.wait(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)

    def _consume(self) -> int:
        """Read the stream until it closes. Returns tweets delivered."""
        delivered = 0
        params = {
            "tweet.fields": "created_at,author_id,text",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        with self._session.get(
            STREAM_URL, params=params, stream=True, timeout=(REQUEST_TIMEOUT, STREAM_READ_TIMEOUT)
        ) as response:
            response.raise_for_status()
            self._connected = True
            self._last_connected_at = time.time()
            self._last_error = ""
            structured_log(logging.INFO, "twitter.stream_connected")
            for raw in response.iter_lines():
                if self._stop.is_set():
                    self._connected = False
                    return delivered
                if not raw:
                    continue  # keep-alive newline
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for item in self._to_items(payload):
                    delivered += 1
                    self._last_item_at = time.time()
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        structured_log(logging.WARNING, "twitter.queue_full")
        self._connected = False
        return delivered

    def _to_items(self, payload: dict[str, Any]) -> list[NewsItem]:
        data = payload.get("data") or {}
        text = str(data.get("text") or "").strip()
        tweet_id = str(data.get("id") or "")
        if not text or not tweet_id:
            return []

        # Log X's own delivery lag (tweet timestamp -> arrival on our socket).
        # created_at has second granularity, so treat this as +/-1s.
        published_at = None
        created_raw = data.get("created_at")
        if created_raw:
            try:
                created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                published_at = created
                lag = (datetime.now(timezone.utc) - created).total_seconds()
                structured_log(
                    logging.INFO, "twitter.delivery_lag", lagSeconds=round(lag, 2)
                )
            except ValueError:
                pass

        users = {u["id"]: u for u in (payload.get("includes", {}).get("users") or [])}
        author = users.get(str(data.get("author_id")), {})
        handle = str(author.get("username") or "unknown")

        players = self._names.find(text) if self._names else []
        # No matched player already has no mechanical roster path. A single
        # matched replacement is not automatically the subject when the post
        # contains absence language about an unmatched nickname or surname.
        subject_confident = len(players) <= 1
        player = players[0] if players else ""
        if len(players) == 1 and ABSENCE_CUE.search(text):
            subject_confident = attributed_absence_subject(text, players) == player
        elif len(players) > 1:
            attributed = attributed_absence_subject(text, players)
            if attributed:
                player = attributed
                subject_confident = True
            else:
                # Keep one report-level alert instead of generating one alert
                # per mention. The pipeline withholds mechanical moves when
                # deterministic attribution cannot identify the subject.
                subject_confident = False

        return [
            NewsItem(
                source="twitter",
                guid=f"twitter:{tweet_id}:{player}",
                player_name=player,
                headline=text[:180],
                body=text,
                url=f"https://x.com/{handle}/status/{tweet_id}",
                published_at=published_at,
                subject_confident=subject_confident,
            )
        ]
