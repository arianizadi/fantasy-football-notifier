"""Beat reporters worth streaming, and NFL player-name extraction from tweets.

Unlike RotoWire, a tweet has no "Player: headline" structure, so the player has
to be recovered from free text before the plays engine can do anything with it.
That is done by scanning word n-grams against the known NFL player set rather
than asking the model, so a hallucinated name can never drive a waiver call.
"""

from __future__ import annotations

from typing import Any

from ..matcher import compact_key, player_name_in_text

# Handles verified against the X API and volume measured over a real 7-day
# window (posts/day excluding retweets and replies), because reads are billed
# per post. Guessed handles are worse than useless: they silently stream
# nothing. Re-verify with bin/measure-reporters.py before adding any.
#
# Deliberately EXCLUDED:
#   RotoWireNFL   50.6/day ($7.59/mo) - posts the identical items to the
#                 RotoWire RSS feed this project already polls for free.
#   Rotoworld_FB  18.1/day ($2.72/mo) - same publisher family, same overlap.
#   underdog__nfl / FantasyPtsNFL / JJZachariason - do not resolve on X.
NATIONAL = [
    "AdamSchefter",    # 16.9/day - usually the origin for league-wide news
    "RapSheet",        # 13.7/day
    "MikeGarafolo",    #  8.3/day
    "JFowlerESPN",     #  5.3/day
    "AlbertBreer",     #  4.4/day
    "CameronWolfe",    #  4.4/day
    "JosinaAnderson",  #  3.0/day
    "TomPelissero",    #  0.3/day - low volume but high signal when he posts
]

# Fantasy-framed accounts that break transaction and depth-chart news the
# national reporters treat as too minor to cover.
FANTASY = [
    "MatthewBerryTMR",  # 6.3/day
    "FieldYates",       # 2.4/day - fast on signings and roster moves
]

# The exact Josh Jacobs contingency report on 2026-08-26 came directly from
# the Packers media availability before it reached general fantasy feeds.
# These four handles all posted that same availability and were resolved
# against X's users/by endpoint on 2026-08-26. Semantic dedupe collapses their
# overlapping football fact while the redundancy protects against one writer
# omitting a player's full name.
PACKERS_BREAKING = [
    "mattschneidman",
    "ByRyanWood",
    "RobDemovsky",
    "by_JBH",
]

# Single-team beat writers. High volume for one team's worth of coverage, so
# only worth the spend if you roster that team heavily.
BEAT_OPTIONAL = [
    "MikeReiss",  # 10.7/day - Patriots
]

ALL_REPORTERS = NATIONAL + FANTASY + PACKERS_BREAKING

# X caps a single filtered-stream rule at 1024 characters.
MAX_RULE_CHARS = 1024


def build_stream_rules(handles: list[str] | None = None) -> list[dict[str, str]]:
    """Pack `from:` clauses into as few rules as the 1024-char cap allows."""
    handles = handles or ALL_REPORTERS
    suffix = " -is:retweet -is:reply"
    rules: list[dict[str, str]] = []
    current: list[str] = []

    def flush(index: int) -> None:
        if current:
            rules.append(
                {"value": f"({' OR '.join(current)}){suffix}", "tag": f"reporters-{index}"}
            )

    for handle in handles:
        clause = f"from:{handle}"
        candidate = current + [clause]
        if len(f"({' OR '.join(candidate)}){suffix}") > MAX_RULE_CHARS:
            flush(len(rules))
            current = [clause]
        else:
            current = candidate
    flush(len(rules))
    return rules


class PlayerNameIndex:
    """Finds NFL player names inside arbitrary text."""

    def __init__(self, player_index: dict[str, Any]) -> None:
        self._names: dict[str, str] = {}
        for record in player_index.values():
            name = record.get("full_name") or ""
            # Only players attached to a team can be depth-charted.
            if not name or not record.get("team"):
                continue
            if record.get("position") not in {"QB", "RB", "WR", "TE", "K"}:
                continue
            self._names[compact_key(name)] = name

    def find(self, text: str) -> list[str]:
        """Return player names mentioned, longest-match-first, in order."""
        words = [w.strip(".,;:!?()[]\"'") for w in (text or "").split()]
        found: list[str] = []
        seen: set[str] = set()
        index = 0
        while index < len(words):
            # Try 3-word names ("Amon-Ra St. Brown") before 2-word ones.
            for size in (3, 2):
                if index + size > len(words):
                    continue
                key = compact_key(" ".join(words[index : index + size]))
                name = self._names.get(key)
                if name and key not in seen and player_name_in_text(name, text):
                    found.append(name)
                    seen.add(key)
                    index += size - 1
                    break
            index += 1
        return found
