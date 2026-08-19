"""Match free-text player names from news feeds onto ESPN roster entries.

Feeds and ESPN disagree on punctuation and suffixes ("C.J. Stroud" vs
"CJ Stroud", "Marvin Harrison Jr." vs "Marvin Harrison"), so both sides are
reduced to the same normalized form before comparison. Matching is exact on
the normalized string: fuzzy matching here would produce false alerts about
players who are not on the roster, which is worse than a missed one.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from .logging_utils import structured_log
from .models import RosterPlayer, RosterSnapshot

SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
PUNCTUATION = re.compile(r"[.'`‘’\-,]")
SLUG_TRAILING_ID = re.compile(r"-\d+$")


def normalize_name(value: str) -> str:
    """Fold a display name into a comparable key."""
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = PUNCTUATION.sub("", stripped.lower())
    tokens = [token for token in cleaned.split() if token]
    while len(tokens) > 2 and tokens[-1] in SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def compact_key(value: str) -> str:
    """Whitespace-insensitive comparison key.

    normalize_name() drops hyphens ("Amon-Ra" -> "amonra") while URL slugs use
    them as word separators ("amon-ra" -> "amon ra"), so the two paths would
    never match on a hyphenated name. Removing spaces as well collapses both
    onto "amonrastbrown".
    """
    return normalize_name(value).replace(" ", "")


def name_from_rotowire_url(url: str) -> str:
    """Recover a player name from a RotoWire player URL slug.

    Used when the "Player: Headline" title format does not parse.
    """
    if "/player/" not in url:
        return ""
    slug = url.rsplit("/player/", 1)[-1].split("?")[0].split("#")[0]
    return SLUG_TRAILING_ID.sub("", slug).replace("-", " ").strip()


class RosterIndex:
    """Normalized-name lookup over a roster snapshot."""

    def __init__(self, snapshot: RosterSnapshot) -> None:
        self._by_name: dict[str, RosterPlayer] = {}
        collisions: list[str] = []

        # Own players win any collision so a bench player on another team can
        # never mask an alert about a starter of yours.
        for player in sorted(snapshot.players, key=lambda entry: not entry.on_my_team):
            key = compact_key(player.name)
            if not key:
                continue
            if key in self._by_name:
                collisions.append(player.name)
                continue
            self._by_name[key] = player

        if collisions:
            structured_log(
                logging.WARNING,
                "matcher.name_collisions",
                collisionCount=len(collisions),
                names=collisions[:10],
            )

    def lookup(self, *candidates: str) -> RosterPlayer | None:
        for candidate in candidates:
            key = compact_key(candidate)
            if key and key in self._by_name:
                return self._by_name[key]
        return None

    def __len__(self) -> int:
        return len(self._by_name)
