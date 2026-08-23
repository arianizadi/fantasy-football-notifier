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
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests

from ..logging_utils import structured_log
from ..models import NewsItem
from .reporters import PlayerNameIndex, build_stream_rules

RULES_URL = "https://api.x.com/2/tweets/search/stream/rules"
STREAM_URL = "https://api.x.com/2/tweets/search/stream"
REQUEST_TIMEOUT = 30
STREAM_READ_TIMEOUT = 90
BACKOFF_START = 5
BACKOFF_MAX = 320
QUEUE_MAXSIZE = 500


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
        # One NewsItem per player mentioned, so a multi-player tweet can produce
        # a correct per-player alert instead of one ambiguous blob.
        targets = players or [""]

        items = []
        for player in targets:
            items.append(
                NewsItem(
                    source="twitter",
                    guid=f"twitter:{tweet_id}:{player}",
                    player_name=player,
                    headline=text[:180],
                    body=text,
                    url=f"https://x.com/{handle}/status/{tweet_id}",
                    published_at=published_at,
                )
            )
        return items
