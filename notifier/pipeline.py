"""Poll loop: fetch -> dedupe -> plays -> classify -> tier -> notify.

Every item is classified, not just ones touching your roster, so general NFL
news still surfaces. Relevance is expressed through tiering and per-tier
severity floors rather than by discarding items before the model sees them.
At ~$0.000025 per call this costs a few dollars a season.
"""

from __future__ import annotations

import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from .classify import classify
from .config import Config, seen_path
from .dedupe import SeenStore
from .logging_utils import structured_log
from .matcher import name_from_rotowire_url
from .models import Alert, NewsItem, RosterSnapshot
from .notify import send_alert, send_plain
from .plays import DepthCharts, LeaguePlays, plays_context_for_model
from .roster import load_snapshot, snapshot_mtime
from .sources import rotowire, sleeper
from .sources.twitter import TwitterStream

# X is the origin for most breaking news; RotoWire writes it up 1-5 minutes
# later. Ordering by priority means the faster source claims the semantic
# dedupe slot and the slower duplicate is suppressed, not the reverse.
SOURCE_PRIORITY = {"twitter": 0, "rotowire": 1}
CLASSIFY_WORKERS = 8

# Before your leagues draft every roster is empty, so roster tiering has
# nothing to bite on and the notifier would either say nothing or forward every
# practice note. Preseason mode instead asks: "would this change how I draft?"
# It raises the severity bar and restricts to players actually worth drafting,
# using Sleeper's overall rank. It engages automatically whenever no rostered
# players exist and disengages by itself after the draft.
PRESEASON_MIN_SEVERITY = 4
PRESEASON_MAX_RANK = 250

ROSTER_STALE_HOURS = 36
TRENDING_REFRESH_SECONDS = 15 * 60
PLAYER_INDEX_REFRESH_SECONDS = 12 * 60 * 60

# Weekday/hour (US/Eastern) windows where NFL news actually breaks. Outside
# these, polling backs off to the idle interval - RotoWire sends no cache
# validators, so every poll is a full fetch and 24/7 fast polling is just rude.
# Monday=0 ... Sunday=6
ACTIVE_WINDOWS = {
    6: range(9, 24),   # Sunday: inactives through the late games
    0: range(11, 24),  # Monday: fallout + MNF
    1: range(11, 20),  # Tuesday: waivers, cuts
    2: range(11, 20),  # Wednesday: first practice reports
    3: range(11, 24),  # Thursday: practice + TNF
    4: range(11, 22),  # Friday: final injury designations
    5: range(11, 20),  # Saturday: elevations, weather
}


EASTERN = ZoneInfo("America/New_York")


def _is_active_window(now: datetime) -> bool:
    """True inside an NFL news window, in US/Eastern (handles EST/EDT)."""
    stamp = now.astimezone(EASTERN)
    return stamp.hour in ACTIVE_WINDOWS.get(stamp.weekday(), range(0))


class Notifier:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        self.seen = SeenStore(seen_path(config))
        self.snapshot: RosterSnapshot = load_snapshot(config)
        self._snapshot_mtime = snapshot_mtime(config)
        self.poller = rotowire.FeedPoller()
        self._tweet_queue: queue.Queue = queue.Queue(maxsize=500)
        # Classification is ~2s of network wait, so a burst of items should
        # fan out rather than serialise. At $0.000025 a call this is free.
        self._pool = ThreadPoolExecutor(
            max_workers=CLASSIFY_WORKERS, thread_name_prefix="classify"
        )
        self.twitter: TwitterStream | None = (
            TwitterStream(config.twitter_bearer_token, self._tweet_queue)
            if config.twitter_bearer_token
            else None
        )

        self._player_index: dict = {}
        self._player_index_at = 0.0
        self._trending: dict[str, int] = {}
        self._trending_at = 0.0
        self.depth: DepthCharts | None = None
        self.preseason = not self.snapshot.drafted_leagues()
        self._rebuild_depth_charts()

    # ---------- state refresh ----------

    def _rebuild_depth_charts(self) -> None:
        if not self._player_index:
            try:
                self._player_index = sleeper.load_player_index(
                    self.config.state_dir, self.session
                )
                self._player_index_at = time.time()
            except (requests.RequestException, ValueError) as error:
                structured_log(logging.WARNING, "sleeper.index_failed", error=str(error))
                return
        self.depth = DepthCharts(self._player_index, self.snapshot)
        if self.twitter is not None:
            self.twitter.set_player_index(self._player_index)

    def _reload_roster_if_changed(self) -> None:
        current = snapshot_mtime(self.config)
        if current > self._snapshot_mtime:
            self.snapshot = load_snapshot(self.config)
            self._snapshot_mtime = current
            was_preseason = self.preseason
            self.preseason = not self.snapshot.drafted_leagues()
            self._rebuild_depth_charts()
            if was_preseason and not self.preseason:
                send_plain(
                    self.session,
                    self.config,
                    "Draft detected - roster filtering is live. "
                    "Switching from preseason mode to full alerts.",
                )
            structured_log(
                logging.INFO,
                "roster.reloaded",
                leagueCount=len(self.snapshot.leagues),
                myPlayerCount=len(self.snapshot.mine()),
            )

    def _refresh_trending(self) -> None:
        if not self.config.watch_trending:
            return
        if (time.time() - self._trending_at) < TRENDING_REFRESH_SECONDS:
            return
        try:
            self._trending = sleeper.trending_adds(
                self.session, self._player_index, limit=self.config.trending_limit
            )
            self._trending_at = time.time()
        except (requests.RequestException, ValueError) as error:
            structured_log(logging.WARNING, "sleeper.trending_failed", error=str(error))

    # ---------- evaluation ----------

    @staticmethod
    def _tier_for(per_league: list[LeaguePlays]) -> str:
        if any(p.subject_state == "mine" for p in per_league):
            return "mine"
        if any(p.claimable for p in per_league):
            return "claimable"
        if any(p.subject_state == "rostered" for p in per_league):
            return "rival"
        return "league"

    def _threshold_for(self, tier: str) -> int:
        if tier == "mine":
            return self.config.min_severity
        if tier == "claimable":
            return max(self.config.min_severity, 3)
        return self.config.min_severity_other

    def _evaluate(self, item: NewsItem) -> Alert | None:
        names = (item.player_name, name_from_rotowire_url(item.url))
        per_league: list[LeaguePlays] = []
        record = None
        if self.depth is not None:
            record, per_league = self.depth.build(
                subject_names=names, snapshot=self.snapshot
            )

        if self.preseason:
            return self._evaluate_preseason(item, record)

        tier = self._tier_for(per_league) if per_league else "league"

        classification = classify(
            self.session, self.config, item, context=plays_context_for_model(per_league)
        )

        threshold = self._threshold_for(tier)
        if classification.severity < threshold:
            structured_log(
                logging.DEBUG,
                "pipeline.below_threshold",
                player=item.player_name,
                tier=tier,
                severity=classification.severity,
                threshold=threshold,
            )
            return None

        # Only show leagues where there is something to do or something at stake.
        relevant = [
            plays
            for plays in per_league
            if plays.has_action or plays.subject_state == "mine"
        ]
        context = (
            self.depth.team_context(record, self.snapshot)
            if self.depth is not None and record is not None
            else None
        )
        return Alert(
            item=item,
            classification=classification,
            tier=tier,
            per_league=relevant,
            context=context,
            all_leagues=self.snapshot.drafted_leagues(),
        )

    def _evaluate_preseason(
        self, item: NewsItem, record: dict | None
    ) -> Alert | None:
        """Pre-draft: only news that would change how you draft."""
        if record is None:
            return None
        rank = record.get("search_rank")
        if rank is None or rank > PRESEASON_MAX_RANK:
            structured_log(
                logging.DEBUG,
                "preseason.not_draftable",
                player=item.player_name,
                searchRank=rank,
            )
            return None

        classification = classify(self.session, self.config, item)
        if classification.severity < PRESEASON_MIN_SEVERITY:
            return None

        return Alert(
            item=item,
            classification=classification,
            tier="preseason",
            per_league=[],
            context=self.depth.team_context(record, self.snapshot)
            if self.depth is not None
            else None,
            all_leagues=self.snapshot.drafted_leagues(),
        )

    # ---------- poll ----------

    def poll_once(self) -> int:
        self._reload_roster_if_changed()
        self._refresh_trending()

        items: list[NewsItem] = []
        feed_was_full = False
        modified = False
        try:
            items, feed_was_full, modified = self.poller.fetch(self.session)
        except requests.RequestException as error:
            structured_log(logging.WARNING, "rotowire.fetch_failed", error=str(error))
            # X is an independent primary source. An RSS outage must not stop
            # queued tweets from reaching the shared evaluation path.
        if not modified and self._tweet_queue.empty():
            return 0

        # Overflow detection must consider only the capacity-limited RSS feed,
        # before tweets are mixed in.
        rss_fresh = [item for item in items if self.seen.is_new(item)]
        if feed_was_full and items and len(rss_fresh) == len(items):
            structured_log(
                logging.WARNING,
                "rotowire.possible_overflow",
                hint="Every item was new; the 5-item feed may have rolled over.",
            )

        # Tweets arrive asynchronously; fold them into the same dedupe and
        # classification path so ordering and suppression stay consistent.
        tweets = []
        while True:
            try:
                tweets.append(self._tweet_queue.get_nowait())
            except queue.Empty:
                break

        fresh = rss_fresh + [item for item in tweets if self.seen.is_new(item)]

        # Fastest source first so it wins the semantic dedupe slot.
        fresh.sort(key=lambda i: SOURCE_PRIORITY.get(i.source, 9))
        for item in fresh:
            self.seen.record(item)

        # Classify concurrently, then serialise the alert decision so the
        # semantic dedupe check and its write cannot interleave.
        alerts = list(self._pool.map(self._evaluate, fresh))

        sent = 0
        for item, alert in zip(fresh, alerts):
            if alert is None:
                continue
            event_type = alert.classification.event_type
            if not self.seen.is_semantically_new(item.player_name, event_type):
                structured_log(
                    logging.INFO,
                    "pipeline.duplicate_event_suppressed",
                    player=item.player_name,
                    eventType=event_type,
                    source=item.source,
                )
                continue
            message_id = send_alert(self.session, self.config, alert)
            if message_id is None:
                continue
            self.seen.record_semantic(item.player_name, event_type)
            sent += 1

        if fresh:
            self.seen.save()
        return sent

    # ---------- lifecycle ----------

    def check_roster_freshness(self) -> None:
        if self.snapshot.generated_at is None:
            return
        age = (datetime.now(timezone.utc) - self.snapshot.generated_at).total_seconds() / 3600
        if age > ROSTER_STALE_HOURS:
            structured_log(logging.WARNING, "roster.stale", ageHours=round(age, 1))
            send_plain(
                self.session,
                self.config,
                f"Roster snapshot is {age:.0f}h old - alerts may use a stale lineup.",
            )

    def _interval(self) -> int:
        if not self.config.adaptive_polling:
            return self.config.poll_seconds
        return (
            self.config.poll_seconds
            if _is_active_window(datetime.now(timezone.utc))
            else self.config.poll_seconds_idle
        )

    def run_forever(self) -> None:
        structured_log(
            logging.INFO,
            "notifier.started",
            leagues=[f"{ref.label} ({ref.provider})" for ref in self.snapshot.leagues],
            myPlayerCount=len(self.snapshot.mine()),
            pollSeconds=self.config.poll_seconds,
            idlePollSeconds=self.config.poll_seconds_idle,
            model=self.config.openrouter_model,
        )
        if self.twitter is not None and self.twitter.sync_rules():
            self.twitter.start()

        if self.preseason:
            structured_log(
                logging.INFO,
                "preseason.mode_active",
                minSeverity=PRESEASON_MIN_SEVERITY,
                maxRank=PRESEASON_MAX_RANK,
            )
            send_plain(
                self.session,
                self.config,
                "<b>Preseason mode.</b> Until your leagues draft you will only "
                f"get severity {PRESEASON_MIN_SEVERITY}+ news about draft-relevant "
                "players. Switches to full roster alerts automatically after a draft.",
            )

        self.check_roster_freshness()
        last_freshness = time.time()
        last_index_refresh = time.time()

        while True:
            started = time.time()
            try:
                self.poll_once()
            except Exception as error:  # noqa: BLE001 - loop must survive
                structured_log(
                    logging.ERROR,
                    "pipeline.poll_failed",
                    error=str(error),
                    errorType=type(error).__name__,
                )

            now = time.time()
            if now - last_freshness > 6 * 3600:
                self.check_roster_freshness()
                last_freshness = now
            if now - last_index_refresh > PLAYER_INDEX_REFRESH_SECONDS:
                # Depth charts shift weekly; drop the cache and rebuild.
                self._player_index = {}
                self._rebuild_depth_charts()
                last_index_refresh = now

            time.sleep(max(1.0, self._interval() - (time.time() - started)))
