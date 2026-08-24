"""Poll loop: fetch -> dedupe -> plays -> classify -> tier -> notify.

Every item is classified, not just ones touching your roster, so general NFL
news still surfaces. Relevance is expressed through tiering and per-tier
severity floors rather than by discarding items before the model sees them.
At ~$0.000025 per call this costs a few dollars a season.
"""

from __future__ import annotations

import fcntl
import html
import logging
import math
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from .classify import classify
from .config import Config, seen_path
from .dedupe import (
    SeenStore,
    event_fact_signature,
    event_facts_equivalent,
    event_status,
)
from .event_store import EventStore
from .health import HEALTH, age_label, duration_label
from .logging_utils import NotifierError, structured_log
from .matcher import compact_key, name_from_rotowire_url
from .models import Alert, Classification, NewsItem, RosterSnapshot
from .notify import retry_after_seconds, send_alert, send_plain, telegram_state
from .outbox import DeliveryOutbox, PendingDelivery
from .player_lookup import format_player_lookup
from .plays import DepthCharts, LeaguePlays, plays_context_for_model, plays_for_event
from .roster import load_snapshot, refresh_drafted_snapshot, snapshot_mtime
from .sources import rotowire, sleeper
from .sources.fantasypros import FantasyProsCache
from .sources.twitter import TwitterStream
from .telegram_control import TelegramControl

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
PRESEASON_MIN_SEVERITY = 3
PRESEASON_MAX_RANK = 250

ROSTER_STALE_HOURS = 36
PLAYER_INDEX_REFRESH_SECONDS = sleeper.PLAYER_INDEX_TTL_SECONDS
PLAYER_INDEX_RETRY_SECONDS = 15 * 60
JIT_ROSTER_REFRESH_MIN_SECONDS = 60
FANTASYPROS_RETRY_BASE_SECONDS = 15 * 60
FANTASYPROS_RETRY_MAX_SECONDS = 6 * 60 * 60
FANTASYPROS_ENRICHMENT_ERROR_LOG_SECONDS = 5 * 60
FANTASYPROS_UNAVAILABLE_ERRORS = frozenset(
    {"dataset_unavailable", "partial_dataset_unavailable"}
)


def _next_player_index_refresh_at(player_index: dict, now: float) -> float:
    """Schedule daily success refreshes and short degraded-state retries."""
    if not player_index or bool(getattr(player_index, "stale", False)):
        return now + PLAYER_INDEX_RETRY_SECONDS
    refreshed_at = getattr(player_index, "refreshed_at", None)
    if refreshed_at is None:
        return now + PLAYER_INDEX_REFRESH_SECONDS
    return refreshed_at.timestamp() + PLAYER_INDEX_REFRESH_SECONDS


def _fantasypros_retry_delay(consecutive_failures: int) -> float:
    """Bound repeated failed batches without changing the healthy cadence."""
    exponent = min(5, max(0, int(consecutive_failures) - 1))
    return float(
        min(
            FANTASYPROS_RETRY_BASE_SECONDS * (2**exponent),
            FANTASYPROS_RETRY_MAX_SECONDS,
        )
    )


def _fantasypros_failure_delay(
    error: str,
    consecutive_failures: int,
    healthy_refresh_seconds: float,
) -> float:
    """Keep unpublished datasets on cadence; back off hard failures."""
    if error in FANTASYPROS_UNAVAILABLE_ERRORS:
        return float(healthy_refresh_seconds)
    return _fantasypros_retry_delay(consecutive_failures)


def _depth_report_text(item: NewsItem) -> str:
    """Factual report text plus RotoWire's URL-attributed article player.

    RotoWire removes ``Player:`` from the parsed headline. When a backup's
    article is re-centered on an injured starter, the source URL still proves
    which backup the article discussed. Add that name only to the depth-chart
    mention check; mutating the stored headline/body would create a false new
    report revision during deployment.
    """
    values = [item.headline, item.body]
    if item.source == "rotowire":
        values.insert(0, name_from_rotowire_url(item.url))
    return " ".join(value for value in values if value)


# A newer delivered clearance/return makes an older queued absence report
# misleading. These narrow transitions are safe to suppress automatically;
# unrelated follow-ups still deliver in chronological order.
SUPERSEDING_EVENTS = {
    "injury": frozenset({"return"}),
    "inactive": frozenset({"return"}),
    "practice_report": frozenset({"inactive", "return"}),
    "suspension": frozenset({"return"}),
    # Do not replay an optimistic clearance after a later re-injury or game
    # absence has already reached the user.
    "return": frozenset({"injury", "inactive"}),
}

_EVENT_STATUS_RANK = {
    "season_out": 100,
    "injured_reserve": 90,
    "inactive": 80,
    "doubtful": 60,
    "questionable": 50,
    "dnp": 40,
    "limited": 30,
    "cleared": 20,
}


def _status_rank(status: str) -> int:
    return _EVENT_STATUS_RANK.get(status, 0)


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_event_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return _aware_utc(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def _report_is_newer(
    newer_item: NewsItem,
    newer_received_at: float,
    older_item: NewsItem,
    older_received_at: float,
) -> bool:
    """Prefer source chronology, then the serialized queue/receipt order."""
    newer_published = _aware_utc(newer_item.published_at)
    older_published = _aware_utc(older_item.published_at)
    if newer_published is not None and older_published is not None:
        if newer_published != older_published:
            return newer_published > older_published
    return newer_received_at > older_received_at


def _same_event_update_supersedes(older: Alert, newer: Alert) -> bool:
    """Whether a chronological update makes an older same-event send stale."""
    if newer.classification.event_type != older.classification.event_type:
        return False

    old_severity = older.classification.severity
    new_severity = newer.classification.severity
    old_status = event_status(older.item, older.classification.event_type)
    new_status = event_status(newer.item, newer.classification.event_type)
    old_rank = _status_rank(old_status)
    new_rank = _status_rank(new_status)

    # A deterministic clearance is a forward transition even though the
    # absence-oriented rank scale is numerically lower.
    if new_status == "cleared" and old_rank >= _status_rank("limited"):
        return True

    # A lower-urgency follow-up cannot erase a stronger pending alert.
    if new_severity < old_severity or new_rank < old_rank:
        return False
    if new_severity > old_severity or new_rank > old_rank:
        return True

    old_facts = event_fact_signature(older.item)
    new_facts = event_fact_signature(newer.item)
    if old_facts == new_facts:
        # Equal concrete condition markers prove corroboration. Two generic
        # usage reports both map to ``unspecified``, however, and may contain
        # distinct actionable facts (for example starter versus goal-line
        # work), so they must remain independently deliverable.
        return event_facts_equivalent(
            old_facts,
            new_facts,
            status=new_status,
        )
    if old_facts == "unspecified":
        return new_facts != "unspecified"
    if new_facts == "unspecified":
        return False
    # A timetable/refinement that retains every prior marker supersedes the
    # less-complete report. Disjoint conditions remain independent facts.
    old_markers = set(old_facts.split("|"))
    new_markers = set(new_facts.split("|"))
    return old_markers < new_markers


def _alert_supersedes(older: Alert, newer: Alert) -> bool:
    newer_event = newer.classification.event_type
    older_event = older.classification.event_type
    if newer_event in SUPERSEDING_EVENTS.get(older_event, frozenset()):
        return True
    return _same_event_update_supersedes(older, newer)

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
        self._process_lock_file = None
        if not config.dry_run:
            lock_path = config.state_dir / "notifier.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = lock_path.open("a+")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                lock_file.close()
                raise NotifierError(
                    "Another notifier process owns the state directory; "
                    "stop it before running a second daemon, --once, or --send backfill."
                ) from error
            self._process_lock_file = lock_file
        self.session = requests.Session()
        self.seen = SeenStore(seen_path(config))
        self.outbox = DeliveryOutbox(config.state_dir)
        self.events = EventStore(config.state_dir, in_memory=config.dry_run)
        self.fantasypros = FantasyProsCache(
            config.state_dir,
            "" if config.dry_run else getattr(config, "fantasypros_api_key", ""),
            int(getattr(config, "espn_year", datetime.now(timezone.utc).year)),
            app_daily_cap=int(getattr(config, "fantasypros_request_limit", 425)),
            refresh_seconds=(
                int(getattr(config, "fantasypros_refresh_hours", 2)) * 3600
            ),
            max_stale_seconds=(
                int(getattr(config, "fantasypros_max_age_hours", 12)) * 3600
            ),
        )
        self._fantasypros_thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._delivery_lock = threading.RLock()
        self._inflight_items: dict[NewsItem, float] = {}
        self.snapshot: RosterSnapshot = load_snapshot(config)
        self._snapshot_mtime = snapshot_mtime(config)
        self._jit_roster_lock = threading.Lock()
        self._last_jit_roster_refresh = 0.0
        self._last_jit_roster_success = 0.0
        self.poller = rotowire.FeedPoller()
        self._tweet_queue: queue.Queue = queue.Queue(maxsize=500)
        # Classification is ~2s of network wait, so a burst of items should
        # fan out rather than serialise. At $0.000025 a call this is free.
        self._pool = ThreadPoolExecutor(
            max_workers=CLASSIFY_WORKERS, thread_name_prefix="classify"
        )
        # X cannot sit behind a burst of slow RSS model calls. Its own pool and
        # queue dispatcher let stream items begin classification immediately.
        self._twitter_pool = ThreadPoolExecutor(
            max_workers=max(2, CLASSIFY_WORKERS // 2),
            thread_name_prefix="classify-twitter",
        )
        self._twitter_dispatcher: threading.Thread | None = None
        self._stop = threading.Event()
        self.twitter: TwitterStream | None = (
            TwitterStream(config.twitter_bearer_token, self._tweet_queue)
            if config.twitter_bearer_token
            else None
        )

        self._player_index: dict = {}
        self.depth: DepthCharts | None = None
        self.preseason = not self.snapshot.drafted_leagues()
        self._rebuild_depth_charts()
        self.telegram_state = telegram_state(config)
        self.telegram_control = TelegramControl(
            config,
            self.telegram_state,
            status_provider=self.status_text,
            player_provider=self.player_text,
            search_provider=self.news_search_text,
            feedback_provider=self.events.record_feedback,
        )
        HEALTH.mark(
            "roster",
            ok=True,
            detail=f"{len(self.snapshot.leagues)} leagues, {len(self.snapshot.mine())} mine",
        )

    # ---------- state refresh ----------

    def _rebuild_depth_charts(self, *, reload_player_index: bool = False) -> bool:
        """Atomically install a fresh player index and derived depth charts.

        A failed network refresh leaves the last good index/depth pair live.
        Callers can then retry soon without degrading player lookup or X name
        matching for the rest of the day.
        """
        with self._state_lock:
            player_index = self._player_index
        if reload_player_index or not player_index:
            try:
                player_index = sleeper.load_player_index(
                    self.config.state_dir,
                    self.session,
                    write_cache=not self.config.dry_run,
                )
            except (requests.RequestException, ValueError) as error:
                HEALTH.mark("sleeper", ok=False, detail=str(error))
                structured_log(logging.WARNING, "sleeper.index_failed", error=str(error))
                return False
        with self._state_lock:
            self._player_index = player_index
            self.depth = DepthCharts(player_index, self.snapshot)
        refreshed_at = getattr(player_index, "refreshed_at", None)
        detail = (
            refreshed_at.isoformat()
            if refreshed_at is not None
            else f"{len(player_index)} players"
        )
        stale = bool(getattr(player_index, "stale", False))
        HEALTH.mark(
            "sleeper",
            ok=not stale,
            detail=f"stale cache; {detail}" if stale else detail,
        )
        if self.twitter is not None:
            self.twitter.set_player_index(player_index)
        return True

    def _reload_roster_if_changed(self) -> None:
        current = snapshot_mtime(self.config)
        if current > self._snapshot_mtime:
            snapshot = load_snapshot(self.config)
            with self._state_lock:
                was_preseason = self.preseason
                self.snapshot = snapshot
                self._snapshot_mtime = current
                self.preseason = not snapshot.drafted_leagues()
                self.depth = DepthCharts(self._player_index, snapshot)
                preseason = self.preseason
            if was_preseason and not preseason:
                send_plain(
                    self.session,
                    self.config,
                    "Draft detected - roster filtering is live. "
                    "Switching from preseason mode to full alerts.",
                )
            structured_log(
                logging.INFO,
                "roster.reloaded",
                leagueCount=len(snapshot.leagues),
                myPlayerCount=len(snapshot.mine()),
            )
            HEALTH.mark(
                "roster",
                ok=True,
                detail=f"{len(snapshot.leagues)} leagues, {len(snapshot.mine())} mine",
            )

    def _refresh_ownership_just_in_time(self) -> bool:
        """Refresh drafted-league ownership before evaluating a waiver candidate."""
        now = time.time()
        if now - self._last_jit_roster_success < JIT_ROSTER_REFRESH_MIN_SECONDS:
            return True
        if now - self._last_jit_roster_refresh < JIT_ROSTER_REFRESH_MIN_SECONDS:
            return False
        with self._jit_roster_lock:
            now = time.time()
            if now - self._last_jit_roster_success < JIT_ROSTER_REFRESH_MIN_SECONDS:
                return True
            if now - self._last_jit_roster_refresh < JIT_ROSTER_REFRESH_MIN_SECONDS:
                return False
            # Throttle failures as well as successes; an outage should not make
            # every simultaneous breaking-news worker hammer active providers.
            self._last_jit_roster_refresh = now
            try:
                with self._state_lock:
                    previous = self.snapshot
                snapshot, written_version = refresh_drafted_snapshot(
                    self.config,
                    previous,
                )
            except Exception as error:  # noqa: BLE001 - keep the alert path alive
                HEALTH.mark("roster", ok=False, detail=str(error))
                structured_log(logging.WARNING, "roster.jit_refresh_failed", error=str(error))
                return False

            with self._state_lock:
                self.snapshot = snapshot
                self._snapshot_mtime = written_version
                self.preseason = not snapshot.drafted_leagues()
                self.depth = DepthCharts(self._player_index, snapshot)
                self._last_jit_roster_success = time.time()
            HEALTH.mark(
                "roster",
                ok=True,
                detail=f"just-in-time; {len(snapshot.leagues)} leagues",
            )
            structured_log(
                logging.INFO,
                "roster.jit_refreshed",
                leagueCount=len(snapshot.leagues),
                playerCount=len(snapshot.players),
            )
            return True

    def _fantasypros_scoring_formats(self) -> tuple[str, ...]:
        """Scoring formats actually used by drafted provider leagues."""
        with self._state_lock:
            snapshot = self.snapshot
            drafted_keys = {league.key for league in snapshot.drafted_leagues()}
            values = {
                scoring
                for league_key, scoring in snapshot.scoring_formats.items()
                if league_key in drafted_keys
            }
        order = ("PPR", "HALF", "STD")
        return tuple(scoring for scoring in order if scoring in values)

    def _enrich_fantasypros(
        self, per_league: list[LeaguePlays]
    ) -> list[LeaguePlays]:
        """Add memory-only ranking context without touching the alert clock."""
        cache = getattr(self, "fantasypros", None)
        if cache is None:
            return per_league

        enriched_plays: list[LeaguePlays] = []
        for plays in per_league:
            scoring = (plays.scoring_format or "").strip().upper()
            if scoring not in {"PPR", "HALF", "STD"} or not plays.beneficiaries:
                enriched_plays.append(plays)
                continue
            beneficiaries = []
            for beneficiary in plays.beneficiaries:
                try:
                    signal = cache.signal(
                        beneficiary.name,
                        scoring=scoring,
                        team=beneficiary.pro_team,
                        position=beneficiary.position,
                    )
                    if signal is None:
                        beneficiaries.append(beneficiary)
                        continue
                    enriched = replace(
                        beneficiary,
                        fantasypros_waiver_rank=signal.waiver_rank,
                        fantasypros_waiver_pos_rank=signal.waiver_pos_rank,
                        fantasypros_ros_rank=signal.ros_rank,
                        fantasypros_ros_pos_rank=signal.ros_pos_rank,
                        fantasypros_scoring=signal.scoring,
                        fantasypros_updated_at=signal.updated_at.isoformat(),
                    )
                except Exception:
                    # This is optional, secondary context. A corrupt cache or
                    # unexpected client implementation must never drop or
                    # delay the breaking alert. Do not include exception text:
                    # custom transports can put credentials in it.
                    now = time.monotonic()
                    last_log = getattr(
                        self,
                        "_last_fantasypros_enrichment_error_log",
                        float("-inf"),
                    )
                    if now - last_log >= FANTASYPROS_ENRICHMENT_ERROR_LOG_SECONDS:
                        self._last_fantasypros_enrichment_error_log = now
                        structured_log(
                            logging.WARNING,
                            "fantasypros.enrichment_skipped",
                            reason="cache_error",
                        )
                    enriched = beneficiary
                beneficiaries.append(enriched)
            enriched_plays.append(replace(plays, beneficiaries=beneficiaries))
        return enriched_plays

    def _fantasypros_refresh_loop(self) -> None:
        """Maintain bulk ranking snapshots off the breaking-news path."""
        consecutive_failures = 0
        while not self._stop.is_set():
            scoring_formats = self._fantasypros_scoring_formats()
            if not scoring_formats:
                self._stop.wait(60)
                continue

            delay = self.fantasypros.seconds_until_refresh(scoring_formats)
            refreshed = True
            if delay <= 0:
                refreshed = self.fantasypros.refresh(scoring_formats)

            status = self.fantasypros.status()
            expected = {
                f"{scoring}:{ranking_type}"
                for scoring in scoring_formats
                for ranking_type in ("WAIVER", "ROS")
            }
            fresh = set(status.datasets_fresh)
            complete = expected.issubset(fresh)
            detail = (
                f"{len(fresh & expected)}/{len(expected)} datasets; "
                f"{status.requests_used}/{status.request_cap} requests/24h"
            )
            if status.last_error:
                detail += f"; {status.last_error}"
            HEALTH.mark("fantasypros", ok=refreshed and complete, detail=detail)

            if not refreshed:
                if status.last_error in FANTASYPROS_UNAVAILABLE_ERRORS:
                    # The endpoint is healthy; the seasonal ranking family is
                    # simply unpublished. Reprobe all due siblings on the
                    # configured healthy cadence instead of degrading to the
                    # six-hour hard-failure backoff.
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                wait_seconds = _fantasypros_failure_delay(
                    status.last_error,
                    consecutive_failures,
                    int(getattr(self.config, "fantasypros_refresh_hours", 2))
                    * 3600,
                )
            else:
                consecutive_failures = 0
                wait_seconds = self.fantasypros.seconds_until_refresh(
                    scoring_formats
                )
                if math.isinf(wait_seconds):
                    return
                # Notice a newly refreshed roster/scoring format promptly
                # without busy-polling the provider.
                wait_seconds = min(max(wait_seconds, 60.0), 5 * 60.0)
            self._stop.wait(wait_seconds)

    def _start_fantasypros_refresher(self) -> None:
        if not self.fantasypros.enabled:
            return
        if self._fantasypros_thread is not None and self._fantasypros_thread.is_alive():
            return
        self._fantasypros_thread = threading.Thread(
            target=self._fantasypros_refresh_loop,
            name="fantasypros-refresh",
            daemon=True,
        )
        self._fantasypros_thread.start()

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

    def _journal_received(self, item: NewsItem) -> None:
        if getattr(self.config, "dry_run", False) or not hasattr(self, "events"):
            return
        try:
            self.events.record_received(item)
        except Exception as error:  # noqa: BLE001 - journal must not block alerts
            structured_log(logging.WARNING, "events.record_failed", error=str(error))

    def _journal_classification(
        self,
        item: NewsItem,
        classification,
        *,
        tier: str,
        outcome: str,
    ) -> None:
        if getattr(self.config, "dry_run", False) or not hasattr(self, "events"):
            return
        try:
            self.events.record_classification(
                item,
                classification,
                tier=tier,
                outcome=outcome,
            )
        except Exception as error:  # noqa: BLE001 - journal must not block alerts
            structured_log(logging.WARNING, "events.classification_failed", error=str(error))

    def _journal_outcome(
        self,
        item: NewsItem,
        outcome: str,
        *,
        tier: str = "",
        message_id: int | None = None,
    ) -> None:
        if getattr(self.config, "dry_run", False) or not hasattr(self, "events"):
            return
        try:
            self.events.mark_outcome(
                item,
                outcome,
                tier=tier,
                message_id=message_id,
            )
        except Exception as error:  # noqa: BLE001 - journal must not block alerts
            structured_log(logging.WARNING, "events.outcome_failed", error=str(error))

    def _recent_news_context(self, item: NewsItem) -> str:
        """Compact, structured history for trajectory-aware classification."""
        if not item.player_name or not hasattr(self, "events"):
            return ""
        try:
            recent = self.events.recent_for_player(
                item.player_name,
                limit=12,
                exclude_guid=item.guid,
                since_hours=72,
            )
        except Exception as error:  # noqa: BLE001 - history is optional grounding
            structured_log(logging.WARNING, "events.context_failed", error=str(error))
            return ""
        facts = []
        seen_kinds: set[tuple[str, str]] = set()
        for event in recent:
            event_type = event.get("event_type")
            if not event_type:
                continue
            if event.get("feedback") in {"wrong", "noisy"}:
                continue
            if event.get("outcome") in {
                "suppressed_duplicate",
                "filtered_unknown_player",
                "filtered_not_draftable",
            }:
                continue
            direction = str(event.get("direction") or "unknown")
            kind = (str(event_type), direction)
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            facts.append(
                f"{event_type}/{direction}/"
                f"severity {event.get('severity') or '?'}: "
                f"{str(event.get('headline') or '')[:140]}"
            )
            if len(facts) >= 3:
                break
        if not facts:
            return ""
        return (
            "Recent saved reports for context (the current report may supersede them): "
            + " | ".join(facts)
        )

    def _evaluate(self, item: NewsItem) -> Alert | None:
        # X callbacks and the roster/index refresh loop run concurrently.
        # Capture a coherent snapshot/depth pair rather than reading one on
        # either side of an atomic state swap.
        with self._state_lock:
            snapshot = self.snapshot
            depth = self.depth
            preseason = self.preseason
        names = (item.player_name, name_from_rotowire_url(item.url))
        per_league: list[LeaguePlays] = []
        record = None
        if depth is not None:
            record, per_league = depth.build(
                subject_names=names,
                snapshot=snapshot,
                report_text=_depth_report_text(item),
            )

        if preseason:
            return self._evaluate_preseason(item, record, snapshot, depth)

        classification_context = "\n".join(
            value
            for value in (
                plays_context_for_model(per_league),
                self._recent_news_context(item),
            )
            if value
        )
        classification = classify(
            self.session,
            self.config,
            item,
            context=classification_context,
        )
        if not item.subject_confident:
            classification = replace(
                classification,
                event_type="other",
                fantasy_impact="",
                is_actionable=False,
                raw={
                    **classification.raw,
                    "direction": "unknown",
                    "subject_attribution": "uncertain",
                },
            )

        per_league = plays_for_event(
            per_league,
            classification.event_type,
            classification.severity,
        )
        if not item.subject_confident:
            # A multi-player X report with ambiguous attribution remains useful
            # news, but no mentioned player may inherit another player's injury
            # or trigger a mechanical pickup/lineup move.
            per_league = [
                replace(plays, beneficiaries=[], bench_options=[])
                for plays in per_league
            ]
        availability_refresh_failed = False
        # Refresh whenever this event has a next-man-up candidate, even when
        # the cached snapshot says that candidate is rostered. A player may
        # have been dropped since the scheduled snapshot and become claimable.
        if any(plays.beneficiaries for plays in per_league) and not self.config.dry_run:
            if self._refresh_ownership_just_in_time():
                # Rebuild from one coherent, just-fetched state pair before an
                # ADD line or free-agent tag can be emitted.
                with self._state_lock:
                    snapshot = self.snapshot
                    depth = self.depth
                if depth is not None:
                    record, refreshed_plays = depth.build(
                        subject_names=names,
                        snapshot=snapshot,
                        report_text=_depth_report_text(item),
                    )
                    per_league = plays_for_event(
                        refreshed_plays,
                        classification.event_type,
                        classification.severity,
                    )
                else:
                    availability_refresh_failed = True
            else:
                availability_refresh_failed = True

            if availability_refresh_failed:
                # Fail closed. The report itself can still alert, and owned
                # players may still have a safe bench substitution, but stale
                # ownership can never produce ADD/FA advice.
                per_league = [replace(plays, beneficiaries=[]) for plays in per_league]
        # Cache-only enrichment happens after the live ownership rebuild. It
        # cannot delay, suppress, or create a pickup recommendation.
        per_league = self._enrich_fantasypros(per_league)
        tier = self._tier_for(per_league) if per_league else "league"

        threshold = self._threshold_for(tier)
        if classification.severity < threshold:
            self._journal_classification(
                item,
                classification,
                tier=tier,
                outcome="filtered_threshold",
            )
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
            depth.team_context(record, snapshot)
            if depth is not None and record is not None
            else None
        )
        alert = Alert(
            item=item,
            classification=classification,
            tier=tier,
            per_league=relevant,
            context=context,
            all_leagues=snapshot.drafted_leagues(),
            availability_refresh_failed=availability_refresh_failed,
        )
        self._journal_classification(
            item,
            classification,
            tier=tier,
            outcome="alert_ready",
        )
        return alert

    def _evaluate_preseason(
        self,
        item: NewsItem,
        record: dict | None,
        snapshot: RosterSnapshot | None = None,
        depth: DepthCharts | None = None,
    ) -> Alert | None:
        """Pre-draft: only news that would change how you draft."""
        # Optional arguments preserve the focused helper/testing API; normal
        # concurrent evaluation passes its already-coherent state capture.
        snapshot = snapshot if snapshot is not None else self.snapshot
        depth = depth if depth is not None else self.depth
        if record is None:
            self._journal_outcome(item, "filtered_unknown_player", tier="preseason")
            return None
        rank = record.get("search_rank")
        if rank is None or rank > PRESEASON_MAX_RANK:
            self._journal_outcome(item, "filtered_not_draftable", tier="preseason")
            structured_log(
                logging.DEBUG,
                "preseason.not_draftable",
                player=item.player_name,
                searchRank=rank,
            )
            return None

        classification = classify(
            self.session,
            self.config,
            item,
            context=(
                "Mode: preseason before fantasy drafts. Describe draft-value consequence only; "
                "do not tell the manager to activate, start, bench, add, drop, or draft a player.\n"
                + self._recent_news_context(item)
            ),
        )
        if not item.subject_confident:
            classification = replace(
                classification,
                event_type="other",
                fantasy_impact="",
                is_actionable=False,
                raw={
                    **classification.raw,
                    "direction": "unknown",
                    "subject_attribution": "uncertain",
                },
            )
        if classification.severity < PRESEASON_MIN_SEVERITY:
            self._journal_classification(
                item,
                classification,
                tier="preseason",
                outcome="filtered_threshold",
            )
            return None

        alert = Alert(
            item=item,
            classification=classification,
            tier="preseason",
            per_league=[],
            context=depth.team_context(record, snapshot)
            if depth is not None
            else None,
            all_leagues=snapshot.drafted_leagues(),
        )
        self._journal_classification(
            item,
            classification,
            tier="preseason",
            outcome="alert_ready",
        )
        return alert

    # ---------- durable delivery ----------

    def _claim_item(self, item: NewsItem) -> bool:
        """Reserve a new item for evaluation without advancing durable dedupe."""
        with self._state_lock:
            if not self.seen.is_new(item) or self.outbox.contains_item(item):
                return False
            # Only an identical raw revision is an early duplicate. Different
            # GUIDs with a shared headline, and a reused GUID with a changed
            # body, need classification before their meaning is knowable.
            if item in self._inflight_items:
                return False
            self._inflight_items[item] = time.time()
        self._journal_received(item)
        article_player = (
            name_from_rotowire_url(item.url) if item.source == "rotowire" else ""
        )
        if (
            article_player
            and item.player_name
            and compact_key(article_player) != compact_key(item.player_name)
        ):
            # Log only after the normalized revision wins the seen/outbox and
            # in-flight checks. Feed servers without useful cache validators
            # can return the same five items every poll; logging during pure
            # normalization would otherwise create thousands of duplicate
            # audit lines per day.
            structured_log(
                logging.INFO,
                "rotowire.subject_reattributed",
                articlePlayer=article_player,
                absenceSubject=item.player_name,
            )
        return True

    def _normalize_source_subject(self, item: NewsItem) -> NewsItem:
        """Apply source-specific deterministic subject attribution."""
        if item.source != "rotowire":
            return item
        with self._state_lock:
            player_index = getattr(self, "_player_index", {})
        return rotowire.reattribute_beneficiary_report(item, player_index)

    def _release_item(self, item: NewsItem) -> None:
        with self._state_lock:
            self._inflight_items.pop(item, None)

    def _observed_at(self, item: NewsItem) -> float:
        with self._state_lock:
            return self._inflight_items.get(item, time.time())

    def _record_terminal_item(self, item: NewsItem) -> None:
        """Finalize a filtered or already-delivered duplicate item."""
        if self.config.dry_run:
            return
        with self._state_lock:
            self.seen.record(item)
            self.seen.save()

    def _semantic_is_new(self, alert: Alert) -> bool:
        classification = alert.classification
        return self.seen.is_semantically_new(
            alert.item.player_name,
            classification.event_type,
            classification.severity,
            event_status(alert.item, classification.event_type),
            event_fact_signature(alert.item),
        )

    def _can_coalesce(self, alert: Alert) -> bool:
        """Whether a semantic repeat can safely enrich its existing message."""
        if self.config.dry_run:
            return False
        state = getattr(self, "telegram_state", None)
        if state is None:
            return False
        try:
            return state.coalescing_target(alert) is not None
        except Exception as error:  # noqa: BLE001 - editing is optional
            structured_log(
                logging.WARNING,
                "notify.coalescing_check_failed",
                error=str(error),
            )
            return False

    def _record_success(self, alert: Alert) -> bool:
        classification = alert.classification
        self.seen.record(alert.item)
        self.seen.record_semantic(
            alert.item.player_name,
            classification.event_type,
            classification.severity,
            event_status(alert.item, classification.event_type),
            event_fact_signature(alert.item),
        )
        return self.seen.save()

    @staticmethod
    def _alert_from_event_row(row: dict) -> tuple[Alert, float] | None:
        """Rebuild only the fields needed to compare a delivered journal row."""
        received = _parse_event_time(row.get("received_at"))
        if received is None:
            return None
        published = _parse_event_time(row.get("published_at"))
        try:
            severity = int(row.get("severity") or 0)
        except (TypeError, ValueError):
            severity = 0
        item = NewsItem(
            source=str(row.get("source") or ""),
            guid=str(row.get("guid") or ""),
            player_name=str(row.get("player_name") or ""),
            headline=str(row.get("headline") or ""),
            body=str(row.get("body") or ""),
            url=str(row.get("url") or ""),
            published_at=published,
        )
        classification = Classification(
            event_type=str(row.get("event_type") or "other"),
            severity=severity,
            fantasy_impact=str(row.get("summary") or ""),
            is_actionable=bool(row.get("is_actionable", False)),
            raw={},
        )
        return Alert(
            item=item,
            classification=classification,
            tier=str(row.get("tier") or "league"),
        ), received.timestamp()

    def _pending_superseding_alert(
        self, pending: PendingDelivery
    ) -> tuple[Alert, bool] | None:
        """Return a newer pending/delivered report and whether it was delivered."""
        alert = pending.alert
        if not alert.item.player_name:
            return None
        rows: list[dict] = []
        if hasattr(self, "events"):
            try:
                rows = self.events.recent_for_player(alert.item.player_name, limit=12)
            except Exception as error:  # noqa: BLE001 - delivery must remain available
                structured_log(
                    logging.WARNING,
                    "delivery.supersession_check_failed",
                    error=str(error),
                )
        pending_received_at = pending.observed_at or pending.queued_at

        # During an outage a state transition can itself still be queued. The
        # older alert must not retry first (for example ADD after a return).
        for candidate in self.outbox.pending_for_player(alert.item.player_name):
            if candidate.delivery_id == pending.delivery_id:
                continue
            candidate_item = candidate.alert.item
            candidate_received_at = candidate.observed_at or candidate.queued_at
            if not _report_is_newer(
                candidate_item,
                candidate_received_at,
                alert.item,
                pending_received_at,
            ):
                continue
            if _alert_supersedes(alert, candidate.alert):
                return candidate.alert, False

        for row in rows:
            if row.get("outcome") != "delivered":
                continue
            restored = self._alert_from_event_row(row)
            if restored is None:
                continue
            delivered, received_at = restored
            same_revision = (
                delivered.item.guid == alert.item.guid
                and delivered.item.headline == alert.item.headline
                and delivered.item.body == alert.item.body
            )
            if same_revision:
                # Telegram accepted this exact revision and the journal was
                # committed before an interrupted outbox removal.
                return delivered, True
            # A reused GUID keeps its original received_at in SQLite. Its
            # updated_at is the only durable chronology for the replacement.
            if delivered.item.guid == alert.item.guid:
                updated = _parse_event_time(row.get("updated_at"))
                if updated is not None:
                    received_at = updated.timestamp()
            if not _report_is_newer(
                delivered.item,
                received_at,
                alert.item,
                pending_received_at,
            ):
                continue
            if _alert_supersedes(alert, delivered):
                return delivered, True
        return None

    def _pending_is_superseded(self, pending: PendingDelivery) -> bool:
        return self._pending_superseding_alert(pending) is not None

    def _superseded_pending_by(
        self, delivered: PendingDelivery
    ) -> list[PendingDelivery]:
        """Find older queued reports made stale by this accepted delivery."""
        def receipt(pending: PendingDelivery) -> float:
            return pending.observed_at or pending.queued_at

        stale: list[PendingDelivery] = []
        for candidate in self.outbox.pending_for_player(
            delivered.alert.item.player_name
        ):
            if candidate.delivery_id == delivered.delivery_id:
                continue
            if not _report_is_newer(
                delivered.alert.item,
                receipt(delivered),
                candidate.alert.item,
                receipt(candidate),
            ):
                continue
            if _alert_supersedes(candidate.alert, delivered.alert):
                stale.append(candidate)
        return stale

    def _revalidate_delayed_alert(self, pending: PendingDelivery) -> Alert:
        """Refresh backup eligibility before retrying a persisted alert."""
        alert = replace(pending.alert, delivery_delayed=True)
        needs_ownership = alert.availability_refresh_failed or any(
            plays.beneficiaries for plays in alert.per_league
        )
        if not needs_ownership:
            return alert

        if not self._refresh_ownership_just_in_time():
            return replace(
                alert,
                per_league=[replace(plays, beneficiaries=[]) for plays in alert.per_league],
                availability_refresh_failed=True,
            )

        with self._state_lock:
            snapshot = self.snapshot
            depth = self.depth
        if depth is None:
            return replace(
                alert,
                per_league=[replace(plays, beneficiaries=[]) for plays in alert.per_league],
                availability_refresh_failed=True,
            )

        names = (
            alert.item.player_name,
            name_from_rotowire_url(alert.item.url),
        )
        record, per_league = depth.build(
            subject_names=names,
            snapshot=snapshot,
            report_text=_depth_report_text(alert.item),
        )
        per_league = plays_for_event(
            per_league,
            alert.classification.event_type,
            alert.classification.severity,
        )
        per_league = self._enrich_fantasypros(per_league)
        relevant = [
            plays
            for plays in per_league
            if plays.has_action or plays.subject_state == "mine"
        ]
        context = depth.team_context(record, snapshot) if record is not None else None
        return replace(
            alert,
            tier=self._tier_for(per_league) if per_league else "league",
            per_league=relevant,
            context=context,
            all_leagues=snapshot.drafted_leagues(),
            availability_refresh_failed=False,
        )

    def _attempt_pending_locked(
        self,
        pending: PendingDelivery,
        *,
        force_semantic: bool = False,
        replay: bool = False,
    ) -> int:
        """Try one persisted alert. Caller serializes delivery decisions."""
        current = self.outbox.get(pending.delivery_id)
        if current is None:
            # An earlier accepted update can retire a later entry that still
            # exists in deliver_pending's due snapshot. Never send that
            # detached object or attempt to reschedule it.
            return 0
        pending = current
        superseding = self._pending_superseding_alert(pending)
        if superseding is not None:
            newer, was_delivered = superseding
            self.seen.record(pending.alert.item)
            saved = self._record_success(newer) if was_delivered else self.seen.save()
            if saved:
                self.outbox.remove(pending.delivery_id)
            if pending.alert.item.guid != newer.item.guid:
                self._journal_outcome(
                    pending.alert.item,
                    "suppressed_superseded",
                    tier=pending.alert.tier,
                )
            return 0
        if pending.attempts > 0 or replay:
            pending.alert = self._revalidate_delayed_alert(pending)

        alert = pending.alert
        if (
            not force_semantic
            and not self._semantic_is_new(alert)
            and not self._can_coalesce(alert)
        ):
            # Another source delivered this event while this attempt was
            # pending. Mark only the raw duplicate seen, then retire it.
            self.seen.record(alert.item)
            if self.seen.save():
                self.outbox.remove(pending.delivery_id)
            self._journal_outcome(
                alert.item,
                "suppressed_duplicate",
                tier=alert.tier,
            )
            return 0

        message_id = send_alert(self.session, self.config, alert)
        if message_id is None:
            self.outbox.mark_failed(
                pending.delivery_id,
                retry_after=retry_after_seconds(),
            )
            self._journal_outcome(alert.item, "pending_retry", tier=alert.tier)
            structured_log(
                logging.WARNING,
                "delivery.deferred",
                player=alert.item.player_name,
                eventType=alert.classification.event_type,
                attempts=pending.attempts,
            )
            return 0

        # Telegram accepted the message. Only now may GUID/content/semantic
        # dedupe advance. If its disk write fails, retain the outbox entry: an
        # at-least-once duplicate is safer than a permanently lost alert.
        superseded = self._superseded_pending_by(pending)
        for stale in superseded:
            self.seen.record(stale.alert.item)
        saved = self._record_success(alert)
        # The journal is the cross-process proof that this exact revision was
        # accepted. Commit it before removing the durable send intent.
        self._journal_outcome(
            alert.item,
            "delivered",
            tier=alert.tier,
            message_id=message_id,
        )
        if saved:
            self.outbox.remove(pending.delivery_id)
            for stale in superseded:
                if stale.alert.item.guid != alert.item.guid:
                    self._journal_outcome(
                        stale.alert.item,
                        "suppressed_superseded",
                        tier=stale.alert.tier,
                    )
                self.outbox.remove(stale.delivery_id)
        return 1

    def _complete_evaluation(self, item: NewsItem, alert: Alert | None) -> int:
        try:
            if alert is None:
                self._record_terminal_item(item)
                return 0

            if self.config.dry_run:
                if not self._semantic_is_new(alert):
                    return 0
                return 1 if send_alert(self.session, self.config, alert) is not None else 0

            with self._delivery_lock:
                semantic_new = self._semantic_is_new(alert)
                coalescing = not semantic_new and self._can_coalesce(alert)
                if not semantic_new and not coalescing:
                    structured_log(
                        logging.INFO,
                        "pipeline.duplicate_event_suppressed",
                        player=item.player_name,
                        eventType=alert.classification.event_type,
                        source=item.source,
                    )
                    self._record_terminal_item(item)
                    self._journal_outcome(item, "suppressed_duplicate", tier=alert.tier)
                    return 0
                try:
                    pending = self.outbox.add(
                        alert,
                        observed_at=self._observed_at(item),
                    )
                except (OSError, RuntimeError) as error:
                    structured_log(logging.ERROR, "outbox.enqueue_failed", error=str(error))
                    return 0
                self._journal_outcome(item, "queued", tier=alert.tier)
                return self._attempt_pending_locked(
                    pending,
                    force_semantic=coalescing,
                )
        finally:
            self._release_item(item)

    def deliver_pending(self) -> int:
        """Retry due outbox entries before processing new feed work."""
        if self.config.dry_run:
            return 0
        sent = 0
        for pending in self.outbox.due():
            with self._delivery_lock:
                sent += self._attempt_pending_locked(pending, replay=True)
        return sent

    def _evaluate_future(self, item: NewsItem, future: Future) -> int:
        try:
            return self._complete_evaluation(item, future.result())
        except Exception as error:  # noqa: BLE001 - one item must not kill dispatch
            self._release_item(item)
            structured_log(
                logging.ERROR,
                "pipeline.evaluate_failed",
                player=item.player_name,
                error=str(error),
                errorType=type(error).__name__,
            )
            return 0

    def _process_items(self, items: list[NewsItem], pool: ThreadPoolExecutor) -> int:
        normalized = [self._normalize_source_subject(item) for item in items]
        claimed = [item for item in normalized if self._claim_item(item)]
        sent = 0
        # Complete all higher-priority source work before a lower-priority
        # source can claim its semantic slot. Within one source, send each
        # result as soon as it finishes instead of waiting for the slowest call.
        priorities = sorted({SOURCE_PRIORITY.get(item.source, 9) for item in claimed})
        for priority in priorities:
            group = [
                item for item in claimed if SOURCE_PRIORITY.get(item.source, 9) == priority
            ]
            futures: dict[Future, NewsItem] = {}
            for item in group:
                try:
                    futures[pool.submit(self._evaluate, item)] = item
                except RuntimeError as error:
                    self._release_item(item)
                    structured_log(
                        logging.ERROR,
                        "pipeline.submit_failed",
                        player=item.player_name,
                        error=str(error),
                    )
            for future in as_completed(futures):
                sent += self._evaluate_future(futures[future], future)
        return sent

    # ---------- immediate X dispatch ----------

    def _submit_tweet(self, item: NewsItem) -> None:
        if not self._claim_item(item):
            return
        try:
            future = self._twitter_pool.submit(self._evaluate, item)
        except RuntimeError as error:
            self._release_item(item)
            structured_log(
                logging.ERROR,
                "twitter.submit_failed",
                player=item.player_name,
                error=str(error),
            )
            return
        future.add_done_callback(lambda result, target=item: self._evaluate_future(target, result))

    def _twitter_dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._tweet_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._submit_tweet(item)

    def _start_twitter_dispatcher(self) -> None:
        if self._twitter_dispatcher is not None and self._twitter_dispatcher.is_alive():
            return
        self._twitter_dispatcher = threading.Thread(
            target=self._twitter_dispatch_loop,
            name="twitter-dispatch",
            daemon=True,
        )
        self._twitter_dispatcher.start()

    # ---------- poll ----------

    def poll_once(self) -> int:
        self._reload_roster_if_changed()
        sent = self.deliver_pending()

        items: list[NewsItem] = []
        feed_was_full = False
        modified = False
        try:
            items, feed_was_full, modified = self.poller.fetch(self.session)
            # Subject normalization must precede the first seen check.  Seen
            # state is recorded from the normalized item, so checking the raw
            # RotoWire title on later polls would otherwise make one unchanged
            # report look perpetually new.
            items = [self._normalize_source_subject(item) for item in items]
            HEALTH.mark(
                "rotowire",
                ok=True,
                detail=f"{len(items)} items; modified={modified}",
            )
        except requests.RequestException as error:
            HEALTH.mark("rotowire", ok=False, detail=str(error))
            structured_log(logging.WARNING, "rotowire.fetch_failed", error=str(error))
            # X is an independent primary source. An RSS outage must not stop
            # queued tweets from reaching the shared evaluation path.
        if not modified and self._tweet_queue.empty():
            return sent

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
        fresh.sort(key=lambda item: SOURCE_PRIORITY.get(item.source, 9))
        return sent + self._process_items(fresh, self._pool)

    # ---------- Telegram read-only controls ----------

    @staticmethod
    def _health_line(label: str, component) -> str:
        if component is None:
            return f"{label}: waiting for first check"
        state = "OK" if component.ok else "ERROR"
        last = age_label(component.last_success_at)
        detail = f" · {html.escape(component.detail, quote=False)}" if not component.ok and component.detail else ""
        return f"{label}: {state} · last success {last}{detail}"

    def status_text(self) -> str:
        """Operational health suitable for Telegram's /status command."""
        components = HEALTH.snapshot()
        with self._state_lock:
            preseason = self.preseason
            player_index = self._player_index
            snapshot = self.snapshot
        lines = [
            "<b>Fantasy notifier status</b>",
            f"Mode: {'preseason' if preseason else 'in-season'}",
            f"Uptime: {duration_label(time.time() - HEALTH.started_at)}",
            f"Pending deliveries: {len(self.outbox)}",
            f"Saved reports: {self.events.count()}",
            self._health_line("RotoWire", components.get("rotowire")),
            self._health_line("Model", components.get("model")),
        ]

        fantasypros = self.fantasypros.status()
        if not fantasypros.enabled:
            fp_label = (
                "disabled"
                if not fantasypros.ledger_trusted or not getattr(
                    self.config, "fantasypros_api_key", ""
                )
                else "stopped"
            )
            lines.append(f"FantasyPros cache: {fp_label}")
        else:
            last = (
                age_label(fantasypros.last_success_at.timestamp())
                if fantasypros.last_success_at is not None
                else "never"
            )
            state = "OK" if fantasypros.datasets_fresh else "WAITING"
            lines.append(
                f"FantasyPros cache: {state} · {len(fantasypros.datasets_fresh)} "
                f"fresh · {fantasypros.requests_used}/{fantasypros.request_cap} "
                f"requests/24h · last fetch {last}"
            )

        sleeper_stamp = getattr(player_index, "refreshed_at", None)
        if sleeper_stamp is not None:
            sleeper_state = "STALE" if getattr(player_index, "stale", False) else "OK"
            lines.append(
                f"Sleeper index: {sleeper_state} · refreshed "
                f"{age_label(sleeper_stamp.timestamp())}"
            )
        else:
            lines.append(self._health_line("Sleeper index", components.get("sleeper")))

        if snapshot.generated_at is not None:
            lines.append(
                f"Rosters: OK · refreshed {age_label(snapshot.generated_at.timestamp())}"
            )
        else:
            lines.append("Rosters: timestamp unavailable")

        if self.twitter is None:
            lines.append("X stream: disabled")
        else:
            x = self.twitter.health_snapshot()
            status = (
                "connected"
                if x["connected"]
                else "reconnecting"
                if x["alive"]
                else "stopped"
            )
            connected = age_label(float(x["last_connected_at"] or 0))
            lines.append(f"X stream: {status} · last connection {connected}")

        telegram_stamp = max(
            self.telegram_state.last_telegram_success,
            components.get("telegram").last_success_at
            if components.get("telegram") is not None
            else 0.0,
        )
        lines.append(f"Telegram: last success {age_label(telegram_stamp)}")
        lines.append(f"Bot controls: {self.telegram_control.status_label}")
        retention = self.telegram_control.auto_delete_seconds
        if retention is None:
            lines.append("Telegram auto-delete: not checked yet")
        elif retention <= 0:
            lines.append("Telegram auto-delete: disabled")
        elif retention % 86400 == 0:
            lines.append(f"Telegram auto-delete: {retention // 86400} days")
        else:
            lines.append(f"Telegram auto-delete: {retention // 3600} hours")
        return "\n".join(lines)

    def player_text(self, query: str) -> str:
        with self._state_lock:
            player_index = self._player_index
            snapshot = self.snapshot
        text = format_player_lookup(
            query,
            player_index,
            snapshot,
            refreshed_at=getattr(player_index, "refreshed_at", None),
        )
        recent = self.events.recent_for_player(query, limit=4)
        if not recent:
            return text
        lines = [text, "", "<b>Recent saved reports</b>"]
        for event in recent:
            severity = event.get("severity")
            prefix = f"{severity}/5 · " if severity is not None else ""
            direction = str(event.get("direction") or "unclassified")
            event_type = str(event.get("event_type") or "news").replace("_", " ")
            headline = html.escape(str(event.get("headline") or "")[:150], quote=False)
            lines.append(
                f"  {prefix}{html.escape(direction)} · {html.escape(event_type)} · {headline}"
            )
        return "\n".join(lines)

    def news_search_text(self, query: str) -> str:
        rows = self.events.search(query, limit=8)
        escaped = html.escape(query, quote=False)
        if not rows:
            return f"No saved reports matched <b>{escaped}</b>."
        lines = [f"<b>Saved reports matching {escaped}</b>"]
        for event in rows:
            player = html.escape(str(event.get("player_name") or "League news"), quote=False)
            severity = event.get("severity")
            rating = f"{severity}/5 · " if severity is not None else ""
            direction = html.escape(str(event.get("direction") or "unclassified"), quote=False)
            headline = html.escape(str(event.get("headline") or "")[:160], quote=False)
            lines += ["", f"<b>{rating}{player}</b> · {direction}", headline]
        return "\n".join(lines)

    # ---------- lifecycle ----------

    def check_roster_freshness(self) -> None:
        with self._state_lock:
            snapshot = self.snapshot
        if snapshot.generated_at is None:
            return
        age = (datetime.now(timezone.utc) - snapshot.generated_at).total_seconds() / 3600
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

    def close(self) -> None:
        """Stop background workers and close local resources."""
        self._stop.set()
        self.telegram_control.stop()
        self.fantasypros.close()
        if self._fantasypros_thread is not None:
            self._fantasypros_thread.join(timeout=2)
        if self.twitter is not None:
            self.twitter.stop()
        if self._twitter_dispatcher is not None:
            self._twitter_dispatcher.join(timeout=2)
        # Journal callbacks may still be running in either pool. Drain active
        # work before closing SQLite so shutdown cannot race a feedback/event
        # write against a closed connection.
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._twitter_pool.shutdown(wait=True, cancel_futures=True)
        self.session.close()
        try:
            self.events.close()
        except Exception:  # noqa: BLE001 - shutdown should remain best effort
            pass
        if self._process_lock_file is not None:
            try:
                fcntl.flock(self._process_lock_file.fileno(), fcntl.LOCK_UN)
                self._process_lock_file.close()
            finally:
                self._process_lock_file = None

    def request_stop(self) -> None:
        """Ask the main loop and background waiters to exit gracefully."""
        self._stop.set()

    def run_forever(self) -> None:
        structured_log(
            logging.INFO,
            "notifier.started",
            leagues=[f"{ref.label} ({ref.provider})" for ref in self.snapshot.leagues],
            myPlayerCount=len(self.snapshot.mine()),
            pollSeconds=self.config.poll_seconds,
            idlePollSeconds=self.config.poll_seconds_idle,
            model=self.config.openrouter_model,
            fantasyProsEnabled=self.fantasypros.enabled,
        )
        self._start_fantasypros_refresher()
        if (
            (self.config.telegram_controls_enabled or self.config.daily_digest_enabled)
            and not self.config.dry_run
        ):
            self.telegram_control.start()
        if self.twitter is not None and self.twitter.sync_rules():
            self._start_twitter_dispatcher()
            self.twitter.start()

        if self.preseason:
            structured_log(
                logging.INFO,
                "preseason.mode_active",
                minSeverity=PRESEASON_MIN_SEVERITY,
                maxRank=PRESEASON_MAX_RANK,
            )

        self.check_roster_freshness()
        last_freshness = time.time()
        # Schedule the next full Sleeper fetch from the represented data age,
        # not process start. Otherwise a 23-hour-old cache loaded at startup
        # would remain in use for almost 48 hours.
        with self._state_lock:
            player_index = self._player_index
        next_index_refresh = _next_player_index_refresh_at(player_index, time.time())

        while not self._stop.is_set():
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
            if now >= next_index_refresh:
                # Sleeper asks clients to fetch the full player map at most
                # once daily. Retain the old map on failure and retry after a
                # bounded delay rather than blanking player lookup for a day.
                if self._rebuild_depth_charts(reload_player_index=True):
                    with self._state_lock:
                        player_index = self._player_index
                    next_index_refresh = _next_player_index_refresh_at(player_index, now)
                else:
                    next_index_refresh = now + PLAYER_INDEX_RETRY_SECONDS

            self._stop.wait(max(1.0, self._interval() - (time.time() - started)))
