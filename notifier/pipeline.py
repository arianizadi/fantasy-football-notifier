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
from .dedupe import SeenStore, event_status, fingerprint
from .event_store import EventStore
from .health import HEALTH, age_label, duration_label
from .logging_utils import NotifierError, structured_log
from .matcher import name_from_rotowire_url
from .models import Alert, NewsItem, RosterSnapshot
from .notify import retry_after_seconds, send_alert, send_plain, telegram_state
from .outbox import DeliveryOutbox, PendingDelivery
from .player_lookup import format_player_lookup
from .plays import DepthCharts, LeaguePlays, plays_context_for_model, plays_for_event
from .roster import load_snapshot, refresh_snapshot, snapshot_mtime
from .sources import rotowire, sleeper
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


def _next_player_index_refresh_at(player_index: dict, now: float) -> float:
    """Schedule daily success refreshes and short degraded-state retries."""
    if not player_index or bool(getattr(player_index, "stale", False)):
        return now + PLAYER_INDEX_RETRY_SECONDS
    refreshed_at = getattr(player_index, "refreshed_at", None)
    if refreshed_at is None:
        return now + PLAYER_INDEX_REFRESH_SECONDS
    return refreshed_at.timestamp() + PLAYER_INDEX_REFRESH_SECONDS

# A newer delivered clearance/return makes an older queued absence report
# misleading. These narrow transitions are safe to suppress automatically;
# unrelated follow-ups still deliver in chronological order.
SUPERSEDING_EVENTS = {
    "injury": frozenset({"return"}),
    "inactive": frozenset({"return"}),
    "practice_report": frozenset({"inactive", "return"}),
    "suspension": frozenset({"return"}),
}

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
        self._state_lock = threading.RLock()
        self._delivery_lock = threading.RLock()
        self._inflight_guids: set[str] = set()
        self._inflight_fingerprints: set[str] = set()
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
        """Refresh all league ownership before evaluating a waiver candidate."""
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
            # every simultaneous breaking-news worker hammer both providers.
            self._last_jit_roster_refresh = now
            try:
                snapshot = refresh_snapshot(self.config)
            except Exception as error:  # noqa: BLE001 - keep the alert path alive
                HEALTH.mark("roster", ok=False, detail=str(error))
                structured_log(logging.WARNING, "roster.jit_refresh_failed", error=str(error))
                return False

            with self._state_lock:
                self.snapshot = snapshot
                self._snapshot_mtime = snapshot_mtime(self.config)
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
                item.guid,
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
                subject_names=names, snapshot=snapshot
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

        per_league = plays_for_event(
            per_league,
            classification.event_type,
            classification.severity,
        )
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
        digest = fingerprint(item)
        with self._state_lock:
            if not self.seen.is_new(item) or self.outbox.contains_item(item):
                return False
            if item.guid in self._inflight_guids or digest in self._inflight_fingerprints:
                return False
            self._inflight_guids.add(item.guid)
            self._inflight_fingerprints.add(digest)
        self._journal_received(item)
        return True

    def _release_item(self, item: NewsItem) -> None:
        with self._state_lock:
            self._inflight_guids.discard(item.guid)
            self._inflight_fingerprints.discard(fingerprint(item))

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
        )

    def _record_success(self, alert: Alert) -> bool:
        classification = alert.classification
        self.seen.record(alert.item)
        self.seen.record_semantic(
            alert.item.player_name,
            classification.event_type,
            classification.severity,
            event_status(alert.item, classification.event_type),
        )
        return self.seen.save()

    def _pending_is_superseded(self, pending: PendingDelivery) -> bool:
        """Whether a newer delivered state transition makes this retry stale."""
        alert = pending.alert
        superseding = SUPERSEDING_EVENTS.get(alert.classification.event_type, frozenset())
        if not superseding or not alert.item.player_name or not hasattr(self, "events"):
            return False
        try:
            rows = self.events.recent_for_player(alert.item.player_name, limit=12)
        except Exception as error:  # noqa: BLE001 - delivery must remain available
            structured_log(logging.WARNING, "delivery.supersession_check_failed", error=str(error))
            return False
        for row in rows:
            if row.get("guid") == alert.item.guid or row.get("outcome") != "delivered":
                continue
            try:
                received = datetime.fromisoformat(str(row.get("received_at") or ""))
                if received.tzinfo is None:
                    received = received.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if (
                received.timestamp() > pending.queued_at
                and str(row.get("event_type") or "") in superseding
            ):
                return True
        return False

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
        record, per_league = depth.build(subject_names=names, snapshot=snapshot)
        per_league = plays_for_event(
            per_league,
            alert.classification.event_type,
            alert.classification.severity,
        )
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
    ) -> int:
        """Try one persisted alert. Caller serializes delivery decisions."""
        if pending.attempts > 0:
            if self._pending_is_superseded(pending):
                self.seen.record(pending.alert.item)
                if self.seen.save():
                    self.outbox.remove(pending.delivery_id)
                self._journal_outcome(
                    pending.alert.item,
                    "suppressed_superseded",
                    tier=pending.alert.tier,
                )
                return 0
            pending.alert = self._revalidate_delayed_alert(pending)

        alert = pending.alert
        if not force_semantic and not self._semantic_is_new(alert):
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
        if self._record_success(alert):
            self.outbox.remove(pending.delivery_id)
        self._journal_outcome(
            alert.item,
            "delivered",
            tier=alert.tier,
            message_id=message_id,
        )
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
                if not self._semantic_is_new(alert):
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
                    pending = self.outbox.add(alert)
                except (OSError, RuntimeError) as error:
                    structured_log(logging.ERROR, "outbox.enqueue_failed", error=str(error))
                    return 0
                self._journal_outcome(item, "queued", tier=alert.tier)
                return self._attempt_pending_locked(pending)
        finally:
            self._release_item(item)

    def deliver_pending(self) -> int:
        """Retry due outbox entries before processing new feed work."""
        if self.config.dry_run:
            return 0
        sent = 0
        for pending in self.outbox.due():
            with self._delivery_lock:
                sent += self._attempt_pending_locked(pending)
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
        claimed = [item for item in items if self._claim_item(item)]
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
        )
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
            send_plain(
                self.session,
                self.config,
                "<b>Preseason mode.</b> Until your leagues draft you will only "
                f"get severity {PRESEASON_MIN_SEVERITY}+ news about draft-relevant "
                "players. Switches to full roster alerts automatically after a draft.",
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
