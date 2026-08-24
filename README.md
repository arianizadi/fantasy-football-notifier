# fantasy-football-notifier

NFL news filtered through your fantasy leagues, delivered to Telegram with
event-aware roster and draft context. The service watches an optional X
reporter stream plus RotoWire RSS, joins each report to Sleeper's NFL player
data and depth chart, and checks ownership across ESPN and Sleeper leagues.

## Current pipeline

```text
X filtered stream ──> immediate work queue ─┐
                                            ├─> source-id dedupe
RotoWire RSS (adaptive polling) ────────────┘
                                                 │
                    player match + Sleeper depth/availability context
                                                 │
                    DeepSeek event classification (bounded parallel calls)
                                                 │
                    deterministic event-aware actions and safety rules
                                                 │
                    cached FantasyPros WAIVER/ROS context (optional)
                                                 │
                    cross-source semantic dedupe + severity/tier gate
                              │                  │
                    local news journal     retry-safe Telegram delivery
```

X uses a long-lived filtered-stream connection. Its queue wakes the processing
loop directly; it does not wait for the next RSS poll. RotoWire remains a
separate fallback source. Source IDs suppress literal repeats, while semantic
dedupe recognizes a later corroboration of the same player event so Telegram
can update the existing alert.

The model classifies the event type, fantasy direction, severity, impact
summary, and whether it is actionable. Code owns the depth chart, roster availability,
event label, and add/start/bench rules. Positive returns therefore cannot be
formatted as injuries or trigger a backup claim merely because the subject was
previously hurt.

## Alert tiers and events

| Tier | Meaning |
|---|---|
| YOUR ROSTER | You own the affected player. |
| WAIVER OPPORTUNITY | The event creates a claimable role in at least one drafted league. |
| RIVAL ROSTER | A rival owns the affected player. |
| LEAGUE NEWS | Relevant NFL news without a direct roster move. |
| PRESEASON - DRAFT IMPACT | Draft advice while configured leagues remain undrafted. |

Event direction is explicit: for example `RETURN`, `INJURY`, `OUT`, `TRADE`,
or `ROLE`. The event subject is highlighted in the depth chart, but an injury
tag appears only when the data or event supports it. Sleeper status is shown
separately from the news classification.

Example return alert:

```text
[4/5] PRESEASON — RETURN - George Kittle
The 49ers activated Kittle from active/PUP.

Model summary: Kittle's expected availability improved.

Draft note: Return news improves availability; confirm full practice and Week 1
status before adjusting draft value.

Backup watch: Jake Tonges is next in Sleeper's TE depth order. No pickup is
recommended from return news; injury or inactive alerts recheck league
availability.

SF TE DEPTH / BACKUP WATCH · SLEEPER
  refreshed 2026-08-23 10:45 PT
  TE1 George Kittle · Sleeper rank #91 · SUBJECT · RETURN
  TE2 Jake Tonges · Sleeper rank #178
  TE3 Luke Farrell · Sleeper rank #491

X source
```

Sleeper's `search_rank` is labelled as a Sleeper rank, not jersey number, ADP,
or positional rank. Context timestamps say when the cached player data was
refreshed. Reserve, taxi, IR, and NFL-inactive players are not offered as
immediately startable replacements.

## League and roster data

The snapshot stores every fantasy team's roster, not only yours. That makes
availability league-specific: a backup can be free in one league and owned in
another.

- ESPN is read directly from ESPN's fantasy league endpoint with the `mTeam`,
  `mRoster`, and `mSettings` views. Private-league authentication still uses
  `ESPN_SWID` and `ESPN_S2`; `ESPN_TEAM_ID` is an optional unambiguous override.
- Sleeper leagues are discovered from `SLEEPER_USERNAME`, optionally filtered
  by `SLEEPER_LEAGUE_IDS`.
- Sleeper's NFL player dataset supplies depth, team, rank, and live status.
- Each league's reception scoring is read from its provider (`PPR`, `HALF`, or
  `STD`) so optional FantasyPros context uses the right rankings per league.

`bin/refresh-roster.py` writes an atomic local snapshot twice a day, and
`bin/check-drafts.py` checks hourly for a newly completed draft. ESPN and
Sleeper drafts are tracked independently. Whenever an event would generate a
waiver recommendation, the notifier refreshes league ownership just in time
before emitting an `ADD` line so a recent rival claim is not presented as a
free agent. If that refresh fails, it fails closed: the news can still alert,
but all `ADD` and free-agent claims are hidden and the message says why.
Delayed outbox retries re-run the same ownership check.

Draft activation is league-specific. If ESPN has drafted while Sleeper is still
pre-draft, ESPN immediately receives full roster and waiver handling while the
empty Sleeper league is excluded from ownership and free-agent calculations.
Just-in-time refreshes skip providers that have no drafted league, so a
pre-draft Sleeper outage cannot suppress a valid ESPN pickup check. Sleeper is
added automatically after its own roster appears.

Each X post is handled as one report, even when it names several players. If
the notifier cannot confidently attribute an injury or absence to one player,
it keeps the news alert neutral and withholds automatic pickup and lineup
recommendations.

Sleeper's full NFL player map is cached for twenty-four hours, matching
Sleeper's documented once-daily guidance. It provides depth and status context,
not medical confirmation; breaking news is never delayed while waiting for a
fresh copy of the full map. A failed daily refresh keeps the last good map live
and retries after fifteen minutes.

### In-season injury pickups

This notifier is intentionally focused on the short window after an injury or
inactive report, while the replacement may still be available. It never adds
or drops a player automatically. When configured, FantasyPros consensus
WAIVER and rest-of-season rankings are added as a cached second opinion; they
do not generate candidates, establish workload, or determine availability.

Before showing a pickup option, the notifier refreshes ownership in every
drafted ESPN and Sleeper league. It keeps the nearest two players after the
affected player in Sleeper's depth order even when their overall search rank is
low. When more than one is free, they are presented as alternatives instead of
two commands to add both:

```text
[4/5] WAIVER OPPORTUNITY — INJURY

LEAGUE-SPECIFIC MOVES
Sunday Crew: PICKUP OPTIONS — Michael Carter (Sleeper depth RB2 · named in report
  · FantasyPros HALF waiver RB34 · ROS RB55) | Bam Knight (Sleeper depth RB3
  · named in report · FantasyPros HALF waiver RB41 · ROS RB63)
  Roster occupancy: Bench 5/5 full · IR 0/1 open (eligibility not checked)
  FantasyPros rank lean: Michael Carter (ranking context only; role unconfirmed)
  FantasyPros cached HALF rankings · provider updated 2026-08-23 10:00 PT;
  may lag this breaking report and do not confirm role or workload.

Backup note: Pickup options are alternatives, not instructions to add both.
Sleeper depth order does not confirm workload or touch share.
```

The alert path never calls the FantasyPros API. X remains the fast trigger,
Sleeper generates the nearest depth options, and a just-in-time ESPN/Sleeper
roster refresh determines whether each option is actually free. The background
cache downloads WAIVER and ROS for each scoring format currently used by a
drafted league every two hours. During the current mixed state, the drafted PPR
ESPN league needs two datasets, or 24 requests per day. Once the half-PPR
Sleeper league drafts, four datasets require 48 requests per day, regardless of
tweet volume. A persistent rolling-24-hour cap defaults to 425, leaving
headroom below the account's stated 500-request plan. Requests are globally
spaced by at least one second and reserved in the ledger before network I/O, so
restarts cannot forget quota use. Repeated failed batches back off from 15
minutes to a six-hour maximum instead of burning the daily budget.

FantasyPros freshness uses the provider's `last_updated_ts`, not local fetch
time. Data older than `FANTASYPROS_MAX_AGE_HOURS` is omitted. A missing key,
quota exhaustion, stale response, malformed payload, timeout, or provider
outage leaves the existing Sleeper-based alert unchanged and can never delay
or suppress it. A valid response that is empty or falls back to a different
ranking family is marked unavailable and never relabeled as WAIVER or ROS.
Displayed rankings are explicitly attributed to FantasyPros.

Bench and IR limits are read separately from each league's provider settings;
the current ESPN and Sleeper leagues can therefore both show `5` bench and `1`
IR without hard-coding those values globally. The IR number is occupancy only:
an open spot does not establish that the injured player is IR-eligible. The
notifier also does not choose a drop candidate for a full bench.

### Preseason mode

Preseason mode is active while no configured league has a roster for the user.
The classifier is explicitly told that the league is pre-draft, and action text
is draft advice rather than lineup instructions such as "activate" or "start."
Players inside Sleeper's top 250 generate alerts at severity 3/5 and above.
The notifier exits global preseason mode when the first provider roster
appears. If another league is still pre-draft, that league stays excluded from
roster actions until its own draft completes; already drafted leagues continue
without waiting for it.

## Delivery reliability

- An item is not terminally marked seen until its alert is delivered or a
  deliberate filter/dedupe decision is recorded.
- Telegram delivery failures remain in a persistent outbox and retry after a
  provider outage or process restart. Retries are labeled as delayed, preserve
  chronology, revalidate waiver availability, and suppress a narrow set of
  older absence reports that a newer delivered return has superseded.
- Transient model calls retry before falling back to a conservative,
  deterministic classification with high-signal severity floors.
- Unambiguous high-impact phrases have deterministic safety floors, so model
  failure cannot silently downgrade events such as a torn ACL or season-ending
  IR below the alert threshold.
- Telegram-bound HTML is escaped and secrets are redacted from structured logs.
- Local state files are written atomically. An exclusive state-directory lock
  enforces one notifier process for queue, Telegram, and dedupe decisions;
  Redis is not required.
- A later non-urgent corroboration for the same player, event, status, and
  condition edits the existing Telegram message within six hours instead of
  adding chat noise. Severity increases, status or condition changes, new
  recovery timetables, and different event types post a new alert so urgent
  transitions are not hidden inside a silent edit. Every raw report remains in
  the local journal either way.

## Saved news and feedback

Every newly claimed live X post or RotoWire report is saved locally in
`state/news-events.sqlite3`. Raw text, URL, matched player, source time, and
filter/delivery outcome are retained for every saved report. Event type,
positive/negative/mixed direction, severity, and summary are added when a
report reaches classification; preseason items rejected before classification
remain explicitly unclassified. Delivered rows also retain the Telegram
message id and useful/wrong/noisy feedback. Report identity includes the
source, source GUID, headline, and body, so an upstream edit under a reused
GUID keeps both revisions and their feedback separate while exact raw
duplicates still collapse. The database uses WAL mode and
full-text search, so `/player Kittle` can include recent reports. The
`/news Kittle` command searches the journal after Telegram's seven-day copies
have expired.

The schema has nullable `embedding_model` and `embedding` fields, but the
notifier does not send the archive to another embedding API by default.
Structured labels are the reliable inputs for alert decisions; embeddings are
useful later for similarity search once there is enough saved history and
feedback to evaluate a specific embedding model. This avoids adding cost and
another outage path before it provides measurable value.

## Telegram setup and seven-day history

Create the bot with BotFather, send it one message, then run:

```bash
./.venv/bin/python bin/setup-telegram.py
```

For a seven-day temporary chat, set Telegram's native timer in the Telegram UI:

1. Open the bot chat and tap the bot name.
2. Open **More** / **Auto-Delete**.
3. Choose **1 week**.

Telegram applies the setting to new messages after the timer is enabled. The
bot cannot configure this native chat timer through the Bot API. Retention is
therefore a manual per-chat Telegram setting, not a notifier environment
variable.

## Setup

```bash
cp .env.example .env
chmod 600 .env
python3 -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python bin/refresh-roster.py
./.venv/bin/python bin/setup-telegram.py
./.venv/bin/python bin/run-notifier.py --prime
./.venv/bin/python bin/run-notifier.py --once --verbose
```

`--prime` records the current feed before the daemon starts, preventing a burst
of old alerts. Do not enable `ESPN_DEBUG`; raw ESPN payloads contain private
member data. `DRY_RUN=true` keeps the event journal in memory and skips roster,
Sleeper-cache, outbox, Telegram-state, and dedupe writes.

Important configuration:

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram destination. |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | Pinned classifier provider/model. |
| `ESPN_LEAGUE_ID`, `ESPN_YEAR`, `ESPN_SWID`, `ESPN_S2` | Private ESPN league access. |
| `ESPN_TEAM_ID` | Optional team-selection override. |
| `SLEEPER_USERNAME`, `SLEEPER_LEAGUE_IDS` | Sleeper league discovery/filter. |
| `TWITTER_BEARER_TOKEN` | Optional usage-billed X filtered stream. |
| `FANTASYPROS_API_KEY` | Optional cached FantasyPros WAIVER/ROS context; never used in the breaking-alert path. |
| `FANTASYPROS_REQUEST_LIMIT` | Persistent rolling-24h application cap; default `425`, maximum `450`. |
| `FANTASYPROS_REFRESH_HOURS`, `FANTASYPROS_MAX_AGE_HOURS` | Ranking refresh cadence and provider-data freshness limit. |
| `MIN_SEVERITY`, `MIN_SEVERITY_OTHER` | Alert floors for your roster vs other news. |
| `POLL_SECONDS`, `POLL_SECONDS_IDLE`, `ADAPTIVE_POLLING` | RotoWire polling cadence. |
| `TELEGRAM_CONTROLS_ENABLED` | Opt in to commands and feedback only when this service exclusively owns the bot's `getUpdates`; default `false`. |
| `PLAYER_THREAD_HOURS` | Reply-chain lifetime; default `168` matches one week. |
| `DAILY_DIGEST_ENABLED`, `DAILY_DIGEST_HOUR`, `DAILY_DIGEST_TIMEZONE` | Scheduled digest settings. |
| `DRY_RUN` | Print alerts without sending them. |

X pricing and quotas can change. Verify current terms before enabling the
stream; `bin/measure-reporters.py` can measure configured account volume.

## Tests and dependency audit

```bash
./.venv/bin/python -m compileall -q notifier bin
./.venv/bin/python -m pytest -q
./.venv/bin/python -m pip_audit -r requirements-dev.txt --strict
```

GitHub Actions runs all three checks on pushes and pull requests. Runtime HTTP
dependencies are pinned to patched releases, and the ESPN adapter is tested
against representative private-league payloads including owner matching,
explicit team selection, lineup slots, and rival rosters.

## Runner deployment

Production is the `arian` user's git checkout at:

```text
/home/arian/services/fantasy-news-notifier
```

Redeploy from a workstation with:

```bash
ssh Runner
~/services/fantasy-news-notifier/deploy/redeploy.sh
```

The script backs up `.env` and `state/`, updates to `origin/main`, installs
dependencies, runs tests, restarts `fantasy-news-notifier`, and verifies that
the unit is active. The credential archive is mode `0600`, and the SQLite
journal is captured through the online backup API so its WAL snapshot is
consistent. The script does not replace or print credentials.

First-time systemd install:

```bash
cd /home/arian/services/fantasy-news-notifier
sudo cp deploy/fantasy-news-notifier.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fantasy-news-notifier
journalctl -t fantasy-news-notify -f
```

The committed service and cron templates use the same Runner user and checkout
path as production. Roster refresh runs at `03:15` and `15:15` UTC; draft-state
checks run hourly at minute 7.

## Bot controls

The long-running systemd service can handle Telegram control updates in a
separate long-poll thread. Telegram permits only one `getUpdates` consumer per
bot token. Because the current chat may also be managed by OpenClaw, controls
default to disabled. Keep them disabled for a shared token, or use a dedicated
notifier bot before setting `TELEGRAM_CONTROLS_ENABLED=true`. One-shot commands
such as `run-notifier.py --once` do not keep that control loop alive.

- `/status` reports source health, pending delivery count, and roster/cache
  refresh times.
- `/player <name>` shows current Sleeper status and depth plus ownership in
  each drafted league and recent saved reports.
- `/news <query>` searches the saved tweet/report journal.
- `/digest` returns a severity-sorted summary of the last 24 hours. The same
  digest is sent automatically at the configured local hour when enabled; its
  outbound scheduler does not consume Telegram updates and still runs when bot
  controls are disabled.
- Feedback buttons under each alert record **Useful**, **Wrong**, or **Too
  noisy** for later tuning. Buttons are shown only when controls are enabled.
- New alerts for the same player reply to the previous alert while it remains
  inside `PLAYER_THREAD_HOURS`, creating a compact per-player update chain.
