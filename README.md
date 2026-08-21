# fantasy-football-notifier

Near-real-time NFL news, filtered to your fantasy rosters across **ESPN and
Sleeper**, delivered to Telegram with the actual waiver move spelled out.

Replaces having push notifications on for two dozen reporters. It does not beat
X's push latency — it beats X's **noise**, and it tells you what to *do*.

## Pipeline

```
X filtered stream (server push)  ─┐
RotoWire RSS (15s / 60s adaptive)─┤
                                  ├─> guid dedupe
                                  │
              plays engine: NFL depth chart x every roster in every league
                                  │
              classify (Flash, reasoning OFF, 8-way parallel)  ~2s
                                  │
              semantic dedupe (player + event, cross-source)
                                  │
              tier + severity gate ──> Telegram alert
                                  │
              second opinion (background) ──> threaded reply
```

**X is the primary source.** Schefter tweets; RotoWire writes it up 1–5 minutes
later. Items are sorted by source priority so the faster source claims the
semantic-dedupe slot and the slower duplicate is suppressed — not the reverse.

The X filtered stream is a **server push** (long-lived chunked HTTP connection),
not polling. There is no webhook to register; the streaming connection *is* the
callback. Pay-per-use permits exactly one concurrent connection, so it is a
singleton with exponential-backoff reconnect.

## Tiers

| Tier | Meaning |
|---|---|
| YOUR ROSTER | you own the player |
| WAIVER OPPORTUNITY | the news frees a claimable backup |
| RIVAL ROSTER | a rival owns the player |
| LEAGUE NEWS | general NFL news |
| PRESEASON - DRAFT IMPACT | pre-draft only |

Messages are plain text, no emoji. Severity leads as `[n/5]` so the Telegram
preview is scannable without decoding symbols. Severity 1-2 arrives silently;
3+ buzzes the phone.

```
[5/5] YOUR ROSTER - Lamar Jackson
Ruled out for Sunday with a knee injury

Bench Jackson; start your backup.

ESPN: ADD Tyler Huntley (QB2) | start Jayden Daniels
SLEEPER: ADD Tyler Huntley (QB2)

source
```

The upstream report body is deliberately omitted - it restates the headline,
and the link carries the detail. Waiver suggestions are suppressed below
severity 3, so a "limited in practice" note never tells you to burn a claim.

## What makes it actionable

Three things joined together:

1. **NFL depth charts** (Sleeper, free) — who is next up
2. **Every team's roster in every league** — whether that backup is claimable
3. **Your roster** — whether you already cover the hole internally

(2) is why the snapshot stores all rosters, not just yours. Availability is
computed **per league**: the same backup can be free in Sleeper and rostered in
ESPN, so each league gets its own verdict in the same alert.

Depth charts are computed in plain code, never asked of the model — a
hallucinated depth chart would produce confidently wrong waiver advice.

## Model choice

One model classifies, and nothing reviews it. A second and third opinion were
built and removed: on a season-ending IR for a Sleeper rank-126 player,
`deepseek-v4-pro` returned **1/5** ("changes nothing for 99% of rosters"),
which is badly wrong, and on an earlier labelled set it scored 9/12 against
Flash's 11/12. Paying a slower model to be less accurate is not a second
opinion, and three severities on one alert is noise, not signal.

The evaluation tooling stays, because the question "is there a better model
for the fast path" is worth re-asking:

```bash
bin/capture-fixtures.py --tweets 60 --rotowire   # store real items for replay
bin/eval-models.py --graded-only --models a,b    # score against ground truth
bin/eval-models.py --free                        # survey free candidates
```

Fixtures are hand-graded on fantasy CONSEQUENCE = player value x event
magnitude, which is the distinction models get wrong: a season-ending injury
to a WR5 and to a WR1 are both "season-ending" but are not the same fantasy
event. Scoring reports exact match and mean absolute error against those
grades, not agreement with another model - agreement only shows which models
are similar, not which are right.

Structure compliance is scored first and weighted hardest. A free model that
returns prose 30% of the time is worse than useless: the failure is silent,
the classifier falls back to severity 3, and the alert reads as if it were
judged. A survey of 12 free candidates found exactly one usable; the rest
404'd, 429'd, or could not parse. Two Lyria **audio** models were scored
before a modality filter was added, because they advertise `response_format`.

## Measured findings

**Reasoning tokens must be disabled.** DeepSeek v4 emits them by default; on
this task that burned 1,500+ tokens, returned *no content at all*, and took
**66s per call**. `"reasoning": {"enabled": false}` is 37× faster and 11×
cheaper for identical output. This is the single most important setting here.

**Model comparison** on a 12-headline labelled set, reasoning off:

| | in-band | latency | cost |
|---|---|---|---|
| flash-0731 | 9–11/12 | 5.3s | $0.00026 |
| flash + reasoning | 9/12 | 20.0s | $0.00092 |
| pro | 9/12 | ~4s | $0.00382 |

All three score the same. Flash runs the fast path (cheapest, no worse). Pro is
the verifier **not** because it is smarter but because it is a different model —
same-model-with-reasoning has correlated errors, so it makes a poor second
opinion.

**A deterministic severity floor** catches the model under-rating unambiguous
news ("torn ACL", "placed on IR", "suspended"). Verification only runs on alerts
that were *sent*, so an under-rated item would otherwise be dropped silently and
never second-guessed.

## X API cost

Handles verified and volume measured over a real 7-day window
(`bin/measure-reporters.py`). Reads bill at $0.005/post.

| | posts/day | $/month | $/season |
|---|---|---|---|
| 10 shipped accounts | 65 | **$9.75** | ~$49 |
| + all 15 candidates | 144 | $21.66 | ~$108 |

Deliberately excluded: `RotoWireNFL` (50.6/day, $7.59/mo) posts *identical*
items to the RSS feed this project already polls for free — verified against
live data. `Rotoworld_FB` overlaps similarly. `underdog__nfl`,
`FantasyPtsNFL`, and `JJZachariason` do not resolve on X at all; guessed
handles stream silently and are worse than useless.

## Latency

Measured on the deployment host (`n=6` classification, `n=3` Telegram):

| leg | median |
|---|---|
| player extraction from tweet text | 0.26 ms |
| classification (Flash, throughput-pinned) | 1,333 ms |
| message formatting | 0.01 ms |
| Telegram `sendMessage` | 299 ms |
| **our pipeline total** | **1.63 s** |

**Not measured:** X's own delivery lag (tweet posted → arrival on our socket).
A 15-minute stream window in preseason produced zero tweets. The stream now
logs `twitter.delivery_lag` on every tweet, so real numbers accumulate on their
own — check with `journalctl -t fantasy-news-notify | grep delivery_lag`.

Realistic end-to-end is therefore **~4-6s from tweet to phone buzz**, of which
1.63s is ours and the rest is X delivery plus Telegram push. That is comparable
to a native X push notification — the gain here is filtering and the play
recommendation, not raw speed.

**Provider routing is pinned.** OpenRouter's default sprayed across 6 providers
with a 8,890 ms tail. `{"sort": "throughput"}` holds max at 1,521 ms — a 5.8x
better worst case, which matters more than median for breaking news.

## Preseason mode

Engages automatically whenever no rostered players exist, and disengages
itself once a draft is detected (it sends a message when it flips). Instead of
roster tiering it asks "would this change how I draft?": severity 4+ **and** a
Sleeper overall rank inside the top 250. Against a live preseason feed this
correctly dropped all five items ("minor hamstring", "says he's fine", "suited
up for practice") while a torn ACL fired at 5/5.

## Draft handling

ESPN and Sleeper drafts can happen on different dates. Configure the provider
credentials once; the hourly draft-state check refreshes the roster snapshot
when a configured draft completes.

Two automated layers cover scheduled and moved drafts:

1. **Normal cron** (03:15 / 15:15 UTC) covers routine roster changes.
2. **`bin/check-drafts.py`, hourly** asks each provider for the real draft state
   and syncs when one completes. This survives a commissioner moving the draft
   and tracks each league independently when drafts occur on different dates.

When a draft is detected the notifier Telegrams you and flips itself out of
preseason mode.

## Design notes

- **RotoWire's feed holds only 5 items** regardless of any `count` parameter,
  and sends no `ETag`/`Last-Modified` (`no-store`), so conditional GET never
  yields a 304. When every item in a poll is new, it logs
  `rotowire.possible_overflow` rather than silently dropping news.
- **Adaptive polling**: 15s inside NFL news windows (US/Eastern via `zoneinfo`),
  60s otherwise.
- **Providers are off the hot path.** `bin/refresh-roster.py` runs on cron; the
  loop reads the snapshot and hot-reloads on mtime change. ESPN request volume
  stays in line with the existing `espn-sync` worker.
- **Classification fans out 8 ways.** It is ~2s of network wait, so a burst
  should parallelise; the alert decision is then serialised so the semantic
  dedupe check and its write cannot interleave.
- **No Redis.** State is a few hundred KB written atomically via `os.replace`
  and read by one process. A daemon would add a failure mode to replace three
  JSON files that are not a bottleneck.

## Setup

```bash
cp .env.example .env && chmod 600 .env
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python bin/refresh-roster.py            # prints every league's roster
./.venv/bin/python bin/setup-telegram.py            # after messaging your bot
./.venv/bin/python bin/measure-reporters.py         # verify + price X accounts
./.venv/bin/python bin/run-notifier.py --prime      # avoid a backlog burst
./.venv/bin/python bin/run-notifier.py --once --verbose
```

### Tests

```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest -q
```

GitHub Actions runs the test suite and bytecode compilation on every push and
pull request without requiring provider credentials.

### systemd deployment

Runner is a git checkout of this repo at
`/home/arian/services/fantasy-news-notifier`, tracking `origin/main`.
Redeploys are a pull, not a file copy:

```bash
ssh Runner
~/services/fantasy-news-notifier/deploy/redeploy.sh
```

That fetches, resets to `origin/main`, installs deps, runs the tests, and
restarts the unit. `.env`, `state/*.json` and `.venv/` are gitignored, so the
hard reset leaves secrets, dedupe state and the roster snapshot untouched.

First-time install:

```bash
sudo cp deploy/fantasy-news-notifier.service /etc/systemd/system/
sudo systemctl enable --now fantasy-news-notifier
journalctl -t fantasy-news-notify -f
```

## Tuning

| Variable | Effect |
|---|---|
| `MIN_SEVERITY` | Floor for your own players (default 2). |
| `MIN_SEVERITY_OTHER` | Floor for rival/league news (default 3). |
| `POLL_SECONDS` / `POLL_SECONDS_IDLE` | Active vs idle RSS cadence. |
| `VERIFY_ENABLED` / `VERIFY_MODEL` | Second opinion on/off and which model. |
| `TWITTER_BEARER_TOKEN` | Blank disables the X stream entirely. |
| `DRY_RUN` | Print alerts instead of sending. |

## Status

Verified against the live RotoWire feed, live Sleeper depth charts (77% coverage
of active QB/RB/WR/TE), configured ESPN and Sleeper leagues, OpenRouter, and an
X filtered-stream connection with rules synced and tweets parsed.

Roster filtering is inert until your leagues draft — both are `pre_draft` and
return empty rosters, and the notifier says so on startup. Re-run
`bin/refresh-roster.py` after each draft.
