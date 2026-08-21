#!/usr/bin/env python3
"""Score candidate models on stored fixtures and record the best.

Free models - especially unreleased stealth ones - vary wildly in whether they
can hold a JSON schema. A model that is fast and free but returns prose 30% of
the time is worse than useless, because the failure is silent: the classifier
falls back to severity 3 and the alert reads as if it were judged.

So structure compliance is scored first and weighted hardest. Agreement with
the reference model is only measured on responses that parsed.

Writes state/model-scores.json, which VERIFY_MODEL=auto reads.

    bin/eval-models.py --free            # all free candidates
    bin/eval-models.py --stealth         # stealth namespace only
    bin/eval-models.py --models a,b,c
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notifier.classify import _extract_json  # noqa: E402
from notifier.openrouter import chat  # noqa: E402
from notifier.config import load_config  # noqa: E402
from notifier.logging_utils import configure_logging  # noqa: E402
from notifier.model_registry import free_candidates  # noqa: E402

FIXTURES = ROOT / "fixtures" / "news-items.json"
SCORES = ROOT / "state" / "model-scores.json"
REQUEST_TIMEOUT = 90

SYSTEM_PROMPT = """You classify NFL news for a fantasy football manager.
Return ONLY JSON: {"severity":int,"event_type":str,"fantasy_impact":str}
severity 1-5 by FANTASY CONSEQUENCE, not dramatic wording:
1=noise 2=worth knowing 3=notable 4=major 5=season-defining
fantasy_impact under 100 characters."""

# A verifier that cannot hold the schema is unusable regardless of judgement.
MIN_SCHEMA_RATE = 0.90


def ask(session, key, model, item, timeout=REQUEST_TIMEOUT):
    prompt = (
        f"Player: {item.get('player_name') or 'unknown'}\n"
        f"Headline: {item.get('headline')}\n"
        f"Report: {(item.get('body') or '')[:600]}"
    )
    started = time.perf_counter()
    try:
        response = chat(
            session,
            key,
            model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            timeout=timeout,
        )
    except requests.RequestException as error:
        return {"ok": False, "reason": type(error).__name__, "latency": time.perf_counter() - started}

    latency = time.perf_counter() - started
    if not response.ok:
        detail = ""
        try:
            detail = str(response.json().get("error", {}).get("message", ""))[:90]
        except ValueError:
            detail = response.text[:90]
        return {"ok": False, "reason": f"http_{response.status_code}", "detail": detail,
                "latency": latency}

    body = response.json()
    cost = (body.get("usage") or {}).get("cost", 0) or 0
    content = body["choices"][0]["message"].get("content")
    if not content:
        return {"ok": False, "reason": "empty_content", "latency": latency, "cost": cost}
    try:
        parsed = _extract_json(content)
        severity = int(parsed.get("severity"))
    except (ValueError, TypeError, KeyError):
        return {"ok": False, "reason": "unparseable", "latency": latency, "cost": cost}
    if not 1 <= severity <= 5:
        return {"ok": False, "reason": "severity_out_of_range", "latency": latency, "cost": cost}
    return {"ok": True, "severity": severity, "latency": latency, "cost": cost}


def evaluate(session, key, model, items, reference):
    schema_ok = 0
    latencies, costs, agree, reasons, exact, errors = [], [], [], {}, [], []
    for item in items:
        result = ask(session, key, model, item)
        latencies.append(result["latency"])
        costs.append(result.get("cost", 0))
        if result["ok"]:
            schema_ok += 1
            expected = item.get("expected_severity")
            baseline = expected if expected is not None else reference.get(item["guid"])
            if baseline is not None:
                agree.append(abs(result["severity"] - baseline) <= 1)
            if expected is not None:
                exact.append(result["severity"] == expected)
                errors.append(abs(result["severity"] - expected))
        else:
            reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
    total = len(items)
    return {
        "model": model,
        "n": total,
        "schema_rate": schema_ok / total if total else 0.0,
        "agreement": (sum(agree) / len(agree)) if agree else None,
        "exact_match": (sum(exact) / len(exact)) if exact else None,
        "mean_abs_error": (sum(errors) / len(errors)) if errors else None,
        "median_latency": statistics.median(latencies) if latencies else None,
        "total_cost": sum(costs),
        "failures": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free", action="store_true")
    parser.add_argument("--stealth", action="store_true")
    parser.add_argument("--models", default="")
    parser.add_argument("--limit", type=int, default=15, help="fixtures per model")
    parser.add_argument("--graded-only", action="store_true",
                        help="score only against hand-graded ground truth")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    configure_logging()
    config = load_config()
    session = requests.Session()

    if not FIXTURES.exists():
        print("no fixtures; run bin/capture-fixtures.py first")
        return 1
    items = json.loads(FIXTURES.read_text())
    items = [i for i in items if i.get("headline")]
    # Hand-graded fixtures first: agreement with a reference model only shows
    # which models are similar, not which are right.
    graded = [i for i in items if i.get("expected_severity") is not None]
    if args.graded_only:
        items = graded
    else:
        items = graded + [i for i in items if i.get("expected_severity") is None]
    items = items[: args.limit]
    print(f"fixtures: {len(items)} ({sum(1 for i in items if i.get('expected_severity') is not None)} graded)")

    candidates = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.free or args.stealth:
        candidates += [m.model_id for m in free_candidates(session, stealth_only=args.stealth)]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        print("no candidates; pass --free, --stealth or --models")
        return 1

    # Reference severities from the production fast path, used to grade any
    # fixture that has not been hand-labelled.
    print(f"reference: {config.openrouter_model}")
    reference = {}
    for item in items:
        result = ask(session, config.openrouter_api_key, config.openrouter_model, item)
        if result["ok"]:
            reference[item["guid"]] = result["severity"]
    print(f"  reference severities: {len(reference)}/{len(items)}\n")

    results = []
    for model in candidates:
        print(f"evaluating {model} ...", flush=True)
        row = evaluate(session, config.openrouter_api_key, model, items, reference)
        results.append(row)
        agree = f"{row['agreement']:.0%}" if row["agreement"] is not None else "n/a"
        exact = f"{row['exact_match']:.0%}" if row["exact_match"] is not None else "n/a"
        mae = f"{row['mean_abs_error']:.2f}" if row["mean_abs_error"] is not None else "n/a"
        lat = f"{row['median_latency']:.1f}s" if row["median_latency"] else "n/a"
        print(f"  schema {row['schema_rate']:.0%}  exact {exact}  within1 {agree}  "
              f"MAE {mae}  {lat}  ${row['total_cost']:.5f}  fails={row['failures']}")

    usable = [r for r in results if r["schema_rate"] >= MIN_SCHEMA_RATE]
    usable.sort(key=lambda r: (r["mean_abs_error"] if r["mean_abs_error"] is not None else 9,
                               -(r["agreement"] or 0), r["median_latency"] or 999))
    best = usable[0]["model"] if usable else None

    SCORES.parent.mkdir(parents=True, exist_ok=True)
    SCORES.write_text(json.dumps({
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "reference_model": config.openrouter_model,
        "fixture_count": len(items),
        "min_schema_rate": MIN_SCHEMA_RATE,
        "best": best,
        "results": results,
    }, indent=1))

    print(f"\nusable (schema >= {MIN_SCHEMA_RATE:.0%}): {[r['model'] for r in usable]}")
    print(f"best: {best or 'NONE - keeping the configured verifier'}")
    print(f"written: {SCORES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
