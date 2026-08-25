#!/usr/bin/env python3
"""Attach conservative urgency baselines to classified historical reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from notifier.embeddings import news_item_from_row  # noqa: E402
from notifier.event_store import (  # noqa: E402
    EventStore,
    archive_urgency_can_write,
    read_all_reports,
)
from notifier.urgency import POLICY_VERSION, archive_rule_urgency  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    state_dir = Path(os.environ.get("NOTIFIER_STATE_DIR", "").strip() or root / "state")
    saved = 0
    skipped = 0
    protected = 0
    unavailable = 0
    store = None if args.dry_run else EventStore(state_dir)
    try:
        rows = (
            read_all_reports(state_dir, limit=max(1, int(args.limit)))
            if args.dry_run
            else store.all_reports(limit=max(1, int(args.limit)))
        )
        for row in rows:
            if (
                not args.force
                and row.get("urgency_policy_version") == POLICY_VERSION
                and row.get("urgency_rule_level")
                and row.get("urgency_event_type")
                and row.get("urgency_action_context")
            ):
                skipped += 1
                continue
            urgency = archive_rule_urgency(row)
            if urgency is None:
                unavailable += 1
                continue
            if args.dry_run:
                if archive_urgency_can_write(row):
                    saved += 1
                else:
                    protected += 1
                continue
            if store is not None and store.record_archive_urgency(
                news_item_from_row(row), urgency
            ):
                saved += 1
            else:
                protected += 1
        print(
            f"complete: reports={len(rows)} saved={saved} skipped={skipped} "
            f"protected_live={protected} unclassified={unavailable} "
            f"policy={POLICY_VERSION} "
            f"dry_run={str(bool(args.dry_run)).lower()}"
        )
        return 0
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
