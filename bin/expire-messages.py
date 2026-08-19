#!/usr/bin/env python3
"""Delete Telegram messages this bot sent once they pass MESSAGE_TTL_HOURS.

Only ids recorded in state/sent-messages.json are touched, so the user's own
messages are never removed. Telegram refuses deletion past 48 hours; anything
that ages out is dropped from tracking instead of retried forever.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.config import load_config  # noqa: E402
from notifier.logging_utils import configure_logging, structured_log  # noqa: E402
from notifier.notify import delete_message, history  # noqa: E402


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    configure_logging()
    config = load_config()

    if config.message_ttl_hours <= 0:
        structured_log(logging.INFO, "expire.disabled")
        return 0

    store = history(config)
    deletable, too_old = store.due(config.message_ttl_hours * 3600)

    session = requests.Session()
    deleted = [mid for mid in deletable if delete_message(session, config, mid)]
    failed = [mid for mid in deletable if mid not in deleted]

    # Stop tracking anything handled or now undeletable.
    store.forget(deleted + too_old + failed)
    store.save()

    structured_log(
        logging.INFO,
        "expire.complete",
        ttlHours=config.message_ttl_hours,
        deleted=len(deleted),
        failed=len(failed),
        pastWindow=len(too_old),
        stillTracked=len(store),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
