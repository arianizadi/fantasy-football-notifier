#!/usr/bin/env python3
"""Long-running news poll loop, intended to be managed by systemd."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from notifier.config import load_config  # noqa: E402
from notifier.logging_utils import NotifierError, configure_logging, structured_log  # noqa: E402
from notifier.pipeline import Notifier  # noqa: E402
from notifier.sources import rotowire  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit")
    parser.add_argument(
        "--prime",
        action="store_true",
        help="Mark everything currently in the feed as seen, then exit (avoids a backlog burst)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    configure_logging(verbose=args.verbose)

    try:
        config = load_config()
        notifier = Notifier(config)

        if args.prime:
            items, _ = rotowire.fetch(notifier.session)
            notifier.seen.prime(items)
            structured_log(logging.INFO, "notifier.primed", itemCount=len(items))
            return 0

        if args.once:
            sent = notifier.poll_once()
            notifier.seen.save()
            structured_log(logging.INFO, "notifier.poll_complete", alertsSent=sent)
            return 0

        notifier.run_forever()
    except NotifierError as error:
        structured_log(logging.ERROR, "notifier.config_error", error=str(error))
        return 1
    except KeyboardInterrupt:
        structured_log(logging.INFO, "notifier.stopped")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
