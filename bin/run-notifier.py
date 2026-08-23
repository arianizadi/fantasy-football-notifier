#!/usr/bin/env python3
"""Long-running news poll loop, intended to be managed by systemd."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from notifier.config import load_config  # noqa: E402
from notifier.logging_utils import NotifierError, configure_logging, structured_log  # noqa: E402
from notifier.pipeline import Notifier  # noqa: E402
from notifier.sources import rotowire  # noqa: E402


def _install_sigterm_handler(notifier: Notifier):
    """Translate systemd's stop signal into the notifier's graceful stop path."""
    def handle(signum, _frame) -> None:
        structured_log(
            logging.INFO,
            "notifier.stop_requested",
            signal=signal.Signals(signum).name,
        )
        notifier.request_stop()

    return signal.signal(signal.SIGTERM, handle)


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

    notifier = None
    try:
        config = load_config()
        notifier = Notifier(config)

        if args.prime:
            items, _ = rotowire.fetch(notifier.session)
            if config.dry_run:
                structured_log(
                    logging.INFO,
                    "notifier.prime_preview",
                    itemCount=len(items),
                )
            else:
                notifier.seen.prime(items)
                structured_log(logging.INFO, "notifier.primed", itemCount=len(items))
            return 0

        if args.once:
            sent = notifier.poll_once()
            if not config.dry_run:
                notifier.seen.save()
            structured_log(logging.INFO, "notifier.poll_complete", alertsSent=sent)
            return 0

        previous_sigterm = _install_sigterm_handler(notifier)
        try:
            notifier.run_forever()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    except NotifierError as error:
        structured_log(logging.ERROR, "notifier.config_error", error=str(error))
        return 1
    except KeyboardInterrupt:
        structured_log(logging.INFO, "notifier.stopped")
        return 0
    finally:
        if notifier is not None:
            notifier.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
