#!/usr/bin/env python3
"""Compatibility no-op; Telegram's native chat timer owns message retention."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifier.logging_utils import configure_logging, structured_log  # noqa: E402


def main() -> int:
    configure_logging()
    structured_log(
        logging.INFO,
        "expire.disabled",
        reason="Telegram native chat auto-delete owns retention",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
