#!/usr/bin/env python3
"""Refresh cached ESPN/Sleeper rosters and capacity. Run from cron."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from notifier.config import load_config  # noqa: E402
from notifier.logging_utils import NotifierError, configure_logging, structured_log  # noqa: E402
from notifier.roster import refresh_snapshot  # noqa: E402


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    configure_logging()
    try:
        config = load_config()
        snapshot = refresh_snapshot(config)
    except NotifierError as error:
        structured_log(logging.ERROR, "roster.refresh_failed", error=str(error))
        return 1
    except Exception as error:  # noqa: BLE001
        structured_log(
            logging.ERROR,
            "roster.refresh_crashed",
            error=str(error),
            errorType=type(error).__name__,
        )
        return 1

    for league in snapshot.leagues:
        mine = snapshot.mine(league.key)
        print(f"\n{league.provider.upper()}  {league.name}  —  {league.my_team_name}  "
              f"({len(mine)} players)")
        capacity = snapshot.capacities.get(league.key)
        scoring_format = snapshot.scoring_formats.get(league.key)
        if scoring_format:
            print(f"  Scoring: {scoring_format}")
        if capacity is not None:
            values = []
            if (
                capacity.bench_used is not None
                and capacity.bench_limit is not None
            ):
                values.append(f"bench {capacity.bench_used}/{capacity.bench_limit}")
            if capacity.ir_used is not None and capacity.ir_limit is not None:
                values.append(f"IR {capacity.ir_used}/{capacity.ir_limit}")
            if values:
                print("  Roster occupancy: " + " · ".join(values))
        for player in sorted(mine, key=lambda e: (e.position, e.name)):
            print(f"  {player.lineup_slot:<6} {player.name:<26} "
                  f"{player.position:<4} {player.pro_team}")
    if not snapshot.mine():
        print("\n(All rosters empty — leagues have not drafted yet.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
