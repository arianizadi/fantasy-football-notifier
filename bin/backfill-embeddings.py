#!/usr/bin/env python3
"""Backfill saved news vectors without classifying or sending any alerts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from notifier.config import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
)
from notifier.embeddings import (  # noqa: E402
    INPUT_VERSION,
    OpenRouterEmbeddingClient,
    canonical_embedding_text,
    news_item_from_row,
)
from notifier.event_store import EventStore  # noqa: E402
from notifier.logging_utils import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    configure_logging(verbose=args.verbose)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 2
    model = os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    dimensions = int(
        os.environ.get("EMBEDDING_DIMENSIONS", "").strip()
        or DEFAULT_EMBEDDING_DIMENSIONS
    )
    timeout = int(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "8").strip() or 8)
    state_dir = Path(os.environ.get("NOTIFIER_STATE_DIR", "").strip() or root / "state")
    batch_size = max(1, min(int(args.batch_size), 64))

    store = EventStore(state_dir)
    client = OpenRouterEmbeddingClient(
        api_key,
        model,
        dimensions,
        timeout_seconds=timeout,
    )
    try:
        rows = store.embedding_backlog(
            model=model,
            dimensions=dimensions,
            input_version=INPUT_VERSION,
            limit=max(1, int(args.limit)),
            force=bool(args.force),
        )
        saved = 0
        prompt_tokens = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            items = [news_item_from_row(row) for row in batch]
            vectors = client.embed_many(
                [canonical_embedding_text(item) for item in items]
            )
            for item, vector in zip(items, vectors, strict=True):
                if store.store_embedding(
                    item,
                    vector.model,
                    vector.blob,
                    provider=vector.provider,
                    dimensions=vector.dimensions,
                    input_version=vector.input_version,
                    input_hash=vector.input_hash,
                ):
                    saved += 1
                prompt_tokens += vector.prompt_tokens
            print(f"embedded {min(start + len(batch), len(rows))}/{len(rows)}")
        total = store.embedding_count(
            model=model,
            dimensions=dimensions,
            input_version=INPUT_VERSION,
        )
        print(
            f"complete: saved={saved} total={total} model={model} "
            f"dimensions={dimensions} prompt_tokens~={prompt_tokens}"
        )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

