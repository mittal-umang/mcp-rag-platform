"""CLI entry point for the ingestion pipeline.

Examples:
    python -m src.ingestion                 # incremental re-index of the default source
    python -m src.ingestion --full          # ignore saved revisions, re-index everything
    python -m src.ingestion --source wikipedia
"""
from __future__ import annotations

import argparse
import asyncio

from src.common.config import settings
from src.common.enums import SourceName
from src.common.logging import configure_logging, get_logger
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.registry import get_source


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest a source into the vector store.")
    parser.add_argument(
        "--source",
        default=settings.ingestion_source.value,
        choices=[s.value for s in SourceName],
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="re-index every document, ignoring saved revisions",
    )
    args = parser.parse_args()

    source = get_source(SourceName(args.source))
    report = asyncio.run(IngestionPipeline(source).run(incremental=not args.full))
    get_logger("ingestion.cli").info("done", source=args.source, indexed=report.indexed,
                                     skipped=report.skipped, chunks=report.chunks)


if __name__ == "__main__":
    main()
