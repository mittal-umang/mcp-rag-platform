"""Source registry: resolve the configured ingestion source from the ``SourceName`` enum.

Mirrors ``agent.providers.registry`` - a dict lookup keyed by the enum, lazy factories
so optional source dependencies stay optional, and a fail-fast completeness check at
import so an enum member without a factory is caught at startup, not mid-ingest.
"""
from __future__ import annotations

from collections.abc import Callable

from src.common.enums import SourceName
from src.ingestion.base import Source

SourceFactory = Callable[[], Source]


def _wikipedia() -> Source:
    from src.ingestion.wikipedia import WikipediaSource

    return WikipediaSource()


_FACTORIES: dict[SourceName, SourceFactory] = {
    SourceName.WIKIPEDIA: _wikipedia,
}

_unregistered = set(SourceName) - _FACTORIES.keys()
if _unregistered:
    raise RuntimeError(
        f"Sources declared in {SourceName.__name__} but not registered: "
        f"{sorted(s.value for s in _unregistered)}"
    )


def get_source(name: SourceName | None = None) -> Source:
    """Instantiate the configured ingestion source (defaults to ``settings.ingestion_source``)."""
    from src.common.config import settings

    selected = name if name is not None else settings.ingestion_source
    return _FACTORIES[selected]()
