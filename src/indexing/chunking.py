"""Naive but sane markdown-aware chunker.

Splits on headings first (so a chunk keeps its section context), then packs
paragraphs up to a target character budget with a small overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


@dataclass
class TextChunk:
    section: str | None
    text: str


def split_sections(markdown: str) -> list[tuple[str | None, str]]:
    """Return (section_title, body) pairs. Text before the first heading has section=None."""
    matches = list(_HEADING.finditer(markdown))
    if not matches:
        return [(None, markdown.strip())]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        sections.append((None, markdown[: matches[0].start()].strip()))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def chunk(markdown: str, target_chars: int = 900, overlap: int = 120) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for section, body in split_sections(markdown):
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        buf = ""
        for para in paras:
            if len(buf) + len(para) + 2 > target_chars and buf:
                chunks.append(TextChunk(section, buf.strip()))
                buf = buf[-overlap:] + "\n\n" + para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf.strip():
            chunks.append(TextChunk(section, buf.strip()))
    return chunks
