from src.indexing.chunking import chunk, split_sections

SAMPLE = """# Title

Intro paragraph.

## Section A

First para of A.

Second para of A.

## Section B

Only para of B.
"""


def test_split_sections_keeps_titles():
    sections = split_sections(SAMPLE)
    titles = [t for t, _ in sections]
    assert "Section A" in titles
    assert "Section B" in titles


def test_chunk_attaches_section():
    chunks = chunk(SAMPLE, target_chars=50, overlap=10)
    assert chunks, "expected at least one chunk"
    assert any(c.section == "Section A" for c in chunks)
    # every chunk carries non-empty text
    assert all(c.text.strip() for c in chunks)
