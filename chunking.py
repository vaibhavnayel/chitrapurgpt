"""Chunks of long articles, so search can find a passage buried deep inside one.

Every article has one embedding of its whole text (kb_article). A long article also gets one embedding per chunk
(kb_chunk): a window of about CHUNK_WORDS words, overlapping the previous one by about CHUNK_OVERLAP words, starting
at a paragraph or sentence start where one is close. Each chunk is embedded with its article's title on top, so a
chunk from the middle of a talk still says what the talk is.

Each search adds up to two articles whose chunks best match the query to the articles it finds as before
(vector_store.PgVectorStore); the readers still get whole articles, so citations work as before. Measured on 160
questions answered deep inside long articles: found 90% -> 94% of the time (with the fuzzy search; embeddings alone
29% -> 71%), nothing the old search found was lost, and about 12% more articles are read per question.

Chunk embeddings are text-embedding-3-large shortened to CHUNK_DIM dimensions (the model is trained so that a prefix
of the vector, re-normalised, is itself a good embedding), which keeps the table and its index small.
"""
from __future__ import annotations

import re

import numpy as np

CHUNK_WORDS = 250
CHUNK_OVERLAP = 40
CHUNK_MIN_WORDS = 300   # shorter articles are covered by their own embedding
CHUNK_DIM = 1024

_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.!?।॥:]['\"”’)]*$")


def chunk_spans(text: str, target: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """Character spans of overlapping windows of about `target` words. A window starts at a paragraph or sentence
    start when one is within 40 words before the plain overlap point; the last window absorbs a short tail."""
    words = [m.span() for m in _WORD.finditer(text)]
    n = len(words)
    if n <= target:
        return [(0, len(text))] if n else []
    starts = {0}
    for i in range(1, n):
        gap = text[words[i - 1][1]:words[i][0]]
        if "\n" in gap or _SENTENCE_END.search(text[words[i - 1][0]:words[i - 1][1]]):
            starts.add(i)
    spans, i = [], 0
    while True:
        end = min(n, i + target)
        if n - end < target // 3:
            end = n
        spans.append((words[i][0], words[end - 1][1]))
        if end >= n:
            return spans
        nxt = end - overlap
        i = next((j for j in range(nxt, max(i + 1, nxt - 40), -1) if j in starts), nxt)


def article_chunks(title: str, text: str) -> list[tuple[int, int, str]]:
    """(start, end, text to embed) for each chunk of an article; none for articles under CHUNK_MIN_WORDS words."""
    if len(text.split()) < CHUNK_MIN_WORDS:
        return []
    head = f"{title.strip()}\n\n" if title.strip() else ""
    return [(s, e, head + text[s:e]) for s, e in chunk_spans(text)]


def shorten(vector, dim: int = CHUNK_DIM) -> list[float]:
    """A text-embedding-3 vector cut to its first `dim` dimensions and re-normalised to unit length (what the API's
    `dimensions` parameter returns)."""
    v = np.asarray(vector, dtype=np.float64)[:dim]
    return (v / (np.linalg.norm(v) or 1.0)).tolist()
