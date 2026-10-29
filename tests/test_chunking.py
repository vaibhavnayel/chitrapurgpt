"""chunking.py: windows over long articles, and shortened embeddings."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chunking  # noqa: E402


def article(paragraphs=40, words=30):
    return "\n\n".join(" ".join(f"w{p}_{i}" for i in range(words - 1)) + "." for p in range(paragraphs))


def test_windows_cover_the_article_in_order_with_overlap():
    text = article()  # 1,200 words in 30-word paragraphs
    spans = chunking.chunk_spans(text, target=250, overlap=40)
    words = lambda s, e: re.findall(r"\S+", text[s:e])  # noqa: E731
    assert spans[0][0] == 0 and spans[-1][1] == len(text.rstrip())
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s1 < s2 < e1 < e2, "each window starts inside the previous one"
        assert 40 <= len(words(s2, e1)) <= 80, "overlap of about 40 words, snapped back to a paragraph start"
        assert text[s2 - 2:s2] == "\n\n", "windows start at a paragraph start when one is close"
    assert all(len(words(s, e)) <= 250 for s, e in spans[:-1])
    assert len(words(*spans[-1])) >= 250 // 3, "no tiny tail window"


def test_short_articles_get_no_chunks_and_long_ones_carry_their_title():
    assert chunking.article_chunks("Short", "one two three. " * 50) == []
    text = article()
    chunks = chunking.article_chunks("Guru Purnima at Vittal", text)
    assert len(chunks) == len(chunking.chunk_spans(text))
    for start, end, embed_text in chunks:
        assert embed_text == "Guru Purnima at Vittal\n\n" + text[start:end]


def test_devanagari_sentence_ends_count_as_sentence_starts():
    text = " ".join((["शब्द"] * 9 + ["अंत।"]) * 60)  # 600 words, a danda every 10 words, no paragraphs
    spans = chunking.chunk_spans(text, target=250, overlap=40)
    assert len(spans) > 1 and all(text[s - 2:s] == "। " for s, _ in spans[1:])


def test_shorten_is_a_renormalised_prefix():
    rng = np.random.default_rng(0)
    v = rng.normal(size=3072)
    v /= np.linalg.norm(v)
    s = np.array(chunking.shorten(v.tolist(), 1024))
    assert s.shape == (1024,) and abs(np.linalg.norm(s) - 1) < 1e-9
    assert np.allclose(s, v[:1024] / np.linalg.norm(v[:1024]))
