"""The compact fuzzy index (retrievers.fuzzy_cache_entry, Vocabulary) gives exactly the scores the original one did.

The original pre-computed, per article, a lowercase copy of the text, a set of its words and every word's phonetic
code; that was most of the app's memory. The compact one keeps the words as numbers into one shared vocabulary and
each phonetic code once. `original_scores` below is the original scoring code, kept as the reference: every
article's score must be identical for every query (spelling variants included), so search results don't change.
"""
import sys
from pathlib import Path

from metaphone import doublemetaphone
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import retrievers  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

QUERIES = [
    "Parijnanashram", "Parijñānāśrama", "Parijnanashrama Swamiji samadhi", "Shirali", "Shiraali temple",
    "Anandashram", "Anandaashram Swami teachings", "Gurupournima", "Guru Purnima celebrations",
    "Chitrapur Math Karla", "Bhanap community", "Konkani bhajan", "गुरुपरंपरा", "Sadyojat Shankarashram",
    "Umabai Bodhamrit", "Shrimat Pandurangashram", "Navaratri Utsava", "Rathotsava Shirali 1985",
    "Mallapur", "Vittal temple", "what happened in 1965", "KSA founded Bombay 1911", "Talmaki Saraswat bank",
    "zzzz qqqq", "the",
]


def original_entry(doc):
    content = doc.page_content.lower()
    meta = {k: str(v).lower() for k, v in doc.metadata.items() if k in ['title', 'summary', 'description']}
    return {"content": content, "content_tokens": set(content.split()),
            "content_phonetic": ' '.join([doublemetaphone(w)[0] for w in content.split() if w]),
            "metadata": meta,
            "metadata_phonetic": {k: ' '.join([doublemetaphone(w)[0] for w in v.split() if w]) for k, v in meta.items()}}


def original_scores(query, entries, content_weight=0.9, metadata_weight=0.1):
    """The original FuzzyMatchRetriever scoring, unchanged."""
    query = query.lower()
    query_tokens = set(query.split())
    query_phonetic = ' '.join([doublemetaphone(w)[0] for w in query.split() if w])
    out = []
    for d in entries:
        token_overlap = len(query_tokens & d['content_tokens']) / len(query_tokens) * 100
        content_ratio = fuzz.partial_ratio(query, d['content'])
        phonetic_score = fuzz.token_set_ratio(query_phonetic, d['content_phonetic'])
        metadata_scores = [fuzz.ratio(query, v) for v in d['metadata'].values()]
        metadata_scores += [fuzz.token_set_ratio(query_phonetic, v) for v in d['metadata_phonetic'].values()]
        metadata_ratio = max(metadata_scores) if metadata_scores else 0
        content_score = (content_ratio + token_overlap + phonetic_score) / 3
        out.append((content_weight * content_score + metadata_weight * metadata_ratio, token_overlap, phonetic_score))
    return out


_ORIGINALS = []


def originals() -> list[Document]:
    """The original Sunbeam articles (read here: other tests swap the in-memory index for tiny fakes)."""
    if not _ORIGINALS:
        _ORIGINALS.extend(retrievers.load_docs_from_jsonl(str(Path(__file__).resolve().parents[1]
                                                              / "knowledge_base.jsonl")))
    return _ORIGINALS


def sample_docs():
    docs = originals()
    return docs[::16] + [Document(page_content="श्री गुरुपरंपरा चरित्र Parijñānāśrama ... --- 3 4 (part 1 of 2)",
                                 metadata={"title": "गुरुपरंपरा - Parichayanam", "summary": None})]


def test_every_score_is_the_same_as_the_original_index():
    docs = sample_docs()
    retriever = retrievers.FuzzyMatchRetriever(documents=docs, k=5)
    reference = [original_entry(d) for d in docs]
    for q in QUERIES:
        new = [(ratio, overlap, phonetic) for _, ratio, overlap, phonetic in retriever.scores(q)]
        assert new == original_scores(q, reference), q


def test_the_results_and_their_order_are_the_same():
    docs = sample_docs()
    retriever = retrievers.FuzzyMatchRetriever(documents=docs, k=5)
    reference = [original_entry(d) for d in docs]
    for q in QUERIES:
        hits = [(s[2], s[1], s[0], i) for i, s in enumerate(original_scores(q, reference)) if s[0] > 30]
        hits.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        assert retriever.invoke(q) == [docs[h[3]] for h in hits[:5]], q


def test_collections_filter_and_new_articles():
    a = Document(page_content="Rathotsava at Shirali", metadata={"title": "Rathotsava"})
    b = Document(page_content="Rathotsava at Karla, a newly coined word: xyzzyplugh", metadata={"title": "Karla"})
    retriever = retrievers.FuzzyMatchRetriever(
        documents=[a, b], k=5, cache=[retrievers.fuzzy_cache_entry(a, "sunbeam"), retrievers.fuzzy_cache_entry(b, "ksa")])
    assert retriever.invoke("Rathotsava") == [a, b]
    assert retriever.with_kbs(["ksa"]).invoke("Rathotsava") == [b]
    assert retriever.invoke("xyzzyplugh") == [b], "words first seen in a new article are found"


def test_the_index_is_smaller():
    docs = originals()[:1500]
    import tracemalloc
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    compact = [retrievers.fuzzy_cache_entry(d, "sunbeam", retrievers.Vocabulary()) for d in docs]
    compact_bytes = tracemalloc.get_traced_memory()[0] - before
    before = tracemalloc.get_traced_memory()[0]
    original = [original_entry(d) for d in docs]
    original_bytes = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    assert compact and original
    assert compact_bytes < original_bytes / 3, (compact_bytes, original_bytes)
