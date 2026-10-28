"""The articles in memory: fuzzy search (the agent's tool) and the Exact/Fuzzy Search commands read them from here.

Loaded from Postgres (kb_article) at startup, in the original knowledge base's order (kb.py), and refreshed after a
publish: only new articles are pre-processed, removed ones are dropped. Each article remembers its knowledge base,
so a search can be limited to the user's (documents(kb_ids), fuzzy_retriever(kb_ids)).
"""
from __future__ import annotations

import logging
import threading
import time

from langchain_core.documents import Document

import kb
from retrievers import FuzzyMatchRetriever, fuzzy_cache_entry

logger = logging.getLogger(__name__)
FUZZY_K = 5


class KnowledgeIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._kbs: list[str] = []
        self._docs: list[Document] = []
        self._cache: dict[str, dict] = {}  # id -> fuzzy_cache_entry (built lazily: it's the slow part)
        self._fuzzy: FuzzyMatchRetriever | None = None
        self.loaded = False

    # ---------- loading ----------

    def set_documents(self, items: list[tuple[str, str, Document]]) -> None:
        """Replace the contents with (id, kb_id, Document) items, in order (tests, scripts)."""
        with self._lock:
            self._ids = [i for i, _, _ in items]
            self._kbs = [k for _, k, _ in items]
            self._docs = [d for _, _, d in items]
            self._cache = {}
            self._fuzzy = None
            self.loaded = True

    def load(self, db: str | None = None, build_fuzzy: bool = True) -> None:
        """Everything from the database (after making sure the schema is up to date)."""
        t = time.monotonic()
        kb.ensure_schema(db)
        keys = kb.article_keys(db)
        rows = kb.load_articles(None, db)
        self.set_documents([(i, k, _document(rows[i])) for i, k in keys if i in rows])
        if build_fuzzy:
            self.fuzzy_retriever()
        logger.info("Loaded %d articles in %.1fs", len(keys), time.monotonic() - t)

    def refresh(self, db: str | None = None) -> tuple[int, int]:
        """Catch up with the database: load new articles, drop removed ones, keep the order. Returns (added,
        removed)."""
        keys = kb.article_keys(db)
        with self._lock:
            have = dict(zip(self._ids, self._docs))
        new_ids = [i for i, _ in keys if i not in have]
        rows = kb.load_articles(new_ids, db) if new_ids else {}
        docs = {**have, **{i: _document(r) for i, r in rows.items()}}
        items = [(i, k, docs[i]) for i, k in keys if i in docs]
        removed = len(set(have) - {i for i, _ in keys})
        with self._lock:
            cache = {i: c for i, c in self._cache.items() if i in docs}
            self._ids, self._kbs, self._docs = [i for i, _, _ in items], [k for _, k, _ in items], [d for _, _, d in items]
            self._cache, self._fuzzy = cache, None
        self.fuzzy_retriever()
        logger.info("Knowledge index refreshed: %d added, %d removed", len(new_ids), removed)
        return len(new_ids), removed

    # ---------- reading ----------

    def kb_ids_present(self) -> set[str]:
        return set(self._kbs)

    def covers_all(self, kb_ids) -> bool:
        """True if these knowledge bases include every one that has articles (then a search needs no filter)."""
        return kb_ids is None or set(kb_ids) >= self.kb_ids_present()

    def documents(self, kb_ids=None) -> list[Document]:
        with self._lock:
            if kb_ids is None:
                return list(self._docs)
            wanted = set(kb_ids)
            return [d for d, k in zip(self._docs, self._kbs) if k in wanted]

    def fuzzy_retriever(self, kb_ids=None) -> FuzzyMatchRetriever:
        """The fuzzy retriever over all articles, or a view limited to some knowledge bases."""
        with self._lock:
            if self._fuzzy is None:
                t = time.monotonic()
                todo = [(i, k, d) for i, k, d in zip(self._ids, self._kbs, self._docs) if i not in self._cache]
                for i, k, d in todo:
                    self._cache[i] = fuzzy_cache_entry(d, k)
                self._fuzzy = FuzzyMatchRetriever(documents=list(self._docs), k=FUZZY_K,
                                                  cache=[self._cache[i] for i in self._ids])
                if todo:
                    logger.info("Fuzzy index: pre-processed %d articles in %.1fs", len(todo), time.monotonic() - t)
            fuzzy = self._fuzzy
        return fuzzy if self.covers_all(kb_ids) else fuzzy.with_kbs(kb_ids)


def _document(row: tuple[str, str, dict]) -> Document:
    _kb_id, content, metadata = row
    return Document(page_content=content, metadata=metadata)


INDEX = KnowledgeIndex()
