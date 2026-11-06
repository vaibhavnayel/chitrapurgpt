from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
import bisect
import json
import threading
from array import array
from pprint import pprint, pformat
import logging
import dotenv
from rapidfuzz import fuzz, process
from metaphone import doublemetaphone

from vector_store import PgVectorStore, clean_metadata

dotenv.load_dotenv()


def add_to_vector_store(documents: list[Document]) -> None:
    vector_db = load_vector_store()

    # Clean documents
    cleaned_documents = [
        Document(
            page_content=doc.page_content,
            metadata=clean_metadata(doc.metadata)
        ) for doc in documents
    ]

    doc_ids = [get_doc_id(doc) for doc in cleaned_documents]
    existing_ids = vector_db.existing_ids(doc_ids)
    
    existing_docs = []
    existing_doc_ids = []
    new_docs = []
    new_doc_ids = []
    
    for doc, doc_id in zip(cleaned_documents, doc_ids):
        if doc_id in existing_ids:
            existing_docs.append(doc)
            existing_doc_ids.append(doc_id)
        else:
            new_docs.append(doc)
            new_doc_ids.append(doc_id)
    logging.info(f"Adding {len(new_docs)} new documents to vector store")
    
    # # Update existing documents
    # if existing_docs:
    #     vector_db.update_documents(
    #         ids=existing_doc_ids,
    #         documents=existing_docs
    #     )
    
    # Add new documents
    if new_docs:
        vector_db.add_documents(
            documents=new_docs,
            ids=new_doc_ids
        )

def load_vector_store() -> PgVectorStore:
    """The knowledge-base vector index in Postgres + pgvector (see vector_store.py)."""
    return PgVectorStore(embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

def get_doc_id(doc: Document) -> str:
    """Generate a unique identifier for a document based on source and title."""
    return f"{doc.metadata['source']}-{doc.metadata['title']}"

def save_doc_to_json(doc: Document, file_path: str) -> None:
    with open(file_path, 'w') as json_file:
        json_file.write(doc.model_dump_json())

def load_doc_from_json(file_path: str) -> Document:
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return Document(**data)
    
def load_docs_from_jsonl(file_path: str) -> list[Document]:
    """Load documents from a JSONL file."""
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

def save_docs_to_jsonl(docs: list[Document], file_path: str) -> None:
    """Save documents to a JSONL file, merging with existing documents if any."""
    # Load existing documents if file exists
    existing_docs = {}
    try:
        docs_in_knowledge_base = load_docs_from_jsonl(file_path)
        for doc in docs_in_knowledge_base:
            existing_docs[get_doc_id(doc)] = doc
    except FileNotFoundError:
        pass
    
    # Add new documents or update existing ones
    for doc in docs:
        existing_docs[get_doc_id(doc)] = doc
    
    # Write all documents back to file
    with open(file_path, 'w') as jsonl_file:
        for doc in existing_docs.values():
            jsonl_file.write(doc.model_dump_json() + '\n')

def format_docs(docs: list[Document]) -> str:
    return "\n".join([f"{i}. {doc.metadata['title']}\n Metadata:\n{pformat(doc.metadata)}\nContent:\n{doc.page_content}\n{'-'*100}" for i, doc in enumerate(docs)])

def deduplicate_docs(docs: list[Document]) -> list[Document]:
    docs = sum(docs, [])
    seen = set()
    unique_docs = []
    for doc in docs:
        doc_id = (doc.page_content, str(doc.metadata))
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    logging.info(f"deduplicated {len(docs)} docs to {len(unique_docs)} docs")
    return unique_docs

class Vocabulary:
    """Every distinct lowercase word in the articles, numbered once. An article keeps its words as a sorted array of
    these numbers instead of a set of strings of its own: the same words in about a fifth of the memory (the sets
    were most of the fuzzy index). Words of removed articles stay; that's small."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._lock = threading.Lock()

    def ids(self, words) -> array:
        """The sorted, distinct numbers of these words (numbering new ones)."""
        get = self._ids.get
        with self._lock:
            out = set()
            for w in words:
                i = get(w)
                if i is None:
                    i = self._ids[w] = len(self._ids)
                out.add(i)
        return array("I", sorted(out))

    def known(self, words) -> list[int]:
        """The numbers of the words seen so far (a word no article has can't be in any of them)."""
        get = self._ids.get
        return [i for i in (get(w) for w in words) if i is not None]


VOCABULARY = Vocabulary()


def _has(ids: array, i: int) -> bool:
    j = bisect.bisect_left(ids, i)
    return j < len(ids) and ids[j] == i


def fuzzy_cache_entry(doc: Document, kb_id: str = "sunbeam", vocabulary: Vocabulary = VOCABULARY) -> dict:
    """What FuzzyMatchRetriever pre-computes for one document (the slow part: phonetic codes of every word).

    Kept small, with exactly the same scores as before (checked by tests/test_fuzzy_index.py): the words as numbers
    (Vocabulary), each distinct phonetic code once (token_set_ratio only looks at which codes occur), and no
    lowercase copy of the text (lowered per search instead; that's fast)."""
    words = doc.page_content.lower().split()
    content_tokens = vocabulary.ids(words)

    # Phonetic codes for the content: each distinct code once
    codes = {doublemetaphone(word)[0] for word in words}
    codes.discard("")
    content_phonetic = ' '.join(sorted(codes))

    # Pre-process metadata
    important_metadata = {
        k: str(v).lower()
        for k, v in doc.metadata.items()
        if k in ['title', 'summary', 'description']
    }

    # Pre-compute phonetic codes for metadata
    metadata_phonetic = {
        k: ' '.join([
            doublemetaphone(word)[0]
            for word in v.split()
            if word
        ])
        for k, v in important_metadata.items()
    }

    return {
        'document': doc,
        'kb_id': kb_id,
        'content_tokens': content_tokens,       # sorted word numbers (Vocabulary)
        'content_phonetic': content_phonetic,
        'metadata': important_metadata,
        'metadata_phonetic': metadata_phonetic
    }


class FuzzyMatchRetriever(BaseRetriever):
    documents: list[Document]
    k: int
    content_weight: float = 0.9
    metadata_weight: float = 0.1
    threshold: float = 30
    # Only documents in these knowledge bases (None: all). A filtered view shares the cache: with_kbs().
    kb_ids: frozenset | None = None

    def __init__(self, **kwargs):
        """Initialize with pre-processed document data for faster matching. `cache` (fuzzy_cache_entry() per
        document, in the same order) skips the pre-processing."""
        cache = kwargs.pop("cache", None)
        super().__init__(**kwargs)
        if cache is None:
            self._initialize_document_cache()
        else:
            self._doc_cache = cache

    def _initialize_document_cache(self):
        """Pre-process documents once during initialization."""
        self._doc_cache = [fuzzy_cache_entry(doc) for doc in self.documents]

    def with_kbs(self, kb_ids) -> "FuzzyMatchRetriever":
        """The same retriever limited to some knowledge bases (no re-processing)."""
        return self.model_copy(update={"kb_ids": None if kb_ids is None else frozenset(kb_ids)})

    def scores(self, query: str):
        """(document entry, match ratio, token overlap, phonetic score) for every document searched."""
        query = query.lower()
        query_tokens = set(query.split())
        query_ids = VOCABULARY.known(query_tokens)
        query_phonetic = ' '.join([
            doublemetaphone(word)[0]
            for word in query.split()
            if word
        ])
        kb_ids = self.kb_ids

        for doc_data in self._doc_cache:
            if kb_ids is not None and doc_data['kb_id'] not in kb_ids:
                continue
            # Token overlap (the query's words found among the article's word numbers)
            ids = doc_data['content_tokens']
            token_overlap = sum(1 for i in query_ids if _has(ids, i)) / len(query_tokens) * 100

            # Content matching
            content_ratio = fuzz.partial_ratio(query, doc_data['document'].page_content.lower())
            
            # Phonetic matching (using pre-computed codes)
            phonetic_score = fuzz.token_set_ratio(query_phonetic, doc_data['content_phonetic'])
            
            # Metadata matching
            metadata_scores = []
            for value in doc_data['metadata'].values():
                fuzzy_score = fuzz.ratio(query, value)
                metadata_scores.append(fuzzy_score)
            
            # Phonetic metadata matching
            for phonetic_value in doc_data['metadata_phonetic'].values():
                phonetic_meta_score = fuzz.token_set_ratio(query_phonetic, phonetic_value)
                metadata_scores.append(phonetic_meta_score)
            
            metadata_ratio = max(metadata_scores) if metadata_scores else 0
            
            # Combined score
            content_score = (content_ratio + token_overlap + phonetic_score) / 3
            match_ratio = (
                self.content_weight * content_score +
                self.metadata_weight * metadata_ratio
            )
            yield doc_data, match_ratio, token_overlap, phonetic_score

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """More efficient implementation using pre-processed data."""
        matching_documents = [
            {"document": doc_data['document'], "ratio": ratio, "token_overlap": overlap, "phonetic_score": phonetic}
            for doc_data, ratio, overlap, phonetic in self.scores(query)
            if ratio > self.threshold
        ]

        # Sort only the documents that passed the threshold
        matching_documents.sort(
            key=lambda x: (x["phonetic_score"], x["token_overlap"], x["ratio"]), 
            reverse=True
        )
        
        return [x["document"] for x in matching_documents][:self.k]

class HybridRetriever(BaseRetriever):
    fuzzy_retriever: FuzzyMatchRetriever
    vector_db_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.fuzzy_retriever.invoke(query) + self.vector_db_retriever.invoke(query)
