import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
import json
from pprint import pprint, pformat


def load_docs_from_jsonl(file_path: str)->list[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

class ExactMatchRetriever(BaseRetriever):
    documents: list[Document]
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Sync implementations for retriever."""
        matching_documents = []
        for document in self.documents:
            for word in query.split():
                metadata_text = ' '.join([str(value) for value in document.metadata.values()])
                count_metadata = metadata_text.lower().count(word.lower())
                count_content = document.page_content.lower().count(word.lower())
                count = count_content + count_metadata
                if count > 0:
                    matching_documents.append({
                        "document": document,
                        "count": count
                    })

        matching_documents.sort(key=lambda x: x["count"], reverse=True)
        return [x["document"] for x in matching_documents][:self.k]

class FuzzyMatchRetriever(BaseRetriever):
    documents: list[Document]
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Sync implementations for retriever using fuzzy matching."""
        from difflib import SequenceMatcher

        matching_documents = []
        for document in self.documents:
            # Calculate fuzzy match ratio for content and metadata
            metadata_text = ' '.join([str(value) for value in document.metadata.values()])
            content_ratio = SequenceMatcher(None, query.lower(), document.page_content.lower()).ratio()
            metadata_ratio = SequenceMatcher(None, query.lower(), metadata_text.lower()).ratio()
            
            # Use the higher of the two ratios
            match_ratio = max(content_ratio, metadata_ratio)
            
            if match_ratio > 0.1: # Threshold to consider a match
                matching_documents.append({
                    "document": document,
                    "ratio": match_ratio
                })

        # Sort by match ratio and return top k
        matching_documents.sort(key=lambda x: x["ratio"], reverse=True)
        return [x["document"] for x in matching_documents][:self.k]

class HybridRetriever(BaseRetriever):
    exact_match_retriever: ExactMatchRetriever
    bm_25_retriever: BM25Retriever
    fuzzy_match_retriever: FuzzyMatchRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.exact_match_retriever.invoke(query) + self.bm_25_retriever.invoke(query) + self.fuzzy_match_retriever.invoke(query)

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
    return unique_docs