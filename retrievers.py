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
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import json
from pprint import pprint, pformat

def clean_metadata(metadata):
    """
    Thoroughly clean metadata for Chroma vector store
    """
    cleaned_metadata = {}
    for key, value in metadata.items():
        if value is None:
            cleaned_metadata[key] = "Unknown"
        elif isinstance(value, (str, int, float, bool)):
            cleaned_metadata[key] = value
        else:
            # Convert complex types to string representation
            cleaned_metadata[key] = str(value)
    
    return cleaned_metadata

# def add_to_vector_store(documents:list[Document])->None:
#     vector_db = load_vector_store()
#     cleaned_documents = [Document(page_content=doc.page_content, metadata=clean_metadata(doc.metadata)) for doc in documents]
#     vector_db.update_documents(
#         documents=cleaned_documents,
#         ids=[get_doc_id(doc) for doc in cleaned_documents]
#     )
def add_to_vector_store(documents: list[Document]) -> None:
    vector_db = load_vector_store()
    
    # Clean documents
    cleaned_documents = [
        Document(
            page_content=doc.page_content, 
            metadata=clean_metadata(doc.metadata)
        ) for doc in documents
    ]
    
    # Generate IDs for the documents
    doc_ids = [get_doc_id(doc) for doc in cleaned_documents]
    
    # First, check which documents already exist
    existing_ids = vector_db.get(ids=doc_ids)['ids']
    
    # Separate documents into existing and new
    existing_docs = []
    existing_doc_ids = []
    new_docs = []
    new_doc_ids = []
    breakpoint()
    
    for doc, doc_id in zip(cleaned_documents, doc_ids):
        if doc_id in existing_ids:
            existing_docs.append(doc)
            existing_doc_ids.append(doc_id)
        else:
            new_docs.append(doc)
            new_doc_ids.append(doc_id)
    
    # Update existing documents
    if existing_docs:
        vector_db.update_documents(
            ids=existing_doc_ids,
            documents=existing_docs
        )
    
    # Add new documents
    if new_docs:
        vector_db.add_documents(
            documents=new_docs,
            ids=new_doc_ids
        )

def load_vector_store()->Chroma:
    return Chroma(
        collection_name="sunbeam_collection",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="./chroma"
    )

def get_doc_id(doc: Document) -> str:
    """Generate a unique identifier for a document based on source and title."""
    return f"{doc.metadata['source']}-{doc.metadata['title']}"

def load_docs_from_jsonl(file_path: str) -> list[Document]:
    """Load documents from a JSONL file."""
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

def save_docs_to_jsonl(array: list[Document], file_path: str) -> None:
    """Save documents to a JSONL file, merging with existing documents if any."""
    # Load existing documents if file exists
    existing_docs = {}
    try:
        existing_array = load_docs_from_jsonl(file_path)
        for doc in existing_array:
            existing_docs[get_doc_id(doc)] = doc
    except FileNotFoundError:
        pass
    
    # Add new documents or update existing ones
    for doc in array:
        existing_docs[get_doc_id(doc)] = doc
    
    # Write all documents back to file
    with open(file_path, 'w') as jsonl_file:
        for doc in existing_docs.values():
            jsonl_file.write(doc.model_dump_json() + '\n')

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
    vector_db_retriever: BaseRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.exact_match_retriever.invoke(query) + self.bm_25_retriever.invoke(query) + self.fuzzy_match_retriever.invoke(query) + self.vector_db_retriever.invoke(query)

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