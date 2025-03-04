from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
import pinecone
from pinecone import Pinecone, ServerlessSpec
import json
import time
from pprint import pprint, pformat
import logging
import dotenv
from rapidfuzz import fuzz, process
from metaphone import doublemetaphone

dotenv.load_dotenv()

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
    existing_ids = sum(list(vector_db._index.list()),[])
    
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

def create_or_fetch_pinecone_index(index_name:str)->pinecone.Index:
    pc = Pinecone()
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name in existing_indexes:
        logging.info(f"Index {index_name} already exists")
        return pc.Index(index_name)
    else:
        logging.info(f"Creating index {index_name}")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)


def load_vector_store()->PineconeVectorStore:
    index = create_or_fetch_pinecone_index("chitrapur-gpt")
    return PineconeVectorStore(index=index, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

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

class FuzzyMatchRetriever(BaseRetriever):
    documents: list[Document]
    k: int
    content_weight: float = 0.9
    metadata_weight: float = 0.1
    threshold: float = 30
    
    def __init__(self, **kwargs):
        """Initialize with pre-processed document data for faster matching."""
        super().__init__(**kwargs)
        self._initialize_document_cache()
    
    def _initialize_document_cache(self):
        """Pre-process documents once during initialization."""
        self._doc_cache = []
        
        for doc in self.documents:
            # Pre-compute lowercase content and tokens
            content = doc.page_content.lower()
            content_tokens = set(content.split())
            
            # Pre-compute phonetic codes for content
            content_phonetic = ' '.join([
                doublemetaphone(word)[0] 
                for word in content.split() 
                if word
            ])
            
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
            
            self._doc_cache.append({
                'document': doc,
                'content': content,
                'content_tokens': content_tokens,
                'content_phonetic': content_phonetic,
                'metadata': important_metadata,
                'metadata_phonetic': metadata_phonetic
            })

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """More efficient implementation using pre-processed data."""
        query = query.lower()
        query_tokens = set(query.split())
        query_phonetic = ' '.join([
            doublemetaphone(word)[0] 
            for word in query.split() 
            if word
        ])
        
        matching_documents = []
        
        # Process documents in batches for better performance
        batch_size = 100
        for i in range(0, len(self._doc_cache), batch_size):
            batch = self._doc_cache[i:i + batch_size]
            
            for doc_data in batch:
                # Token overlap (pre-computed sets)
                token_overlap = len(query_tokens & doc_data['content_tokens']) / len(query_tokens) * 100
                
                # Content matching
                content_ratio = fuzz.partial_ratio(query, doc_data['content'])
                
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
                
                if match_ratio > self.threshold:
                    matching_documents.append({
                        "document": doc_data['document'],
                        "ratio": match_ratio,
                        "token_overlap": token_overlap,
                        "phonetic_score": phonetic_score
                    })
        
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
