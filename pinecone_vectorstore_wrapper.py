"""
Custom Pinecone VectorStore wrapper compatible with LangChain 1.0 and Pinecone SDK 5.x
"""
from typing import Any, Iterable, List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings


class PineconeVectorStore(VectorStore):
    """Pinecone vector store compatible with LangChain 1.0.x and Pinecone SDK 5.x"""
    
    def __init__(
        self, 
        index: Any,  # pinecone.Index type
        embedding: Embeddings,
        text_key: str = "text",
        namespace: str = ""
    ):
        """Initialize the Pinecone vector store.
        
        Args:
            index: Pinecone index instance
            embedding: Embedding model to use
            text_key: Key to store text content in metadata
            namespace: Pinecone namespace to use
        """
        self._index = index
        self._embedding = embedding
        self._text_key = text_key
        self._namespace = namespace
    
    @staticmethod
    def _maximal_marginal_relevance(
        query_embedding: List[float],
        embedding_list: List[List[float]],
        k: int = 4,
        lambda_mult: float = 0.5
    ) -> List[int]:
        """Calculate maximal marginal relevance.
        
        Args:
            query_embedding: Query embedding
            embedding_list: List of embeddings to select from
            k: Number of embeddings to return
            lambda_mult: Diversity parameter (0=max diversity, 1=min diversity)
            
        Returns:
            List of indices of selected embeddings
        """
        import numpy as np
        
        if len(embedding_list) == 0:
            return []
        if k >= len(embedding_list):
            return list(range(len(embedding_list)))
        
        # Convert to numpy arrays
        query_emb = np.array(query_embedding)
        embeddings = np.array(embedding_list)
        
        # Calculate cosine similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        
        # Calculate similarity to query
        similarity_to_query = np.array([
            cosine_similarity(query_emb, emb) for emb in embeddings
        ])
        
        # Initialize with most similar document
        selected = [int(np.argmax(similarity_to_query))]
        
        # Calculate pairwise similarities
        for _ in range(k - 1):
            if len(selected) >= len(embedding_list):
                break
                
            best_score = -float('inf')
            best_idx = -1
            
            for i in range(len(embeddings)):
                if i in selected:
                    continue
                
                # Relevance to query
                relevance = similarity_to_query[i]
                
                # Max similarity to already selected documents
                max_sim_to_selected = max(
                    cosine_similarity(embeddings[i], embeddings[j])
                    for j in selected
                )
                
                # MMR score
                score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx != -1:
                selected.append(best_idx)
        
        return selected
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add texts to the vector store.
        
        Args:
            texts: Iterable of strings to add
            metadatas: Optional list of metadatas associated with the texts
            ids: Optional list of unique IDs for the texts
            namespace: Optional namespace to use
            
        Returns:
            List of IDs of the added texts
        """
        texts_list = list(texts)
        if not texts_list:
            return []
            
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(texts_list))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts_list]
        
        # Generate embeddings
        embeddings = self._embedding.embed_documents(texts_list)
        
        # Prepare vectors for upsert
        vectors = []
        for id, text, embedding, metadata in zip(ids, texts_list, embeddings, metadatas):
            # Store text in metadata
            metadata_copy = metadata.copy()
            metadata_copy[self._text_key] = text
            
            vectors.append({
                "id": id,
                "values": embedding,
                "metadata": metadata_copy
            })
        
        # Upsert to Pinecone
        ns = namespace or self._namespace
        self._index.upsert(vectors=vectors, namespace=ns)
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.
        
        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            filter: Optional filter dict
            namespace: Optional namespace to use
            
        Returns:
            List of Documents most similar to the query
        """
        # Generate query embedding
        query_embedding = self._embedding.embed_query(query)
        
        # Query Pinecone
        ns = namespace or self._namespace
        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=ns,
            filter=filter
        )
        
        # Convert to Documents
        docs = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {}).copy()
            text = metadata.pop(self._text_key, "")
            docs.append(Document(page_content=text, metadata=metadata))
        
        return docs
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Return docs and relevance scores.
        
        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            filter: Optional filter dict
            namespace: Optional namespace to use
            
        Returns:
            List of tuples of (Document, score)
        """
        # Generate query embedding
        query_embedding = self._embedding.embed_query(query)
        
        # Query Pinecone
        ns = namespace or self._namespace
        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=ns,
            filter=filter
        )
        
        # Convert to Documents with scores
        docs_and_scores = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {}).copy()
            text = metadata.pop(self._text_key, "")
            score = match.get('score', 0.0)
            docs_and_scores.append((
                Document(page_content=text, metadata=metadata),
                score
            ))
        
        return docs_and_scores
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance (MMR).
        
        MMR selects docs based on query similarity and diversity.
        
        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            fetch_k: Number of Documents to fetch to pass to MMR algorithm
            lambda_mult: Diversity factor (0=max diversity, 1=min diversity)
            filter: Optional filter dict
            namespace: Optional namespace to use
            
        Returns:
            List of Documents selected by MMR
        """
        # Generate query embedding
        query_embedding = self._embedding.embed_query(query)
        
        # Fetch more documents than needed (must request values for MMR)
        ns = namespace or self._namespace
        results = self._index.query(
            vector=query_embedding,
            top_k=fetch_k,
            include_metadata=True,
            include_values=True,  # Request vectors for MMR calculation
            namespace=ns,
            filter=filter
        )
        
        # Extract embeddings and create documents
        matches = results.get('matches', [])
        if not matches:
            return []
        
        # Get embeddings for MMR calculation
        embeddings = [match.get('values', []) for match in matches]
        
        # Filter out empty embeddings
        valid_matches = [(match, emb) for match, emb in zip(matches, embeddings) if emb]
        
        # If no embeddings returned, fall back to simple similarity search
        if not valid_matches:
            docs = []
            for match in matches[:k]:
                metadata = match.get('metadata', {}).copy()
                text = metadata.pop(self._text_key, "")
                docs.append(Document(page_content=text, metadata=metadata))
            return docs
        
        matches = [m for m, _ in valid_matches]
        embeddings = [e for _, e in valid_matches]
        
        # Calculate MMR ourselves
        mmr_selected = self._maximal_marginal_relevance(
            query_embedding, embeddings, k=k, lambda_mult=lambda_mult
        )
        
        # Return selected documents
        docs = []
        for i in mmr_selected:
            match = matches[i]
            metadata = match.get('metadata', {}).copy()
            text = metadata.pop(self._text_key, "")
            docs.append(Document(page_content=text, metadata=metadata))
        
        return docs
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        index: Any = None,
        namespace: str = "",
        **kwargs: Any
    ) -> "PineconeVectorStore":
        """Create a vector store from a list of texts.
        
        Args:
            texts: List of texts to add
            embedding: Embedding model to use
            metadatas: Optional list of metadatas
            ids: Optional list of IDs
            index: Pinecone index instance
            namespace: Pinecone namespace
            
        Returns:
            PineconeVectorStore instance
        """
        if index is None:
            raise ValueError("index parameter is required")
        
        store = cls(index=index, embedding=embedding, namespace=namespace, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[List[str]] = None,
        index: Any = None,
        namespace: str = "",
        **kwargs: Any
    ) -> "PineconeVectorStore":
        """Create a vector store from a list of documents.
        
        Args:
            documents: List of Documents to add
            embedding: Embedding model to use
            ids: Optional list of IDs
            index: Pinecone index instance
            namespace: Pinecone namespace
            
        Returns:
            PineconeVectorStore instance
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            ids=ids,
            index=index,
            namespace=namespace,
            **kwargs
        )

