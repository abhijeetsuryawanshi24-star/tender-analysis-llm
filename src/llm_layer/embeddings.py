"""
Embedding Generation Module
Generates vector embeddings for tender documents for semantic search
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from loguru import logger
import pickle
import os

logger.add("logs/embeddings.log", rotation="500 MB")

class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using sentence transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently
        """
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks
        """
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.generate_embeddings_batch(chunk_texts)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        return chunks


class VectorStore:
    """
    Base class for vector database operations
    """
    
    def __init__(self):
        pass
    
    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict]) -> None:
        raise NotImplementedError
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        raise NotImplementedError
    
    def delete_vectors(self, ids: List[str]) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """
    Chroma vector database implementation
    """
    
    def __init__(self, collection_name: str = "tender_documents"):
        try:
            import chromadb
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized Chroma with collection: {collection_name}")
        except ImportError:
            logger.error("Chroma not installed. Install with: pip install chromadb")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]], document_id: str) -> None:
        """
        Add document chunks to Chroma
        """
        try:
            ids = [f"{document_id}_chunk_{chunk['chunk_id']}" for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]
            embeddings = [chunk['embedding'].tolist() for chunk in chunks]
            metadatas = [chunk.get('metadata', {}) for chunk in chunks]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to Chroma for document {document_id}")
        except Exception as e:
            logger.error(f"Error adding chunks to Chroma: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, query_text: str = None) -> List[Dict]:
        """
        Search for similar documents
        """
        try:
            if query_text:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=top_k
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
            
            # Format results
            formatted_results = []
            for i, (doc_id, distance, metadata, document) in enumerate(
                zip(results['ids'][0], results['distances'][0], 
                    results['metadatas'][0], results['documents'][0])
            ):
                formatted_results.append({
                    'id': doc_id,
                    'content': document,
                    'distance': float(distance),
                    'metadata': metadata,
                    'relevance_score': 1 - float(distance)  # Convert distance to similarity
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching Chroma: {str(e)}")
            raise
    
    def delete_documents(self, document_id: str) -> None:
        """
        Delete all chunks of a document
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(where={"document_id": document_id})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted document {document_id} from Chroma")
        except Exception as e:
            logger.error(f"Error deleting from Chroma: {str(e)}")
            raise


class FAISSVectorStore(VectorStore):
    """
    FAISS vector database implementation for local deployment
    """
    
    def __init__(self, embedding_dim: int = 384, save_path: str = "./data/embeddings/faiss_index"):
        try:
            import faiss
            self.faiss = faiss
            self.embedding_dim = embedding_dim
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata_store = {}
            self.save_path = save_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            logger.info(f"Initialized FAISS index with dimension: {embedding_dim}")
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]], document_id: str) -> None:
        """
        Add chunks to FAISS index
        """
        try:
            embeddings = np.array([chunk['embedding'] for chunk in chunks])
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                idx = start_idx + i
                self.metadata_store[idx] = {
                    'document_id': document_id,
                    'chunk_id': chunk['chunk_id'],
                    'content': chunk['content'],
                    'metadata': chunk.get('metadata', {})
                }
            
            logger.info(f"Added {len(chunks)} chunks to FAISS for document {document_id}")
            self.save_index()
        except Exception as e:
            logger.error(f"Error adding chunks to FAISS: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search in FAISS index
        """
        try:
            query_vector = query_embedding.astype('float32').reshape(1, -1)
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx in self.metadata_store:
                    metadata = self.metadata_store[idx]
                    results.append({
                        'id': f"{metadata['document_id']}_chunk_{metadata['chunk_id']}",
                        'content': metadata['content'],
                        'distance': float(distance),
                        'metadata': metadata['metadata'],
                        'relevance_score': 1 / (1 + float(distance))  # Convert L2 distance to similarity
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            raise
    
    def save_index(self) -> None:
        """
        Save index and metadata to disk
        """
        try:
            self.faiss.write_index(self.index, f"{self.save_path}.index")
            with open(f"{self.save_path}.metadata", 'wb') as f:
                pickle.dump(self.metadata_store, f)
            logger.info(f"Saved FAISS index to {self.save_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def load_index(self) -> None:
        """
        Load index and metadata from disk
        """
        try:
            if os.path.exists(f"{self.save_path}.index"):
                self.index = self.faiss.read_index(f"{self.save_path}.index")
                with open(f"{self.save_path}.metadata", 'rb') as f:
                    self.metadata_store = pickle.load(f)
                logger.info(f"Loaded FAISS index from {self.save_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise
