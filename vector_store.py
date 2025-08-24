import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from langchain.schema import Document
import uuid
import time

class VectorStore:
    def __init__(self, collection_name: str = "documents"):
        print("Initializing vector store...")
        start_time = time.time()
        
        # Initialize ChromaDB (persistent storage)
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model (SentenceTransformers as required)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        init_time = time.time() - start_time
        print(f"Vector store initialized in {init_time:.2f}s")
    
    def add_documents(self, documents: List[Document]) -> Dict:
        """Add documents to vector store with timing info"""
        print(f"Adding {len(documents)} documents to vector store...")
        start_time = time.time()
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings with timing
        embedding_start = time.time()
        embeddings = self.embedding_model.encode(texts).tolist()
        embedding_time = time.time() - embedding_start
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Add to ChromaDB
        storage_start = time.time()
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        storage_time = time.time() - storage_start
        
        total_time = time.time() - start_time
        
        return {
            "total_docs": len(documents),
            "embedding_time": embedding_time,
            "storage_time": storage_time,
            "total_time": total_time,
            "db_size": self.collection.count()
        }
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        """Search for similar documents with cosine similarity"""
        print(f"Searching for top {k} relevant chunks...")
        start_time = time.time()
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        search_time = time.time() - start_time
        
        # Format results with similarity scores
        search_results = []
        if results['documents'][0]:
            for i in range(len(results['documents'][0])):
                similarity = 1 - results['distances'][0][i]  # Convert distance to similarity
                search_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,
                    'search_time': search_time
                })
        
        print(f"Search completed in {search_time:.2f}s")
        return search_results
    
    def get_stats(self):
        """Get vector database statistics"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_db": "ChromaDB"
        }

