import requests
import json
import time
from typing import List, Dict
from vector_store import VectorStore

class ChatBot:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.vector_store = VectorStore()
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"ChatBot initialized with model: {model_name}")
        
        # Test Ollama connection
        self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'].split(':')[0] for m in models]
                if self.model_name not in model_names:
                    print(f"Warning: {self.model_name} not found. Available models: {model_names}")
                else:
                    print(f"✓ Ollama connected. Model {self.model_name} is available.")
            else:
                print("Warning: Could not connect to Ollama")
        except Exception as e:
            print(f"Warning: Ollama connection test failed: {e}")
    
    def generate_response(self, prompt: str) -> Dict:
        """Generate response using Ollama with timing"""
        start_time = time.time()
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=data, timeout=60)
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result['response'],
                    "generation_time": generation_time,
                    "model_used": self.model_name,
                    "success": True
                }
            else:
                return {
                    "response": f"Error: HTTP {response.status_code}",
                    "generation_time": generation_time,
                    "success": False
                }
        except Exception as e:
            return {
                "response": f"Error connecting to Ollama: {str(e)}",
                "generation_time": time.time() - start_time,
                "success": False
            }
    
    def create_rag_prompt(self, query: str, context_docs: List[dict]) -> str:
        """Create RAG prompt with retrieved context"""
        context = "\n\n".join([
            f"[Source {i+1}]: {doc['content']}"
            for i, doc in enumerate(context_docs)
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context. If the answer is not in the context, say "I don't have enough information in the provided documents to answer that question."

CONTEXT:
{context}

QUESTION: {query}

ANSWER (be specific and cite which sources you used):"""
        
        return prompt
    
    def chat(self, query: str, k: int = 3) -> Dict:
        """Main RAG pipeline: Retrieve + Generate"""
        print(f"\n🔍 Processing query: '{query[:50]}...'")
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        relevant_docs = self.vector_store.search(query, k=k)
        retrieval_time = time.time() - retrieval_start
        
        if not relevant_docs:
            return {
                "answer": "No relevant documents found in the database.",
                "sources": [],
                "retrieval_time": retrieval_time,
                "generation_time": 0,
                "total_time": retrieval_time
            }
        
        # Step 2: Create prompt with context
        prompt = self.create_rag_prompt(query, relevant_docs)
        
        # Step 3: Generate response
        generation_result = self.generate_response(prompt)
        
        total_time = retrieval_time + generation_result["generation_time"]
        
        return {
            "answer": generation_result["response"],
            "sources": relevant_docs,
            "retrieval_time": retrieval_time,
            "generation_time": generation_result["generation_time"],
            "total_time": total_time,
            "model_used": self.model_name,
            "success": generation_result["success"]
        }

