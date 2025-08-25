import os
from typing import List
# Fixed imports for newer LangChain versions:
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Initialize document processor with chunking parameters"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        print(f"Document processor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension"""
        print(f"Loading document: {file_path}")
        _, ext = os.path.splitext(file_path)
        
        try:
            if ext.lower() == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext.lower() in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            documents = loader.load()
            print(f"Loaded {len(documents)} pages/sections from {os.path.basename(file_path)}")
            return documents
            
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into manageable chunks for better retrieval"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"Document split into {len(chunks)} chunks")
            
            # Add chunk information to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            return chunks
            
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            raise
    
    def process_file(self, file_path: str) -> List[Document]:
        """Complete processing pipeline: load → split → return chunks"""
        try:
            print(f"🔄 Processing file: {os.path.basename(file_path)}")
            
            # Step 1: Load document
            documents = self.load_document(file_path)
            
            # Step 2: Split into chunks
            chunks = self.split_documents(documents)
            
            # Step 3: Add file info to metadata
            for chunk in chunks:
                chunk.metadata['source_file'] = os.path.basename(file_path)
                chunk.metadata['file_type'] = os.path.splitext(file_path)[1]
            
            print(f"✅ Successfully processed {os.path.basename(file_path)}")
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing file {file_path}: {str(e)}")
            raise

    def get_processing_stats(self, chunks: List[Document]) -> dict:
        """Get statistics about processed chunks"""
        if not chunks:
            return {"total_chunks": 0, "avg_chunk_size": 0, "total_characters": 0}
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_size = total_chars // len(chunks) if chunks else 0
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": avg_size,
            "total_characters": total_chars,
            "chunk_size_range": f"{min(len(c.page_content) for c in chunks)}-{max(len(c.page_content) for c in chunks)}"
        }
