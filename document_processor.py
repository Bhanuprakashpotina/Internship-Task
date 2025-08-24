import os
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        print(f"Document processor initialized with chunk_size={chunk_size}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension"""
        print(f"Loading document: {file_path}")
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext.lower() in ['.txt', '.md']:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        documents = loader.load()
        print(f"Loaded {len(documents)} pages/sections")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into manageable chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"Document split into {len(chunks)} chunks")
        return chunks
    
    def process_file(self, file_path: str) -> List[Document]:
        """Complete processing pipeline for ingestion"""
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)
        return chunks

