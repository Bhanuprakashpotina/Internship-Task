import gradio as gr
import os
import time
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_bot import ChatBot

class RAGApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.chat_bot = ChatBot()
        
    def upload_and_process_file(self, file):
        """Handle file upload and processing with detailed feedback"""
        if file is None:
            return "❌ No file uploaded"
        
        try:
            print(f"Processing uploaded file: {file.name}")
            
            # Process document
            start_time = time.time()
            chunks = self.doc_processor.process_file(file.name)
            processing_time = time.time() - start_time
            
            # Add to vector store
            ingestion_info = self.vector_store.add_documents(chunks)
            
            # Create detailed feedback
            feedback = f"""✅ **File processed successfully!**

📄 **File**: {os.path.basename(file.name)}
🔢 **Chunks created**: {len(chunks)}
⚡ **Processing time**: {processing_time:.2f}s
🧠 **Embedding time**: {ingestion_info['embedding_time']:.2f}s
💾 **Storage time**: {ingestion_info['storage_time']:.2f}s
📊 **Total documents in DB**: {ingestion_info['db_size']}

Ready to answer questions about this document!"""
            
            return feedback
            
        except Exception as e:
            return f"❌ **Error processing file**: {str(e)}"
    
    def ask_question(self, question, k_value):
        """Handle question asking with comprehensive response"""
        if not question.strip():
            return "Please enter a question.", "No sources available."
        
        try:
            # Get response from RAG system
            response = self.chat_bot.chat(question, k=int(k_value))
            
            if not response['success']:
                return f"❌ Error: {response['answer']}", "No sources"
            
            # Format main answer
            answer = f"""**Answer:**
{response['answer']}

---
**Performance:**
⚡ Retrieval: {response['retrieval_time']:.2f}s
🤖 Generation: {response['generation_time']:.2f}s
⏱️ Total: {response['total_time']:.2f}s
🧠 Model: {response['model_used']}"""
            
            # Format sources
            if response['sources']:
                sources_text = "**📚 Sources Used:**\n\n"
                for i, source in enumerate(response['sources'], 1):
                    similarity_pct = source['similarity'] * 100
                    preview = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                    
                    sources_text += f"""**Source {i}** (Similarity: {similarity_pct:.1f}%)
{preview}

---
"""
            else:
                sources_text = "No sources found."
            
            return answer, sources_text
            
        except Exception as e:
            return f"❌ Error: {str(e)}", "Error occurred"
    
    def get_database_info(self):
        """Get current database statistics"""
        try:
            stats = self.vector_store.get_stats()
            return f"""**📊 Database Statistics:**

📄 **Total document chunks**: {stats['total_chunks']}
🧠 **Embedding model**: {stats['embedding_model']}
💾 **Vector database**: {stats['vector_db']}
📍 **Storage location**: ./data/chroma_db/

**Status**: {'✅ Ready' if stats['total_chunks'] > 0 else '⚠️ No documents uploaded'}"""
        except Exception as e:
            return f"❌ Error getting stats: {str(e)}"

# Initialize the app
app = RAGApp()

# Create Gradio interface
with gr.Blocks(title="Local AI Q&A Bot", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🤖 Local AI Q&A Bot with RAG
    ### Upload documents and ask questions - everything runs locally!
    
    Built with: **Ollama** + **LangChain** + **ChromaDB** + **SentenceTransformers**
    """)
    
    with gr.Tab("📤 Upload Documents"):
        gr.Markdown("### Upload PDF, TXT, or Markdown files")
        
        with gr.Row():
            file_input = gr.File(
                label="Choose Document", 
                file_types=[".pdf", ".txt", ".md"]
            )
        
        upload_btn = gr.Button("🔄 Process Document", variant="primary", size="lg")
        upload_output = gr.Markdown(label="Processing Status")
        
        upload_btn.click(
            app.upload_and_process_file,
            inputs=[file_input],
            outputs=[upload_output]
        )
    
    with gr.Tab("❓ Ask Questions"):
        gr.Markdown("### Ask anything about your uploaded documents")
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is the main topic of the document?",
                    lines=3
                )
            with gr.Column(scale=1):
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Sources to retrieve"
                )
        
        ask_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Markdown(label="Answer")
            with gr.Column():
                sources_output = gr.Markdown(label="Sources")
        
        ask_btn.click(
            app.ask_question,
            inputs=[question_input, k_slider],
            outputs=[answer_output, sources_output]
        )
        
        # Sample questions
        gr.Markdown("""
        **💡 Try these sample questions:**
        - "What is the main topic discussed?"
        - "Summarize the key points"
        - "What are the conclusions?"
        - "List the important facts mentioned"
        """)
    
    with gr.Tab("📊 Database Info"):
        gr.Markdown("### Vector Database Statistics")
        
        info_btn = gr.Button("🔄 Refresh Info", variant="secondary")
        info_output = gr.Markdown()
        
        # Auto-load info on tab open
        demo.load(app.get_database_info, outputs=[info_output])
        info_btn.click(app.get_database_info, outputs=[info_output])

if __name__ == "__main__":
    print("🚀 Starting Local AI Q&A Bot...")
    print("📝 Make sure Ollama is running: ollama serve")
    demo.launch(share=True, debug=True)

