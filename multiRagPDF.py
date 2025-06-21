import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms import OpenAI  # Replace with your preferred LLM
from langchain.chains import RetrievalQA

# 1. Define Qwen3-Embedding wrapper for LangChain
class Qwen3Embedding(Embeddings):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding"):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def embed_documents(self, texts):
        # Batch encode, mean pool, return numpy vectors
        import torch
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings

# 2. Load and chunk PDFs
def load_and_chunk_pdfs(pdf_paths):
    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

# 3. Embed and store in FAISS
def create_vector_store(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

# 4. RAG Chatbot loop
def main(pdf_files):
    # Step 1: Load and chunk
    docs = load_and_chunk_pdfs(pdf_files)
    
    # Step 2: Embedding
    embeddings = Qwen3Embedding()  # Replace with your checkpoint if needed
    
    # Step 3: Vector store
    vectordb = create_vector_store(docs, embeddings)
    
    # Step 4: Setup LLM and QA chain
    llm = OpenAI(temperature=0)  # Or your own LLM
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vectordb.as_retriever(search_kwargs={"k": 4})
    )
    
    print("Multi-PDF RAG Chatbot is ready. Ask questions!")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa.run(query)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    # List of PDF files to load
    pdf_files = ["startup.pdf", "HAL Digital Concierge - BRD V2.0.pdf"]  # Update with your PDF paths
    main(pdf_files)