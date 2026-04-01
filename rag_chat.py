import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
PDF_PATH = "my_document.pdf"  # REPLACE THIS with your PDF filename
MODEL_NAME = "llama3.2:1b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_PATH = "./chroma_db"

def main():
    # 1. Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"Error: File '{PDF_PATH}' not found. Please add a PDF file.")
        return

    print("Step 1: Loading and splitting PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    
    # Split text into chunks (small enough for the model to digest)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    print(f"   Split into {len(chunks)} chunks.")

    # 2. Embed and Store (Vector Database)
    print("Step 2: Creating embeddings and vector store (this may take a moment)...")
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Create Vector Store (Chroma)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=VECTOR_STORE_PATH
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

    # 3. Setup the Chat Chain
    print("Step 3: Initializing Chat Chain...")
    llm = ChatOllama(model=MODEL_NAME)

    # Prompt Template
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # RAG Pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Chat Loop
    print("\n--- System Ready. Ask questions about your PDF. Type 'exit' to quit. ---")
    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        print("Bot: ", end="", flush=True)
        # Stream the response
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    main()