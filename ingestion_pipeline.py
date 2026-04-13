import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()

def load_documents(docs_path="docx"):
    print(f"Loading documents from {docs_path}....")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    loader = DirectoryLoader(path=docs_path,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i,  doc in enumerate(documents[:2]):
        print(f"\ndocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n ---Chunk {i+1} ---")
            print(f"Source: {len(chunk.page_content)} characters.")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and stroing in chromadb...")
    openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not openrouter_api_key:
        raise EnvironmentError("Missing OPEN_ROUTER_API_KEY in .env")

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    print("---Creating Vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("---Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Main Function")
    #1. Load all the files
    documents= load_documents(docs_path="docx")
    #2. Chunking the files
    chunks=split_documents(documents)
    #3. Embedding and storing in Vector DB
    vectorstore=create_vector_store(chunks)


if __name__ == "__main__":
    main()
