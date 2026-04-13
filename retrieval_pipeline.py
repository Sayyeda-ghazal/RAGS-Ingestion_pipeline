from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
if not openrouter_api_key:
    raise EnvironmentError("Missing OPEN_ROUTER_API_KEY in .env")

persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )

db = Chroma(
    persist_directory= persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "what was micorsoft's first launched product?"

retriver = db.as_retriever(search_kwargs={"k":10})

relevant_docs = retriver.invoke(query)

print(f"User Query: {query}")
print("---Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
