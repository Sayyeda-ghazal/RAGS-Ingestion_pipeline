from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
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

retriver = db.as_retriever(search_kwargs={"k":3})

relevant_docs = retriver.invoke(query)

print(f"User Query: {query}")
print("---Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"-{doc.page_content} for doc in relevant_docs"])}
Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the document. say, i don't have enough information t answer the question based on the provided documents
"""

model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=256,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

result= model.invoke(messages)
print("\n--- Generated Response ---")
print("Content only:")
print(result.content)
