from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

load_dotenv()

openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
if not openrouter_api_key:
    raise EnvironmentError("Missing OPEN_ROUTER_API_KEY in .env")


persistent_directory="db/chorma_db"
embeddings= OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=256,
)
query = "what was micorsoft's first launched product?"

chat_history = []

def ask_question(user_question):
    print(f"\n --- You asked: {user_question} ---")
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone.")
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k":3})
    docs= retriever.invoke(search_question)

    print(f" found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Doc {i}: {preview}...")
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

def start_chat():
    print(" Ask me questions! Type 'quit' to exit")
    while True:
        question = input("\n your question: ")
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()