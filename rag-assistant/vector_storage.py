from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

load_dotenv()
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str = "liquid/lfm-2.5-embedding-350m:free"):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        import requests
        import os

        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts
            }
        )

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.embed_documents([text])[0]

# # Running in-memory vector store
# from langchain_chroma import Chroma
# vector_store = Chroma(collection_name="my_collection", embedding_function=embeddings)

embeddings = OpenRouterEmbeddings(model="liquid/lfm-2.5-embedding-350m:free")

# Running with data persistence
from langchain_chroma import Chroma
from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]

import tempfile
# Add items to the vector store

dir = "./chroma_db"

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=dir
)

print(f"Vector store created with {len(documents)} documents and persisted to {dir}.")

query = "What did I have for breakfast?"
# # perfrom similarity search
# results = vector_store.similarity_search(query, k=3)

# for i, result in enumerate(results):
#     print(f"Result {i + 1}: Content: {result.page_content}, Metadata: {result.metadata}")
#     print()

# # Perform similarity search with scores
# results = vector_store.similarity_search_with_score(query, k=3)

# for i, (result, score) in enumerate(results):
#     print(f"Result {i + 1}: Content: {result.page_content}, Metadata: {result.metadata}, Score: {score}")
#     print()
    
# metadata filtering
filtered_results = vector_store.similarity_search(query, k=3, filter={"source": "tweet"})

for i, result in enumerate(filtered_results):
    print(f"Filtered Result {i + 1}: Content: {result.page_content}, Metadata: {result.metadata}")
    print()

