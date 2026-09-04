import requests
import os
from langchain_core.embeddings import Embeddings


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