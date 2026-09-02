import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash-fin:free", temperature=0)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
response_openrouter = llm.invoke(messages)
print(f"Response from ChatOpenRouter: {response_openrouter}")

embedding = ChatOpenRouter(model_kwargs={"input": "What is the capital of France?"}, model="liquid/lfm-2.5-embedding-350m:free")
print(f"Embedding from ChatOpenRouter: {embedding}")

import requests

response = requests.post(
  "https://openrouter.ai/api/v1/embeddings",
  headers={
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
  },
  json={
    "model": "liquid/lfm-2.5-embedding-350m:free",
    "input": "What is the capital of France?"
  }
)

data = response.json()
embedding = data["data"][0]["embedding"]
print(f"Embedding dimension: {len(embedding)}")
# print(embedding)
# print(data)