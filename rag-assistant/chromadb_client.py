#uv pip install chromadb
import chromadb_client

chroma_client = chromadb_client.Client()
collection_name = "my_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Define a sample document to add to the collection
documents = [
    {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog", "metadata": {"topic": "animal"}},
    {"id": "doc2", "text": "A journey of a thousand miles begins with a single step", "metadata": {"topic": "travel"}},
    {"id": "doc3", "text": "To be or not to be, that is the question", "metadata": {"topic": "philosophy"}}
]


for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], metadatas=[doc["metadata"]])

# query the collection
def query():
    query = "quick brown fox"
    results = collection.query(query_texts=[query], n_results=2)
    print(results)

# similarity
def compute_similarity():
    similarity_scores = []
    # similarity from distance
    # similarity = 1/ (1 + distance)
    # or
    # similarity = 1 - (distance / max_distance)
    results = collection.query(query_texts=["quick brown fox"], n_results=2)
    for distance in results['distances'][0]:
        similarity = 1 / (1 + distance)
        similarity_scores.append(similarity)

    print("Similarity Scores:", similarity_scores)

# metadata filtering
def filter_by_metadata():
    query = "quick brown fox"
    results = collection.query(query_texts=[query], n_results=2, where={"topic": "animal"})
    print(results)



filter_by_metadata()