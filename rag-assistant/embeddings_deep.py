import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from open_router import OpenRouterEmbeddings


load_dotenv()

embeddings = OpenRouterEmbeddings(model="liquid/lfm-2.5-embedding-350m:free")

def basic_embeddings():
    # sample text to generate embeddings for
    text = "What is machine learning?"
    
    # Generate embeddings for the sample texts
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimentions: {len(single_embedding)}")
    print(f"First 5 values of the embedding: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")
    
def batch_embeddings():
    # sample texts to generate embeddings for
    texts = [
        "What is machine learning?",
        "What is deep learning?",
        "What is natural language processing?"
    ]
    
    # Generate embeddings for the sample texts
    batch_embeddings = embeddings.embed_documents(texts)
    for i, emb in enumerate(batch_embeddings):
        print(f"Text: {texts[i]}")
        print(f"Vector dimentions: {len(emb)}")
        print(f"First 5 values of the embedding: {emb[:5]}")
        print(f"Vector norm: {np.linalg.norm(emb):.4f}")
        print("-" * 50)

def similariy_search():
    # sample texts to generate embeddings for
    docs = [
        "python is a programming language",
        "Javascript is used for web development",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning is a subset of machine learning",
        "Cats are great pets"
    ]
    
    query = "What programming languages exist?"
    
    
    # Generate embeddings for the sample texts
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)
    

    def cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]
    
    # rank documents by the similarity 
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    
    print(f"Query: {query}")
    print("Ranked documents by similarity:")
    for doc, score in ranked_docs:
        print(f"Similarity Score: {score:.4f} | Document: {doc}")
    
if __name__ == "__main__":
    similariy_search()