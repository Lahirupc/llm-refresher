from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
from open_router import OpenRouterEmbeddings

load_dotenv()

def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrivers using weighted Reciprocal Rank Fusion."""
    doc_scores = {} # page_content -> (score, doc)
    
    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0/(rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]




document_1 = Document(
    page_content="Product SKU-7742X is a high-quality wireless Bluetooth speaker with excellent sound clarity and long battery life.",
    metadata={"type": "product"},
    id=1,
)

document_2 = Document(
    page_content="For the connectivity issues, please ensure that your device's Bluetooth is turned on and that the speaker is in pairing mode. If problems persist, try resetting the speaker by holding down the power button for 10 seconds.",
    metadata={"type": "troubleshooting"},
    id=2,
)

document_3 = Document(
    page_content="Error code E_CONN_REFUSED indicates that the device is unable to establish a connection with the server. This may be due to network issues or server downtime. Please check your internet connection and try again later.",
    metadata={"type": "error"},
    id=3,
)

document_4 = Document(
    page_content="The authentication process requires a valid API key. Please ensure that you have provided the correct API key in your request headers. If you continue to experience issues, contact support for assistance.",
    metadata={"type": "auth"},
    id=4,
)

document_5 = Document(
    page_content="Router configuration settings can be accessed through the admin panel. To log in, enter the router's IP address (192.168.1.1) in your web browser and use the default username and password provided in the user manual. For security reasons, it is recommended to change the default credentials after the initial setup.",
    metadata={"type": "config"},
    id=5,
)

document_6 = Document(
    page_content="WCAG 2.1 compliance ensures that web content is accessible to all users, including those with disabilities. It includes guidelines for text alternatives, keyboard navigation, and color contrast. Adhering to these standards helps create an inclusive online experience.",
    metadata={"type": "compliance"},
    id=6,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,

]


class HybridRetriever:
    """Production hybrid retriever with BM25 + Vector search"""

    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int = 4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
        
        # Initialize embeddings
        self.embeddings = OpenRouterEmbeddings()
        
        # Create vector store and retriever
        self.vectorstore = Chroma.from_documents(
            documents, self.embeddings, collection_name="hybrid_search"
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)
        
    def search(self, query: str) -> List[Document]:
        """Run hybrid search using weighted RRF"""
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        )
        
    def add_documents(self, documents: List[Document]):
        """Add new documents to both retrievers"""
        self.vectorstore.add_documents(documents)
        
        # Recreate BM25 (it doesn't support incremental adds)
        all_docs = self.vectorstore.get()
        self.bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=doc) for doc in all_docs["documents"]], k=self.k
        )
        
        
# Usage
retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
results = retriever.search("SKU-7742X specification")

for doc in results:
    print("-" *60)
    print(doc.page_content[:100])
