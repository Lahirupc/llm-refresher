import tempfile

from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever


from dotenv import load_dotenv
from open_router import OpenRouterEmbeddings

load_dotenv()

embeddings = OpenRouterEmbeddings(model="liquid/lfm-2.5-embedding-350m:free")


document_1 = Document(
    page_content="Product SKU-7743X is a high-quality wireless Bluetooth speaker with excellent sound clarity and long battery life.",
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


print(f"Loaded {len(documents)} documents for embedding generation.")

vector_store = Chroma.from_documents(
    documents, 
    embeddings, 
    collection_name="my_collection", 
    persist_directory=tempfile.mkdtemp()
)

# create a retriever from the vector store
vector_retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}
)

print('Vector retriever ready.')


# BM25 works on the raw text
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3   
)

print("BM25 retriver ready")

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
    search_kwargs={"k": 3}
)

print("Hybrid retriever ready")

def test_query(query, name, retriever):
    """
    Test a query against a retriever and print the results.
    """
    print(f"\n--- Testing {name} ---")
    results = retriever.invoke(query)
    print(f"\n{name} - Query: {query}")
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f"Result {i+1}: {preview}")
    return results

# Test queries designed to challenge the vector search
test_queries = [
    "SKU-7743X specification",
    "Error code E_CONN_REFUSED",
    "How do I authenticate?",
    "WCAG compliance",
    "router configuration",
]

for query in test_queries:
    print("=" *60)

    # vector only
    vector_results = test_query(query, 'VECTOR', vector_retriever)

    # BM25 only
    bm25_results = test_query(query, 'BM25', bm25_retriever)

    # Hybrid
    hybrid_results = test_query(query, 'HYBRID', ensemble_retriever)



