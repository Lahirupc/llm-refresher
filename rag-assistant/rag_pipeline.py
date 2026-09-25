import tempfile
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from open_router import OpenRouterEmbeddings

load_dotenv()

# read .md file as knowledge base
with open("./docs/langchain_knowledge_base.md", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = f.read()

embeddings_model = OpenRouterEmbeddings(model="liquid/lfm-2.5-embedding-350m:free")

def create_kb():
    """Create a vector store from knowledge base."""
    
    # split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"})
    
    chunks = splitter.split_documents([doc])
    
    # create a vector store from the chunks
    vector_store = Chroma.from_documents(chunks, embeddings_model, collection_name="my_collection", persist_directory=tempfile.mkdtemp())
    
    return vector_store

def demo_basic_rag():
    
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash-fin:free", temperature=0)
    
    # RAG prompt template
    system_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Answer:
    
    
    Make sure to answer in a concise manner, and if you don't know the answer, say "I don't know".
    """)

    def format_docs(docs):
        return "\n\n".join([f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        |system_prompt
        | llm
        | StrOutputParser()
    )
    
        
    # Test the RAG chain
    # Test
    questions = ["What is are runnables?", "Who created LangChain?", "What is LangGraph used for?"]

        
        
    print("Basic RAG Demo:\n")
    for q in questions:
            answer = rag_chain.invoke(q)
            print(f"Question: {q}\nAnswer: {answer}\n")
            
def demo_rag_with_sources():

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context below. Include which sources you used.

        Context:
        {context}

        Question: {question}

        Answer (include sources):"""
            )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")
