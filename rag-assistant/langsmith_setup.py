import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from langsmith import traceable

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

@traceable(name="basic_chaining")
def demo_basic_tracing():
    """Basic Langsmith tracing."""
    
    llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash-fin:free", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    print("Running chain with Langsmith tracing enabled")
    
    result = chain.invoke({"topic": "machine learning"})
    
    print(f"Result: {result}")

@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """Name your runs for easier identification"""
    
    llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash-fin:free", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize: {text}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    print("Running chain with Langsmith tracing enabled")
    
    result = chain.invoke({"text": """Machine learning is an area of computer science which enables
                           machines to train models by looking at data."""})
    
    print(f"Result: {result}")



@traceable(name="trace_with_metadata_demo", tags=["metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Name your runs for easier identification"""
    
    llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash-fin:free", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize: {text}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    print("Running chain with Langsmith tracing enabled")
    
    result = chain.invoke("Hello from user {user_id}")
    
    print(f"Result: {result}")



if __name__ == "__main__":
    demo_basic_tracing()
    demo_named_runs()
    demo_trace_with_metadata(user_id="user_123", request_type="greeting")



