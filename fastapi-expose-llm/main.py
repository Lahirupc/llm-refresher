import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from logging import getLogger
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# Access the values using os.environ
api_key = os.environ.get("GOOGLE_API_KEY")

app = FastAPI(title="Text Processing API")

# disable AFC
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    # stream_usage=True,
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    api_key=api_key,  # If you prefer to pass api key in directly
    # base_url="...",
    # organization="...",
    # other params...
)

# tools with pydantic models
# class Add(BaseModel):
#     """
#     Add two numbers
#     """
#     a: int = Field(description="First number")
#     b: int = Field(description="Second number")

# class Multiply(BaseModel):
#     """
#     Multiply two numbers
#     """
#     a: int = Field(description="First number")
#     b: int = Field(description="Second number")

@tool
def add(a: int, b: int) -> int:
    """
    Add two numbers
    """
    return a + b +1

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers
    """
    return a * b

# bind tools to model
tools = [add, multiply]
model = model.bind_tools(tools)


# request and response models
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    translated_text: str = Field(description="Translated user sentence")
    is_formal: bool = Field(description="Whether the translated text is formal")

class CalculatedResponse(BaseModel):
    result: int = Field(description="The result of the calculation")
    answer_in_words: str = Field(description="The answer in words")




@app.get("/")
async def root():   
    return {"message": "API is running. Send a POST request to /translate"}

@app.post("/translate", response_model=TextResponse)
def handle_text(request: TextRequest):
    """
    Accepts text and responds with text
    """
    messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", request.text),
    ]

    try:
        # structured output
        structured_model = model.with_structured_output(TextResponse, method="json_mode")
        output_parser = PydanticOutputParser(pydantic_object=TextResponse)
        format_instructions = output_parser.get_format_instructions()
        structured_response = structured_model.invoke(f"{messages} {format_instructions}")
        logger.debug(f"translate - request: {messages} {format_instructions}")

        logger.info("Successfully received response from LLM")
        logger.debug(f"translate - response: {structured_response}")
        # enforce the response model
        return TextResponse(
            translated_text=structured_response.translated_text,
            is_formal=structured_response.is_formal
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error calling LLM")



@app.post("/calculate", response_model=CalculatedResponse)
def handle_text(request: TextRequest):
    """
    Accepts text and responds with text
    """
    messages = [
    (
        "system",
        "You are a helpful assistant that calculates numbers. Use the tools to answer the user's request.",
    ),
    ("human", request.text),
    ]

    try:
        # structured output
        structured_model = model.with_structured_output(CalculatedResponse, method="json_mode")
        output_parser = PydanticOutputParser(pydantic_object=CalculatedResponse)
        format_instructions = output_parser.get_format_instructions()
        structured_response = structured_model.invoke(f"{messages} {format_instructions}")
        logger.debug(f"calculate - request: {messages} {format_instructions}")

        logger.info("Successfully received response from LLM")
        logger.debug(f"calculate - response: {structured_response}")
        # enforce the response model
        return CalculatedResponse(
            result=structured_response.result,
            answer_in_words=structured_response.answer_in_words
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error calling LLM")