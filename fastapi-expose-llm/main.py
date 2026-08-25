import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from logging import getLogger
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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




class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    translated_text: str = Field(
            description="Translated user sentence"
        )
    is_formal: bool = Field(
        description="Whether the translated text is formal"
    )

structured_model = model.with_structured_output(TextResponse, method="json_mode")

@app.post("/question", response_model=TextResponse)
async def handle_text(request: TextRequest):
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
        output_parser = PydanticOutputParser(pydantic_object=TextResponse)
        format_instructions = output_parser.get_format_instructions()
        logger.info(f"Format instructions: {format_instructions}")

        ai_msg = await structured_model.ainvoke(f"{messages} {format_instructions}")

        logger.info("Successfully received response from LLM")
        logger.debug(ai_msg)
        # enforce the response model
        return TextResponse(
            translated_text=ai_msg.translated_text,
            is_formal=ai_msg.is_formal
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error calling LLM")


@app.get("/")
async def root():
    return {"message": "API is running. Send a POST request to /question"}
