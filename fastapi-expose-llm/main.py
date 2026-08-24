import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from logging import getLogger

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
    text: str

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
        ai_msg = await model.ainvoke(messages)
        logger.info("Successfully received response from LLM")
        logger.debug(f"LLM content: {ai_msg.content}")
        return TextResponse(text=f"{ai_msg.content[0]['text']}")
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error calling LLM")


@app.get("/")
async def root():
    return {"message": "API is running. Send a POST request to /question"}
