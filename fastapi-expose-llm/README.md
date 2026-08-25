This is an API that exposes an LLM through FastAPI.

To run this API in dev mode:

1. Install uv globally
2. Install dependencies using `uv sync` (this will create a virtual environment on the root directory)
3. Run the API using `uv run fastapi dev`

This has structured output enabled using PydanticOutputParser. Which enforces the model to return a json object that conforms to the Pydantic model.

To test this API, you can use the following curl command:

```bash
curl -X POST http://localhost:8000/question -H "Content-Type: application/json" -d "{\"text\": \"Hello, how are you?\"}"
```

