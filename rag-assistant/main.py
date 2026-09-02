from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain_core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openrouter import ChatOpenRouter

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    # # Test openAI
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # response_openai = llm.invoke("Say 'setup complete!' in one word")
    # print(f"Response from ChatOpenAI: {response_openai}")

    # # Test Anthropic
    # llm = ChatAnthropic(model="claude-sonnet-4-5-20230929", temperature=0)
    # response_anthropic = llm.invoke("Say 'setup complete!' in one word")
    # print(f"Response from ChatAnthropic: {response_anthropic}")

    # Test OpenRouter
    llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash-fin:free", temperature=0)
    response_openrouter = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenRouter: {response_openrouter}")


if __name__ == "__main__":
    main()
