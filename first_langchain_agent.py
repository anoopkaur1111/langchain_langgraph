from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search(query: str) -> str:
    """
    Tool to search the web for information.
    
    Args:        query (str): The search query.
    Returns:        str: The search results.
    """
    print(f"Searching for: {query}")
    # return "Pitt weather is not good today"
    return tavily.search(query)

llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Welcome to the LangChain Agent!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather like in Pittsburgh today?")})
    print(f"Agent response: {result}")

if __name__ == "__main__":
    main()
