from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))]
#this adds new argument when making the call - search_depth and in this case it was fast
llm = ChatOpenAI()
agent = create_agent(model=llm, tools=tools)

def main():
    print("Welcome to the LangChain Agent!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather like in Pittsburgh today?")})
    print(f"Agent response: {result}")

if __name__ == "__main__":
    main()
