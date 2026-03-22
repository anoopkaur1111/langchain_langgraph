from typing import List
from pydantic import BaseModel,Field
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source use by the agent"""

    url: str=Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer: str=Field(description="The agent's answer to the question")
    sources: List[Source]=Field(default_factory=list, description="The list of sources used to answer the question")

llm = ChatOpenAI()
tools = [TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Welcome to the LangChain Agent!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather like in Pittsburgh today?")})
    print(f"Agent response: {result}")

if __name__ == "__main__":
    main()
