from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS=10
model = "qwen2.5:3b"

@tool
def get_product_price(product: str) -> float:
    """Look up the product price in the database and return it."""
    print(f"executing get product price with product: {product}")
    prices = {"laptop": 1299.99, "headphones": 199.99, "keyboard": 89.99}
    return prices.get(product,0)

@tool
def get_discount(price:float, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"executing get discount with price: {price} and discount tier: {discount_tier}")
    discounts_percent = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discounts_percent.get(discount_tier,0)
    return round(price*(1-discount/100),2)

### Agent loop
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price, get_discount]
    tool_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{model}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Initial question: {question}")
    
    content = ("you are a helpful shopping assistant... You have access to product catalog tool and a discount tool" \
    "STRICT RULES - you must folllow exacly:" \
    "1. NEVER guess or assume product price" \
    "You must call get_product price tool to get the price of the product" \
    "2. Only call apply_discount AFTER you have received a prioce from get_product_price returned by get_product_price, do not cook up some numbers"\
    "3. Never calculate discount yourself, always use the discount tool"\
      "4. if user does not specify discount tier, askthem which tier they want to use, do not assume"        )

    messages = [SystemMessage(content=content), HumanMessage(content=question)]

    for i in range(1,MAX_ITERATIONS+1):
        print(f"\n--- Iteration {i} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print("No tool calls, final answer:", ai_message.content)
            return ai_message.content
        tool_call = tool_calls[0]
        tool_name=  tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        observation = tool_to_use.invoke(tool_args)
        print(f"Tool {tool_name} returned observation: {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))


    print(messages)
    print("Max iterations reached without a final answer.")

if __name__ == "__main__":
    print("Welcome to the E-commerce Agent!"
          )
    result = run_agent("What is the price of a laptop with a gold discount?")
    print(f"Final result: {result}")
    