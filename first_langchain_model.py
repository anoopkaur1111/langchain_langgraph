from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
def main():
    print("hello langchain")
    information=" Sarah Janet Maas (born March 5, 1986)[3][4] is an American fantasy author known for her series Throne of Glass, A Court of Thorns and Roses,[5] and Crescent City. As of 2024, she has sold more than 75 million copies of her books and her work has been translated into 40 languages.[6]"

    summary_template = """
    given the information {information} about a person I want you to create:
    1. short summary
    2. two interesting facts about them
"""
    summary_prompt_Template = PromptTemplate(
        input_variables=['information'], template = summary_template
    )
    llm = ChatOpenAI(temperature=0,model="gpt-5")
    chain = summary_prompt_Template | llm
    response = chain.invoke(input = {"information": information})
    print(response.content)
if __name__ =='__main__':
    main()