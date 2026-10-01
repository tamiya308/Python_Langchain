import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 1. Define Name Chain (Modified to block conversational commentary)
name_prompt = PromptTemplate.from_template(
    "I want to open a restaurant for {cuisine} food. Suggest 1 fancy name for it. "
    "Do not include any quotes, introductory conversational filler, or explanations. "
    "Output ONLY the name itself."
)
name_chain = name_prompt | llm | StrOutputParser()

# 2. Define Menu Chain 
menu_prompt = PromptTemplate.from_template(
    "Suggest 5 menu items for a {cuisine} restaurant named {restaurant_name}. "
    "Make the items fit the cuisine and the restaurant's name and theme. "
    "Do not include any introductory text, conversational greetings, or polite filler. "
    "Start your response directly with the list of items."
)
menu_chain = menu_prompt | llm | StrOutputParser()

# 3. Modern Sequential Chain capturing both variables
full_chain = (
    RunnablePassthrough.assign(restaurant_name=name_chain)
    | {
        "restaurant_name": lambda x: x["restaurant_name"],
        "menu_items": menu_chain
      }
)

# 4. Run the chain and print the clean format
response = full_chain.invoke({"cuisine": "Italian"})

print(f"Restaurant Name: {response['restaurant_name']}\n")
print(f"Menu Items:\n{response['menu_items']}")