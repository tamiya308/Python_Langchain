import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Initialize the LLM (Setting temperature lower for more stable generation)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 2. Define the first chain (Restaurant Name)
# Added StrOutputParser() so the output is plain text, not a complex AI Message object
name_prompt = PromptTemplate.from_template(
    "I want to open a restaurant for {cuisine} food. Suggest 1 fancy name for it."
)
name_chain = name_prompt | llm | StrOutputParser()

# 3. Define the second chain (Menu Items)
menu_prompt = PromptTemplate.from_template(
    "Suggest 5 menu items for a {cuisine} restaurant named {restaurant_name}. "
    "Make the items fit the cuisine and the restaurant's name and theme."
)
menu_chain = menu_prompt | llm | StrOutputParser()

# 4. Construct the Sequential Chain
# RunnablePassthrough.assign keeps the original 'cuisine' variable and adds 'restaurant_name' to the dictionary
full_chain = (
    RunnablePassthrough.assign(restaurant_name=name_chain)
    | menu_chain
)

# 5. Run the sequential chain
response = full_chain.invoke({"cuisine": "Italian"})
print(response)