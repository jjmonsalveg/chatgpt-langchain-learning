import os

from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# if you load the OPENAI_API_KEY the parameter is not necessary
llm = OpenAI(openai_api_key=api_key)
result = llm("Write a very very short poem")
print(result)
