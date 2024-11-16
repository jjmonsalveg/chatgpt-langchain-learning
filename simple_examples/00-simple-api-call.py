import argparse
import os

from dotenv import load_dotenv
from langchain.llms import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")

args = parser.parse_args()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# if you load the OPENAI_API_KEY the parameter is not necessary
llm = OpenAI(openai_api_key=api_key)
result = llm("Write a very very short poem")
