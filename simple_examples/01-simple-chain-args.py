import argparse
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")

args = parser.parse_args()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# if you load the OPENAI_API_KEY the parameter is not necessary
llm = OpenAI()
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very very short {language} function that will {task}",
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
result = code_chain.run(language=args.language, task=args.task)
print(result)
