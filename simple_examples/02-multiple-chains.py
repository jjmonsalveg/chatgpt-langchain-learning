import argparse
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")

args = parser.parse_args()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI()
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very very short {language} function that will {task}",
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)

result = chain({"language": args.language, "task": args.task})

print(">>>>> Generated code:")
print(result["code"])

print("\n>>>>> Generated test:")
print(result["test"])
