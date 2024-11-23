from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

load_dotenv()

chat = ChatOpenAI(verbose=True)

# return_messages creates objects to save the results
memory = ConversationSummaryMemory(
    memory_key="messages", return_messages=True, llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        MessagesPlaceholder(
            variable_name="messages"
        ),  # This set up the input extracted from memory
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">> ")
    result = chain({"content": content})

    print(result["text"])
