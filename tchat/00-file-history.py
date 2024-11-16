from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

load_dotenv()

chat = ChatOpenAI()

# return_messages creates objects to save the results
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"), # persistence in secondary memory
    memory_key="messages",
    return_messages=True,
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

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input(">> ")
    result = chain({"content": content})

    print(result["text"])
