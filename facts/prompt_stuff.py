from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Database was already created and populated we just connect to it (playground.py was ran)
db = Chroma(persist_directory="emb", embedding_function=embeddings)
retriever = db.as_retriever()

# stuff is Take some context from the vector store and "stuff" it into the prompt
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.run("What is an interesting fact about the English language?")

print(result)
