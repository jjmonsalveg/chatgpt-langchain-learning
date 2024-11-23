from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
embeddings = OpenAIEmbeddings()

# this calculate some embeddings using openAI
emb =embeddings.embed_query("hi there")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# reach openAI with docs
db = Chroma.from_documents(docs, embeddings, persist_directory="emb")


# for doc in docs:
#     print(doc.page_content)
#     print("\n")

# results = db.similarity_search_with_score(
#     "What is an interesting fact about the English language?",
#     k=2 # give N results (optional parameter)
# )
# for result in results:
#     print("\n")
#     print(result[1]) # score
#     print(result[0].page_content)


results = db.similarity_search(
    "What is an interesting fact about the English language?",
)
for result in results:
    print("\n")
    print(result.page_content)