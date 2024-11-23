# In case you want to debug
import langchain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

langchain.debug = True
load_dotenv()

llm = OpenAI()
embeddings = OpenAIEmbeddings()

# Database was already created and populated we just connect to it (playground.py was ran)
db = Chroma(persist_directory="emb", embedding_function=embeddings)
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="map_rerank", verbose=True
)

result = chain.run("What is an interesting fact about the English language?")

print(result)

# DEBUG TRACE
# [chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
# {
#   "query": "What is an interesting fact about the English language?"
# }
# [chain/start] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain] Entering Chain run with input:
# [inputs]
# /Users/jjmonsalve/.local/share/virtualenvs/chatgpt-langchain-learning-Ylf5VL38/lib/python3.11/site-packages/langchain/chains/llm.py:344: UserWarning: The apply_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
#   warnings.warn(
# [chain/start] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain] Entering Chain run with input:
# {
#   "input_list": [
#     {
#       "context": "1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"\n2. An ostrich's eye is bigger than its brain.\n3. Honey is the only natural food that is made without destroying any kind of life.",
#       "question": "What is an interesting fact about the English language?"
#     },
#     {
#       "context": "4. A snail can sleep for three years.\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\n6. The elephant is the only mammal that can't jump.",
#       "question": "What is an interesting fact about the English language?"
#     },
#     {
#       "context": "86. Broccoli and cauliflower are the only vegetables that are flowers.\n87. The dot over an 'i' or 'j' is called a tittle.\n88. A group of owls is called a parliament.",
#       "question": "What is an interesting fact about the English language?"
#     },
#     {
#       "context": "118. The original Star-Spangled Banner was sewn in Baltimore.\n119. The average adult spends more time on the toilet than they do exercising.",
#       "question": "What is an interesting fact about the English language?"
#     }
#   ]
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 5:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nIn addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:\n\nQuestion: [question here]\nHelpful Answer: [answer here]\nScore: [score between 0 and 100]\n\nHow to determine the score:\n- Higher is a better answer\n- Better responds fully to the asked question, with sufficient level of detail\n- If you do not know the answer based on the context, that should be a score of 0\n- Don't be overconfident!\n\nExample #1\n\nContext:\n---------\nApples are red\n---------\nQuestion: what color are apples?\nHelpful Answer: red\nScore: 100\n\nExample #2\n\nContext:\n---------\nit was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv\n---------\nQuestion: what type was the car?\nHelpful Answer: a sports car or an suv\nScore: 60\n\nExample #3\n\nContext:\n---------\nPears are either red or orange\n---------\nQuestion: what color are apples?\nHelpful Answer: This document does not answer the question\nScore: 0\n\nBegin!\n\nContext:\n---------\n1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"\n2. An ostrich's eye is bigger than its brain.\n3. Honey is the only natural food that is made without destroying any kind of life.\n---------\nQuestion: What is an interesting fact about the English language?\nHelpful Answer:"
#   ]
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 6:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nIn addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:\n\nQuestion: [question here]\nHelpful Answer: [answer here]\nScore: [score between 0 and 100]\n\nHow to determine the score:\n- Higher is a better answer\n- Better responds fully to the asked question, with sufficient level of detail\n- If you do not know the answer based on the context, that should be a score of 0\n- Don't be overconfident!\n\nExample #1\n\nContext:\n---------\nApples are red\n---------\nQuestion: what color are apples?\nHelpful Answer: red\nScore: 100\n\nExample #2\n\nContext:\n---------\nit was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv\n---------\nQuestion: what type was the car?\nHelpful Answer: a sports car or an suv\nScore: 60\n\nExample #3\n\nContext:\n---------\nPears are either red or orange\n---------\nQuestion: what color are apples?\nHelpful Answer: This document does not answer the question\nScore: 0\n\nBegin!\n\nContext:\n---------\n4. A snail can sleep for three years.\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\n6. The elephant is the only mammal that can't jump.\n---------\nQuestion: What is an interesting fact about the English language?\nHelpful Answer:"
#   ]
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 7:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nIn addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:\n\nQuestion: [question here]\nHelpful Answer: [answer here]\nScore: [score between 0 and 100]\n\nHow to determine the score:\n- Higher is a better answer\n- Better responds fully to the asked question, with sufficient level of detail\n- If you do not know the answer based on the context, that should be a score of 0\n- Don't be overconfident!\n\nExample #1\n\nContext:\n---------\nApples are red\n---------\nQuestion: what color are apples?\nHelpful Answer: red\nScore: 100\n\nExample #2\n\nContext:\n---------\nit was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv\n---------\nQuestion: what type was the car?\nHelpful Answer: a sports car or an suv\nScore: 60\n\nExample #3\n\nContext:\n---------\nPears are either red or orange\n---------\nQuestion: what color are apples?\nHelpful Answer: This document does not answer the question\nScore: 0\n\nBegin!\n\nContext:\n---------\n86. Broccoli and cauliflower are the only vegetables that are flowers.\n87. The dot over an 'i' or 'j' is called a tittle.\n88. A group of owls is called a parliament.\n---------\nQuestion: What is an interesting fact about the English language?\nHelpful Answer:"
#   ]
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 8:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nIn addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:\n\nQuestion: [question here]\nHelpful Answer: [answer here]\nScore: [score between 0 and 100]\n\nHow to determine the score:\n- Higher is a better answer\n- Better responds fully to the asked question, with sufficient level of detail\n- If you do not know the answer based on the context, that should be a score of 0\n- Don't be overconfident!\n\nExample #1\n\nContext:\n---------\nApples are red\n---------\nQuestion: what color are apples?\nHelpful Answer: red\nScore: 100\n\nExample #2\n\nContext:\n---------\nit was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv\n---------\nQuestion: what type was the car?\nHelpful Answer: a sports car or an suv\nScore: 60\n\nExample #3\n\nContext:\n---------\nPears are either red or orange\n---------\nQuestion: what color are apples?\nHelpful Answer: This document does not answer the question\nScore: 0\n\nBegin!\n\nContext:\n---------\n118. The original Star-Spangled Banner was sewn in Baltimore.\n119. The average adult spends more time on the toilet than they do exercising.\n---------\nQuestion: What is an interesting fact about the English language?\nHelpful Answer:"
#   ]
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 5:llm:OpenAI] [765ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": " \"Dreamt\" is the only English word that ends with the letters \"mt.\"\nScore: 100",
#         "generation_info": {
#           "finish_reason": "stop",
#           "logprobs": null
#         },
#         "type": "Generation"
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "completion_tokens": 85,
#       "total_tokens": 1473,
#       "prompt_tokens": 1388
#     },
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 6:llm:OpenAI] [765ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": " The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\nScore: 80",
#         "generation_info": {
#           "finish_reason": "stop",
#           "logprobs": null
#         },
#         "type": "Generation"
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {},
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 7:llm:OpenAI] [766ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": " The dot over an 'i' or 'j' is called a tittle.\nScore: 80",
#         "generation_info": {
#           "finish_reason": "stop",
#           "logprobs": null
#         },
#         "type": "Generation"
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {},
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain > 8:llm:OpenAI] [766ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": " This document does not answer the question\nScore: 0",
#         "generation_info": {
#           "finish_reason": "stop",
#           "logprobs": null
#         },
#         "type": "Generation"
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {},
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [chain/end] [1:chain:RetrievalQA > 3:chain:MapRerankDocumentsChain > 4:chain:LLMChain] [767ms] Exiting Chain run with output:
# {
#   "outputs": [
#     {
#       "text": " \"Dreamt\" is the only English word that ends with the letters \"mt.\"\nScore: 100"
#     },
#     {
#       "text": " The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\nScore: 80"