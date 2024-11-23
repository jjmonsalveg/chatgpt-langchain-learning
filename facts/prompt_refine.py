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

# Emulate a conversation providing in sequence more context
chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="refine", verbose=True
)

result = chain.run("What is an interesting fact about the English language?")

print(result)

# DEBUG TRACE
# [chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
# {
#   "query": "What is an interesting fact about the English language?"
# }
# [chain/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain] Entering Chain run with input:
# [inputs]
# [chain/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 4:chain:LLMChain] Entering Chain run with input:
# {
#   "context_str": "1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"\n2. An ostrich's eye is bigger than its brain.\n3. Honey is the only natural food that is made without destroying any kind of life.",
#   "question": "What is an interesting fact about the English language?"
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 4:chain:LLMChain > 5:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Context information is below. \n------------\n1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"\n2. An ostrich's eye is bigger than its brain.\n3. Honey is the only natural food that is made without destroying any kind of life.\n------------\nGiven the context information and not prior knowledge, answer the question: What is an interesting fact about the English language?"
#   ]
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 4:chain:LLMChain > 5:llm:OpenAI] [936ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" ",
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
#       "total_tokens": 111,
#       "completion_tokens": 30,
#       "prompt_tokens": 81
#     },
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [chain/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 4:chain:LLMChain] [937ms] Exiting Chain run with output:
# {
#   "text": "\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" "
# }
# [chain/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 6:chain:LLMChain] Entering Chain run with input:
# {
#   "context_str": "4. A snail can sleep for three years.\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\n6. The elephant is the only mammal that can't jump.",
#   "existing_answer": "\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" ",
#   "question": "What is an interesting fact about the English language?"
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 6:chain:LLMChain > 7:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "The original question is as follows: What is an interesting fact about the English language?\nWe have provided an existing answer: \nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" \nWe have the opportunity to refine the existing answer (only if needed) with some more context below.\n------------\n4. A snail can sleep for three years.\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\n6. The elephant is the only mammal that can't jump.\n------------\nGiven the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer."
#   ]
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 6:chain:LLMChain > 7:llm:OpenAI] [1.77s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. ",
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
#       "total_tokens": 295,
#       "completion_tokens": 138,
#       "prompt_tokens": 157
#     },
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [chain/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 6:chain:LLMChain] [1.77s] Exiting Chain run with output:
# {
#   "text": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. "
# }
# [chain/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 8:chain:LLMChain] Entering Chain run with input:
# {
#   "context_str": "86. Broccoli and cauliflower are the only vegetables that are flowers.\n87. The dot over an 'i' or 'j' is called a tittle.\n88. A group of owls is called a parliament.",
#   "existing_answer": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. ",
#   "question": "What is an interesting fact about the English language?"
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 8:chain:LLMChain > 9:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "The original question is as follows: What is an interesting fact about the English language?\nWe have provided an existing answer: \n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. \nWe have the opportunity to refine the existing answer (only if needed) with some more context below.\n------------\n86. Broccoli and cauliflower are the only vegetables that are flowers.\n87. The dot over an 'i' or 'j' is called a tittle.\n88. A group of owls is called a parliament.\n------------\nGiven the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer."
#   ]
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 8:chain:LLMChain > 9:llm:OpenAI] [2.06s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" ",
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
#       "total_tokens": 444,
#       "completion_tokens": 188,
#       "prompt_tokens": 256
#     },
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [chain/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 8:chain:LLMChain] [2.06s] Exiting Chain run with output:
# {
#   "text": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" "
# }
# [chain/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 10:chain:LLMChain] Entering Chain run with input:
# {
#   "context_str": "118. The original Star-Spangled Banner was sewn in Baltimore.\n119. The average adult spends more time on the toilet than they do exercising.",
#   "existing_answer": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" ",
#   "question": "What is an interesting fact about the English language?"
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 10:chain:LLMChain > 11:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "The original question is as follows: What is an interesting fact about the English language?\nWe have provided an existing answer: \n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" \nWe have the opportunity to refine the existing answer (only if needed) with some more context below.\n------------\n118. The original Star-Spangled Banner was sewn in Baltimore.\n119. The average adult spends more time on the toilet than they do exercising.\n------------\nGiven the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer."
#   ]
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 10:chain:LLMChain > 11:llm:OpenAI] [2.66s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" Furthermore, the original Star-Spangled Banner, the national anthem of the United States, was sewn in Baltimore. And on a quirky note, the average adult spends more time on the toilet than they do exercising.",
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
#       "total_tokens": 523,
#       "completion_tokens": 231,
#       "prompt_tokens": 292
#     },
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [chain/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain > 10:chain:LLMChain] [2.66s] Exiting Chain run with output:
# {
#   "text": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" Furthermore, the original Star-Spangled Banner, the national anthem of the United States, was sewn in Baltimore. And on a quirky note, the average adult spends more time on the toilet than they do exercising."
# }
# [chain/end] [1:chain:RetrievalQA > 3:chain:RefineDocumentsChain] [7.43s] Exiting Chain run with output:
# {
#   "output_text": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" Furthermore, the original Star-Spangled Banner, the national anthem of the United States, was sewn in Baltimore. And on a quirky note, the average adult spends more time on the toilet than they do exercising."
# }
# [chain/end] [1:chain:RetrievalQA] [8.07s] Exiting Chain run with output:
# {
#   "result": "\n\nOne interesting fact about the English language is that \"Dreamt\" is the only word in the language that ends with the letters \"mt.\" This is similar to the word \"slept,\" as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, \"pneumonoultramicroscopicsilicovolcanoconiosis,\" contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word \"elephant\" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a \"tittle,\" and a group of owls is called a \"parliament.\" Furthermore, the original Star-Spangled Banner, the national anthem of the United States, was sewn in Baltimore. And on a quirky note, the average adult spends more time on the toilet than they do exercising."
# }


# One interesting fact about the English language is that "Dreamt" is the only word in the language that ends with the letters "mt." This is similar to the word "slept," as snails can also sleep for long periods of time, up to three years. Additionally, the longest word in the English language, "pneumonoultramicroscopicsilicovolcanoconiosis," contains a total of 45 letters and is a medical term for a lung disease caused by inhaling very fine silica particles. In comparison, the word "elephant" is relatively short and simple, but it is the only mammal that cannot jump. Another interesting fact is that broccoli and cauliflower are the only vegetables that are actually flowers. Additionally, the dot over an 'i' or 'j' is called a "tittle," and a group of owls is called a "parliament." Furthermore, the original Star-Spangled Banner, the national anthem of the United States, was sewn in Baltimore. And on a quirky note, the average adult spends more time on the toilet than they do exercising.

