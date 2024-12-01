from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# In case you want to debug
# import langchain
# langchain.debug = True
load_dotenv()

llm = OpenAI()
embeddings = OpenAIEmbeddings()

# Database was already created and populated we just connect to it (playground.py was ran)
db = Chroma(persist_directory="emb", embedding_function=embeddings)
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="map_reduce", verbose=True
)

result = chain.run("What is an interesting fact about the English language?")

print(result)

# DEBUG TRACE
# [chain/start] [1:chain:RetrievalQA] Entering Chain run with input:
# {
#   "query": "What is an interesting fact about the English language?"
# }
# [chain/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain] Entering Chain run with input:
# [inputs]
# [chain/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain] Entering Chain run with input:
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
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 5:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following portion of a long document to see if any of the text is relevant to answer the question. \nReturn any relevant text verbatim.\n1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"\n2. An ostrich's eye is bigger than its brain.\n3. Honey is the only natural food that is made without destroying any kind of life.\nQuestion: What is an interesting fact about the English language?\nRelevant text, if any:"
#   ]
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 6:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following portion of a long document to see if any of the text is relevant to answer the question. \nReturn any relevant text verbatim.\n4. A snail can sleep for three years.\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\n6. The elephant is the only mammal that can't jump.\nQuestion: What is an interesting fact about the English language?\nRelevant text, if any:"
#   ]
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 7:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following portion of a long document to see if any of the text is relevant to answer the question. \nReturn any relevant text verbatim.\n86. Broccoli and cauliflower are the only vegetables that are flowers.\n87. The dot over an 'i' or 'j' is called a tittle.\n88. A group of owls is called a parliament.\nQuestion: What is an interesting fact about the English language?\nRelevant text, if any:"
#   ]
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 8:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Use the following portion of a long document to see if any of the text is relevant to answer the question. \nReturn any relevant text verbatim.\n118. The original Star-Spangled Banner was sewn in Baltimore.\n119. The average adult spends more time on the toilet than they do exercising.\nQuestion: What is an interesting fact about the English language?\nRelevant text, if any:"
#   ]
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 5:llm:OpenAI] [1.11s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": " 1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"",
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
#       "prompt_tokens": 376,
#       "total_tokens": 458,
#       "completion_tokens": 82
#     },
#     "model_name": "gpt-3.5-turbo-instruct"
#   },
#   "run": null
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 6:llm:OpenAI] [1.11s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'",
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
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 7:llm:OpenAI] [1.11s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "\n87. The dot over an 'i' or 'j' is called a tittle.",
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
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain > 8:llm:OpenAI] [1.11s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": " There is no relevant text in this portion of the document.",
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
# [chain/end] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 4:chain:LLMChain] [1.11s] Exiting Chain run with output:
# {
#   "outputs": [
#     {
#       "text": " 1. \"Dreamt\" is the only English word that ends with the letters \"mt.\""
#     },
#     {
#       "text": "\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'"
#     },
#     {
#       "text": "\n87. The dot over an 'i' or 'j' is called a tittle."
#     },
#     {
#       "text": " There is no relevant text in this portion of the document."
#     }
#   ]
# }
# [chain/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 9:chain:LLMChain] Entering Chain run with input:
# {
#   "question": "What is an interesting fact about the English language?",
#   "summaries": " 1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"\n\n\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\n\n\n87. The dot over an 'i' or 'j' is called a tittle.\n\n There is no relevant text in this portion of the document."
# }
# [llm/start] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 9:chain:LLMChain > 10:llm:OpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "Given the following extracted parts of a long document and a question, create a final answer. \nIf you don't know the answer, just say that you don't know. Don't try to make up an answer.\n\nQUESTION: Which state/country's law governs the interpretation of the contract?\n=========\nContent: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.\n\nContent: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.\n\nContent: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,\n=========\nFINAL ANSWER: This Agreement is governed by English law.\n\nQUESTION: What did the president say about Michael Jackson?\n=========\nContent: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.\n\nContent: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.\n\nContent: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.\n\nContent: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.\n=========\nFINAL ANSWER: The president did not mention Michael Jackson.\n\nQUESTION: What is an interesting fact about the English language?\n=========\n 1. \"Dreamt\" is the only English word that ends with the letters \"mt.\"\n\n\n5. The longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'\n\n\n87. The dot over an 'i' or 'j' is called a tittle.\n\n There is no relevant text in this portion of the document.\n=========\nFINAL ANSWER:"
#   ]
# }
# [llm/end] [1:chain:RetrievalQA > 3:chain:MapReduceDocumentsChain > 9:chain:LLMChain > 10:llm:OpenAI] [826ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": " There is no relevant text in this portion of the document.",
#         "generation_info": {
#           "finish_reason": "stop",
#           "logprobs": null
#         },
#         "type": "Generation"
#       }
#     ]
#   ],
