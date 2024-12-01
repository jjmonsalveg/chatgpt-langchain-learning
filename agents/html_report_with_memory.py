import langchain
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from report import write_report_tool
from tools.sql import describe_tables_tool, run_query_tool, tables

load_dotenv()
langchain.debug = True

chat = ChatOpenAI()
tables = tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to a SQLite database.\n"
                f"The database has tables of: {tables}\n"
                "Do not make any assumptions about what tables exist"
                "or what columns exist. Instead, use the 'describe_tables' function"
            )
        ),
        # chat_history match with the ConversarionBufferMemory key
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        # This is by convention and we need it. (Similar to memory)
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# return_messages return the result as string
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [run_query_tool, describe_tables_tool, write_report_tool]
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools, memory=memory)

agent_executor.run("How many orders are there? Write the result to an html report.")

agent_executor.run("Repeat the exact same process for users")


# output:
# [chain/start] [1:chain:AgentExecutor] Entering Chain run with input:
# {
#   "input": "How many orders are there? Write the result to an html report.",
#   "chat_history": []
# }
# [llm/start] [1:chain:AgentExecutor > 2:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: How many orders are there? Write the result to an html report."
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 2:llm:ChatOpenAI] [713ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "",
#         "generation_info": {
#           "finish_reason": "function_call",
#           "logprobs": null
#         },
#         "type": "ChatGeneration",
#         "message": {
#           "lc": 1,
#           "type": "constructor",
#           "id": [
#             "langchain",
#             "schema",
#             "messages",
#             "AIMessage"
#           ],
#           "kwargs": {
#             "content": "",
#             "additional_kwargs": {
#               "function_call": {
#                 "name": "run_sqlite_query",
#                 "arguments": "{\"query\":\"SELECT COUNT(*) FROM orders\"}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 189,
#       "completion_tokens": 20,
#       "total_tokens": 209,
#       "prompt_tokens_details": {
#         "cached_tokens": 0,
#         "audio_tokens": 0
#       },
#       "completion_tokens_details": {
#         "reasoning_tokens": 0,
#         "audio_tokens": 0,
#         "accepted_prediction_tokens": 0,
#         "rejected_prediction_tokens": 0
#       }
#     },
#     "model_name": "gpt-3.5-turbo",
#     "system_fingerprint": null
#   },
#   "run": null
# }
# [tool/start] [1:chain:AgentExecutor > 3:tool:run_sqlite_query] Entering Tool run with input:
# "{'query': 'SELECT COUNT(*) FROM orders'}"
# [tool/end] [1:chain:AgentExecutor > 3:tool:run_sqlite_query] [1ms] Exiting Tool run with output:
# "[(1500,)]"
# [llm/start] [1:chain:AgentExecutor > 4:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: How many orders are there? Write the result to an html report.\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT COUNT(*) FROM orders\"}'}\nFunction: [[1500]]"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 4:llm:ChatOpenAI] [650ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "",
#         "generation_info": {
#           "finish_reason": "function_call",
#           "logprobs": null
#         },
#         "type": "ChatGeneration",
#         "message": {
#           "lc": 1,
#           "type": "constructor",
#           "id": [
#             "langchain",
#             "schema",
#             "messages",
#             "AIMessage"
#           ],
#           "kwargs": {
#             "content": "",
#             "additional_kwargs": {
#               "function_call": {
#                 "name": "write_report",
#                 "arguments": "{\"filename\":\"orders_report.html\",\"html\":\"<h1>Total number of orders: 1500</h1>\"}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 223,
#       "completion_tokens": 34,
#       "total_tokens": 257,
#       "prompt_tokens_details": {
#         "cached_tokens": 0,
#         "audio_tokens": 0
#       },
#       "completion_tokens_details": {
#         "reasoning_tokens": 0,
#         "audio_tokens": 0,
#         "accepted_prediction_tokens": 0,
#         "rejected_prediction_tokens": 0
#       }
#     },
#     "model_name": "gpt-3.5-turbo",
#     "system_fingerprint": null
#   },
#   "run": null
# }
# [tool/start] [1:chain:AgentExecutor > 5:tool:write_report] Entering Tool run with input:
# "{'filename': 'orders_report.html', 'html': '<h1>Total number of orders: 1500</h1>'}"
# [tool/end] [1:chain:AgentExecutor > 5:tool:write_report] [1ms] Exiting Tool run with output:
# "None"
# [llm/start] [1:chain:AgentExecutor > 6:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: How many orders are there? Write the result to an html report.\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT COUNT(*) FROM orders\"}'}\nFunction: [[1500]]\nAI: {'name': 'write_report', 'arguments': '{\"filename\":\"orders_report.html\",\"html\":\"<h1>Total number of orders: 1500</h1>\"}'}\nFunction: null"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 6:llm:ChatOpenAI] [827ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "I have generated the report. You can download it from the following link: [orders_report.html](sandbox:/orders_report.html)",
#         "generation_info": {
#           "finish_reason": "stop",
#           "logprobs": null
#         },
#         "type": "ChatGeneration",
#         "message": {
#           "lc": 1,
#           "type": "constructor",
#           "id": [
#             "langchain",
#             "schema",
#             "messages",
#             "AIMessage"
#           ],
#           "kwargs": {
#             "content": "I have generated the report. You can download it from the following link: [orders_report.html](sandbox:/orders_report.html)",
#             "additional_kwargs": {}
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 266,
#       "completion_tokens": 27,
#       "total_tokens": 293,
#       "prompt_tokens_details": {
#         "cached_tokens": 0,
#         "audio_tokens": 0
#       },
#       "completion_tokens_details": {
#         "reasoning_tokens": 0,
#         "audio_tokens": 0,
#         "accepted_prediction_tokens": 0,
#         "rejected_prediction_tokens": 0
#       }
#     },
#     "model_name": "gpt-3.5-turbo",
#     "system_fingerprint": null
#   },
#   "run": null
# }
# [chain/end] [1:chain:AgentExecutor] [2.20s] Exiting Chain run with output:
# {
#   "output": "I have generated the report. You can download it from the following link: [orders_report.html](sandbox:/orders_report.html)"
# }
# [chain/start] [1:chain:AgentExecutor] Entering Chain run with input:
# [inputs]
# [llm/start] [1:chain:AgentExecutor > 2:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: How many orders are there? Write the result to an html report.\nAI: I have generated the report. You can download it from the following link: [orders_report.html](sandbox:/orders_report.html)\nHuman: Repeat the exact same process for users"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 2:llm:ChatOpenAI] [370ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "",
#         "generation_info": {
#           "finish_reason": "function_call",
#           "logprobs": null
#         },
#         "type": "ChatGeneration",
#         "message": {
#           "lc": 1,
#           "type": "constructor",
#           "id": [
#             "langchain",
#             "schema",
#             "messages",
#             "AIMessage"
#           ],
#           "kwargs": {
#             "content": "",
#             "additional_kwargs": {
#               "function_call": {
#                 "name": "describe_tables",
#                 "arguments": "{\"table_names\":[\"users\"]}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 230,
#       "completion_tokens": 15,
#       "total_tokens": 245,
#       "prompt_tokens_details": {
#         "cached_tokens": 0,
#         "audio_tokens": 0
#       },
#       "completion_tokens_details": {
#         "reasoning_tokens": 0,
#         "audio_tokens": 0,
#         "accepted_prediction_tokens": 0,
#         "rejected_prediction_tokens": 0
#       }
#     },
#     "model_name": "gpt-3.5-turbo",
#     "system_fingerprint": null
#   },
#   "run": null
# }
# [tool/start] [1:chain:AgentExecutor > 3:tool:describe_tables] Entering Tool run with input:
# "{'table_names': ['users']}"
# [tool/end] [1:chain:AgentExecutor > 3:tool:describe_tables] [0ms] Exiting Tool run with output:
# "CREATE TABLE users (
#     id INTEGER PRIMARY KEY,
#     name TEXT,
#     email TEXT UNIQUE,
#     password TEXT
#     )"
# [llm/start] [1:chain:AgentExecutor > 4:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: How many orders are there? Write the result to an html report.\nAI: I have generated the report. You can download it from the following link: [orders_report.html](sandbox:/orders_report.html)\nHuman: Repeat the exact same process for users\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"users\"]}'}\nFunction: CREATE TABLE users (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    email TEXT UNIQUE,\n    password TEXT\n    )"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 4:llm:ChatOpenAI] [427ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "",
#         "generation_info": {
#           "finish_reason": "function_call",
#           "logprobs": null
#         },
#         "type": "ChatGeneration",
#         "message": {
#           "lc": 1,
#           "type": "constructor",
#           "id": [
#             "langchain",
#             "schema",
#             "messages",
#             "AIMessage"
#           ],
#           "kwargs": {
#             "content": "",
#             "additional_kwargs": {
#               "function_call": {
#                 "name": "run_sqlite_query",
#                 "arguments": "{\"query\":\"SELECT COUNT(*) FROM users\"}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 278,
#       "completion_tokens": 20,
#       "total_tokens": 298,
#       "prompt_tokens_details": {
#         "cached_tokens": 0,
#         "audio_tokens": 0
#       },
#       "completion_tokens_details": {
#         "reasoning_tokens": 0,
#         "audio_tokens": 0,
#         "accepted_prediction_tokens": 0,
#         "rejected_prediction_tokens": 0
#       }
#     },
#     "model_name": "gpt-3.5-turbo",
#     "system_fingerprint": null
#   },
#   "run": null
# }
# [tool/start] [1:chain:AgentExecutor > 5:tool:run_sqlite_query] Entering Tool run with input:
# "{'query': 'SELECT COUNT(*) FROM users'}"
# [tool/end] [1:chain:AgentExecutor > 5:tool:run_sqlite_query] [3ms] Exiting Tool run with output:
# "[(2000,)]"
# [llm/start] [1:chain:AgentExecutor > 6:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: How many orders are there? Write the result to an html report.\nAI: I have generated the report. You can download it from the following link: [orders_report.html](sandbox:/orders_report.html)\nHuman: Repeat the exact same process for users\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"users\"]}'}\nFunction: CREATE TABLE users (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    email TEXT UNIQUE,\n    password TEXT\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT COUNT(*) FROM users\"}'}\nFunction: [[2000]]"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 6:llm:ChatOpenAI] [619ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "",
#         "generation_info": {
#           "finish_reason": "function_call",
#           "logprobs": null
#         },
#         "type": "ChatGeneration",
#         "message": {
#           "lc": 1,
#           "type": "constructor",
#           "id": [
#             "langchain",
#             "schema",
#             "messages",
#             "AIMessage"
#           ],
#           "kwargs": {
#             "content": "",
#             "additional_kwargs": {
#               "function_call": {
#                 "name": "write_report",
#                 "arguments": "{\"filename\":\"users_report.html\",\"html\":\"<h1>Number of Users</h1><p>There are 2000 users in the database.</p>\"}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 312,
#       "completion_tokens": 44,
#       "total_tokens": 356,
#       "prompt_tokens_details": {
#         "cached_tokens": 0,
#         "audio_tokens": 0
#       },
#       "completion_tokens_details": {
#         "reasoning_tokens": 0,
#         "audio_tokens": 0,
#         "accepted_prediction_tokens": 0,
#         "rejected_prediction_tokens": 0
#       }
#     },
#     "model_name": "gpt-3.5-turbo",
#     "system_fingerprint": null
#   },
#   "run": null
# }
# [tool/start] [1:chain:AgentExecutor > 7:tool:write_report] Entering Tool run with input:
# "{'filename': 'users_report.html', 'html': '<h1>Number of Users</h1><p>There are 2000 users in the database.</p>'}"
# [tool/end] [1:chain:AgentExecutor > 7:tool:write_report] [0ms] Exiting Tool run with output:
# "None"
# [llm/start] [1:chain:AgentExecutor > 8:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: How many orders are there? Write the result to an html report.\nAI: I have generated the report. You can download it from the following link: [orders_report.html](sandbox:/orders_report.html)\nHuman: Repeat the exact same process for users\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"users\"]}'}\nFunction: CREATE TABLE users (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    email TEXT UNIQUE,\n    password TEXT\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT COUNT(*) FROM users\"}'}\nFunction: [[2000]]\nAI: {'name': 'write_report', 'arguments': '{\"filename\":\"users_report.html\",\"html\":\"<h1>Number of Users</h1><p>There are 2000 users in the database.</p>\"}'}\nFunction: null"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 8:llm:ChatOpenAI] [592ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "I have generated the report for the number of users. You can download it from the following link: [users_report.html](sandbox:/users_report.html)",
#         "generation_info": {
#           "finish_reason": "stop",
#           "logprobs": null
#         },
#         "type": "ChatGeneration",
#         "message": {
#           "lc": 1,
#           "type": "constructor",
#           "id": [
#             "langchain",
#             "schema",
#             "messages",
#             "AIMessage"
#           ],
#           "kwargs": {
#             "content": "I have generated the report for the number of users. You can download it from the following link: [users_report.html](sandbox:/users_report.html)",
#             "additional_kwargs": {}
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 365,
#       "completion_tokens": 32,
#       "total_tokens": 397,
#       "prompt_tokens_details": {
#         "cached_tokens": 0,
#         "audio_tokens": 0
#       },
#       "completion_tokens_details": {
#         "reasoning_tokens": 0,
#         "audio_tokens": 0,
#         "accepted_prediction_tokens": 0,
#         "rejected_prediction_tokens": 0
#       }
#     },
#     "model_name": "gpt-3.5-turbo",
#     "system_fingerprint": null
#   },
#   "run": null
# }
# [chain/end] [1:chain:AgentExecutor] [2.02s] Exiting Chain run with output:
# {
#   "output": "I have generated the report for the number of users. You can download it from the following link: [users_report.html](sandbox:/users_report.html)"
# }
