import langchain
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
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
        HumanMessagePromptTemplate.from_template("{input}"),
        # This is by convention and we need it. (Similar to memory)
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

tools = [run_query_tool, describe_tables_tool, write_report_tool]
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools,
)

# agent_executor.run("How many users have provided a shipping address?")
# agent_executor.run("How many users are in the database?")
agent_executor.run(
    "Summarize the top 5 most popular products. Write the results to a report file."
)
# output for: Summarize the top 5 most popular products. Write the results to a report file.
# [chain/start] [1:chain:AgentExecutor] Entering Chain run with input:
# {
#   "input": "Summarize the top 5 most popular products. Write the results to a report file."
# }
# [llm/start] [1:chain:AgentExecutor > 2:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: Summarize the top 5 most popular products. Write the results to a report file."
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 2:llm:ChatOpenAI] [986ms] Exiting LLM run with output:
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
#                 "arguments": "{\"table_names\":[\"products\",\"order_products\"]}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 194,
#       "completion_tokens": 18,
#       "total_tokens": 212,
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
# "{'table_names': ['products', 'order_products']}"
# [tool/end] [1:chain:AgentExecutor > 3:tool:describe_tables] [1ms] Exiting Tool run with output:
# "CREATE TABLE products (
#     id INTEGER PRIMARY KEY,
#     name TEXT,
#     price REAL
#     )
# CREATE TABLE order_products (
#     id INTEGER PRIMARY KEY,
#     order_id INTEGER,
#     product_id INTEGER,
#     amount INTEGER
#     )"
# [llm/start] [1:chain:AgentExecutor > 4:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: Summarize the top 5 most popular products. Write the results to a report file.\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\",\"order_products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )\nCREATE TABLE order_products (\n    id INTEGER PRIMARY KEY,\n    order_id INTEGER,\n    product_id INTEGER,\n    amount INTEGER\n    )"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 4:llm:ChatOpenAI] [556ms] Exiting LLM run with output:
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
#                 "arguments": "{\"query\":\"SELECT product_id, COUNT(product_id) AS num_orders FROM order_products GROUP BY product_id ORDER BY num_orders DESC LIMIT 5\"}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 267,
#       "completion_tokens": 41,
#       "total_tokens": 308,
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
# "{'query': 'SELECT product_id, COUNT(product_id) AS num_orders FROM order_products GROUP BY product_id ORDER BY num_orders DESC LIMIT 5'}"
# [tool/end] [1:chain:AgentExecutor > 5:tool:run_sqlite_query] [8ms] Exiting Tool run with output:
# "[(3929, 10), (1437, 9), (3192, 8), (3115, 8), (2091, 8)]"
# [llm/start] [1:chain:AgentExecutor > 6:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: Summarize the top 5 most popular products. Write the results to a report file.\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\",\"order_products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )\nCREATE TABLE order_products (\n    id INTEGER PRIMARY KEY,\n    order_id INTEGER,\n    product_id INTEGER,\n    amount INTEGER\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT product_id, COUNT(product_id) AS num_orders FROM order_products GROUP BY product_id ORDER BY num_orders DESC LIMIT 5\"}'}\nFunction: [[3929, 10], [1437, 9], [3192, 8], [3115, 8], [2091, 8]]"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 6:llm:ChatOpenAI] [380ms] Exiting LLM run with output:
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
#                 "arguments": "{\"table_names\":[\"products\"]}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 353,
#       "completion_tokens": 15,
#       "total_tokens": 368,
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
# [tool/start] [1:chain:AgentExecutor > 7:tool:describe_tables] Entering Tool run with input:
# "{'table_names': ['products']}"
# [tool/end] [1:chain:AgentExecutor > 7:tool:describe_tables] [1ms] Exiting Tool run with output:
# "CREATE TABLE products (
#     id INTEGER PRIMARY KEY,
#     name TEXT,
#     price REAL
#     )"
# [llm/start] [1:chain:AgentExecutor > 8:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: Summarize the top 5 most popular products. Write the results to a report file.\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\",\"order_products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )\nCREATE TABLE order_products (\n    id INTEGER PRIMARY KEY,\n    order_id INTEGER,\n    product_id INTEGER,\n    amount INTEGER\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT product_id, COUNT(product_id) AS num_orders FROM order_products GROUP BY product_id ORDER BY num_orders DESC LIMIT 5\"}'}\nFunction: [[3929, 10], [1437, 9], [3192, 8], [3115, 8], [2091, 8]]\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 8:llm:ChatOpenAI] [585ms] Exiting LLM run with output:
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
#                 "arguments": "{\"query\":\"SELECT * FROM products WHERE id IN (3929, 1437, 3192, 3115, 2091)\"}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 396,
#       "completion_tokens": 42,
#       "total_tokens": 438,
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
# [tool/start] [1:chain:AgentExecutor > 9:tool:run_sqlite_query] Entering Tool run with input:
# "{'query': 'SELECT * FROM products WHERE id IN (3929, 1437, 3192, 3115, 2091)'}"
# [tool/end] [1:chain:AgentExecutor > 9:tool:run_sqlite_query] [2ms] Exiting Tool run with output:
# "[(1437, 'Refined Chips', 468.43), (2091, 'Sleek Pizza', 658.65), (3115, 'Fresh Bike', 586.29), (3192, 'Generic Ball', 146.88), (3929, 'Unbranded Pizza', 570.36)]"
# [llm/start] [1:chain:AgentExecutor > 10:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: Summarize the top 5 most popular products. Write the results to a report file.\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\",\"order_products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )\nCREATE TABLE order_products (\n    id INTEGER PRIMARY KEY,\n    order_id INTEGER,\n    product_id INTEGER,\n    amount INTEGER\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT product_id, COUNT(product_id) AS num_orders FROM order_products GROUP BY product_id ORDER BY num_orders DESC LIMIT 5\"}'}\nFunction: [[3929, 10], [1437, 9], [3192, 8], [3115, 8], [2091, 8]]\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT * FROM products WHERE id IN (3929, 1437, 3192, 3115, 2091)\"}'}\nFunction: [[1437, \"Refined Chips\", 468.43], [2091, \"Sleek Pizza\", 658.65], [3115, \"Fresh Bike\", 586.29], [3192, \"Generic Ball\", 146.88], [3929, \"Unbranded Pizza\", 570.36]]"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 10:llm:ChatOpenAI] [3.15s] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "The top 5 most popular products are:\n\n1. Product ID: 3929, Name: Unbranded Pizza, Price: $570.36\n2. Product ID: 1437, Name: Refined Chips, Price: $468.43\n3. Product ID: 3192, Name: Generic Ball, Price: $146.88\n4. Product ID: 3115, Name: Fresh Bike, Price: $586.29\n5. Product ID: 2091, Name: Sleek Pizza, Price: $658.65\n\nI will now write these results to a report file.",
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
#             "content": "The top 5 most popular products are:\n\n1. Product ID: 3929, Name: Unbranded Pizza, Price: $570.36\n2. Product ID: 1437, Name: Refined Chips, Price: $468.43\n3. Product ID: 3192, Name: Generic Ball, Price: $146.88\n4. Product ID: 3115, Name: Fresh Bike, Price: $586.29\n5. Product ID: 2091, Name: Sleek Pizza, Price: $658.65\n\nI will now write these results to a report file.",
#             "additional_kwargs": {
#               "function_call": {
#                 "name": "write_report",
#                 "arguments": "{\"filename\":\"top_5_popular_products_report.html\",\"html\":\"<h1>Top 5 Most Popular Products</h1><p><strong>1. Product ID:</strong> 3929, <strong>Name:</strong> Unbranded Pizza, <strong>Price:</strong> $570.36</p><p><strong>2. Product ID:</strong> 1437, <strong>Name:</strong> Refined Chips, <strong>Price:</strong> $468.43</p><p><strong>3. Product ID:</strong> 3192, <strong>Name:</strong> Generic Ball, <strong>Price:</strong> $146.88</p><p><strong>4. Product ID:</strong> 3115, <strong>Name:</strong> Fresh Bike, <strong>Price:</strong> $586.29</p><p><strong>5. Product ID:</strong> 2091, <strong>Name:</strong> Sleek Pizza, <strong>Price:</strong> $658.65</p>\"}"
#               }
#             }
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 518,
#       "completion_tokens": 362,
#       "total_tokens": 880,
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
# [tool/start] [1:chain:AgentExecutor > 11:tool:write_report] Entering Tool run with input:
# "{'filename': 'top_5_popular_products_report.html', 'html': '<h1>Top 5 Most Popular Products</h1><p><strong>1. Product ID:</strong> 3929, <strong>Name:</strong> Unbranded Pizza, <strong>Price:</strong> $570.36</p><p><strong>2. Product ID:</strong> 1437, <strong>Name:</strong> Refined Chips, <strong>Price:</strong> $468.43</p><p><strong>3. Product ID:</strong> 3192, <strong>Name:</strong> Generic Ball, <strong>Price:</strong> $146.88</p><p><strong>4. Product ID:</strong> 3115, <strong>Name:</strong> Fresh Bike, <strong>Price:</strong> $586.29</p><p><strong>5. Product ID:</strong> 2091, <strong>Name:</strong> Sleek Pizza, <strong>Price:</strong> $658.65</p>'}"
# [tool/end] [1:chain:AgentExecutor > 11:tool:write_report] [1ms] Exiting Tool run with output:
# "None"
# [llm/start] [1:chain:AgentExecutor > 12:llm:ChatOpenAI] Entering LLM run with input:
# {
#   "prompts": [
#     "System: You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function\nHuman: Summarize the top 5 most popular products. Write the results to a report file.\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\",\"order_products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )\nCREATE TABLE order_products (\n    id INTEGER PRIMARY KEY,\n    order_id INTEGER,\n    product_id INTEGER,\n    amount INTEGER\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT product_id, COUNT(product_id) AS num_orders FROM order_products GROUP BY product_id ORDER BY num_orders DESC LIMIT 5\"}'}\nFunction: [[3929, 10], [1437, 9], [3192, 8], [3115, 8], [2091, 8]]\nAI: {'name': 'describe_tables', 'arguments': '{\"table_names\":[\"products\"]}'}\nFunction: CREATE TABLE products (\n    id INTEGER PRIMARY KEY,\n    name TEXT,\n    price REAL\n    )\nAI: {'name': 'run_sqlite_query', 'arguments': '{\"query\":\"SELECT * FROM products WHERE id IN (3929, 1437, 3192, 3115, 2091)\"}'}\nFunction: [[1437, \"Refined Chips\", 468.43], [2091, \"Sleek Pizza\", 658.65], [3115, \"Fresh Bike\", 586.29], [3192, \"Generic Ball\", 146.88], [3929, \"Unbranded Pizza\", 570.36]]\nAI: The top 5 most popular products are:\n\n1. Product ID: 3929, Name: Unbranded Pizza, Price: $570.36\n2. Product ID: 1437, Name: Refined Chips, Price: $468.43\n3. Product ID: 3192, Name: Generic Ball, Price: $146.88\n4. Product ID: 3115, Name: Fresh Bike, Price: $586.29\n5. Product ID: 2091, Name: Sleek Pizza, Price: $658.65\n\nI will now write these results to a report file.{'name': 'write_report', 'arguments': '{\"filename\":\"top_5_popular_products_report.html\",\"html\":\"<h1>Top 5 Most Popular Products</h1><p><strong>1. Product ID:</strong> 3929, <strong>Name:</strong> Unbranded Pizza, <strong>Price:</strong> $570.36</p><p><strong>2. Product ID:</strong> 1437, <strong>Name:</strong> Refined Chips, <strong>Price:</strong> $468.43</p><p><strong>3. Product ID:</strong> 3192, <strong>Name:</strong> Generic Ball, <strong>Price:</strong> $146.88</p><p><strong>4. Product ID:</strong> 3115, <strong>Name:</strong> Fresh Bike, <strong>Price:</strong> $586.29</p><p><strong>5. Product ID:</strong> 2091, <strong>Name:</strong> Sleek Pizza, <strong>Price:</strong> $658.65</p>\"}'}\nFunction: null"
#   ]
# }
# [llm/end] [1:chain:AgentExecutor > 12:llm:ChatOpenAI] [575ms] Exiting LLM run with output:
# {
#   "generations": [
#     [
#       {
#         "text": "The top 5 most popular products have been summarized and written to a report file. You can download the report file [here](sandbox:/top_5_popular_products_report.html).",
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
#             "content": "The top 5 most popular products have been summarized and written to a report file. You can download the report file [here](sandbox:/top_5_popular_products_report.html).",
#             "additional_kwargs": {}
#           }
#         }
#       }
#     ]
#   ],
#   "llm_output": {
#     "token_usage": {
#       "prompt_tokens": 892,
#       "completion_tokens": 38,
#       "total_tokens": 930,
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
# [chain/end] [1:chain:AgentExecutor] [6.26s] Exiting Chain run with output:
# {
#   "output": "The top 5 most popular products have been summarized and written to a report file. You can download the report file [here](sandbox:/top_5_popular_products_report.html)."
# }
