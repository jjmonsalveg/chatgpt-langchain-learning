from dotenv import load_dotenv
from handlers.chat_model_start_handler import ChatModelStartHandler
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
# langchain.debug = True

handler = ChatModelStartHandler()
chat = ChatOpenAI(callbacks=[handler])
tables = tables()
prompt = ChatPromptTemplate(
    input_variables=["input"],
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
    ],
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

# agent_executor.run("How many users have provided a shipping address?")
# agent_executor.run("How many users are in the database?")
agent_executor.run("How many orders are there? Write the result to an html report.")

agent_executor.run("Repeat the exact same process for users")

# output before adding pyboxen:
# > Entering new AgentExecutor chain...
# [[SystemMessage(content="You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function"), HumanMessage(content='How many orders are there? Write the result to an html report.')]]

# Invoking: `run_sqlite_query` with `{'query': 'SELECT COUNT(*) FROM orders'}`


# [(1500,)][[SystemMessage(content="You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function"), HumanMessage(content='How many orders are there? Write the result to an html report.'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'run_sqlite_query', 'arguments': '{"query":"SELECT COUNT(*) FROM orders"}'}}), FunctionMessage(content='[[1500]]', name='run_sqlite_query')]]

# Invoking: `write_report` with `{'filename': 'orders_report.html', 'html': '<h1>Number of Orders</h1><p>There are 1500 orders in the database.</p>'}`


# None[[SystemMessage(content="You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function"), HumanMessage(content='How many orders are there? Write the result to an html report.'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'run_sqlite_query', 'arguments': '{"query":"SELECT COUNT(*) FROM orders"}'}}), FunctionMessage(content='[[1500]]', name='run_sqlite_query'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'write_report', 'arguments': '{"filename":"orders_report.html","html":"<h1>Number of Orders</h1><p>There are 1500 orders in the database.</p>"}'}}), FunctionMessage(content='null', name='write_report')]]
# The number of orders in the database is 1500. I have written the result to an HTML report. You can download the report from the following link: [orders_report.html](sandbox:/orders_report.html)

# > Finished chain.


# > Entering new AgentExecutor chain...
# [[SystemMessage(content="You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function"), HumanMessage(content='How many orders are there? Write the result to an html report.'), AIMessage(content='The number of orders in the database is 1500. I have written the result to an HTML report. You can download the report from the following link: [orders_report.html](sandbox:/orders_report.html)'), HumanMessage(content='Repeat the exact same process for users')]]

# Invoking: `run_sqlite_query` with `{'query': 'SELECT COUNT(*) FROM users'}`


# [(2000,)][[SystemMessage(content="You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function"), HumanMessage(content='How many orders are there? Write the result to an html report.'), AIMessage(content='The number of orders in the database is 1500. I have written the result to an HTML report. You can download the report from the following link: [orders_report.html](sandbox:/orders_report.html)'), HumanMessage(content='Repeat the exact same process for users'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'run_sqlite_query', 'arguments': '{"query":"SELECT COUNT(*) FROM users"}'}}), FunctionMessage(content='[[2000]]', name='run_sqlite_query')]]

# Invoking: `write_report` with `{'filename': 'users_report.html', 'html': '<h1>Number of Users</h1><p>The number of users in the database is 2000.</p>'}`


# None[[SystemMessage(content="You are an AI that has access to a SQLite database.\nThe database has tables of: users\naddresses\nproducts\ncarts\norders\norder_products\nDo not make any assumptions about what tables existor what columns exist. Instead, use the 'describe_tables' function"), HumanMessage(content='How many orders are there? Write the result to an html report.'), AIMessage(content='The number of orders in the database is 1500. I have written the result to an HTML report. You can download the report from the following link: [orders_report.html](sandbox:/orders_report.html)'), HumanMessage(content='Repeat the exact same process for users'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'run_sqlite_query', 'arguments': '{"query":"SELECT COUNT(*) FROM users"}'}}), FunctionMessage(content='[[2000]]', name='run_sqlite_query'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'write_report', 'arguments': '{"filename":"users_report.html","html":"<h1>Number of Users</h1><p>The number of users in the database is 2000.</p>"}'}}), FunctionMessage(content='null', name='write_report')]]
# I have created a report on the number of users in the database. You can download the report from the following link: [users_report.html](sandbox:/users_report.html)

# > Finished chain.
