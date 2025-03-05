import os
from pprint import pprint

import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from steps import respond_to_user_message
from search_engine import search_knowledge_base

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == (os.getenv("USERNAME"), os.getenv("PASSWORD")):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
    
commands = [
    {"id": "Keyword Search", "icon": "file-search", "description": "Keyword search across all documents"},
]

@cl.on_chat_start
async def on_chat_start():
    prompt = """
You are a helpful research assistant who can answer questions about the chitrapur saraswat religious community by thoroughly studying magazine articles in your knowledge base. 
you have access to a search_knowledge_base tool that can search the knowledge base to get relevant magazine articles. 
You may use this tool multiple times to get more information.
When responding to the user, add citations to the sources you used to answer in wikipedia format. sources should not be duplicated. citation indices should be sequential starting at 1.
Before you give your answer to the user, think about your reasoning step by step and draft your answer, making sure citation formatting is correct.
Put your final answer in <final_answer></final_answer> tags. Don't forget to close the tags.
Only answer questions with information from the knowledge base. Don't use your own knowledge to answer the question.

Here is some relevant information about the chitrapur saraswat samaj:
- the main holy site of the chitrapur saraswat samaj is the chitrapur math.
- the religious leader of the chitrapur saraswat samaj is called swamiji.

<citation example>
After looking at the knowledge base, I found the following information:
more information goes here...

<final_answer>
The first president of the chitrapur saraswat samaj was <name> [1] and was appointed on <date> [2]. Additionally, the second president was <name> [1].

sources:
[1] <title of article 1>, <document 1> page <start page number>-<end page number>
[2] <title of article 2>, <document 2> page <start page number>-<end page number>
</final_answer>

</citation_example>
    """
    cl.user_session.set("messages", [SystemMessage(content=prompt)])
    await cl.context.emitter.set_commands(commands)

@cl.on_message
async def on_message(message: cl.Message):
    if message.command == "Keyword Search":
        query = message.content
        search_results = search_knowledge_base(query)
        await cl.Message(content=search_results).send()
    else:
        messages = cl.user_session.get("messages")

        messages.append(HumanMessage(content=message.content))
        messages.append(await respond_to_user_message(messages))
        
        cl.user_session.set("messages", messages)
        await cl.Message(content=messages[-1].content).send()
