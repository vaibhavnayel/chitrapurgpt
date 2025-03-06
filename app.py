import os
from pprint import pprint

import chainlit as cl
from chainlit.types import ThreadDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from steps import respond_to_user_message
from search_engine import search_knowledge_base

    
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User) -> cl.User | None:
    if default_user.identifier in open("email_whitelist.txt").read().split(","):
        return default_user
    else:
        return None
    
commands = [
    {"id": "Exact Search", "icon": "crosshair", "description": "Exact search across all documents. 'sans' will not match 'sanskrit'."},
    {"id": "Fuzzy Search", "icon": "search", "description": "Fuzzy search across all documents. 'sans' will match 'sanskar' and 'sanskrit'."},
]

system_prompt = """
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
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [SystemMessage(content=system_prompt)])
    await cl.context.emitter.set_commands(commands)

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("messages", [SystemMessage(content=system_prompt)])
    
    for message in thread["steps"]:
        if message["type"] == "user_message":
            cl.user_session.get("messages").append(HumanMessage(content=message["output"]))
        elif message["type"] == "assistant_message":
            cl.user_session.get("messages").append(AIMessage(content=message["output"]))
    await cl.context.emitter.set_commands(commands)


@cl.on_message
async def on_message(message: cl.Message):
    if message.command == "Fuzzy Search":
        search_results = search_knowledge_base(message.content, exact=False)
        await cl.Message(content=search_results, tags=["command_output"]).send()
    elif message.command == "Exact Search":
        search_results = search_knowledge_base(message.content, exact=True)
        await cl.Message(content=search_results, tags=["command_output"]).send()
    else:
        messages = cl.user_session.get("messages")

        messages.append(HumanMessage(content=message.content))
        messages.append(await respond_to_user_message(messages))
        
        cl.user_session.set("messages", messages)
        await cl.Message(content=messages[-1].content).send()
