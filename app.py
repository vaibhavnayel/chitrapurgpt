from pprint import pprint

import chainlit as cl

from steps import respond_to_user_message

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("messages", [])


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("messages")
    messages = await respond_to_user_message(message.content, message_history)
    
    cl.user_session.set("messages", messages)
    
    pprint([(msg.type, msg.content) for msg in messages])

    await cl.Message(content=messages[-1].content).send()
