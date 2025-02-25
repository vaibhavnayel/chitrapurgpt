import os
from pprint import pprint

import chainlit as cl

from steps import respond_to_user_message

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
