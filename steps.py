from typing import Annotated

import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from retrievers import format_docs, load_docs_from_jsonl, ExactMatchRetriever, BM25Retriever, HybridRetriever, deduplicate_docs, FuzzyMatchRetriever


documents = load_docs_from_jsonl('knowledge_base.jsonl')
exact_match_retriever = ExactMatchRetriever(documents=documents, k=5)
bm_25_retriever = BM25Retriever.from_documents(documents=documents, k=5)
fuzzy_match_retriever = FuzzyMatchRetriever(documents=documents, k=5)
retriever = HybridRetriever(exact_match_retriever=exact_match_retriever, bm_25_retriever=bm_25_retriever, fuzzy_match_retriever=fuzzy_match_retriever)

@tool
@cl.step(name="knowledge base search engine")
@traceable
async def research_assistant_tool(research_instructions: Annotated[str, "the research question(s) to be answered"]) -> str:
    """Answer research question(s) by looking at the knowledge base"""

    # parse the question to generate queries
    search_queries = await generate_search_queries(research_instructions)

    docs = await retriever.abatch(search_queries)
    docs = deduplicate_docs(docs)

    #filter docs with LLM
    docs = await filter_docs(docs, research_instructions)

    # read the filtered docs and answer the question
    return await answer_question(docs, research_instructions)


class Queries(BaseModel):
    reasoning: str = Field(description="the reasoning for the queries")
    queries: list[str] = Field(description="a list of queries to search the knowledge base")

@traceable
@cl.step(name="search queries generator")
async def generate_search_queries(research_instructions: str) -> list[str]:
    parser_prompt = """
You are a research assistant whose task it is to look at the question and parse it into a list of queries to search the knowledge base.
Try to make queries short and concise.
    """
    messages = [
        SystemMessage(content=parser_prompt),
        HumanMessage(content=f"here is the question: {research_instructions}")
    ]
    # return (model="gpt-4o", temperature=0).with_structured_output(Queries).ainvoke(messages).queries
    # return await ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(Queries).ainvoke(messages).queries
    response = await ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0).with_structured_output(Queries).ainvoke(messages)
    return response.queries

class Filters(BaseModel):
    reasoning: str = Field(description="the reasoning for the filtered documents")
    filter_indices: list[int] = Field(description="a list of indices (starting from 0) of the documents that are relevant to the question")

@traceable
@cl.step(name="document filter")
async def filter_docs(docs: list[Document], research_instructions: str) -> list[Document]:
    filter_prompt = """
You are a research assistant whose task it is to filter the documents to find the most relevant ones. 
note that the docs may contain garbled text. you should ignore the garbled text.
    """
    formatted_docs = format_docs(docs)
    messages = [
        SystemMessage(content=filter_prompt),
        HumanMessage(content=f"Here is the question for which you must find relevant documents: {research_instructions}\n\nHere are the documents: \n{formatted_docs}")
    ]
    # llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Filters)
    # llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).with_structured_output(Filters)
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0).with_structured_output(Filters)
    filters = await llm.ainvoke(messages)
    return [docs[i] for i in filters.filter_indices]

@traceable
@cl.step(name="document analysis")
async def answer_question(docs: list[Document], research_instructions: str) -> str:
    answer_prompt = f"""
You are a research assistant whose task is to read the filtered documents and answer the user query: {research_instructions}
after constructing your answer, add citations to the sources you used to answer the question as you would see on wikipedia.

<example>
The capital of X is Y [1] and the capital of Z is W. [2]

sources:
[1] <title of article 1>, <document 1> page <start page number>-<end page number>
[2] <title of article 2>, <document 2> page <start page number>-<end page number>
</example>
"""
    formatted_docs = format_docs(docs)
    messages = [
        SystemMessage(content=answer_prompt),
        HumanMessage(content=f"here are the documents: \n{formatted_docs}")
    ]
    # return await ChatOpenAI(model="gpt-4-turbo", temperature=0).ainvoke(messages).content
    # return await ChatGroq(model="llama-3.3-70b-versatile", temperature=0).ainvoke(messages).content
    response = await ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0).ainvoke(messages)
    return response.content

@traceable
async def handle_tool_call(messages: list[BaseMessage]) -> list[BaseMessage]:
    while messages[-1].tool_calls:
        for tool_call in messages[-1].tool_calls:
            tool_response = await research_assistant_tool.ainvoke(tool_call["args"])
            tool_message = ToolMessage(content=tool_response, tool_call_id=tool_call["id"])
            messages.append(tool_message)
        response = await ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([research_assistant_tool]).ainvoke(messages)
        # response = await ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools([research_assistant_tool]).ainvoke(messages)
        # response = await ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0).bind_tools([research_assistant_tool]).ainvoke(messages)
        messages.append(response)
    return messages

@traceable
async def respond_to_user_message(user_message: str, message_history: list[BaseMessage]) -> str:
    if not message_history:
        prompt = """
You are a helpful assistant who can answer questions about the chitrapur saraswat religious community by looking at books and magazines in your knowledge base. 
you have access to a research assistant tool that can search the knowledge base and answer questions you ask it. 
When responding to the user, add citations to the sources you used to answer (you will get this from the research assistant tool).
        """
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message)
        ]
    else: 
        messages = message_history + [HumanMessage(content=user_message)]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([research_assistant_tool])
    # llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools([research_assistant_tool])
    # llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0).bind_tools([research_assistant_tool])
    response = await llm.ainvoke(messages)
    messages.append(response)

    return await handle_tool_call(messages)
