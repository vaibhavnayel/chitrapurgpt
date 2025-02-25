import asyncio
from typing import Annotated

import chainlit as cl
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from retrievers import format_docs, load_docs_from_jsonl, ExactMatchRetriever, BM25Retriever, HybridRetriever, deduplicate_docs, FuzzyMatchRetriever, load_vector_store


documents = load_docs_from_jsonl('knowledge_base.jsonl')
exact_match_retriever = ExactMatchRetriever(documents=documents, k=10)
bm_25_retriever = BM25Retriever.from_documents(documents=documents, k=10)
fuzzy_match_retriever = FuzzyMatchRetriever(documents=documents, k=10)
vector_db_retriever = load_vector_store().as_retriever(search_type="mmr",search_kwargs={"k": 10, "fetch_k": 40})
retriever = HybridRetriever(exact_match_retriever=exact_match_retriever, bm_25_retriever=bm_25_retriever, fuzzy_match_retriever=fuzzy_match_retriever, vector_db_retriever=vector_db_retriever)

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
    
    # Split docs into batches of 10
    batch_size = 10
    doc_batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    
    # Prepare batch of messages for LLM
    batch_messages = []
    for doc_batch in doc_batches:
        formatted_docs = format_docs(doc_batch)
        messages = [
            SystemMessage(content=filter_prompt),
            HumanMessage(content=f"Here is the question for which you must find relevant documents: {research_instructions}\n\nHere are the documents: \n{formatted_docs}")
        ]
        batch_messages.append(messages)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0).with_structured_output(Filters)
    batch_filters = await llm.abatch(batch_messages)
    
    # Combine filtered documents from all batches
    filtered_docs = []
    for i, filters in enumerate(batch_filters):
        for idx in filters.filter_indices:
            filtered_docs.append(doc_batches[i][idx])
    
    return filtered_docs

@traceable
@cl.step(name="document analysis")
async def answer_question(docs: list[Document], research_instructions: str) -> str:
    # Split docs into batches of 10
    batch_size = 10
    doc_batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    
    # Process each batch in parallel
    batch_messages = []
    for doc_batch in doc_batches:
        batch_messages.append(doc_batch)
    
    chunk_answers = await asyncio.gather(
        *[process_document_chunk(batch, research_instructions) for batch in batch_messages]
    )
    
    if len(chunk_answers) == 1:
        return chunk_answers[0]
    
    # Combine all chunk answers
    return await combine_chunk_answers(chunk_answers, research_instructions)

@traceable
async def process_document_chunk(doc_batch: list[Document], research_instructions: str) -> str:
    """Process a single batch of documents and generate an answer with citations."""
    chunk_answer_prompt = f"""
You are a research assistant whose task is to read the filtered documents and answer the user query: {research_instructions}
After constructing your answer, add citations to the sources you used to answer the question as you would see on Wikipedia.
The citation indices must be sequential and start from 1. Don't skip any numbers in the citation step.
First think step by step about the documents and then put your final answer to the user in <answer></answer> tags. Answer tags must appear only at the end of your response.

<example>

<answer>
The capital of X is Y [1] and the capital of Z is W. [2]

sources:
[1] <title of article 1>, <document 1> page <start page number>-<end page number>
[2] <title of article 2>, <document 2> page <start page number>-<end page number>
</answer>

</example>
"""
    formatted_docs = format_docs(doc_batch)
    messages = [
        SystemMessage(content=chunk_answer_prompt),
        HumanMessage(content=f"Here are the documents: \n{formatted_docs}")
    ]
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
    response = await llm.ainvoke(messages)
    
    if "<answer>" in response.content and "</answer>" in response.content:
        return response.content.split("<answer>")[1].split("</answer>")[0]
    return response.content

@traceable
async def combine_chunk_answers(chunk_answers: list[str], research_instructions: str) -> str:
    """Combine multiple chunk answers into a single coherent answer with renumbered citations."""
    final_answer_prompt = f"""
You are a research assistant whose task is to combine multiple partial answers into a comprehensive response to the user query: {research_instructions}
Each partial answer has its own citations. You need to merge these answers and renumber all citations sequentially starting from 1.
First think step by step about how to combine the information, then put your final answer to the user in <answer></answer> tags.
Make sure all citations are properly renumbered and all sources are included in the final list.

<example>

<answer>
The capital of X is Y [1] and the capital of Z is W [2]. Additionally, the population of Y is 5 million [3].

sources:
[1] <title of article 1>, <document 1> page <start page number>-<end page number>
[2] <title of article 2>, <document 2> page <start page number>-<end page number>
[3] <title of article 3>, <document 3> page <start page number>-<end page number>
</answer>

</example>
"""
    messages = [
        SystemMessage(content=final_answer_prompt),
        HumanMessage(content=f"Here are the partial answers to combine:\n\n" + "\n\n---\n\n".join(chunk_answers))
    ]
    
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
    response = await llm.ainvoke(messages)
    
    if "<answer>" in response.content and "</answer>" in response.content:
        return response.content.split("<answer>")[1].split("</answer>")[0]
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
