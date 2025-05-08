import logging
import re
import chainlit as cl
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from retrievers import format_docs, deduplicate_docs, load_vector_store, FuzzyMatchRetriever, load_docs_from_jsonl, HybridRetriever

fuzzy_retriever = FuzzyMatchRetriever(documents=load_docs_from_jsonl("knowledge_base.jsonl"), k=5)
vector_db_retriever = load_vector_store().as_retriever(search_type="mmr",search_kwargs={"k": 5, "fetch_k": 20})
retriever = HybridRetriever(fuzzy_retriever=fuzzy_retriever, vector_db_retriever=vector_db_retriever)

@tool
@cl.step(name="knowledge base search engine")
@traceable
async def search_knowledge_base(research_query: str) -> str:
    """
    Search the knowledge base to get relevant magazine articles.
    The query should be a question that you want to answer.
    """

    queries = await generate_search_queries(research_query)

    # search the knowledge base
    docs = retriever.batch(queries)
    docs = deduplicate_docs(docs)

    #contextual compression
    contextualized_docs = await contextualize_docs(docs, research_query)

    return format_docs(contextualized_docs)


class Queries(BaseModel):
    reasoning: str = Field(description="the reasoning for the queries")
    queries: list[str] = Field(description="a list of queries to search the knowledge base")

@traceable
@cl.step(name="search query generator")
async def generate_search_queries(research_instructions: str) -> list[str]:
    parser_prompt = """
You are a research assistant whose task it is to look at the question and parse it into a list of queries to search the knowledge base.
Try to make queries short and concise. They will be search with a mix of exact match, fuzzy match and vector search.
    """
    messages = [
        SystemMessage(content=parser_prompt),
        HumanMessage(content=f"here is the question: {research_instructions}")
    ]
    response = await ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0).with_structured_output(Queries).ainvoke(messages)
    logging.info(f" generated queries: {response.queries}")
    return response.queries

class RelevantPassages(BaseModel):
    reasoning: str = Field(description="reasoning for the compressed document")
    passages_in_context: list[str] = Field(description="summarised passages with quotes that are relevant to the query. If there is no relevant information, say 'no relevant information found'")

@traceable
@cl.step(name="document analysis")
async def contextualize_docs(docs: list[Document], query: str) -> list[Document]:
    prompt = f"""
Your task is to extract information relevant to the query from the following document.
Your extractions should accurately summarise the document in the context of the query while also quoting the passage verbatim. 
Make sure that the quoted passages are talking about the same topic, person or place as the query.
Do not make up answers or include information that is not present in the document.
If you find that the document is not relevant to the query, the passage should say "no relevant information found".
It is also essential that you don't miss any important information and don't include any information that is not present in the document, otherwise the user will not get a complete answer to their question.
The query is: {query}
    """
    messages_batch = [[SystemMessage(content=prompt), HumanMessage(content=f"Here is the document: {format_docs([doc])}")] for doc in docs]
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=8000).with_structured_output(RelevantPassages).withretry()
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0, max_tokens=8000).with_structured_output(RelevantPassages).with_retry()
    # llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0, max_tokens=8000).with_structured_output(RelevantPassages).with_retry()
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_tokens=8000).with_structured_output(RelevantPassages).with_retry()
    compressed_docs = await llm.abatch(messages_batch)

    contextualized_docs = []
    for compressed_doc, doc in zip(compressed_docs, docs):
        if compressed_doc and compressed_doc.passages_in_context:
            doc.page_content = f"Here are the relevant passages from this document: {'\n'.join(compressed_doc.passages_in_context)}"
            contextualized_docs.append(doc)
    logging.info(f"number of contextualized docs: {len(contextualized_docs)}")
    return contextualized_docs

@traceable
async def handle_tool_call(messages: list[BaseMessage]) -> str:
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, max_tokens=8000).bind_tools([search_knowledge_base], parallel_tool_calls=False)

    for tool_call in messages[-1].tool_calls:
        if tool_call["name"] == "search_knowledge_base":
            logging.info(f"tool call: {tool_call}")
            tool_response = await search_knowledge_base.ainvoke(tool_call["args"])
            messages.append(ToolMessage(content=tool_response, tool_call_id=tool_call["id"]))
    messages.append(await llm.ainvoke(messages))

    return messages

def parse_final_answer(message: BaseMessage) -> BaseMessage:
    match = re.search(r'<final_answer>(.*?)</final_answer>', message.content, re.DOTALL)
    if match:
        message.content = match.group(1)
    return message

@traceable
async def respond_to_user_message(messages: list[BaseMessage]) -> str:
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, max_tokens=8000).bind_tools([search_knowledge_base], parallel_tool_calls=False)
    messages.append(await llm.ainvoke(messages))

    while messages[-1].tool_calls:
        messages = await handle_tool_call(messages)

    return parse_final_answer(messages[-1])
