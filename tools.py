from langchain_core.tools import tool
from typing import Annotated
from langchain_community.retrievers.bm25 import BM25Retriever

from retrievers import load_docs_from_jsonl, ExactMatchRetriever, BM25Retriever, HybridRetriever, deduplicate_docs
from steps import generate_search_queries, filter_docs, answer_question


documents = load_docs_from_jsonl('knowledge_base.jsonl')
exact_match_retriever = ExactMatchRetriever(documents=documents, k=5)
bm_25_retriever = BM25Retriever.from_documents(documents=documents, k=5)
retriever = HybridRetriever(exact_match_retriever=exact_match_retriever, bm_25_retriever=bm_25_retriever)
@tool
def research_assistant_tool(research_instructions: Annotated[str, "the research question(s) to be answered"]) -> str:
    """Answer research question(s) by looking at the knowledge base"""

    # parse the question to generate queries
    search_queries = generate_search_queries(research_instructions)

    docs = retriever.batch(search_queries)
    docs = deduplicate_docs(docs)

    #filter docs with LLM
    docs = filter_docs(docs, research_instructions)

    # read the filtered docs and answer the question
    return answer_question(docs, research_instructions)

