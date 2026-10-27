import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import unidecode
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import re
from multiprocessing import Pool, cpu_count
import os
from glob import glob
import dotenv
from tqdm import tqdm, trange
import logging
from tiktoken import get_encoding
from retrievers import save_doc_to_json, load_doc_from_json, save_docs_to_jsonl, load_docs_from_jsonl, add_to_vector_store, get_doc_id
import models

dotenv.load_dotenv()

# USD per million tokens (input, output), for the cost print only.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (5.0, 25.0), "claude-haiku-5-5": (0.10, 0.50)}

class ArticleBoundary(BaseModel):
    title: str = Field(description="The title of the article. MUST be ASCII characters only.")
    author: str|None = Field(description="The author of the article, leave empty if not found")
    summary: str|None = Field(description="A short summary of the article. 1 to 5 sentences.")
    published_date: str = Field(description="date of publication of the article. DD-MM-YYYY format. use first of the month if exact date isn't found.")
    start_line: int = Field(description="The line number of the first line of the article")
    end_line: int = Field(description="The line number of the last line of the article")
    start_page: int = Field(description="The page number of the first page of the article")
    end_page: int = Field(description="The page number of the last page of the article")

class ArticleBoundaries(BaseModel):
    # Was `scratchpad`: Claude Sonnet 5.5 stops with stop_reason="refusal" (no output) whenever a structured-output
    # schema has a field of that name. Nothing reads this field; the model also thinks natively now.
    notes: str = Field(description="notes about the articles (titles, page ranges, the publication date) before listing them")
    article_boundaries: list[ArticleBoundary]

def sanitize_string(text):
    return unidecode.unidecode(text)

def add_line_numbers(text):
    lines = text.split('\n')
    numbered_lines = [f"{i+1}. {line}" for i, line in enumerate(lines)]
    return '\n'.join(numbered_lines)

def extract_pdf_text(path):
    try:
        docs = PyPDFLoader(path, mode='single',extraction_mode='layout',pages_delimiter="\n------- PAGE END -------\n").load()
    except:
        logging.warning(f"Failed to load {path} with layout mode, trying plain mode")
        docs = PyPDFLoader(path, mode='single',extraction_mode='plain',pages_delimiter="\n------- PAGE END -------\n").load()

    full_text = docs[0].page_content
    full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
    sanitized_text = sanitize_string(full_text)
    return sanitized_text


def print_cost(usage: dict, model: str = models.INGEST_MODEL):
    prompt, completion = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    print(f"Tokens -> Prompt: {prompt:6d} | Completion: {completion:7d} | Total: {prompt + completion:6d}")
    if model in PRICES:
        p_in, p_out = (p / 1e6 for p in PRICES[model])
        print(f"Costs -> Prompt: ${prompt * p_in:.4f} | Completion: ${completion * p_out:.4f} | Total: ${prompt * p_in + completion * p_out:.4f}")

def parse_articles(full_text:str, article_boundaries:list[ArticleBoundary], path:str) -> list[Document]:
    docs = []
    for article_boundary in article_boundaries:
        article_text = '\n'.join(full_text.split('\n')[article_boundary.start_line:article_boundary.end_line])
        metadata = {
            "source": path.split("/")[-1],
            "published_date": article_boundary.published_date,
            "title": article_boundary.title,
            "author": article_boundary.author,
            "summary": article_boundary.summary,
            "start_line": article_boundary.start_line,
            "end_line": article_boundary.end_line,
            "start_page": article_boundary.start_page,
            "end_page": article_boundary.end_page,
        }
        docs.append(Document(page_content=article_text, metadata=metadata))
    return docs

async def extract_article_boundaries(text:str) -> list[ArticleBoundary]:
    system_prompt = """
ROLE:
You are a PDF parser. You are given text from a PDF file and you need to parse it into articles based on line numbers.

INSTRUCTIONS:
- Start and end articles on page ends (marked by ------- PAGE END -------) unless the article starts or ends in the middle of a page.
- ignore pages with only photos + captions. Photos are marked with brackets () and there will be a caption above or below it. Here is an example:
    <example>
    At Ãdi-Sthala at Vittal - Mrittikã-haran on 2-12-2024.
    (Photos of this event by Prashant Haridas and Ashwin Cherka)

    Samuhika prãrtana at Srimath Ananteshwara Temple Vittal
                        before Dhwajarohana (Kodi).
    </example>
- articles are usually 1-15 pages long.
- articles may or may not have authors but there will always be a title.
- The first page is the cover page and it is an article (title should be "Cover Page").
- pay attention to the table of contents (this is also an article) and the page numbers in it.
- the published date will most likely be found on the cover page. you must use this as the published_date for all articles.
    """

    # CGPT_INGEST_MODEL (default Sonnet 5.5): adaptive thinking, so no temperature, and its thinking counts against
    # max_tokens (kept under the SDK's limit for a non-streaming call). Sonnet 5.5 rejects the forced tool call
    # that function-calling structured output uses, so ask for the JSON schema directly.
    llm = models.chat_model(models.INGEST_MODEL, max_tokens=20000).with_structured_output(
        ArticleBoundaries, method="json_schema", include_raw=True).with_retry()

    text_with_line_numbers = add_line_numbers(text)

    output = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=text_with_line_numbers)])
    article_boundaries = output['parsed'].article_boundaries
    print_cost(output['raw'].usage_metadata or {})
    print(f"extracted {len(article_boundaries)} article boundaries")

    return article_boundaries

def filter_existing_docs(paths:list[str], knowledge_base_path:str)->list[str]:
    existing_docs = load_docs_from_jsonl(knowledge_base_path)
    existing_filenames = {os.path.basename(doc.metadata.get('source')) for doc in existing_docs}
    return [path for path in paths if os.path.basename(path) not in existing_filenames]

def save_pdf_text_if_not_exists(path):
    save_path = f"documents/txts/{os.path.basename(path).replace('.pdf', '.txt')}"
    if os.path.exists(save_path):
        return
    text = extract_pdf_text(path)
    with open(save_path, "w") as f:
        f.write(text)

async def main():
    # convert pdfs to txt
    pdf_paths = glob("documents/documents_renamed/*.pdf")
    knowledge_base_path = "knowledge_base.jsonl"
    
    # with Pool(processes=cpu_count()) as pool:
    #     pool.map(save_pdf_text_if_not_exists, pdf_paths)

    # # filter existing txts from knowledge_base
    # txt_paths = glob("documents/txts/*.txt")
    # txt_paths = filter_existing_docs(txt_paths, knowledge_base_path)

    # if not txt_paths:
    #     print("No new documents to process")
    #     exit()

    # async def process_txt_file(txt_path):
    #     try:
    #         logging.info(f"extracting article boundaries for {txt_path}")
    #         with open(txt_path, "r") as f:
    #             text = f.read()
    #         article_boundaries = await extract_article_boundaries(text)
    #         articles = parse_articles(text, article_boundaries, txt_path)
    #         for article in articles:
    #             save_doc_to_json(article, f"documents/articles/{get_doc_id(article)}.json")
    #     except Exception as e:
    #         logging.error(f"Error processing {txt_path}: {str(e)}")
    #         raise
    
    # # Process files in batches of 3 with progress tracking
    # for i in tqdm(range(0, len(txt_paths), 3), desc="Processing text files"):
    #     batch = txt_paths[i:i+3]
    #     try:
    #         await asyncio.gather(*[process_txt_file(txt_path) for txt_path in batch])
    #     except Exception as e:
    #         logging.error(f"Batch processing failed: {str(e)}")
    #         # Continue with next batch instead of stopping completely
    #         continue
    
    # move articles to knowledge base
    article_paths = glob("documents/articles/*.json")
    articles = [load_doc_from_json(path) for path in article_paths]
    encoding = get_encoding("cl100k_base")
    articles = [article for article in articles if len(encoding.encode(article.page_content)) < 9000]
    save_docs_to_jsonl(articles, knowledge_base_path)

    # add articles to vector store
    batch_size = 100
    for i in trange(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        add_to_vector_store(batch)
    

if __name__ == "__main__":
    asyncio.run(main())