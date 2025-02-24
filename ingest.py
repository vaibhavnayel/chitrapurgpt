from concurrent.futures import ProcessPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import unidecode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel, Field
import re
from glob import glob
import json
import traceback
import dotenv
dotenv.load_dotenv()

def sanitize_string(text):
    return unidecode.unidecode(text)

def add_line_numbers(text):
    lines = text.split('\n')
    numbered_lines = [f"{i+1}. {line}" for i, line in enumerate(lines)]
    return '\n'.join(numbered_lines)

def preprocess_pdf(path):
    try:
        docs = PyPDFLoader(path, mode='single',extraction_mode='layout',pages_delimiter="\n------- PAGE END -------\n").load()
    except:
        docs = PyPDFLoader(path, mode='single',extraction_mode='plain',pages_delimiter="\n------- PAGE END -------\n").load()

    full_text = docs[0].page_content
    full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
    sanitized_text = sanitize_string(full_text)
    return sanitized_text

class Article(BaseModel):
    title: str = Field(description="The title of the article")
    author: str|None = Field(description="The author of the article, leave empty if not found")
    summary: str|None = Field(description="A short summary of the article. 1 to 5 sentences.")
    published_date: str = Field(description="date of publication of the article. DD-MM-YYYY format. use first of the month if exact date isn't found.")
    start_line: int = Field(description="The line number of the first line of the article")
    end_line: int = Field(description="The line number of the last line of the article")
    start_page: int = Field(description="The page number of the first page of the article")
    end_page: int = Field(description="The page number of the last page of the article")

class ArticleList(BaseModel):
    scratchpad: str = Field(description="space for thinking and reasoning about the articles before extraction")
    articles: list[Article]

def print_cost(cb):
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Cost: ${cb.prompt_tokens * 3e-6:.6f}")
    print(f"Completion Cost: ${cb.completion_tokens * 15e-6:.6f}")
    print(f"Total Cost: ${cb.prompt_tokens * 3e-6 + cb.completion_tokens * 15e-6:.6f}")

def extract_articles(path:str) -> list[Document]:
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
    # llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=4096).with_structured_output(ArticleList)
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, max_tokens=8000).with_structured_output(ArticleList, include_raw=True)
    
    text = preprocess_pdf(path)
    text_with_line_numbers = add_line_numbers(text)

    with get_openai_callback() as cb:
        output = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=text_with_line_numbers)])
        articles = output['parsed'].articles
        print_cost(cb)
    print(f"extracted {len(articles)} articles")

    docs = []
    for article in articles:
        article_text = '\n'.join(text.split('\n')[article.start_line:article.end_line])
        metadata = {
            "source": path.split("/")[-1],
            "published_date": article.published_date,
            "title": article.title,
            "author": article.author,
            "summary": article.summary,
            "start_line": article.start_line,
            "end_line": article.end_line,
            "start_page": article.start_page,
            "end_page": article.end_page,
        }
        docs.append(Document(page_content=article_text, metadata=metadata))

    return docs

def save_docs_to_jsonl(array:list[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.model_dump_json() + '\n')

def load_docs_from_jsonl(file_path: str)->list[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


if __name__ == "__main__":
    paths = glob("/Users/personal/projects/chitrapurgpt/documents/*.pdf")

    def process_file(path):
        print(f"Extracting articles from {path}")
        try:
            return extract_articles(path)
        except Exception as e:
            print(f"Error extracting articles from {path}: {e}")
            with open('errors.txt', 'a') as f:
                f.write(f"{path}\n{traceback.format_exc()}\n\n-----------------------------------------------------------------------------------")
            return []

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_file, paths))
    
    articles = []
    for result in results:
        articles.extend(result)

    save_docs_to_jsonl(articles,'knowledge_base.jsonl')