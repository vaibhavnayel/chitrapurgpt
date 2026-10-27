"""Network shims for running the app inside a sandbox whose only egress is an HTTPS proxy. Eval and E2E only:
production never imports this. Applied by eval/run_eval.py, and by tests/e2e/sitecustomize.py for the E2E server.

- pinecone-env-proxy: every other client here (httpx, requests) honours HTTPS_PROXY, but the Pinecone SDK uses
  its own urllib3 pool and only takes an explicit proxy_url, so pass it one.
- embeddings-no-local-tiktoken: OpenAIEmbeddings tokenizes queries locally with tiktoken (to split texts over 8191
  tokens), and tiktoken downloads its vocabulary on first use. Where that download is blocked, send the query text
  instead: the API tokenizes it with the same cl100k vocabulary, so a short query gets the same embedding.
"""
import os

APPLIED: list[str] = []


def route_pinecone_through_env_proxy() -> None:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        return
    from urllib.parse import unquote, urlsplit

    import pinecone
    import urllib3

    parts = urlsplit(proxy)
    bare = f"{parts.scheme}://{parts.hostname}:{parts.port}"
    headers = (urllib3.make_headers(proxy_basic_auth=f"{unquote(parts.username)}:{unquote(parts.password or '')}")
               if parts.username else None)
    original_init = pinecone.Pinecone.__init__

    def init_with_proxy(self, *args, **kwargs):
        kwargs.setdefault("proxy_url", bare)
        if headers:
            kwargs.setdefault("proxy_headers", headers)
        original_init(self, *args, **kwargs)

    pinecone.Pinecone.__init__ = init_with_proxy
    APPLIED.append("pinecone-env-proxy")


def embed_without_local_tokenizer_if_unavailable() -> None:
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return
    except Exception:  # noqa: BLE001
        pass
    from langchain_openai import OpenAIEmbeddings

    original_init = OpenAIEmbeddings.__init__

    def init_without_tokenizer(self, *args, **kwargs):
        kwargs.setdefault("check_embedding_ctx_length", False)
        original_init(self, *args, **kwargs)

    OpenAIEmbeddings.__init__ = init_without_tokenizer
    APPLIED.append("embeddings-no-local-tiktoken")


def apply() -> list[str]:
    if not APPLIED:
        route_pinecone_through_env_proxy()
        embed_without_local_tokenizer_if_unavailable()
    return APPLIED
