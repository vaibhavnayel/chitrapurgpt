"""Network shims for running the app inside a sandbox whose only egress is an HTTPS proxy. Eval and E2E only:
production never imports this. Applied by eval/run_eval.py, and by tests/e2e/sitecustomize.py for the E2E server.

- embeddings-no-local-tiktoken: OpenAIEmbeddings tokenizes queries locally with tiktoken (to split texts over 8191
  tokens), and tiktoken downloads its vocabulary on first use. Where that download is blocked, send the query text
  instead: the API tokenizes it with the same cl100k vocabulary, so a short query gets the same embedding.
"""
APPLIED: list[str] = []


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
        embed_without_local_tokenizer_if_unavailable()
    return APPLIED
