"""Shared LLM access for generation, enrichment and the RAGAS judge."""

import os
from functools import lru_cache

from config import (
    EVAL_EMBEDDING_MODEL,
    LLM_API_FORMAT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT,
)


def is_configured() -> bool:
    return os.getenv("RAG_OFFLINE", "0") != "1" and bool(
        LLM_API_KEY and LLM_API_KEY != "sk-..." and LLM_MODEL
    )


def validate_config() -> None:
    if not is_configured():
        raise ValueError("Điền LLM_API_KEY và LLM_MODEL trong .env trước khi gọi LLM.")
    if LLM_API_FORMAT not in {"anthropic", "openai"}:
        raise ValueError("LLM_API_FORMAT phải là anthropic hoặc openai.")
    if LLM_TIMEOUT <= 0:
        raise ValueError("LLM_TIMEOUT phải lớn hơn 0.")


def api_base_url() -> str:
    """SDKs add endpoint paths; do not accidentally produce /v1/v1/messages."""
    base = LLM_BASE_URL.rstrip("/")
    if LLM_API_FORMAT == "anthropic":
        return base.removesuffix("/v1")
    return base if base.endswith("/v1") else base + "/v1"


@lru_cache(maxsize=1)
def get_client():
    validate_config()
    kwargs = {
        "api_key": LLM_API_KEY,
        "base_url": api_base_url(),
        "timeout": LLM_TIMEOUT,
        "max_retries": 1,
    }
    if LLM_API_FORMAT == "anthropic":
        from anthropic import Anthropic

        return Anthropic(**kwargs)
    from openai import OpenAI

    return OpenAI(**kwargs)


def generate_text(system: str, prompt: str, max_tokens: int = 512) -> str:
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    if LLM_API_FORMAT == "anthropic":
        response = client.messages.create(
            model=LLM_MODEL,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
    else:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
    if not text.strip():
        raise ValueError("LLM trả về nội dung rỗng.")
    return text.strip()


def safe_error(error: Exception) -> str:
    """Keep provider errors useful without logging the configured API key."""
    message = str(error)
    return message.replace(LLM_API_KEY, "[REDACTED]") if LLM_API_KEY else message


def get_eval_models():
    """Explicit LLM + local embeddings prevent RAGAS's OpenAI defaults."""
    validate_config()
    if LLM_API_FORMAT == "anthropic":
        from langchain_anthropic import ChatAnthropic

        judge = ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=LLM_API_KEY,
            anthropic_api_url=api_base_url(),
            max_tokens=2048,
            timeout=LLM_TIMEOUT,
            max_retries=1,
        )
    else:
        from langchain_openai import ChatOpenAI

        judge = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=LLM_API_KEY,
            openai_api_base=api_base_url(),
            max_tokens=2048,
            request_timeout=LLM_TIMEOUT,
            max_retries=1,
        )
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EVAL_EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return judge, embeddings
