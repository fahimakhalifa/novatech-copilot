"""The one place chat models are created."""
from functools import lru_cache
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import get_settings

ModelRole = Literal["agent", "fast", "judge"]

FALLBACK_ON = (ModelRateLimitError, ModelAPIError)


def model_chain(role: ModelRole) -> list[str]:
    s = get_settings()
    raw = {"agent": s.agent_models, "fast": s.fast_models, "judge": s.judge_model}[role]
    return [name.strip() for name in raw.split(",") if name.strip()]


@lru_cache
def _model(name: str, last_in_chain: bool) -> BaseChatModel:
    s = get_settings()
    api_key = s.groq_api_key if name.startswith("groq:") else s.google_api_key
    extra = {}
    if name.startswith("groq:qwen"):
        # Qwen3 writes its reasoning into the answer text as <think>...</think> unless told not to.
        extra["reasoning_format"] = "hidden"
    elif "gpt-oss" in name:
        extra["reasoning_effort"] = "low"
    return init_chat_model(name, api_key=api_key, max_retries=4 if last_in_chain else 0, **extra)


def get_llm(role: ModelRole, *, tools: list[dict] | None = None,
            schema: type[BaseModel] | None = None) -> Runnable:
    """A model (with fallbacks) for `role`, optionally with tools or structured output."""
    names = model_chain(role)
    models: list[Runnable] = [_model(n, i == len(names) - 1) for i, n in enumerate(names)]
    if tools:
        models = [m.bind_tools(tools) for m in models]
    if schema:
        models = [m.with_structured_output(schema, method="json_schema") if n.startswith("groq:")
                  else m.with_structured_output(schema) for m, n in zip(models, names)]
    primary, *fallbacks = models
    return primary.with_fallbacks(fallbacks, exceptions_to_handle=FALLBACK_ON) if fallbacks else primary
