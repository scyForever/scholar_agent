from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from dataclasses import dataclass
from uuid import UUID
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from config.settings import settings

try:
    from api_keys import get_api_key
except ImportError:  # pragma: no cover
    def get_api_key(provider_name: str) -> str:
        cfg = settings.provider_configs.get(provider_name, {})
        key_name = str(cfg.get("api_key_name") or f"{provider_name.upper()}_API_KEY")
        return os.getenv(key_name, "")

try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = None

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object
    LLMResult = Any
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None


def _langchain_available() -> bool:
    return all(component is not None for component in (ChatOpenAI, ChatPromptTemplate, StrOutputParser))


StructuredSchemaT = TypeVar("StructuredSchemaT", bound="BaseModel") # type: ignore


def _resolve_langchain_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


@dataclass(slots=True)
class ProviderStatus:
    available: bool = True
    failure_count: int = 0
    last_failure_at: float = 0.0
    last_success_at: float = 0.0


class LLMProvider(ABC):
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        priority: int,
        timeout: int,
        max_retries: int,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url
        self.priority = priority
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def call(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: str = "text",
        purpose: str = "",
    ) -> str:
        raise NotImplementedError

    def call_structured(
        self,
        prompt: str,
        schema: Type[StructuredSchemaT],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        purpose: str = "",
        method: str = "function_calling",
    ) -> StructuredSchemaT:
        raw = self.call(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
            purpose=purpose,
        )
        payload = json.loads(raw)
        return schema.model_validate(payload)

    def create_chat_model(
        self,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        callbacks: Optional[List[Any]] = None,
    ) -> Any:
        raise NotImplementedError


class LangChainOpenAICompatibleProvider(LLMProvider):
    def __init__(self, *args: Any, api_key: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not _langchain_available():
            raise RuntimeError("LangChain 依赖未安装，无法初始化 LangChain provider。")
        self.api_key = api_key
        self.client = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=_resolve_langchain_base_url(self.base_url),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def call(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: str = "text",
        purpose: str = "",
    ) -> str:
        _ = purpose
        prompt_template, payload = self._build_prompt(prompt, system_prompt)
        bind_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            bind_kwargs["response_format"] = {"type": "json_object"}
        chain = prompt_template | self.client.bind(**bind_kwargs) | StrOutputParser()
        return str(chain.invoke(payload))

    def call_structured(
        self,
        prompt: str,
        schema: Type[StructuredSchemaT],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        purpose: str = "",
        method: str = "function_calling",
    ) -> StructuredSchemaT:
        _ = purpose
        prompt_template, payload = self._build_prompt(prompt, system_prompt)
        runnable = self.client.bind(
            temperature=temperature,
            max_tokens=max_tokens,
        ).with_structured_output(
            schema,
            method=method,
            strict=False,
        )
        chain = prompt_template | runnable
        result = chain.invoke(payload)
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> tuple[Any, Dict[str, str]]:
        if system_prompt:
            return (
                ChatPromptTemplate.from_messages(
                    [
                        ("system", "{system_prompt}"),
                        ("human", "{prompt}"),
                    ]
                ),
                {"system_prompt": system_prompt, "prompt": prompt},
            )
        return (
            ChatPromptTemplate.from_messages([("human", "{prompt}")]),
            {"prompt": prompt},
        )

    def create_chat_model(
        self,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        callbacks: Optional[List[Any]] = None,
    ) -> Any:
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=_resolve_langchain_base_url(self.base_url),
            timeout=self.timeout,
            max_retries=self.max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=list(callbacks or []),
        )


class MockProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(
            name="mock",
            model="mock-model",
            base_url="",
            priority=0,
            timeout=0,
            max_retries=0,
        )

    def call(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: str = "text",
        purpose: str = "",
    ) -> str:
        _ = (system_prompt, temperature, max_tokens, purpose)
        if response_format == "json":
            return json.dumps(
                {
                    "intent": "search_papers",
                    "confidence": 0.35,
                    "summary": "Mock response generated without external LLM.",
                },
                ensure_ascii=False,
            )
        return (
            "这是一个 Mock LLM 响应，用于在未配置外部模型时保持项目链路可运行。"
            f"\n\n用户输入摘要：{prompt[:240]}"
        )

    def call_structured(
        self,
        prompt: str,
        schema: Type[StructuredSchemaT],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        purpose: str = "",
        method: str = "function_calling",
    ) -> StructuredSchemaT:
        _ = (system_prompt, temperature, max_tokens, purpose, method)
        payload = json.loads(self.call(prompt, response_format="json"))
        return schema.model_validate(payload)

    def create_chat_model(
        self,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        callbacks: Optional[List[Any]] = None,
    ) -> Any:
        _ = (temperature, max_tokens, callbacks)
        raise RuntimeError("Mock provider does not support LangChain tool-calling agents.")


class LangChainTraceCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    def __init__(
        self,
        manager: "LLMManager",
        provider: LLMProvider,
        *,
        purpose: str,
        max_tokens: int,
    ) -> None:
        self.manager = manager
        self.provider = provider
        self.purpose = purpose
        self.max_tokens = max_tokens
        self._started_at: Dict[UUID, float] = {}
        self._call_ids: Dict[UUID, int] = {}
        self._prompt_preview: Dict[UUID, str] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = (serialized, parent_run_id, tags, metadata, kwargs)
        prompt_preview = self._messages_preview(messages)
        call_id = self.manager._next_call_id()
        self._started_at[run_id] = time.perf_counter()
        self._call_ids[run_id] = call_id
        self._prompt_preview[run_id] = prompt_preview
        try:
            self.manager._consume_budget(self.purpose)
            self.manager._trace_llm_event(
                call_id=call_id,
                phase="started",
                status="running",
                provider=self.provider,
                prompt=prompt_preview,
                response_format="agent_messages",
                max_tokens=self.max_tokens,
                purpose=self.purpose,
                requested_provider=self.provider.name,
                attempt=1,
                fallback=False,
            )
        except Exception as exc:
            self.manager._trace_llm_event(
                call_id=call_id,
                phase="completed",
                status="error",
                provider=self.provider,
                prompt=prompt_preview,
                response_format="agent_messages",
                max_tokens=self.max_tokens,
                purpose=self.purpose,
                requested_provider=self.provider.name,
                attempt=1,
                fallback=False,
                error=str(exc)[:400],
                error_type=type(exc).__name__,
            )
            raise

    def on_llm_end(
        self,
        response: LLMResult, # type: ignore
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = (parent_run_id, tags, kwargs)
        call_id = self._call_ids.pop(run_id, self.manager._next_call_id())
        started_at = self._started_at.pop(run_id, time.perf_counter())
        prompt_preview = self._prompt_preview.pop(run_id, "")
        self.manager._trace_llm_event(
            call_id=call_id,
            phase="completed",
            status="success",
            provider=self.provider,
            prompt=prompt_preview,
            response_format="agent_messages",
            max_tokens=self.max_tokens,
            purpose=self.purpose,
            requested_provider=self.provider.name,
            attempt=1,
            fallback=False,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            response_preview=self._response_preview(response),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        _ = (parent_run_id, tags, kwargs)
        call_id = self._call_ids.pop(run_id, self.manager._next_call_id())
        started_at = self._started_at.pop(run_id, time.perf_counter())
        prompt_preview = self._prompt_preview.pop(run_id, "")
        self.manager._trace_llm_event(
            call_id=call_id,
            phase="completed",
            status="error",
            provider=self.provider,
            prompt=prompt_preview,
            response_format="agent_messages",
            max_tokens=self.max_tokens,
            purpose=self.purpose,
            requested_provider=self.provider.name,
            attempt=1,
            fallback=False,
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            error=str(error)[:400],
            error_type=type(error).__name__,
        )

    def _messages_preview(self, messages: list[list[Any]]) -> str:
        parts: List[str] = []
        for batch in messages:
            for message in batch:
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    merged = []
                    for item in content:
                        if isinstance(item, str):
                            merged.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("text"), str):
                            merged.append(str(item["text"]))
                    content = "\n".join(merged)
                parts.append(str(content))
        return "\n".join(part for part in parts if part)[:240]

    def _response_preview(self, response: LLMResult) -> str: # type: ignore
        try:
            generations = response.generations or []
            if not generations or not generations[0]:
                return ""
            generation = generations[0][0]
            message = getattr(generation, "message", None)
            content = getattr(message, "content", "")
            if isinstance(content, list):
                chunks: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        chunks.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        chunks.append(str(item["text"]))
                return "\n".join(chunks)[:240]
            return str(content)[:240]
        except Exception:
            return ""


class LLMManager:
    def __init__(self) -> None:
        self.providers: Dict[str, LLMProvider] = {"mock": MockProvider()}
        self.provider_status: Dict[str, ProviderStatus] = {"mock": ProviderStatus()}
        self._failure_threshold = settings.llm_failure_threshold
        self._recovery_time = settings.llm_recovery_time
        self._trace_id_var: ContextVar[str] = ContextVar("llm_trace_id", default="")
        self._tracer_var: ContextVar[Any] = ContextVar("llm_tracer", default=None)
        self._call_index_var: ContextVar[int] = ContextVar("llm_call_index", default=0)
        self._stage_var: ContextVar[str] = ContextVar("llm_stage", default="")
        self._budget_limit_var: ContextVar[int | None] = ContextVar("llm_budget_limit", default=None)
        self._budget_used_var: ContextVar[int] = ContextVar("llm_budget_used", default=0)
        self._init_providers()

    def bind_trace(self, tracer: Any, trace_id: str) -> Tuple[Token[str], Token[Any], Token[int]]:
        return (
            self._trace_id_var.set(trace_id),
            self._tracer_var.set(tracer),
            self._call_index_var.set(0),
        )

    def reset_trace(self, tokens: Tuple[Token[str], Token[Any], Token[int]]) -> None:
        trace_id_token, tracer_token, call_index_token = tokens
        self._trace_id_var.reset(trace_id_token)
        self._tracer_var.reset(tracer_token)
        self._call_index_var.reset(call_index_token)

    def bind_budget(self, max_calls: int | None) -> Tuple[Token[int | None], Token[int]]:
        limit = int(max_calls) if isinstance(max_calls, int) and max_calls > 0 else None
        return (
            self._budget_limit_var.set(limit),
            self._budget_used_var.set(0),
        )

    def reset_budget(self, tokens: Tuple[Token[int | None], Token[int]]) -> None:
        limit_token, used_token = tokens
        self._budget_limit_var.reset(limit_token)
        self._budget_used_var.reset(used_token)

    def bind_stage(self, stage: str) -> Token[str]:
        return self._stage_var.set(str(stage or ""))

    def reset_stage(self, token: Token[str]) -> None:
        self._stage_var.reset(token)

    def get_budget_status(self) -> Dict[str, int | None]:
        limit = self._budget_limit_var.get()
        used = self._budget_used_var.get()
        remaining = None if limit is None else max(limit - used, 0)
        return {"limit": limit, "used": used, "remaining": remaining}

    def _consume_budget(self, purpose: str) -> None:
        limit = self._budget_limit_var.get()
        if limit is None:
            return
        used = self._budget_used_var.get()
        if used >= limit:
            label = purpose or "未命名调用"
            raise RuntimeError(f"LLM call budget exceeded: used {used}/{limit}, blocked purpose={label}")
        self._budget_used_var.set(used + 1)

    def _init_providers(self) -> None:
        if not _langchain_available():
            return
        for name, cfg in settings.provider_configs.items():
            api_key = get_api_key(name)
            base_url = str(cfg.get("base_url") or "").strip()
            if not api_key or not base_url:
                continue
            self.providers[name] = LangChainOpenAICompatibleProvider(
                name=name,
                model=str(cfg.get("model", "")),
                base_url=base_url,
                priority=int(cfg.get("priority", 1)),
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
                api_key=api_key,
            )
            self.provider_status[name] = ProviderStatus()

    def call(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: str = "text",
        purpose: str = "",
        budgeted: bool = False,
    ) -> str:
        if provider:
            if budgeted:
                self._consume_budget(purpose)
            return self._invoke_provider(
                provider_name=provider,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                purpose=purpose,
                requested_provider=provider,
                attempt=1,
                fallback=False,
            )
        return self.call_with_fallback(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            purpose=purpose,
            budgeted=budgeted,
        )

    def call_with_fallback(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: str = "text",
        purpose: str = "",
        budgeted: bool = False,
    ) -> str:
        if budgeted:
            self._consume_budget(purpose)
        healthy_providers = self._get_healthy_providers()
        for attempt, provider_name in enumerate(healthy_providers, start=1):
            try:
                result = self._invoke_provider(
                    provider_name=provider_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    purpose=purpose,
                    requested_provider=None,
                    attempt=attempt,
                    fallback=len(healthy_providers) > 1,
                )
                self._record_success(provider_name)
                return result
            except Exception:
                self._record_failure(provider_name)
        return self._invoke_provider(
            provider_name="mock",
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            purpose=purpose,
            requested_provider=None,
            attempt=len(healthy_providers) + 1,
            fallback=True,
        )

    def call_json(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        purpose: str = "",
        budgeted: bool = False,
    ) -> Dict[str, Any]:
        raw = self.call(
            prompt,
            provider=provider,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
            purpose=purpose,
            budgeted=budgeted,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def call_structured(
        self,
        prompt: str,
        schema: Type[StructuredSchemaT],
        *,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        purpose: str = "",
        budgeted: bool = False,
        methods: Optional[List[str]] = None,
    ) -> StructuredSchemaT:
        if BaseModel is None:
            raise RuntimeError("Pydantic 不可用，无法执行 structured output。")
        structured_methods = list(methods or ["function_calling", "json_schema", "json_mode"])
        if provider:
            if budgeted:
                self._consume_budget(purpose)
            return self._invoke_provider_structured(
                provider_name=provider,
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose=purpose,
                requested_provider=provider,
                attempt=1,
                fallback=False,
                methods=structured_methods,
            )

        if budgeted:
            self._consume_budget(purpose)
        healthy_providers = self._get_healthy_providers()
        for attempt, provider_name in enumerate(healthy_providers, start=1):
            try:
                result = self._invoke_provider_structured(
                    provider_name=provider_name,
                    prompt=prompt,
                    schema=schema,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    purpose=purpose,
                    requested_provider=None,
                    attempt=attempt,
                    fallback=len(healthy_providers) > 1,
                    methods=structured_methods,
                )
                self._record_success(provider_name)
                return result
            except Exception:
                self._record_failure(provider_name)

        return self._invoke_provider_structured(
            provider_name="mock",
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose,
            requested_provider=None,
            attempt=len(healthy_providers) + 1,
            fallback=True,
            methods=["function_calling"],
        )

    def _get_healthy_providers(self) -> List[str]:
        now = time.time()
        candidates = []
        for name, provider in self.providers.items():
            if name == "mock":
                continue
            status = self.provider_status[name]
            if not status.available:
                if now - status.last_failure_at >= self._recovery_time:
                    status.available = True
                    status.failure_count = 0
                else:
                    continue
            candidates.append((1 if status.last_success_at > 0 else 0, status.last_success_at, provider.priority, name))
        candidates.sort(reverse=True)
        return [name for _, _, _, name in candidates]

    def _record_failure(self, provider_name: str) -> None:
        status = self.provider_status[provider_name]
        status.failure_count += 1
        status.last_failure_at = time.time()
        if status.failure_count >= self._failure_threshold:
            status.available = False

    def _record_success(self, provider_name: str) -> None:
        status = self.provider_status[provider_name]
        status.failure_count = 0
        status.available = True
        status.last_success_at = time.time()

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        status = {
            name: {
                "available": provider_status.available,
                "failure_count": provider_status.failure_count,
                "last_failure_at": provider_status.last_failure_at,
                "last_success_at": provider_status.last_success_at,
                "model": self.providers[name].model,
            }
            for name, provider_status in self.provider_status.items()
        }
        if "mock" in status:
            status["mock"]["framework"] = "langchain" if _langchain_available() else "mock-only"
        return status

    def get_healthy_provider_names(self) -> List[str]:
        return list(self._get_healthy_providers())

    def get_verified_provider_names(self) -> List[str]:
        verified: List[str] = []
        for name, status in self.provider_status.items():
            if name == "mock":
                continue
            if status.available and status.last_success_at > 0:
                verified.append(name)
        verified.sort(
            key=lambda item: (
                self.provider_status[item].last_success_at,
                self.providers[item].priority,
            ),
            reverse=True,
        )
        return verified

    def has_verified_provider(self) -> bool:
        return bool(self.get_verified_provider_names())

    def record_provider_failure(self, provider_name: str) -> None:
        if provider_name in self.provider_status:
            self._record_failure(provider_name)

    def record_provider_success(self, provider_name: str) -> None:
        if provider_name in self.provider_status:
            self._record_success(provider_name)

    def create_langchain_chat_model(
        self,
        provider_name: str,
        *,
        purpose: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Any:
        provider = self.providers.get(provider_name)
        if provider is None:
            raise KeyError(f"Unknown provider: {provider_name}")
        callbacks: List[Any] = []
        if BaseCallbackHandler is not None and provider_name != "mock":
            callbacks.append(
                LangChainTraceCallbackHandler(
                    self,
                    provider,
                    purpose=purpose,
                    max_tokens=max_tokens,
                )
            )
        return provider.create_chat_model(
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

    def reset_failures(self, provider_name: str) -> None:
        if provider_name in self.provider_status:
            self.provider_status[provider_name] = ProviderStatus()

    def _invoke_provider(
        self,
        provider_name: str,
        prompt: str,
        *,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format: str,
        purpose: str,
        requested_provider: Optional[str],
        attempt: int,
        fallback: bool,
    ) -> str:
        provider = self.providers[provider_name]
        call_id = self._next_call_id()
        self._trace_llm_event(
            call_id=call_id,
            phase="started",
            status="running",
            provider=provider,
            prompt=prompt,
            response_format=response_format,
            max_tokens=max_tokens,
            purpose=purpose,
            requested_provider=requested_provider,
            attempt=attempt,
            fallback=fallback,
        )
        started = time.perf_counter()
        try:
            result = provider.call(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                purpose=purpose,
            )
            self._trace_llm_event(
                call_id=call_id,
                phase="completed",
                status="success",
                provider=provider,
                prompt=prompt,
                response_format=response_format,
                max_tokens=max_tokens,
                purpose=purpose,
                requested_provider=requested_provider,
                attempt=attempt,
                fallback=fallback,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                response_preview=result[:240],
            )
            return result
        except Exception as exc:
            self._trace_llm_event(
                call_id=call_id,
                phase="completed",
                status="error",
                provider=provider,
                prompt=prompt,
                response_format=response_format,
                max_tokens=max_tokens,
                purpose=purpose,
                requested_provider=requested_provider,
                attempt=attempt,
                fallback=fallback,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc)[:400],
                error_type=type(exc).__name__,
            )
            raise

    def _invoke_provider_structured(
        self,
        provider_name: str,
        prompt: str,
        schema: Type[StructuredSchemaT],
        *,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        purpose: str,
        requested_provider: Optional[str],
        attempt: int,
        fallback: bool,
        methods: List[str],
    ) -> StructuredSchemaT:
        provider = self.providers[provider_name]
        call_id = self._next_call_id()
        self._trace_llm_event(
            call_id=call_id,
            phase="started",
            status="running",
            provider=provider,
            prompt=prompt,
            response_format=f"structured:{schema.__name__}",
            max_tokens=max_tokens,
            purpose=purpose,
            requested_provider=requested_provider,
            attempt=attempt,
            fallback=fallback,
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        last_method = ""

        for method in methods:
            last_method = method
            try:
                result = provider.call_structured(
                    prompt,
                    schema,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    purpose=purpose,
                    method=method,
                )
                self._trace_llm_event(
                    call_id=call_id,
                    phase="completed",
                    status="success",
                    provider=provider,
                    prompt=prompt,
                    response_format=f"structured:{schema.__name__}",
                    max_tokens=max_tokens,
                    purpose=purpose,
                    requested_provider=requested_provider,
                    attempt=attempt,
                    fallback=fallback,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    response_preview=self._structured_preview(result),
                    structured_method=method,
                )
                return result
            except Exception as exc:
                last_error = exc

        try:
            last_method = "json_fallback"
            payload = self.call_json(
                prompt,
                provider=provider_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose=purpose,
                budgeted=False,
            )
            result = schema.model_validate(payload)
            self._trace_llm_event(
                call_id=call_id,
                phase="completed",
                status="success",
                provider=provider,
                prompt=prompt,
                response_format=f"structured:{schema.__name__}",
                max_tokens=max_tokens,
                purpose=purpose,
                requested_provider=requested_provider,
                attempt=attempt,
                fallback=fallback,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                response_preview=self._structured_preview(result),
                structured_method=last_method,
            )
            return result
        except Exception as exc:
            last_error = exc

        self._trace_llm_event(
            call_id=call_id,
            phase="completed",
            status="error",
            provider=provider,
            prompt=prompt,
            response_format=f"structured:{schema.__name__}",
            max_tokens=max_tokens,
            purpose=purpose,
            requested_provider=requested_provider,
            attempt=attempt,
            fallback=fallback,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(last_error)[:400] if last_error is not None else "",
            error_type=type(last_error).__name__ if last_error is not None else "",
            structured_method=last_method,
        )
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Structured call failed for provider {provider_name}")

    def _next_call_id(self) -> int:
        call_id = self._call_index_var.get() + 1
        self._call_index_var.set(call_id)
        return call_id

    def _trace_llm_event(
        self,
        *,
        call_id: int,
        phase: str,
        status: str,
        provider: LLMProvider,
        prompt: str,
        response_format: str,
        max_tokens: int,
        purpose: str,
        requested_provider: Optional[str],
        attempt: int,
        fallback: bool,
        latency_ms: Optional[float] = None,
        response_preview: str = "",
        error: str = "",
        error_type: str = "",
        structured_method: str = "",
        http_status: Optional[int] = None,
        error_response_preview: str = "",
    ) -> None:
        trace_id = self._trace_id_var.get()
        tracer = self._tracer_var.get()
        if not trace_id or tracer is None:
            return
        stage = self._stage_var.get()

        tracer.trace_step(
            trace_id,
            "llm",
            {
                "call_id": call_id,
                "stage": stage,
                "purpose": purpose,
                "requested_provider": requested_provider or "auto",
                "prompt_preview": prompt[:240],
                "response_format": response_format,
                "structured_method": structured_method,
                "max_tokens": max_tokens,
                "budget_limit": self._budget_limit_var.get(),
                "budget_used": self._budget_used_var.get(),
            },
            {
                "call_id": call_id,
                "stage": stage,
                "phase": phase,
                "status": status,
                "provider": provider.name,
                "model": provider.model,
                "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
                "response_preview": response_preview,
                "error": error,
                "error_type": error_type,
                "structured_method": structured_method,
                "http_status": http_status,
                "error_response_preview": error_response_preview,
                "budget_remaining": self.get_budget_status()["remaining"],
            },
            metadata={"attempt": attempt, "fallback": fallback},
        )

    def _structured_preview(self, result: Any) -> str:
        if BaseModel is not None and isinstance(result, BaseModel):
            payload = result.model_dump(mode="json")
            return json.dumps(payload, ensure_ascii=False)[:240]
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)[:240]
        return str(result)[:240]
