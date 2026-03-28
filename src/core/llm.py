from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from api_keys import get_api_key
from config.settings import settings


def _resolve_chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _extract_text_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(text, dict) and isinstance(text.get("value"), str):
                parts.append(str(text["value"]))
                continue
            nested = item.get("content")
            if isinstance(nested, str):
                parts.append(nested)
        merged = "\n".join(part for part in parts if part)
        return merged or None
    return None


def _extract_response_text(body: Dict[str, Any]) -> Optional[str]:
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                extracted = _extract_text_from_content(message.get("content"))
                if extracted is not None:
                    return extracted
            legacy_text = first_choice.get("text")
            if isinstance(legacy_text, str):
                return legacy_text

    output_text = body.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = body.get("output")
    if isinstance(output, list):
        extracted = _extract_text_from_content(output)
        if extracted is not None:
            return extracted
    return None


@dataclass(slots=True)
class ProviderStatus:
    available: bool = True
    failure_count: int = 0
    last_failure_at: float = 0.0


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


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, *args: Any, api_key: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.api_key = api_key

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
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        _ = purpose
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                content = _extract_response_text(body)
                if content is None:
                    raise RuntimeError(
                        f"Provider {self.name} returned no text content: "
                        f"{json.dumps(body, ensure_ascii=False)[:1000]}"
                    )
                return content
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep((attempt + 1) * 2)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Provider {self.name} call failed")


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


class LLMManager:
    def __init__(self) -> None:
        self.providers: Dict[str, LLMProvider] = {"mock": MockProvider()}
        self.provider_status: Dict[str, ProviderStatus] = {"mock": ProviderStatus()}
        self._failure_threshold = settings.llm_failure_threshold
        self._recovery_time = settings.llm_recovery_time
        self._trace_id_var: ContextVar[str] = ContextVar("llm_trace_id", default="")
        self._tracer_var: ContextVar[Any] = ContextVar("llm_tracer", default=None)
        self._call_index_var: ContextVar[int] = ContextVar("llm_call_index", default=0)
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
        for name, cfg in settings.provider_configs.items():
            api_key = get_api_key(name)
            base_url = str(cfg.get("base_url") or "").strip()
            if not api_key or not base_url:
                continue
            self.providers[name] = OpenAICompatibleProvider(
                name=name,
                model=str(cfg.get("model", "")),
                base_url=_resolve_chat_completions_url(base_url),
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
            candidates.append((provider.priority, name))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [name for _, name in candidates]

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

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "available": status.available,
                "failure_count": status.failure_count,
                "last_failure_at": status.last_failure_at,
                "model": self.providers[name].model,
            }
            for name, status in self.provider_status.items()
        }

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
            response = getattr(exc, "response", None)
            http_status = None
            error_response_preview = ""
            if response is not None:
                http_status = getattr(response, "status_code", None)
                try:
                    error_response_preview = str(response.text)[:400]
                except Exception:
                    error_response_preview = ""
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
                http_status=http_status,
                error_response_preview=error_response_preview,
            )
            raise

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
        http_status: Optional[int] = None,
        error_response_preview: str = "",
    ) -> None:
        trace_id = self._trace_id_var.get()
        tracer = self._tracer_var.get()
        if not trace_id or tracer is None:
            return

        tracer.trace_step(
            trace_id,
            "llm",
            {
                "call_id": call_id,
                "purpose": purpose,
                "requested_provider": requested_provider or "auto",
                "prompt_preview": prompt[:240],
                "response_format": response_format,
                "max_tokens": max_tokens,
                "budget_limit": self._budget_limit_var.get(),
                "budget_used": self._budget_used_var.get(),
            },
            {
                "call_id": call_id,
                "phase": phase,
                "status": status,
                "provider": provider.name,
                "model": provider.model,
                "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
                "response_preview": response_preview,
                "error": error,
                "error_type": error_type,
                "http_status": http_status,
                "error_response_preview": error_response_preview,
                "budget_remaining": self.get_budget_status()["remaining"],
            },
            metadata={"attempt": attempt, "fallback": fallback},
        )
