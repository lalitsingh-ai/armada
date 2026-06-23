"""Chat clients.

Two implementations share one interface (`chat`): `LlamaCppClient` talks to a real llama.cpp
server over the OpenAI-compatible API, and `MockClient` is a deterministic, offline model driven
by each task's `mock_plan`. The mock models the prompt-prefix cache so that the caching
optimization is visible (and testable) even without a server.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx

from .tasks import MockPlan

# Synthetic throughput used only by the mock to produce plausible, self-consistent latencies.
_MOCK_PREFILL_TPS = 1000.0
_MOCK_DECODE_TPS = 50.0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float | None = None


def _estimate_tokens_text(text: str) -> int:
    return max(1, len(text) // 4)


def _estimate_tokens_messages(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += _estimate_tokens_text(content)
        for call in msg.get("tool_calls", []) or []:
            fn = call.get("function", {})
            total += _estimate_tokens_text(fn.get("name", "") + str(fn.get("arguments", "")))
    return total


class MockClient:
    """A deterministic, offline 'model' that replays each task's planned tool calls."""

    def __init__(self, prompt_cache: bool = True, sim_speed: float = 1.0):
        self.prompt_cache = prompt_cache
        # sim_speed scales the simulated wall time (1.0 = realistic, 0.0 = instant for tests).
        self.sim_speed = sim_speed

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        mock_plan: MockPlan | None = None,
    ) -> ChatResponse:
        plan = mock_plan or MockPlan(tool_calls=[], final="")
        n_done = sum(1 for m in messages if m.get("role") == "tool")

        prompt_tokens = _estimate_tokens_messages(messages)
        if self.prompt_cache and n_done > 0:
            # The whole prefix except the freshly appended tool result is served from KV cache.
            cached_tokens = _estimate_tokens_messages(messages[:-1])
        else:
            cached_tokens = 0
        new_prompt_tokens = max(prompt_tokens - cached_tokens, 1)

        if n_done < len(plan.tool_calls):
            planned = plan.tool_calls[n_done]
            args_json = json.dumps(planned.arguments)
            completion_tokens = _estimate_tokens_text(planned.name + args_json)
            response = ChatResponse(
                content=None,
                tool_calls=[ToolCall(id=f"call_{n_done}", name=planned.name, arguments=planned.arguments)],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
        else:
            completion_tokens = _estimate_tokens_text(plan.final)
            response = ChatResponse(
                content=plan.final,
                tool_calls=[],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )

        latency = new_prompt_tokens / _MOCK_PREFILL_TPS + completion_tokens / _MOCK_DECODE_TPS
        response.latency_s = latency
        response.ttft_s = new_prompt_tokens / _MOCK_PREFILL_TPS
        if self.sim_speed > 0:
            await asyncio.sleep(latency * self.sim_speed)
        else:
            await asyncio.sleep(0)
        return response

    async def aclose(self) -> None:  # symmetry with the real client
        return None


class LlamaCppClient:
    """Async client for a llama.cpp server's OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, base_url: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        mock_plan: MockPlan | None = None,  # ignored; kept for interface symmetry
    ) -> ChatResponse:
        payload = {
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_prompt": True,
            "timings_per_token": True,
        }
        start = time.perf_counter()
        resp = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        latency = time.perf_counter() - start
        data = resp.json()
        return _parse_openai_response(data, latency)

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_openai_response(data: dict, latency: float) -> ChatResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls: list[ToolCall] = []
    for idx, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=call.get("id", f"call_{idx}"), name=fn.get("name", ""), arguments=args))

    usage = data.get("usage", {}) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    # Tokens actually evaluated (prompt_n) < prompt_tokens means the KV prefix cache was hit.
    timings = data.get("timings", {}) or {}
    cached_tokens = 0
    ttft_s: float | None = None
    details = usage.get("prompt_tokens_details") or {}
    if "cached_tokens" in details:
        cached_tokens = int(details["cached_tokens"])
    elif "prompt_n" in timings:
        cached_tokens = max(prompt_tokens - int(timings["prompt_n"]), 0)
    if "prompt_ms" in timings:
        ttft_s = float(timings["prompt_ms"]) / 1000.0

    return ChatResponse(
        content=message.get("content"),
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        latency_s=latency,
        ttft_s=ttft_s,
    )
