"""The agent loop: a ReAct-style tool-calling cycle.

One agent solves one task by repeatedly calling the model, executing any requested tools, and
feeding the results back until the model returns a final answer. The system prompt is deliberately
substantial because it (plus the tool schemas) is the *repeated prefix* whose caching Armada
measures.
"""

from __future__ import annotations

import json
import time

from .client import ChatResponse, ToolCall
from .metrics import StepMetrics, TaskResult
from .tasks import Task
from .tools import TOOLS_SCHEMA, execute_tool

MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are Armada, a precise tool-using assistant running on an Arm-based CPU server. "
    "You answer questions by calling the available tools rather than guessing. "
    "Rules:\n"
    "1. Use the `calculator` tool for any arithmetic; never compute by hand.\n"
    "2. Use the `kv_lookup` tool to retrieve facts by key (for example 'population_of_paris').\n"
    "3. Call one tool at a time and wait for its result before deciding the next step.\n"
    "4. When you have enough information, reply with a short final answer that states the result "
    "explicitly.\n"
    "Always be concise and accurate."
)


def _assistant_tool_message(tool_calls: list[ToolCall]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ],
    }


async def run_task(task: Task, client, *, temperature: float = 0.0, max_tokens: int = 512) -> TaskResult:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    steps: list[StepMetrics] = []
    tool_calls_made = 0
    start = time.perf_counter()

    for _ in range(MAX_STEPS):
        resp: ChatResponse = await client.chat(
            messages,
            TOOLS_SCHEMA,
            temperature=temperature,
            max_tokens=max_tokens,
            mock_plan=task.mock_plan,
        )
        steps.append(
            StepMetrics(
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cached_tokens=resp.cached_tokens,
                latency_s=resp.latency_s,
                ttft_s=resp.ttft_s,
            )
        )

        if resp.tool_calls:
            messages.append(_assistant_tool_message(resp.tool_calls))
            for tc in resp.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                tool_calls_made += 1
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        final = resp.content or ""
        return TaskResult(
            task_id=task.id,
            success=task.check(final),
            steps=steps,
            wall_time_s=time.perf_counter() - start,
            tool_calls=tool_calls_made,
        )

    return TaskResult(
        task_id=task.id,
        success=False,
        steps=steps,
        wall_time_s=time.perf_counter() - start,
        tool_calls=tool_calls_made,
        error="max steps exceeded",
    )
