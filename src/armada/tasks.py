"""The agent task suite.

Tasks are deterministic and require tool use. Each task carries a ``mock_plan`` describing the
exact sequence of tool calls and the final answer, so the deterministic mock model can drive the
full agent loop without a real LLM. In real (server) mode the ``mock_plan`` is ignored and the
model decides for itself; ``check()`` verifies the final answer either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ToolCallPlan:
    name: str
    arguments: dict


@dataclass
class MockPlan:
    """Deterministic script for the mock model: tool calls in order, then a final answer."""

    tool_calls: list[ToolCallPlan] = field(default_factory=list)
    final: str = ""


@dataclass
class Task:
    id: str
    prompt: str
    expected: str  # canonical answer that must appear in the final response
    mock_plan: MockPlan

    def check(self, final_answer: str) -> bool:
        """True if the expected answer appears in the model's final text (digits-normalized)."""
        want = _normalize(self.expected)
        got = _normalize(final_answer)
        return want in got


def _normalize(text: str) -> str:
    # Drop thousands separators so "16,100,000" matches "16100000"; lowercase for words.
    return re.sub(r"(?<=\d),(?=\d)", "", text).lower().strip()


def _tc(name: str, **arguments) -> ToolCallPlan:
    return ToolCallPlan(name=name, arguments=arguments)


def default_suite() -> list[Task]:
    return [
        Task(
            id="calc-simple",
            prompt="What is (1234 * 56) + 789? Use the calculator tool.",
            expected="69893",
            mock_plan=MockPlan(
                tool_calls=[_tc("calculator", expression="(1234 * 56) + 789")],
                final="The result is 69893.",
            ),
        ),
        Task(
            id="calc-chain",
            prompt="Compute 2 to the power of 10, then multiply that result by 3.",
            expected="3072",
            mock_plan=MockPlan(
                tool_calls=[
                    _tc("calculator", expression="2 ** 10"),
                    _tc("calculator", expression="1024 * 3"),
                ],
                final="2^10 is 1024, and multiplied by 3 that is 3072.",
            ),
        ),
        Task(
            id="lookup-calc",
            prompt="What is the population of Paris multiplied by 2?",
            expected="4280000",
            mock_plan=MockPlan(
                tool_calls=[
                    _tc("kv_lookup", key="population_of_paris"),
                    _tc("calculator", expression="2140000 * 2"),
                ],
                final="The population of Paris is 2,140,000, so doubled it is 4280000.",
            ),
        ),
        Task(
            id="lookup-simple",
            prompt="What is the capital of Japan?",
            expected="tokyo",
            mock_plan=MockPlan(
                tool_calls=[_tc("kv_lookup", key="capital_of_japan")],
                final="The capital of Japan is Tokyo.",
            ),
        ),
        Task(
            id="lookup-lookup-calc",
            prompt="Add the population of Paris and the population of Tokyo.",
            expected="16100000",
            mock_plan=MockPlan(
                tool_calls=[
                    _tc("kv_lookup", key="population_of_paris"),
                    _tc("kv_lookup", key="population_of_tokyo"),
                    _tc("calculator", expression="2140000 + 13960000"),
                ],
                final="Paris (2,140,000) plus Tokyo (13,960,000) is 16100000.",
            ),
        ),
        Task(
            id="calc-percent",
            prompt="What is 15 percent of 200?",
            expected="30",
            mock_plan=MockPlan(
                tool_calls=[_tc("calculator", expression="200 * 15 / 100")],
                final="15% of 200 is 30.",
            ),
        ),
    ]


SUITES: dict[str, callable] = {
    "default": default_suite,
}


def load_suite(name: str) -> list[Task]:
    if name not in SUITES:
        raise ValueError(f"unknown task suite '{name}'; available: {sorted(SUITES)}")
    return SUITES[name]()
