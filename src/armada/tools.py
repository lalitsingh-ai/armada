"""Hermetic agent tools.

These tools are intentionally offline and deterministic so the benchmark is reproducible and
network-independent. The calculator uses a restricted AST evaluator (never ``eval``) to avoid
code-injection — tool arguments come from an LLM and must be treated as untrusted input.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

# --- safe calculator --------------------------------------------------------------

_ALLOWED_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_POW_EXPONENT = 64  # guard against huge exponentiation (CPU/memory DoS)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow):
            exponent = _eval_node(node.right)
            if abs(exponent) > _MAX_POW_EXPONENT:
                raise ValueError("exponent too large")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+ - * / // % ** and parentheses)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
    except ZeroDivisionError:
        return "error: division by zero"
    except (ValueError, SyntaxError, TypeError) as exc:
        return f"error: {exc}"
    # Present whole numbers without a trailing .0
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --- hermetic knowledge lookup ----------------------------------------------------

FACTS: dict[str, str] = {
    "capital_of_france": "Paris",
    "capital_of_japan": "Tokyo",
    "population_of_paris": "2140000",
    "population_of_tokyo": "13960000",
    "speed_of_light_m_s": "299792458",
    "days_in_leap_year": "366",
}


def kv_lookup(key: str) -> str:
    """Look up a fact by key from a fixed, offline knowledge base."""
    return FACTS.get(key.strip().lower(), "error: unknown key")


# --- registry + OpenAI-compatible schema -----------------------------------------

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "kv_lookup": kv_lookup,
}

TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '(1234 * 56) + 789'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kv_lookup",
            "description": "Look up a fact by key from an offline knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Fact key such as 'capital_of_france' or 'population_of_paris'."
                        ),
                    }
                },
                "required": ["key"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call to its implementation. Unknown tools return an error string."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"error: unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as exc:
        return f"error: bad arguments for {name}: {exc}"
