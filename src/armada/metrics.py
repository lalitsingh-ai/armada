"""Metrics and the cost model.

A run produces a list of `TaskResult`s. `RunMetrics` aggregates them into the numbers that
actually matter for an agentic workload on Arm: cost-per-task, tasks-per-dollar, sustained
concurrent agents within an SLA, latency percentiles, throughput, and how much work the
prompt-prefix cache saved.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class StepMetrics:
    """One LLM call inside an agent's loop."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from the KV cache (not recomputed)
    latency_s: float = 0.0
    ttft_s: float | None = None  # time to first token, when the server reports it


@dataclass
class TaskResult:
    """The outcome of a single agent solving a single task."""

    task_id: str
    success: bool = False
    steps: list[StepMetrics] = field(default_factory=list)
    wall_time_s: float = 0.0
    tool_calls: int = 0
    error: str | None = None

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def cached_tokens(self) -> int:
        return sum(s.cached_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached_tokens=self.cached_tokens,
            n_steps=self.n_steps,
        )
        return d


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in [0, 100]). Returns None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil(pct / 100.0 * len(ordered))
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


@dataclass
class RunMetrics:
    config_name: str
    platform: str
    arm_features: list[str]
    concurrency: int
    usd_per_hour: float
    instance_label: str
    latency_sla_s: float
    total_wall_s: float
    tasks: list[TaskResult] = field(default_factory=list)

    # --- counts -----------------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def successful(self) -> list[TaskResult]:
        return [t for t in self.tasks if t.success]

    @property
    def n_success(self) -> int:
        return len(self.successful)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_tasks if self.n_tasks else 0.0

    # --- latency ----------------------------------------------------------------
    @property
    def task_latencies(self) -> list[float]:
        return [t.wall_time_s for t in self.successful]

    @property
    def p50_task_latency_s(self) -> float | None:
        return percentile(self.task_latencies, 50)

    @property
    def p95_task_latency_s(self) -> float | None:
        return percentile(self.task_latencies, 95)

    @property
    def within_sla_rate(self) -> float:
        if not self.successful:
            return 0.0
        ok = sum(1 for t in self.successful if t.wall_time_s <= self.latency_sla_s)
        return ok / len(self.successful)

    @property
    def sustained_agents(self) -> float:
        """Estimated concurrent agents served within the latency SLA."""
        return round(self.concurrency * self.within_sla_rate, 1)

    # --- tokens / throughput ----------------------------------------------------
    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.tasks)

    @property
    def total_completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.tasks)

    @property
    def total_cached_tokens(self) -> int:
        return sum(t.cached_tokens for t in self.tasks)

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of prompt tokens that were served from the KV prefix cache."""
        return self.total_cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens else 0.0

    @property
    def gen_tokens_per_sec(self) -> float:
        return self.total_completion_tokens / self.total_wall_s if self.total_wall_s else 0.0

    @property
    def tasks_per_sec(self) -> float:
        return self.n_success / self.total_wall_s if self.total_wall_s else 0.0

    # --- cost -------------------------------------------------------------------
    @property
    def _billable_hours(self) -> float:
        return self.total_wall_s / 3600.0

    @property
    def cost_per_task_usd(self) -> float | None:
        if self.usd_per_hour <= 0 or self.n_success == 0:
            return None
        return self._billable_hours * self.usd_per_hour / self.n_success

    @property
    def tasks_per_dollar(self) -> float | None:
        if self.usd_per_hour <= 0:
            return None
        spend = self._billable_hours * self.usd_per_hour
        return self.n_success / spend if spend > 0 else None

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "platform": self.platform,
            "arm_features": self.arm_features,
            "concurrency": self.concurrency,
            "instance_label": self.instance_label,
            "usd_per_hour": self.usd_per_hour,
            "latency_sla_s": self.latency_sla_s,
            "total_wall_s": self.total_wall_s,
            "n_tasks": self.n_tasks,
            "n_success": self.n_success,
            "success_rate": self.success_rate,
            "p50_task_latency_s": self.p50_task_latency_s,
            "p95_task_latency_s": self.p95_task_latency_s,
            "sustained_agents": self.sustained_agents,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
            "gen_tokens_per_sec": self.gen_tokens_per_sec,
            "tasks_per_sec": self.tasks_per_sec,
            "cost_per_task_usd": self.cost_per_task_usd,
            "tasks_per_dollar": self.tasks_per_dollar,
            "tasks": [t.to_dict() for t in self.tasks],
        }
