"""Concurrent fleet runner.

Runs the whole task suite with up to `concurrency` agents in flight at once (a semaphore caps
parallelism), measures total wall-clock time, and assembles a `RunMetrics`.
"""

from __future__ import annotations

import asyncio
import time

from .agent import run_task
from .config import Config
from .cpuinfo import CpuInfo
from .metrics import RunMetrics
from .tasks import load_suite


async def run_benchmark(cfg: Config, client, cpu: CpuInfo) -> RunMetrics:
    suite = load_suite(cfg.tasks.suite)
    all_tasks = suite * max(cfg.run.repeats, 1)

    sem = asyncio.Semaphore(cfg.run.concurrency)

    async def worker(task):
        async with sem:
            return await run_task(
                task,
                client,
                temperature=cfg.model.temperature,
                max_tokens=cfg.model.max_tokens,
            )

    start = time.perf_counter()
    results = await asyncio.gather(*(worker(t) for t in all_tasks))
    total_wall = time.perf_counter() - start

    return RunMetrics(
        config_name=cfg.run.name,
        platform=cpu.summary(),
        arm_features=cpu.accel_features,
        concurrency=cfg.run.concurrency,
        usd_per_hour=cfg.cost.usd_per_hour,
        instance_label=cfg.cost.instance_label,
        latency_sla_s=cfg.run.latency_sla_s,
        total_wall_s=total_wall,
        tasks=list(results),
    )
