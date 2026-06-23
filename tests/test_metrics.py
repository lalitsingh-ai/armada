from armada.metrics import RunMetrics, StepMetrics, TaskResult, percentile


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 100) == 4


def _task(latency: float, success: bool = True, cached: int = 0, prompt: int = 100, gen: int = 20):
    return TaskResult(
        task_id="t",
        success=success,
        steps=[StepMetrics(prompt_tokens=prompt, completion_tokens=gen, cached_tokens=cached)],
        wall_time_s=latency,
    )


def _run(tasks, total_wall, usd_per_hour=0.0, sla=15.0, concurrency=4):
    return RunMetrics(
        config_name="t",
        platform="test",
        arm_features=[],
        concurrency=concurrency,
        usd_per_hour=usd_per_hour,
        instance_label="test",
        latency_sla_s=sla,
        total_wall_s=total_wall,
        tasks=tasks,
    )


def test_cost_model():
    run = _run([_task(1.0), _task(1.0)], total_wall=3600.0, usd_per_hour=1.0)
    assert run.n_success == 2
    assert run.cost_per_task_usd == 0.5
    assert run.tasks_per_dollar == 2.0


def test_cost_disabled_when_no_price():
    run = _run([_task(1.0)], total_wall=3600.0, usd_per_hour=0.0)
    assert run.cost_per_task_usd is None
    assert run.tasks_per_dollar is None


def test_cache_ratio_and_sla():
    run = _run(
        [_task(1.0, cached=50, prompt=100), _task(100.0, cached=0, prompt=100)],
        total_wall=10.0,
        sla=15.0,
    )
    # 50 cached out of 200 prompt tokens total.
    assert abs(run.cache_hit_ratio - 0.25) < 1e-9
    # One of two tasks is within the 15s SLA -> half the fleet sustained.
    assert run.sustained_agents == 2.0
    assert run.success_rate == 1.0
