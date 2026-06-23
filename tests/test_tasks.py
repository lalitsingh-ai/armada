import asyncio

from armada.agent import run_task
from armada.client import MockClient
from armada.tasks import default_suite


def _solve(task):
    client = MockClient(prompt_cache=True, sim_speed=0.0)
    return asyncio.run(run_task(task, client))


def test_each_task_succeeds():
    for task in default_suite():
        result = _solve(task)
        assert result.success, f"task {task.id} did not produce the expected answer"
        assert result.tool_calls == len(task.mock_plan.tool_calls)
        # First LLM call is never cached; later calls reuse the prefix.
        if task.mock_plan.tool_calls:
            assert result.cached_tokens > 0
