import asyncio

from armada.client import MockClient
from armada.config import Config
from armada.cpuinfo import detect
from armada.runner import run_benchmark


def _bench(prompt_cache: bool = True):
    cfg = Config()
    cfg.run.concurrency = 4
    cfg.server.prompt_cache = prompt_cache
    client = MockClient(prompt_cache=prompt_cache, sim_speed=0.0)

    async def go():
        try:
            return await run_benchmark(cfg, client, detect())
        finally:
            await client.aclose()

    return asyncio.run(go())


def test_benchmark_all_pass():
    run = _bench()
    assert run.n_tasks == 6
    assert run.success_rate == 1.0
    assert run.gen_tokens_per_sec >= 0.0


def test_prefix_cache_on_beats_off():
    on = _bench(prompt_cache=True)
    off = _bench(prompt_cache=False)
    assert on.cache_hit_ratio > 0.0
    assert off.cache_hit_ratio == 0.0
    assert on.total_cached_tokens > off.total_cached_tokens
