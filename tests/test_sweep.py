import asyncio

from armada.client import MockClient
from armada.compare import (
    make_variants,
    render_compare_markdown,
    write_compare,
)
from armada.config import Config
from armada.cpuinfo import detect
from armada.runner import run_benchmark


def _run(overrides):
    cfg = Config.load(None, list(overrides) + ["run.concurrency=4", "server.parallel=4"])
    client = MockClient(prompt_cache=cfg.server.prompt_cache, sim_speed=0.0)

    async def go():
        try:
            return await run_benchmark(cfg, client, detect())
        finally:
            await client.aclose()

    return asyncio.run(go())


def _sweep(kind):
    return [(v.label, _run(v.overrides).to_dict()) for v in make_variants(kind)]


def test_make_variants_kinds():
    assert [v.label for v in make_variants("cache")] == ["cache-on", "cache-off"]
    assert len(make_variants("concurrency")) == 4

    import pytest

    with pytest.raises(ValueError):
        make_variants("nope")


def test_cache_sweep_shows_the_win():
    by_label = dict(_sweep("cache"))
    assert by_label["cache-on"]["cache_hit_ratio"] > by_label["cache-off"]["cache_hit_ratio"]
    assert by_label["cache-off"]["cache_hit_ratio"] == 0.0


def test_compare_markdown_has_headline_for_two():
    items = _sweep("cache")
    md = render_compare_markdown(items, title="cache sweep")
    assert "Armada comparison" in md
    assert "Headline:" in md
    assert "x the throughput" in md
    # one header row + divider + one row per variant
    assert md.count("| cache-on |") == 1
    assert md.count("| cache-off |") == 1


def test_concurrency_sweep_no_headline():
    md = render_compare_markdown(_sweep("concurrency"), title="concurrency sweep")
    assert "Headline:" not in md  # only emitted for exactly two variants


def test_write_compare_emits_files(tmp_path):
    items = _sweep("cache")
    out = write_compare(items, tmp_path / "cmp", title="cache sweep")
    assert (out / "compare.json").is_file()
    assert (out / "compare.md").is_file()

    import json

    payload = json.loads((out / "compare.json").read_text())
    assert payload["title"] == "cache sweep"
    assert [v["label"] for v in payload["variants"]] == ["cache-on", "cache-off"]
    # the heavy per-task list is stripped from the comparison summary
    assert all("tasks" not in v for v in payload["variants"])
