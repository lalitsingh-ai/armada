import asyncio

from armada.client import MockClient
from armada.compare import (
    make_variants,
    render_compare_html,
    render_compare_markdown,
    render_compare_mermaid,
    render_compare_svg,
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
    assert (out / "compare.svg").is_file()
    assert (out / "compare.html").is_file()

    import json

    payload = json.loads((out / "compare.json").read_text())
    assert payload["title"] == "cache sweep"
    assert [v["label"] for v in payload["variants"]] == ["cache-on", "cache-off"]
    # the heavy per-task list is stripped from the comparison summary
    assert all("tasks" not in v for v in payload["variants"])
    # charts are embedded in the published markdown
    assert "```mermaid" in (out / "compare.md").read_text()


def test_mermaid_chart_is_wellformed():
    md = render_compare_mermaid(_sweep("cache"))
    assert "xychart-beta" in md
    assert '"cache-on"' in md and '"cache-off"' in md
    assert "tasks/s" in md


def test_svg_chart_is_wellformed():
    svg = render_compare_svg(_sweep("cache"))
    assert svg.startswith("<svg") and svg.strip().endswith("</svg>")
    assert "<rect" in svg
    # the winning bar is highlighted green
    assert "#2da44e" in svg


def test_html_report_is_self_contained():
    html = render_compare_html(_sweep("cache"), title="cache sweep")
    # a complete HTML document
    assert html.startswith("<!doctype html>") and html.strip().endswith("</html>")
    # the table and both variants are present
    assert "<table" in html and "</table>" in html
    assert "cache-on" in html and "cache-off" in html
    # the headline and the charts are inlined (SVG, not <img src>)
    assert "the throughput" in html
    assert "<svg" in html
    # genuinely self-contained: no external scripts, styles, or fetched assets
    assert "<script" not in html
    assert "<link" not in html
    assert "src=" not in html
    assert "https://" not in html
