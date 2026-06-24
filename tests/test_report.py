import asyncio

from armada.client import MockClient
from armada.config import Config
from armada.cpuinfo import detect
from armada.report import load_results, render_html, write_results
from armada.runner import run_benchmark


def _run():
    cfg = Config()
    cfg.run.concurrency = 4
    cfg.cost.usd_per_hour = 0.05
    cfg.cost.instance_label = "test box"
    client = MockClient(prompt_cache=cfg.server.prompt_cache, sim_speed=0.0)

    async def go():
        try:
            return await run_benchmark(cfg, client, detect())
        finally:
            await client.aclose()

    return asyncio.run(go())


def test_render_html_is_self_contained():
    data = _run().to_dict()
    html = render_html(data)

    # a complete HTML document
    assert html.startswith("<!doctype html>") and html.strip().endswith("</html>")
    # headline economics surfaced as stat cards
    assert "Cost / task" in html and "Tasks / $" in html
    # every task appears in the per-task table
    for t in data["tasks"]:
        assert t["task_id"] in html
    # genuinely self-contained: no external scripts, styles, or fetched assets
    assert "<script" not in html
    assert "<link" not in html
    assert "src=" not in html
    assert "https://" not in html


def test_render_html_escapes_untrusted_text():
    data = _run().to_dict()
    data["instance_label"] = "<script>alert(1)</script>"
    html = render_html(data)
    # the raw tag must not survive into the document
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_results_emits_html(tmp_path):
    run = _run()
    out = write_results(run, tmp_path / "r")

    assert (out / "results.json").is_file()
    assert (out / "report.md").is_file()
    assert (out / "report.html").is_file()

    # the saved json round-trips through the HTML renderer
    data = load_results(out)
    assert render_html(data).startswith("<!doctype html>")
