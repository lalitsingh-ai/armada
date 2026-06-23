"""Report generation: JSON for machines, Markdown + a console table for humans."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .metrics import RunMetrics


def _fmt_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.6f}"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def summary_rows(run: RunMetrics) -> list[tuple[str, str]]:
    return [
        ("Platform", run.platform),
        ("Arm accel features", ", ".join(run.arm_features) or "none"),
        ("Instance", f"{run.instance_label} (${run.usd_per_hour:.4f}/hr)"),
        ("Concurrency (fleet size)", str(run.concurrency)),
        ("Tasks", f"{run.n_success}/{run.n_tasks} ok ({_pct(run.success_rate)})"),
        ("Total wall time", f"{run.total_wall_s:.2f}s"),
        ("Throughput", f"{run.tasks_per_sec:.2f} tasks/s, {run.gen_tokens_per_sec:.1f} gen tok/s"),
        ("Task latency p50 / p95", f"{_fmt_num(run.p50_task_latency_s)}s / {_fmt_num(run.p95_task_latency_s)}s"),
        ("Sustained agents (within SLA)", f"{run.sustained_agents} (SLA {run.latency_sla_s:.0f}s)"),
        ("Prefix cache hit ratio", _pct(run.cache_hit_ratio)),
        ("Cost per task", _fmt_usd(run.cost_per_task_usd)),
        ("Tasks per dollar", _fmt_num(run.tasks_per_dollar, 0)),
    ]


def render_console(run: RunMetrics, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"Armada — {run.config_name}", show_header=False, title_style="bold")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for label, value in summary_rows(run):
        table.add_row(label, value)
    console.print(table)


def render_markdown(run: RunMetrics) -> str:
    lines = [
        f"# Armada benchmark — `{run.config_name}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for label, value in summary_rows(run):
        lines.append(f"| {label} | {value} |")

    lines += [
        "",
        "## Per-task results",
        "",
        "| Task | OK | Steps | Tool calls | Prompt tok | Gen tok | Cached tok | Wall (s) |",
        "| --- | :-: | -: | -: | -: | -: | -: | -: |",
    ]
    for t in run.tasks:
        ok = "✅" if t.success else "❌"
        lines.append(
            f"| {t.task_id} | {ok} | {t.n_steps} | {t.tool_calls} | "
            f"{t.prompt_tokens} | {t.completion_tokens} | {t.cached_tokens} | {t.wall_time_s:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_results(run: RunMetrics, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(run), encoding="utf-8")
    return out


def load_results(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    return json.loads(p.read_text(encoding="utf-8"))


def render_dict_console(data: dict, console: Console | None = None) -> None:
    """Render a previously saved results.json (a plain dict) as a console table."""
    console = console or Console()
    table = Table(title=f"Armada — {data.get('config_name', '?')}", show_header=False, title_style="bold")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    rows = [
        ("Platform", str(data.get("platform"))),
        ("Arm accel features", ", ".join(data.get("arm_features") or []) or "none"),
        ("Instance", f"{data.get('instance_label')} (${data.get('usd_per_hour', 0):.4f}/hr)"),
        ("Concurrency (fleet size)", str(data.get("concurrency"))),
        ("Tasks", f"{data.get('n_success')}/{data.get('n_tasks')} ok"),
        ("Total wall time", f"{data.get('total_wall_s', 0):.2f}s"),
        ("Throughput", f"{data.get('tasks_per_sec', 0):.2f} tasks/s, {data.get('gen_tokens_per_sec', 0):.1f} gen tok/s"),
        ("Task latency p50 / p95", f"{_fmt_num(data.get('p50_task_latency_s'))}s / {_fmt_num(data.get('p95_task_latency_s'))}s"),
        ("Sustained agents", f"{data.get('sustained_agents')} (SLA {data.get('latency_sla_s', 0):.0f}s)"),
        ("Prefix cache hit ratio", _pct(data.get("cache_hit_ratio", 0.0))),
        ("Cost per task", _fmt_usd(data.get("cost_per_task_usd"))),
        ("Tasks per dollar", _fmt_num(data.get("tasks_per_dollar"), 0)),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)
