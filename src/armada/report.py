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


def _esc(value: object) -> str:
    """Escape text for safe inclusion in HTML."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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
    data = run.to_dict()
    (out / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(run), encoding="utf-8")
    (out / "report.html").write_text(render_html(data), encoding="utf-8")
    return out


def load_results(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _dict_summary_rows(data: dict) -> list[tuple[str, str]]:
    """Human-readable (label, value) rows for a saved results.json dict."""
    return [
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


def render_dict_console(data: dict, console: Console | None = None) -> None:
    """Render a previously saved results.json (a plain dict) as a console table."""
    console = console or Console()
    table = Table(title=f"Armada — {data.get('config_name', '?')}", show_header=False, title_style="bold")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for label, value in _dict_summary_rows(data):
        table.add_row(label, value)
    console.print(table)


# Inline dark-theme stylesheet (a plain string so the f-string below stays brace-free).
_REPORT_CSS = """
  :root { color-scheme: dark; }
  body { margin: 0; padding: 2rem; background: #0d1117; color: #c9d1d9;
         font-family: system-ui, "Segoe UI", Helvetica, Arial, sans-serif; }
  .wrap { max-width: 880px; margin: 0 auto; }
  h1 { color: #e6edf3; font-size: 1.4rem; margin: 0 0 .25rem; }
  .sub { color: #8b949e; margin: 0 0 1.25rem; }
  .cards { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0 1.5rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: .7rem .9rem; min-width: 120px; }
  .card .k { color: #8b949e; font-size: .72rem; text-transform: uppercase; letter-spacing: .03em; }
  .card .v { color: #2da44e; font-size: 1.25rem; font-weight: 700; margin-top: .15rem; }
  h2 { color: #e6edf3; font-size: 1.05rem; margin: 1.5rem 0 .5rem; }
  table { width: 100%; border-collapse: collapse; font-size: .9rem; }
  th, td { padding: .45rem .6rem; text-align: right; border-bottom: 1px solid #21262d; }
  thead th { color: #8b949e; font-weight: 600; border-bottom: 2px solid #30363d; }
  tbody th { text-align: left; color: #e6edf3; font-weight: 500; }
  td.ok { color: #2da44e; }
  td.no { color: #f85149; }
  footer { color: #6e7681; font-size: .8rem; margin-top: 1.5rem; }
"""


def render_html(data: dict) -> str:
    """A single self-contained dark-theme HTML page for one run.

    Stat cards for the headline economics, the full summary table, and a per-task breakdown.
    No external CSS/JS/fonts or network calls — safe to open offline or screenshot.
    """
    name = _esc(data.get("config_name", "run"))
    platform = _esc(data.get("platform", "?"))
    features = _esc(", ".join(data.get("arm_features") or []) or "none")

    cards = [
        ("Cost / task", _fmt_usd(data.get("cost_per_task_usd"))),
        ("Tasks / $", _fmt_num(data.get("tasks_per_dollar"), 0)),
        ("Gen tok/s", _fmt_num(data.get("gen_tokens_per_sec"), 1)),
        ("Cache hit", _pct(data.get("cache_hit_ratio", 0.0))),
        ("p95 latency", f"{_fmt_num(data.get('p95_task_latency_s'))}s"),
        ("Sustained", _fmt_num(data.get("sustained_agents"), 1)),
    ]
    cards_html = (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
            for k, v in cards
        )
        + "</div>"
    )

    summary_html = (
        "<h2>Summary</h2><table><tbody>"
        + "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in _dict_summary_rows(data))
        + "</tbody></table>"
    )

    tasks = data.get("tasks") or []
    tasks_html = ""
    if tasks:
        head = (
            "<tr><th>Task</th><th>OK</th><th>Steps</th><th>Tool calls</th>"
            "<th>Prompt tok</th><th>Gen tok</th><th>Cached tok</th><th>Wall (s)</th></tr>"
        )
        body = ""
        for t in tasks:
            ok_cls, ok_txt = ("ok", "yes") if t.get("success") else ("no", "no")
            body += (
                f"<tr><th>{_esc(t.get('task_id'))}</th>"
                f'<td class="{ok_cls}">{ok_txt}</td>'
                f"<td>{int(t.get('n_steps', 0))}</td>"
                f"<td>{int(t.get('tool_calls', 0))}</td>"
                f"<td>{int(t.get('prompt_tokens', 0))}</td>"
                f"<td>{int(t.get('completion_tokens', 0))}</td>"
                f"<td>{int(t.get('cached_tokens', 0))}</td>"
                f"<td>{float(t.get('wall_time_s', 0.0)):.2f}</td></tr>"
            )
        tasks_html = (
            "<h2>Per-task results</h2><table><thead>"
            + head
            + "</thead><tbody>"
            + body
            + "</tbody></table>"
        )

    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Armada — {name}</title>\n<style>{_REPORT_CSS}</style>\n</head>\n<body>\n"
        '<div class="wrap">\n'
        f"  <h1>Armada — {name}</h1>\n"
        f'  <p class="sub">{platform} · Arm features: {features}</p>\n'
        f"  {cards_html}\n  {summary_html}\n  {tasks_html}\n"
        "  <footer>Generated by Armada · self-contained, no external assets.</footer>\n"
        "</div>\n</body>\n</html>\n"
    )
