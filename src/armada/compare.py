"""Multi-run comparison.

Two things judges want to see at a glance:

* **cache on/off** (and concurrency) on the *same* machine — does the optimization actually move
  the numbers? Produced by ``armada sweep``.
* **Arm vs x86** on the *same* workload — is Arm cheaper/faster per task? Produced by
  ``armada compare`` over two saved result sets.

Both reduce to the same thing: a list of ``(label, result_dict)`` rendered as one table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

# (label, result-dict) where result-dict is RunMetrics.to_dict() or a loaded results.json
Item = tuple[str, dict]


@dataclass
class Variant:
    """One leg of a sweep: a label plus the config overrides that define it."""

    label: str
    overrides: list[str] = field(default_factory=list)


def make_variants(kind: str) -> list[Variant]:
    """Named sweep recipes. ``cache`` proves the prefix-cache win; ``concurrency`` shows scaling."""
    if kind == "cache":
        return [
            Variant("cache-on", ["server.prompt_cache=true"]),
            Variant("cache-off", ["server.prompt_cache=false"]),
        ]
    if kind == "concurrency":
        return [
            Variant(f"agents={n}", [f"run.concurrency={n}", f"server.parallel={n}"])
            for n in (1, 2, 4, 8)
        ]
    raise ValueError(f"unknown sweep kind: {kind!r} (expected 'cache' or 'concurrency')")


def _fmt_usd(value: object) -> str:
    return "n/a" if value is None else f"${float(value):,.6f}"


def _fmt_num(value: object, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):,.{digits}f}"


def _pct(value: object) -> str:
    return f"{(float(value) if value is not None else 0.0) * 100:.1f}%"


# Column header -> formatter over a result dict. Order defines the table layout.
_COLUMNS: list[tuple[str, object]] = [
    ("tasks/s", lambda d: _fmt_num(d.get("tasks_per_sec"))),
    ("gen tok/s", lambda d: _fmt_num(d.get("gen_tokens_per_sec"), 1)),
    ("cache hit", lambda d: _pct(d.get("cache_hit_ratio"))),
    ("p95 (s)", lambda d: _fmt_num(d.get("p95_task_latency_s"))),
    ("wall (s)", lambda d: _fmt_num(d.get("total_wall_s"))),
    ("sustained", lambda d: _fmt_num(d.get("sustained_agents"), 1)),
    ("cost/task", lambda d: _fmt_usd(d.get("cost_per_task_usd"))),
    ("tasks/$", lambda d: _fmt_num(d.get("tasks_per_dollar"), 0)),
]


def _headline(items: list[Item]) -> str | None:
    """For a two-way comparison, state the throughput ratio in one line."""
    if len(items) != 2:
        return None
    (label_a, a), (label_b, b) = items
    ta = a.get("tasks_per_sec") or 0.0
    tb = b.get("tasks_per_sec") or 0.0
    if ta <= 0 or tb <= 0:
        return None
    if ta >= tb:
        faster, slower, ratio = label_a, label_b, ta / tb
    else:
        faster, slower, ratio = label_b, label_a, tb / ta
    return f"**Headline:** `{faster}` delivers **{ratio:.2f}x the throughput** of `{slower}`."


def _xml_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _mermaid_bar(title: str, labels: list[str], values: list[float], y_label: str) -> str:
    xs = ", ".join(f'"{label.replace(chr(34), chr(39))}"' for label in labels)
    ys = ", ".join(f"{v:.2f}" for v in values)
    return (
        "```mermaid\n"
        "xychart-beta\n"
        f'    title "{title.replace(chr(34), chr(39))}"\n'
        f"    x-axis [{xs}]\n"
        f'    y-axis "{y_label}"\n'
        f"    bar [{ys}]\n"
        "```"
    )


def render_compare_mermaid(items: list[Item]) -> str:
    """Mermaid bar charts (render natively on GitHub + in Actions job summaries)."""
    labels = [label for label, _ in items]
    charts = [
        _mermaid_bar(
            "Throughput (tasks/s) — higher is better",
            labels,
            [float(d.get("tasks_per_sec") or 0.0) for _, d in items],
            "tasks/s",
        )
    ]
    tpd = [d.get("tasks_per_dollar") for _, d in items]
    if any(v is not None for v in tpd):
        charts.append(
            _mermaid_bar(
                "Tasks per dollar — higher is better",
                labels,
                [float(v or 0.0) for v in tpd],
                "tasks/$",
            )
        )
    return "\n\n".join(charts)


def render_compare_svg(
    items: list[Item],
    metric: str = "tasks_per_sec",
    title: str = "Throughput (tasks/s) — higher is better",
    unit: str = "tasks/s",
) -> str:
    """A dependency-free horizontal bar chart (for slides / Devpost). Best bar is highlighted."""
    labels = [str(label) for label, _ in items]
    values = [float(d.get(metric) or 0.0) for _, d in items]
    n = len(values)
    vmax = max(values) if values and max(values) > 0 else 1.0

    pad_l, pad_r, pad_t, pad_b = 180, 80, 52, 18
    bar_h, gap, plot_w = 30, 16, 440
    width = pad_l + plot_w + pad_r
    height = pad_t + n * bar_h + max(n - 1, 0) * gap + pad_b
    best = max(range(n), key=lambda i: values[i]) if n else -1

    rows: list[str] = []
    y = pad_t
    for i, (label, val) in enumerate(zip(labels, values)):
        w = max(1.0, val / vmax * plot_w)
        color = "#2da44e" if i == best else "#8b949e"
        mid = y + bar_h / 2 + 4
        rows.append(
            f'<text x="{pad_l - 10}" y="{mid:.0f}" text-anchor="end" font-size="13" '
            f'fill="#c9d1d9">{_xml_escape(label)}</text>'
        )
        rows.append(
            f'<rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>'
        )
        rows.append(
            f'<text x="{pad_l + w + 8:.1f}" y="{mid:.0f}" font-size="13" '
            f'fill="#c9d1d9">{val:,.2f}</text>'
        )
        y += bar_h + gap

    body = "\n  ".join(rows)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="system-ui,Segoe UI,Helvetica,Arial,sans-serif">\n'
        f'  <rect width="{width}" height="{height}" fill="#0d1117"/>\n'
        f'  <text x="{pad_l}" y="30" font-size="16" font-weight="700" '
        f'fill="#e6edf3">{_xml_escape(title)}</text>\n'
        f'  <text x="{width - pad_r}" y="30" text-anchor="end" font-size="12" '
        f'fill="#8b949e">{_xml_escape(unit)}</text>\n'
        f"  {body}\n"
        f"</svg>\n"
    )


def render_compare_console(items: list[Item], console: Console | None = None,
                           title: str = "Comparison") -> None:
    console = console or Console()
    table = Table(title=title, title_style="bold")
    table.add_column("Variant", style="cyan", no_wrap=True)
    for header, _ in _COLUMNS:
        table.add_column(header, justify="right", style="white")
    for label, data in items:
        table.add_row(label, *[fmt(data) for _, fmt in _COLUMNS])
    console.print(table)


def render_compare_markdown(items: list[Item], title: str = "Comparison") -> str:
    header = " | ".join(h for h, _ in _COLUMNS)
    divider = " | ".join("-:" for _ in _COLUMNS)
    lines = [
        f"# Armada comparison — {title}",
        "",
        f"| Variant | {header} |",
        f"| --- | {divider} |",
    ]
    for label, data in items:
        cells = " | ".join(fmt(data) for _, fmt in _COLUMNS)
        lines.append(f"| {label} | {cells} |")
    note = _headline(items)
    if note:
        lines += ["", note]
    lines.append("")
    return "\n".join(lines)


def write_compare(items: list[Item], out_dir: str | Path, title: str = "Comparison") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = [
        {"label": label, **{k: v for k, v in data.items() if k != "tasks"}}
        for label, data in items
    ]
    payload = {"title": title, "variants": summary}
    (out / "compare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = (
        render_compare_markdown(items, title)
        + "\n## Charts\n\n"
        + render_compare_mermaid(items)
        + "\n"
    )
    (out / "compare.md").write_text(markdown, encoding="utf-8")
    (out / "compare.svg").write_text(render_compare_svg(items), encoding="utf-8")
    return out
