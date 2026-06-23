"""Command-line interface: ``armada bench | report | info``."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from . import __version__
from .client import LlamaCppClient, MockClient
from .config import Config
from .cpuinfo import detect
from .metrics import RunMetrics
from .report import load_results, render_console, render_dict_console, write_results
from .runner import run_benchmark

console = Console()


async def _run_and_close(cfg: Config, client, cpu) -> RunMetrics:
    try:
        return await run_benchmark(cfg, client, cpu)
    finally:
        await client.aclose()


def _default_out(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return str(Path("results") / f"{name}-{stamp}")


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config, args.set)
    cpu = detect()
    out = args.out or _default_out(cfg.run.name)

    console.print(f"[bold]Armada[/bold] {__version__} — running '{cfg.run.name}' on {cpu.summary()}")

    if args.mock:
        console.print("[yellow]mock mode[/yellow]: deterministic offline model (no server)")
        client = MockClient(prompt_cache=cfg.server.prompt_cache, sim_speed=args.sim_speed)
        run = asyncio.run(_run_and_close(cfg, client, cpu))
    else:
        from .server import LlamaCppServer  # imported lazily so mock mode needs no server deps

        console.print(f"starting llama.cpp server on {cfg.server.base_url} ...")
        with LlamaCppServer(cfg.server):
            client = LlamaCppClient(cfg.server.base_url, cfg.server.startup_timeout_s)
            run = asyncio.run(_run_and_close(cfg, client, cpu))

    out_path = write_results(run, out)
    render_console(run, console)
    console.print(f"\n[green]wrote[/green] {out_path}/results.json and {out_path}/report.md")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    data = load_results(args.path)
    render_dict_console(data, console)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cpu = detect()
    console.print(f"[bold]Armada[/bold] {__version__}")
    console.print(cpu.summary())
    console.print(f"all features: {', '.join(cpu.features) or 'none detected'}")
    try:
        from .server import find_binary

        console.print(f"llama-server: {find_binary()}")
    except FileNotFoundError as exc:
        console.print(f"[yellow]llama-server:[/yellow] {exc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="armada", description="Arm agent-serving efficiency lab")
    parser.add_argument("--version", action="version", version=f"armada {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run the agent fleet benchmark")
    b.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    b.add_argument("--mock", action="store_true", help="use the deterministic offline model")
    b.add_argument("--out", default=None, help="output directory (default: results/<name>-<ts>)")
    b.add_argument("--sim-speed", type=float, default=1.0, help="mock wall-time scale (0 = instant)")
    b.add_argument("--set", action="append", default=[], metavar="key.path=value",
                   help="override a config value (repeatable)")
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("report", help="render a saved results directory or results.json")
    r.add_argument("path", help="results dir or results.json")
    r.set_defaults(func=cmd_report)

    i = sub.add_parser("info", help="show detected CPU/Arm features and server binary")
    i.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
