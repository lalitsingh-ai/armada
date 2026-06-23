"""llama.cpp server lifecycle.

Builds an Arm-tuned ``llama-server`` command line, launches it, waits for the ``/health`` endpoint,
and shuts it down cleanly. Real-mode only — the mock path never touches this module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import httpx

from .config import ServerCfg


def find_binary() -> str:
    """Locate llama-server: $LLAMA_SERVER_BIN, a local build, or PATH."""
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env and os.path.exists(env):
        return env
    local = os.path.join("llama.cpp", "build", "bin", "llama-server")
    if os.path.exists(local):
        return local
    found = shutil.which("llama-server")
    if found:
        return found
    raise FileNotFoundError(
        "llama-server not found. Build it with scripts/build_llama_cpp.sh or set $LLAMA_SERVER_BIN."
    )


def build_args(cfg: ServerCfg, binary: str) -> list[str]:
    args = [
        binary,
        "-m", cfg.model_path,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "-c", str(cfg.ctx_size),
        "-np", str(cfg.parallel),
    ]
    if cfg.threads:
        args += ["-t", str(cfg.threads)]
    if cfg.continuous_batching:
        args += ["-cb"]
    args += cfg.extra_args
    return args


def wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    url = f"{base_url.rstrip('/')}/health"
    last_err: Exception | None = None
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError as exc:  # not up yet
                last_err = exc
            time.sleep(1.0)
    raise TimeoutError(f"llama-server did not become healthy at {url} within {timeout_s}s: {last_err}")


class LlamaCppServer:
    """Context manager that runs llama-server for the duration of a benchmark."""

    def __init__(self, cfg: ServerCfg, binary: str | None = None):
        self.cfg = cfg
        self.binary = binary or find_binary()
        self._proc: subprocess.Popen | None = None

    @property
    def command(self) -> list[str]:
        return build_args(self.cfg, self.binary)

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        wait_for_health(self.cfg.base_url, self.cfg.startup_timeout_s)

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "LlamaCppServer":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
