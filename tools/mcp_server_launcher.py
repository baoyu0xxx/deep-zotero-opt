from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_CANDIDATES = (ROOT.parent,)


def _resolve_deep_zotero_repo() -> Path:
    explicit = os.environ.get("DEEP_ZOTERO_REPO")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return candidate

    for candidate in DEFAULT_REPO_CANDIDATES:
        python_exe = candidate / ".venv" / "Scripts" / "python.exe"
        src_dir = candidate / "src" / "deep_zotero"
        if python_exe.exists() and src_dir.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the deep-zotero-opt repository. Set DEEP_ZOTERO_REPO "
        "to the repository root."
    )


DEEP_ZOTERO_REPO = _resolve_deep_zotero_repo()
DEEP_ZOTERO_SRC = DEEP_ZOTERO_REPO / "src"
TARGET_PYTHON = DEEP_ZOTERO_REPO / ".venv" / "Scripts" / "python.exe"
DEFAULT_CONFIG_PATH = Path("~/.config/deep-zotero/config.json").expanduser()
DEFAULT_QWEN_EMBED_SERVER = ROOT / "qwen_embedding_server.py"
# Keep the Hugging Face cache outside the repo so startup does not depend on
# workspace ACLs or checked-in runtime directories.
DEFAULT_QWEN_CACHE_DIR = Path.home() / ".cache" / "deep-zotero" / "hf-cache"
QWEN_LOG_PATH = DEEP_ZOTERO_REPO / ".runtime-data" / "mcp" / "qwen-embedding-server.log"

logger = logging.getLogger("mcp_server_launcher")
_server_process: subprocess.Popen | None = None
_embedding_process: subprocess.Popen | None = None
_embedding_log_handle = None
_shutting_down = False
PARENT_MONITOR_ENV = "DEEP_ZOTERO_PARENT_MONITOR"
SHOW_BANNER_ENV = "FASTMCP_SHOW_SERVER_BANNER"


def _configure_logging() -> None:
    level = logging.WARNING if any(name.startswith("CODEX_") for name in os.environ) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _default_parent_monitor_enabled() -> bool:
    # Codex-managed MCP launches can have a shallow or synthetic process tree on
    # Windows; avoid guessing ancestors in that environment unless explicitly opted in.
    return not any(name.startswith("CODEX_") for name in os.environ)


def _probe_embedding_endpoint(base_url: str | None, timeout: float = 5.0) -> bool:
    if not base_url:
        return False
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 0) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _build_default_start_command(config: dict) -> list[str] | None:
    if not DEFAULT_QWEN_EMBED_SERVER.exists():
        return None
    if not TARGET_PYTHON.exists():
        return None

    base_url = config.get("embedding_base_url") or "http://127.0.0.1:8000/v1"
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)

    DEFAULT_QWEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return [
        str(TARGET_PYTHON),
        str(DEFAULT_QWEN_EMBED_SERVER),
        "--model",
        config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        "--host",
        host,
        "--port",
        str(port),
        "--cache-dir",
        str(DEFAULT_QWEN_CACHE_DIR),
    ]


def _cleanup_children() -> None:
    global _shutting_down, _embedding_log_handle
    if _shutting_down:
        return
    _shutting_down = True

    for process in (_server_process, _embedding_process):
        if process is None:
            continue
        if process.poll() is not None:
            continue
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    if _embedding_log_handle is not None:
        try:
            _embedding_log_handle.close()
        except Exception:
            pass
        _embedding_log_handle = None


def _register_signal_handlers() -> None:
    def _handle_signal(signum, _frame) -> None:
        logger.info("Received signal %s, shutting down child processes", signum)
        _cleanup_children()
        raise SystemExit(0)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handle_signal)


def _start_parent_monitor() -> None:
    if not _env_flag(PARENT_MONITOR_ENV, _default_parent_monitor_enabled()):
        logger.info("Parent monitor disabled via %s", PARENT_MONITOR_ENV)
        return

    target_pid = os.getppid()

    def monitor() -> None:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, target_pid)
            if handle:
                INFINITE = 0xFFFFFFFF
                kernel32.WaitForSingleObject(handle, INFINITE)
                kernel32.CloseHandle(handle)
        else:
            while True:
                time.sleep(1.0)
                try:
                    os.kill(target_pid, 0)
                except (OSError, PermissionError):
                    break

        _cleanup_children()
        os._exit(0)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()


def _ensure_embedding_service(config: dict) -> None:
    global _embedding_process, _embedding_log_handle

    if config.get("embedding_provider") != "openai_compatible":
        return

    base_url = config.get("embedding_base_url") or "http://127.0.0.1:8000/v1"
    if _probe_embedding_endpoint(base_url):
        logger.info("Embedding endpoint already reachable: %s", base_url)
        return

    if not config.get("embedding_auto_start", False):
        logger.warning("Embedding endpoint is down and embedding_auto_start is false: %s", base_url)
        return

    start_command = config.get("embedding_start_command") or _build_default_start_command(config)
    if not start_command:
        raise RuntimeError(
            "embedding_auto_start is true but no start command is available. "
            "Set embedding_start_command in ~/.config/deep-zotero/config.json "
            "or keep qwen_embedding_server.py in this project."
        )

    QWEN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _embedding_log_handle = QWEN_LOG_PATH.open("a", encoding="utf-8")
    logger.info("Starting embedding service: %s", start_command)
    _embedding_process = subprocess.Popen(
        start_command,
        cwd=str(ROOT),
        shell=isinstance(start_command, str),
        stdout=_embedding_log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    deadline = time.monotonic() + float(config.get("embedding_startup_timeout", 180.0))
    while time.monotonic() < deadline:
        if _probe_embedding_endpoint(base_url):
            logger.info("Embedding endpoint is ready: %s", base_url)
            return
        if _embedding_process.poll() is not None:
            raise RuntimeError(
                f"Embedding service exited early with code {_embedding_process.returncode}. "
                f"Check {QWEN_LOG_PATH}"
            )
        time.sleep(2)

    raise RuntimeError(
        f"Timed out waiting for embedding endpoint: {base_url}. "
        f"Check {QWEN_LOG_PATH}"
    )


def _launch_deep_zotero_server() -> int:
    global _server_process

    if not TARGET_PYTHON.exists():
        raise FileNotFoundError(f"Target Python not found: {TARGET_PYTHON}")
    if not DEEP_ZOTERO_SRC.exists():
        raise FileNotFoundError(f"deep-zotero src not found: {DEEP_ZOTERO_SRC}")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{DEEP_ZOTERO_SRC}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(DEEP_ZOTERO_SRC)
    )
    env[SHOW_BANNER_ENV] = "false"
    env[PARENT_MONITOR_ENV] = "0"
    env.setdefault("FASTMCP_LOG_LEVEL", "ERROR")

    _server_process = subprocess.Popen(
        [str(TARGET_PYTHON), "-m", "deep_zotero.server"],
        cwd=str(DEEP_ZOTERO_REPO),
        env=env,
    )
    return _server_process.wait()


def main() -> int:
    _configure_logging()
    atexit.register(_cleanup_children)
    _register_signal_handlers()
    _start_parent_monitor()

    config = _load_config()
    _ensure_embedding_service(config)
    return _launch_deep_zotero_server()


if __name__ == "__main__":
    raise SystemExit(main())
