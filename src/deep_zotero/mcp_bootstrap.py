from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from .config import Config
from .embedder import create_embedder


PARENT_MONITOR_ENV = "DEEP_ZOTERO_PARENT_MONITOR"
SHOW_BANNER_ENV = "FASTMCP_SHOW_SERVER_BANNER"

logger = logging.getLogger("deep_zotero.mcp_bootstrap")
_bootstrapped = False
_embedding_probe: dict[str, str | bool | None] = {"ready": None, "error": None}


def configure_logging(config: Config | None = None) -> None:
    level = logging.WARNING if any(name.startswith("CODEX_") for name in os.environ) else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if config is not None:
        config.ensure_runtime_dirs()
        handlers.append(logging.FileHandler(config.runtime_log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def default_parent_monitor_enabled() -> bool:
    return not any(name.startswith("CODEX_") for name in os.environ)


def register_signal_handlers() -> None:
    def handle_signal(signum, _frame) -> None:
        logger.info("Received signal %s, shutting down deep-zotero", signum)
        raise SystemExit(0)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, handle_signal)


def start_parent_monitor(enabled: bool = True) -> None:
    if not enabled:
        logger.info("Parent monitor disabled for non-stdio transport")
        return
    if not env_flag(PARENT_MONITOR_ENV, default_parent_monitor_enabled()):
        logger.info("Parent monitor disabled via %s", PARENT_MONITOR_ENV)
        return

    target_pid = os.getppid()

    def monitor() -> None:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            synchronize = 0x00100000
            handle = kernel32.OpenProcess(synchronize, False, target_pid)
            if handle:
                infinite = 0xFFFFFFFF
                kernel32.WaitForSingleObject(handle, infinite)
                kernel32.CloseHandle(handle)
        else:
            while True:
                time.sleep(1.0)
                try:
                    os.kill(target_pid, 0)
                except (OSError, PermissionError):
                    break
        os._exit(0)

    threading.Thread(target=monitor, daemon=True).start()


def prepare_server_environment(config: Config) -> None:
    config.ensure_runtime_dirs()
    os.environ[SHOW_BANNER_ENV] = "false"
    os.environ.setdefault("FASTMCP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TMP", str(config.runtime_tmp_dir))
    os.environ.setdefault("TEMP", str(config.runtime_tmp_dir))
    os.environ.setdefault("TMPDIR", str(config.runtime_tmp_dir))
    os.environ.setdefault("HF_HOME", str(config.model_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(config.model_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(config.runtime_data_dir / "cache"))
    os.environ.setdefault("PADDLE_HOME", str(config.runtime_data_dir / "paddle"))


def probe_embedding_readiness(config: Config, run_warmup: bool = True) -> tuple[bool, str | None]:
    try:
        embedder = create_embedder(config)
        if run_warmup:
            embedder.embed_query("deep-zotero startup readiness check")
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def collect_health_status(config: Config | None = None) -> dict:
    config = config or Config.load()
    config.ensure_runtime_dirs()

    if _embedding_probe["ready"] is None:
        ready, error = probe_embedding_readiness(config, run_warmup=False)
    else:
        ready = bool(_embedding_probe["ready"])
        error = str(_embedding_probe["error"]) if _embedding_probe["error"] else None

    chroma_exists = config.chroma_db_path.exists()
    chroma_entries = None
    if chroma_exists:
        try:
            chroma_entries = len(list(config.chroma_db_path.iterdir()))
        except OSError:
            chroma_entries = None

    return {
        "project_root": str(config.project_root),
        "config_path": str(config.config_path) if config.config_path else None,
        "paths": config.path_report(),
        "checks": {
            "zotero_data_dir_exists": config.zotero_data_dir.exists(),
            "zotero_sqlite_exists": (config.zotero_data_dir / "zotero.sqlite").exists(),
            "chroma_dir_exists": chroma_exists,
            "chroma_dir_entries": chroma_entries,
            "embedding_model_ready": ready,
        },
        "embedding": {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "dimensions": config.embedding_dimensions,
            "batch_size": config.embedding_batch_size,
            "device": config.embedding_device,
            "allow_download": config.embedding_allow_download,
            "model_cache_dir": str(config.model_cache_dir),
            "error": error,
        },
        "validation_errors": config.validate(),
    }


def bootstrap_mcp_server(transport: str = "stdio", config_path: Path | str | None = None) -> Config:
    global _bootstrapped
    if _bootstrapped:
        return Config.load(config_path)

    config = Config.load(config_path)
    configure_logging(config)
    register_signal_handlers()
    start_parent_monitor(enabled=transport == "stdio")

    prepare_server_environment(config)

    errors = config.validate()
    if errors:
        raise RuntimeError("deep-zotero configuration is invalid: " + "; ".join(errors))

    ready, error = probe_embedding_readiness(config, run_warmup=True)
    _embedding_probe["ready"] = ready
    _embedding_probe["error"] = error
    if not ready:
        raise RuntimeError(
            "Failed to initialize in-process embedding model. "
            f"provider={config.embedding_provider}, model={config.embedding_model}, "
            f"device={config.embedding_device or 'auto'}, cache_dir={config.model_cache_dir}. "
            f"error={error}. "
            "Check sentence-transformers/torch installation, model cache integrity, and device settings."
        )

    _bootstrapped = True
    return config
