"""Configuration management and stable project-root path resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import sys


DEFAULT_CONFIG_ENV = "DEEP_ZOTERO_CONFIG"
DEFAULT_CONFIG_BASENAME = "config.json"
DEFAULT_USER_CONFIG_PATH = Path("~/.config/deep-zotero/config.json").expanduser()
PROJECT_ROOT_MARKERS = ("pyproject.toml", "config.example.json", "README.md")
DEFAULT_CHROMA_DB_PATH = "./.runtime-data/chroma_qwen"
DEFAULT_RUNTIME_DATA_DIR = "./.runtime-data"
DEFAULT_RUNTIME_TMP_DIR = "./.runtime-tmp"
DEFAULT_SCRATCH_TMP_DIR = "./.tmp"
DEFAULT_MODEL_CACHE_DIR = "./.runtime-data/hf-cache"
DEFAULT_RUNTIME_LOG_PATH = "./.runtime-data/logs/deep-zotero.log"
DEFAULT_OCR_DEBUG_DIR = "./.runtime-data/ocr-debug"
REMOVED_SIDECAR_KEYS = (
    "embedding_base_url",
    "embedding_api_key",
    "embedding_auto_start",
    "embedding_start_command",
    "embedding_startup_timeout",
    "embedding_log_path",
    "embedding_query_instruction",
)


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    module_path = Path(__file__).resolve()
    candidates.extend(module_path.parents)

    executable = Path(sys.executable).resolve()
    candidates.extend(executable.parents)

    if hasattr(sys, "argv") and sys.argv and sys.argv[0]:
        try:
            argv0 = Path(sys.argv[0]).resolve()
        except OSError:
            argv0 = None
        if argv0 is not None:
            candidates.extend(argv0.parents)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def discover_project_root() -> Path:
    """Find a stable project root without relying on current cwd."""
    for candidate in _candidate_roots():
        if all((candidate / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return candidate

    src_dir = Path(__file__).resolve().parents[1]
    if src_dir.name == "src":
        return src_dir.parent
    return Path(__file__).resolve().parent


def resolve_project_path(value: str | Path, project_root: Path) -> Path:
    """Resolve path-like config values against project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _default_config_candidates(project_root: Path) -> list[Path]:
    return [project_root / DEFAULT_CONFIG_BASENAME, DEFAULT_USER_CONFIG_PATH]


@dataclass
class Config:
    """Application configuration."""

    zotero_data_dir: Path
    chroma_db_path: Path
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    gemini_api_key: str | None
    embedding_provider: str
    embedding_timeout: float
    embedding_max_retries: int
    rerank_alpha: float
    rerank_section_weights: dict[str, float] | None
    rerank_journal_weights: dict[str, float] | None
    rerank_enabled: bool
    oversample_multiplier: int
    oversample_topic_factor: int
    stats_sample_limit: int
    ocr_language: str
    openalex_email: str | None
    vision_enabled: bool
    vision_model: str
    anthropic_api_key: str | None
    embedding_batch_size: int = 8
    embedding_device: str | None = None
    embedding_allow_download: bool = False
    project_root: Path = field(default_factory=discover_project_root)
    config_path: Path | None = None
    runtime_data_dir: Path = field(default_factory=lambda: Path(DEFAULT_RUNTIME_DATA_DIR))
    runtime_tmp_dir: Path = field(default_factory=lambda: Path(DEFAULT_RUNTIME_TMP_DIR))
    scratch_tmp_dir: Path = field(default_factory=lambda: Path(DEFAULT_SCRATCH_TMP_DIR))
    model_cache_dir: Path = field(default_factory=lambda: Path(DEFAULT_MODEL_CACHE_DIR))
    runtime_log_path: Path = field(default_factory=lambda: Path(DEFAULT_RUNTIME_LOG_PATH))
    ocr_debug_dir: Path = field(default_factory=lambda: Path(DEFAULT_OCR_DEBUG_DIR))
    raw_config: dict[str, object] = field(default_factory=dict)
    raw_path_values: dict[str, str] = field(default_factory=dict)
    deprecated_config_keys: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Load config from file and environment with stable path resolution."""
        project_root = discover_project_root()
        explicit_path = Path(path).expanduser().resolve() if path is not None else None

        if explicit_path is not None:
            config_path = explicit_path
        else:
            env_path = os.environ.get(DEFAULT_CONFIG_ENV)
            config_path = Path(env_path).expanduser().resolve() if env_path else None
            if config_path is None:
                for candidate in _default_config_candidates(project_root):
                    if candidate.exists():
                        config_path = candidate.resolve()
                        break
            if config_path is None:
                config_path = DEFAULT_USER_CONFIG_PATH

        data: dict[str, object] = {}
        if config_path.exists():
            with config_path.open(encoding="utf-8-sig") as handle:
                data = json.load(handle)

        deprecated_keys = [key for key in REMOVED_SIDECAR_KEYS if key in data]

        def _path_value(name: str, default: str) -> tuple[str, Path]:
            raw = str(data.get(name, default))
            return raw, resolve_project_path(raw, project_root)

        raw_zotero_data_dir, zotero_data_dir = _path_value("zotero_data_dir", "~/Zotero")
        raw_chroma_db_path, chroma_db_path = _path_value("chroma_db_path", DEFAULT_CHROMA_DB_PATH)
        raw_runtime_data_dir, runtime_data_dir = _path_value("runtime_data_dir", DEFAULT_RUNTIME_DATA_DIR)
        raw_runtime_tmp_dir, runtime_tmp_dir = _path_value("runtime_tmp_dir", DEFAULT_RUNTIME_TMP_DIR)
        raw_scratch_tmp_dir, scratch_tmp_dir = _path_value("scratch_tmp_dir", DEFAULT_SCRATCH_TMP_DIR)
        raw_model_cache_dir, model_cache_dir = _path_value("model_cache_dir", DEFAULT_MODEL_CACHE_DIR)
        raw_runtime_log_path, runtime_log_path = _path_value("runtime_log_path", DEFAULT_RUNTIME_LOG_PATH)
        raw_ocr_debug_dir, ocr_debug_dir = _path_value("ocr_debug_dir", DEFAULT_OCR_DEBUG_DIR)

        return cls(
            zotero_data_dir=zotero_data_dir,
            chroma_db_path=chroma_db_path,
            embedding_model=str(data.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B")),
            embedding_dimensions=int(data.get("embedding_dimensions", 1024)),
            chunk_size=int(data.get("chunk_size", 400)),
            chunk_overlap=int(data.get("chunk_overlap", 100)),
            gemini_api_key=data.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"),
            embedding_provider=str(data.get("embedding_provider", "qwen_inprocess")),
            embedding_timeout=float(data.get("embedding_timeout", 120.0)),
            embedding_max_retries=int(data.get("embedding_max_retries", 3)),
            rerank_alpha=float(data.get("rerank_alpha", 0.7)),
            rerank_section_weights=data.get("rerank_section_weights"),
            rerank_journal_weights=data.get("rerank_journal_weights"),
            rerank_enabled=bool(data.get("rerank_enabled", True)),
            oversample_multiplier=int(data.get("oversample_multiplier", 3)),
            oversample_topic_factor=int(data.get("oversample_topic_factor", 5)),
            stats_sample_limit=int(data.get("stats_sample_limit", 10000)),
            ocr_language=str(data.get("ocr_language", "eng")),
            openalex_email=data.get("openalex_email") or os.environ.get("OPENALEX_EMAIL"),
            vision_enabled=bool(data.get("vision_enabled", True)),
            vision_model=str(data.get("vision_model", "claude-haiku-4-5-20251001")),
            anthropic_api_key=data.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY"),
            embedding_batch_size=int(data.get("embedding_batch_size", 8)),
            embedding_device=(str(data["embedding_device"]) if data.get("embedding_device") else None),
            embedding_allow_download=bool(data.get("embedding_allow_download", False)),
            project_root=project_root,
            config_path=config_path,
            runtime_data_dir=runtime_data_dir,
            runtime_tmp_dir=runtime_tmp_dir,
            scratch_tmp_dir=scratch_tmp_dir,
            model_cache_dir=model_cache_dir,
            runtime_log_path=runtime_log_path,
            ocr_debug_dir=ocr_debug_dir,
            raw_config=data,
            raw_path_values={
                "zotero_data_dir": raw_zotero_data_dir,
                "chroma_db_path": raw_chroma_db_path,
                "runtime_data_dir": raw_runtime_data_dir,
                "runtime_tmp_dir": raw_runtime_tmp_dir,
                "scratch_tmp_dir": raw_scratch_tmp_dir,
                "model_cache_dir": raw_model_cache_dir,
                "runtime_log_path": raw_runtime_log_path,
                "ocr_debug_dir": raw_ocr_debug_dir,
            },
            deprecated_config_keys=deprecated_keys,
        )

    def ensure_runtime_dirs(self) -> None:
        """Create project-local runtime directories used by app."""
        for path in (
            self.runtime_data_dir,
            self.runtime_tmp_dir,
            self.scratch_tmp_dir,
            self.model_cache_dir,
            self.runtime_log_path.parent,
            self.ocr_debug_dir,
            self.chroma_db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def path_report(self) -> dict[str, dict[str, str]]:
        """Return raw and resolved path values for diagnostics."""
        resolved = {
            "zotero_data_dir": self.zotero_data_dir,
            "chroma_db_path": self.chroma_db_path,
            "runtime_data_dir": self.runtime_data_dir,
            "runtime_tmp_dir": self.runtime_tmp_dir,
            "scratch_tmp_dir": self.scratch_tmp_dir,
            "model_cache_dir": self.model_cache_dir,
            "runtime_log_path": self.runtime_log_path,
            "ocr_debug_dir": self.ocr_debug_dir,
        }
        return {
            name: {
                "raw": self.raw_path_values.get(name, str(path)),
                "resolved": str(path),
            }
            for name, path in resolved.items()
        }

    def validate(self) -> list[str]:
        """Return validation errors."""
        errors: list[str] = []
        if not self.zotero_data_dir.exists():
            errors.append(f"Zotero data dir not found: {self.zotero_data_dir}")
        if not (self.zotero_data_dir / "zotero.sqlite").exists():
            errors.append(f"Zotero database not found: {self.zotero_data_dir / 'zotero.sqlite'}")

        valid_providers = ("qwen_inprocess", "gemini", "local")
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            errors.append("GEMINI_API_KEY not set (required for embedding_provider='gemini')")
        elif self.embedding_provider not in valid_providers:
            errors.append(
                f"Invalid embedding_provider: {self.embedding_provider}. "
                "Must be 'qwen_inprocess', 'gemini', or 'local'"
            )

        if self.deprecated_config_keys:
            joined = ", ".join(sorted(self.deprecated_config_keys))
            errors.append(
                "Deprecated sidecar embedding config keys detected: "
                f"{joined}. Remove these keys and use embedding_provider='qwen_inprocess'."
            )

        return errors
