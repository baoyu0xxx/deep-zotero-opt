"""Configuration management."""
from dataclasses import dataclass
from pathlib import Path
import json
import os


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
    # Embedding provider: "gemini" (API) or "local" (ChromaDB default all-MiniLM-L6-v2)
    embedding_provider: str
    # Embedding settings
    embedding_timeout: float
    embedding_max_retries: int
    # Reranking settings
    rerank_alpha: float
    rerank_section_weights: dict[str, float] | None
    rerank_journal_weights: dict[str, float] | None  # Use "unknown" for null quartile
    rerank_enabled: bool
    oversample_multiplier: int
    oversample_topic_factor: int  # Additional factor for search_topic
    stats_sample_limit: int
    # OCR settings (language passed through to pymupdf-layout)
    ocr_language: str
    # OpenAlex settings
    openalex_email: str | None  # Optional email for polite pool (10 req/sec vs 1 req/sec)
    # Vision extraction settings
    vision_enabled: bool
    vision_model: str
    anthropic_api_key: str | None
    # OpenAI-compatible embedding endpoint settings
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_query_instruction: str | None = None
    embedding_batch_size: int = 8
    embedding_auto_start: bool = False
    embedding_startup_timeout: float = 180.0
    embedding_start_command: str | None = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        """Load config from file and/or environment."""
        if path is not None:
            config_path = Path(path).expanduser()
        else:
            config_path = Path("~/.config/deep-zotero/config.json").expanduser()

        data = {}
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)

        return cls(
            zotero_data_dir=Path(data.get("zotero_data_dir", "~/Zotero")).expanduser(),
            chroma_db_path=Path(data.get("chroma_db_path", "~/.local/share/deep-zotero/chroma")).expanduser(),
            embedding_model=data.get("embedding_model", "gemini-embedding-001"),
            embedding_dimensions=data.get("embedding_dimensions", 768),
            chunk_size=data.get("chunk_size", 400),
            chunk_overlap=data.get("chunk_overlap", 100),
            gemini_api_key=data.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"),
            # Embedding provider: "gemini" or "local"
            embedding_provider=data.get("embedding_provider", "gemini"),
            # Embedding settings
            embedding_timeout=data.get("embedding_timeout", 120.0),
            embedding_max_retries=data.get("embedding_max_retries", 3),
            # Reranking settings
            rerank_alpha=data.get("rerank_alpha", 0.7),
            rerank_section_weights=data.get("rerank_section_weights"),
            rerank_journal_weights=data.get("rerank_journal_weights"),
            rerank_enabled=data.get("rerank_enabled", True),
            oversample_multiplier=data.get("oversample_multiplier", 3),
            oversample_topic_factor=data.get("oversample_topic_factor", 5),
            stats_sample_limit=data.get("stats_sample_limit", 10000),
            # OCR settings — language passed through to pymupdf-layout
            ocr_language=data.get("ocr_language", "eng"),
            # OpenAlex settings
            openalex_email=data.get("openalex_email") or os.environ.get("OPENALEX_EMAIL"),
            # Vision extraction settings
            vision_enabled=data.get("vision_enabled", True),
            vision_model=data.get("vision_model", "claude-haiku-4-5-20251001"),
            anthropic_api_key=data.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY"),
            # OpenAI-compatible embedding settings
            embedding_base_url=data.get("embedding_base_url") or os.environ.get("EMBEDDING_BASE_URL"),
            embedding_api_key=data.get("embedding_api_key") or os.environ.get("EMBEDDING_API_KEY"),
            embedding_query_instruction=data.get("embedding_query_instruction"),
            embedding_batch_size=data.get("embedding_batch_size", 8),
            embedding_auto_start=data.get("embedding_auto_start", False),
            embedding_startup_timeout=data.get("embedding_startup_timeout", 180.0),
            embedding_start_command=data.get("embedding_start_command") or os.environ.get("EMBEDDING_START_COMMAND"),
        )

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if not self.zotero_data_dir.exists():
            errors.append(f"Zotero data dir not found: {self.zotero_data_dir}")
        if not (self.zotero_data_dir / "zotero.sqlite").exists():
            errors.append(f"Zotero database not found: {self.zotero_data_dir / 'zotero.sqlite'}")

        valid_providers = ("gemini", "local", "openai_compatible")

        # Only require API key for Gemini provider
        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            errors.append("GEMINI_API_KEY not set (required for embedding_provider='gemini')")
        elif self.embedding_provider == "openai_compatible":
            if not self.embedding_base_url:
                errors.append("embedding_base_url not set (required for embedding_provider='openai_compatible')")
            if not self.embedding_api_key:
                errors.append("embedding_api_key not set (required for embedding_provider='openai_compatible')")
        elif self.embedding_provider not in valid_providers:
            errors.append(
                f"Invalid embedding_provider: {self.embedding_provider}. "
                "Must be 'gemini', 'local', or 'openai_compatible'"
            )

        return errors
