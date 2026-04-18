"""Embedding services: Qwen in-process, Gemini API, and local fallback."""
from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding fails after retries."""


class Embedder:
    """
    Gemini embedding wrapper using gemini-embedding-001.

    Uses asymmetric embeddings: different task types for docs vs queries.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        from google import genai

        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries

    def _embed_batch_with_timeout(
        self, batch: list[str], task_type: str, batch_num: int, total_batches: int
    ) -> list[list[float]]:
        from google.genai import types

        total_chars = sum(len(t) for t in batch)
        logger.debug(
            "Embedding batch %d/%d: %d texts, %d chars total",
            batch_num,
            total_batches,
            len(batch),
            total_chars,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.client.models.embed_content,
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dimensions,
                        ),
                    )
                    response = future.result(timeout=self.timeout)

                return [e.values for e in response.embeddings]
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Batch %d/%d timed out after %.1fs (attempt %d/%d)",
                    batch_num,
                    total_batches,
                    self.timeout,
                    attempt,
                    self.max_retries,
                )
            except Exception as exc:
                logger.warning(
                    "Batch %d/%d failed (attempt %d/%d): %s: %s",
                    batch_num,
                    total_batches,
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    exc,
                )

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        raise EmbeddingError(
            f"Batch {batch_num}/{total_batches} failed after "
            f"{self.max_retries} attempts ({len(batch)} texts)"
        )

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        batch_size = 100
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(
                self._embed_batch_with_timeout(
                    batch=batch,
                    task_type=task_type,
                    batch_num=(i // batch_size) + 1,
                    total_batches=total_batches,
                )
            )
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query], task_type="RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, task_type="RETRIEVAL_DOCUMENT")


class LocalEmbedder:
    """
    Local embedding using ChromaDB default function (all-MiniLM-L6-v2).
    """

    def __init__(self):
        import chromadb.utils.embedding_functions as ef

        self._ef = ef.DefaultEmbeddingFunction()
        self.dimensions = 384

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        if not texts:
            return []
        return [[float(v) for v in e] for e in self._ef(texts)]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


def _snapshot_has_weights(snapshot_dir: Path) -> bool:
    return any((snapshot_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))


def _model_cache_root(cache_dir: Path | None, model_name: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"models--{model_name.replace('/', '--')}"


def _find_complete_snapshot(cache_dir: Path | None, model_name: str) -> Path | None:
    model_cache = _model_cache_root(cache_dir, model_name)
    if model_cache is None:
        return None
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        return None
    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir() and _snapshot_has_weights(snapshot):
            return snapshot
    return None


def _default_hf_cache_dir() -> Path:
    if os.name == "nt":
        return Path.home() / ".cache" / "huggingface" / "hub"
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _resolve_qwen_model_source(
    model_name: str,
    cache_dir: Path | None,
    allow_download: bool,
) -> tuple[str, dict[str, Any]]:
    model_kwargs: dict[str, Any] = {}

    dedicated_snapshot = _find_complete_snapshot(cache_dir, model_name)
    if dedicated_snapshot is not None:
        logger.info("Using dedicated cached snapshot: %s", dedicated_snapshot)
        if cache_dir is not None:
            os.environ.setdefault("HF_HOME", str(cache_dir))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        return str(dedicated_snapshot), model_kwargs

    shared_snapshot = _find_complete_snapshot(_default_hf_cache_dir(), model_name)
    if shared_snapshot is not None:
        logger.info("Using shared Hugging Face snapshot: %s", shared_snapshot)
        return str(shared_snapshot), model_kwargs

    if cache_dir is not None:
        model_kwargs["cache_folder"] = str(cache_dir)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    return model_name, model_kwargs


class QwenInProcessEmbedder:
    """In-process Qwen embedding model using SentenceTransformer."""

    def __init__(
        self,
        model: str,
        dimensions: int,
        cache_dir: Path | None,
        device: str | None,
        batch_size: int,
        allow_download: bool,
        max_retries: int,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.dimensions = dimensions
        self.device = device
        self.batch_size = max(1, batch_size)
        self.allow_download = allow_download
        self.max_retries = max(1, max_retries)
        self._torch = torch

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        model_source, model_kwargs = _resolve_qwen_model_source(model, cache_dir, allow_download)
        if device:
            model_kwargs["device"] = device
        if (
            model_source == model
            and _find_complete_snapshot(cache_dir, model) is not None
            and not allow_download
        ):
            model_kwargs["local_files_only"] = True
            os.environ["HF_HUB_OFFLINE"] = "1"

        self._model = SentenceTransformer(model_source, **model_kwargs)

    def _clear_cuda_cache(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        current_batch_size = max(1, min(self.batch_size, len(texts)))
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                vectors = self._model.encode(
                    texts,
                    batch_size=current_batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return [[float(v) for v in vector] for vector in vectors]
            except self._torch.OutOfMemoryError as exc:
                last_error = exc
                self._clear_cuda_cache()
                if current_batch_size == 1:
                    break
                next_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    "Qwen in-process embedding OOM (attempt %d/%d): %d -> %d batch size",
                    attempt,
                    self.max_retries,
                    current_batch_size,
                    next_batch_size,
                )
                current_batch_size = next_batch_size
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Qwen in-process embedding failed (attempt %d/%d): %s: %s",
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise EmbeddingError(
            "Qwen in-process embedding failed after "
            f"{self.max_retries} attempts: {type(last_error).__name__ if last_error else 'unknown'}: {last_error}"
        )

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        del task_type  # Symmetric embedding model.
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            results.extend(self._encode_batch(texts[i : i + self.batch_size]))
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query], task_type="RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, task_type="RETRIEVAL_DOCUMENT")


def create_embedder(config: "Config"):
    """Create embedder based on config.embedding_provider."""
    if config.embedding_provider == "local":
        logger.info("Using local embeddings (all-MiniLM-L6-v2, 384 dimensions)")
        return LocalEmbedder()

    if config.embedding_provider == "gemini":
        logger.info(
            "Using Gemini embeddings (%s, %d dimensions)",
            config.embedding_model,
            config.embedding_dimensions,
        )
        return Embedder(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            api_key=config.gemini_api_key,
            timeout=config.embedding_timeout,
            max_retries=config.embedding_max_retries,
        )

    if config.embedding_provider == "qwen_inprocess":
        logger.info(
            "Using Qwen in-process embeddings (%s, %d dimensions)",
            config.embedding_model,
            config.embedding_dimensions,
        )
        return QwenInProcessEmbedder(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            cache_dir=config.model_cache_dir,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
            allow_download=config.embedding_allow_download,
            max_retries=config.embedding_max_retries,
        )

    raise ValueError(
        f"Invalid embedding_provider: {config.embedding_provider}. "
        "Must be 'qwen_inprocess', 'local', or 'gemini'"
    )
