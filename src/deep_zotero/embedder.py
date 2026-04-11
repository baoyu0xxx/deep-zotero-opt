"""Embedding services: Gemini API, OpenAI-compatible API, and local fallback."""
import logging
import time
import concurrent.futures
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding fails after retries."""


class Embedder:
    """
    Gemini embedding wrapper using gemini-embedding-001.

    This model is #1 on MTEB Multilingual leaderboard.
    Uses asymmetric embeddings: different task types for docs vs queries.

    Output dimensions: configurable (768 default, up to 3072)
    Max input: 2048 tokens per text
    Batch size: up to 100 texts
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
        # Uses GEMINI_API_KEY env var if api_key not provided
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
        """Embed a single batch with timeout, retry, and logging."""
        from google.genai import types
        total_chars = sum(len(t) for t in batch)
        logger.debug(
            f"Embedding batch {batch_num}/{total_batches}: "
            f"{len(batch)} texts, {total_chars} chars total"
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

                logger.debug(
                    f"Batch {batch_num}/{total_batches} succeeded "
                    f"(attempt {attempt}), got {len(response.embeddings)} embeddings"
                )
                return [e.values for e in response.embeddings]

            except concurrent.futures.TimeoutError:
                logger.warning(
                    f"Batch {batch_num}/{total_batches} timed out after "
                    f"{self.timeout}s (attempt {attempt}/{self.max_retries})"
                )
            except Exception as e:
                logger.warning(
                    f"Batch {batch_num}/{total_batches} failed "
                    f"(attempt {attempt}/{self.max_retries}): {type(e).__name__}: {e}"
                )

            if attempt < self.max_retries:
                backoff = 2 ** attempt
                logger.info(f"Retrying in {backoff}s...")
                time.sleep(backoff)

        raise EmbeddingError(
            f"Batch {batch_num}/{total_batches} failed after "
            f"{self.max_retries} attempts ({len(batch)} texts, {total_chars} chars)"
        )

    def embed(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed
            task_type: One of:
                - "RETRIEVAL_DOCUMENT": For indexing documents (default)
                - "RETRIEVAL_QUERY": For search queries
                - "SEMANTIC_SIMILARITY": For comparing texts
                - "CLASSIFICATION": For classification

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails after retries
        """
        if not texts:
            return []

        results = []
        batch_size = 100  # Gemini limit
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.debug(
            f"Embedding {len(texts)} texts in {total_batches} batch(es), "
            f"task_type={task_type}"
        )

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            batch_results = self._embed_batch_with_timeout(
                batch, task_type, batch_num, total_batches
            )
            results.extend(batch_results)

        return results

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.

        Uses RETRIEVAL_QUERY task type for asymmetric retrieval.
        """
        return self.embed([query], task_type="RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed documents for indexing.

        Uses RETRIEVAL_DOCUMENT task type.
        """
        return self.embed(texts, task_type="RETRIEVAL_DOCUMENT")


class LocalEmbedder:
    """
    Local embedding using ChromaDB's default function (all-MiniLM-L6-v2).

    Benefits:
    - No API key required
    - Works offline
    - ~90MB model, downloaded automatically on first use
    - 384 dimensions (vs Gemini's 768)

    Note: Uses symmetric embeddings (same for docs and queries).
    """

    def __init__(self):
        import chromadb.utils.embedding_functions as ef
        self._ef = ef.DefaultEmbeddingFunction()
        self.dimensions = 384  # all-MiniLM-L6-v2 output size

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Embed texts. task_type is ignored (symmetric model)."""
        if not texts:
            return []
        # ChromaDB's DefaultEmbeddingFunction returns numpy arrays with np.float32
        # Convert to native Python floats for ChromaDB compatibility
        return [[float(v) for v in e] for e in self._ef(texts)]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query."""
        return self.embed([query])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for indexing."""
        return self.embed(texts)


class OpenAICompatibleEmbedder:
    """Embedding client for OpenAI-compatible /v1/embeddings endpoints."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        dimensions: int,
        query_instruction: str | None = None,
        batch_size: int = 8,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.dimensions = dimensions
        self.query_instruction = query_instruction
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries

    def _prepare_texts(self, texts: list[str], task_type: str) -> list[str]:
        if task_type == "RETRIEVAL_QUERY" and self.query_instruction:
            return [f"{self.query_instruction}\n{text}" for text in texts]
        return texts

    def _embed_batch(self, batch: list[str], task_type: str, batch_num: int, total_batches: int) -> list[list[float]]:
        prepared = self._prepare_texts(batch, task_type)
        total_chars = sum(len(text) for text in batch)
        logger.debug(
            "OpenAI-compatible embedding batch %d/%d: %d texts, %d chars",
            batch_num,
            total_batches,
            len(batch),
            total_chars,
        )
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=prepared,
                )
                embeddings = [item.embedding for item in response.data]
                logger.debug(
                    "OpenAI-compatible embedding batch %d/%d succeeded: %d vectors",
                    batch_num,
                    total_batches,
                    len(embeddings),
                )
                return embeddings
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "OpenAI-compatible embedding batch %d/%d failed "
                    "(attempt %d/%d, %d texts, %d chars): %s: %s",
                    batch_num,
                    total_batches,
                    attempt,
                    self.max_retries,
                    len(batch),
                    total_chars,
                    type(exc).__name__,
                    exc,
                )
                if len(batch) > 1 and _is_retryable_embedding_error(exc):
                    mid = max(1, len(batch) // 2)
                    logger.warning(
                        "OpenAI-compatible embedding batch %d/%d shrinking from %d to %d+%d texts",
                        batch_num,
                        total_batches,
                        len(batch),
                        mid,
                        len(batch) - mid,
                    )
                    left = self._embed_batch(batch[:mid], task_type, batch_num, total_batches)
                    right = self._embed_batch(batch[mid:], task_type, batch_num, total_batches)
                    return left + right
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        raise EmbeddingError(
            f"OpenAI-compatible embedding batch {batch_num}/{total_batches} "
            f"failed after {self.max_retries} attempts "
            f"({len(batch)} texts, {total_chars} chars): "
            f"{type(last_exc).__name__ if last_exc else 'unknown'}: {last_exc}"
        )

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        if not texts:
            return []
        results = []
        batch_size = self.batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self._embed_batch(batch, task_type, i // batch_size + 1, total_batches))
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query], task_type="RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, task_type="RETRIEVAL_DOCUMENT")


def _is_retryable_embedding_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if response_status in {429, 500, 502, 503, 504}:
            return True

    body = str(exc).lower()
    return any(
        marker in body
        for marker in (
            "cuda out of memory",
            "cuda_oom",
            "out of memory",
            "embedding request exhausted cuda memory",
            "server error",
        )
    )


def create_embedder(config: "Config"):
    """Create embedder based on config.embedding_provider.

    Args:
        config: Application configuration

    Returns:
        Embedder instance (Gemini API or LocalEmbedder)

    Raises:
        ValueError: If embedding_provider is invalid
    """
    if config.embedding_provider == "local":
        logger.info("Using local embeddings (all-MiniLM-L6-v2, 384 dimensions)")
        return LocalEmbedder()
    elif config.embedding_provider == "gemini":
        logger.info(f"Using Gemini embeddings ({config.embedding_model}, {config.embedding_dimensions} dimensions)")
        return Embedder(
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
            api_key=config.gemini_api_key,
            timeout=config.embedding_timeout,
            max_retries=config.embedding_max_retries,
        )
    elif config.embedding_provider == "openai_compatible":
        logger.info(
            "Using OpenAI-compatible embeddings (%s, %d dimensions, %s)",
            config.embedding_model,
            config.embedding_dimensions,
            config.embedding_base_url,
        )
        if not config.embedding_base_url or not config.embedding_api_key:
            raise ValueError(
                "embedding_base_url and embedding_api_key are required for "
                "embedding_provider='openai_compatible'"
            )
        return OpenAICompatibleEmbedder(
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
            dimensions=config.embedding_dimensions,
            query_instruction=config.embedding_query_instruction,
            batch_size=config.embedding_batch_size,
            timeout=config.embedding_timeout,
            max_retries=config.embedding_max_retries,
        )
    else:
        raise ValueError(
            f"Invalid embedding_provider: {config.embedding_provider}. "
            "Must be 'gemini', 'local', or 'openai_compatible'"
        )
