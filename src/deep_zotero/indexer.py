"""Indexing pipeline orchestration."""
import hashlib
import json
import logging
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from .config import Config
from .zotero_client import ZoteroClient
from .pdf_processor import extract_document
from .chunker import Chunker
from .embedder import create_embedder
from .vector_store import VectorStore
from .journal_ranker import JournalRanker
from .models import ZoteroItem

logger = logging.getLogger(__name__)
_EXTRACTION_CACHE_VERSION = 1
_EMBEDDING_CACHE_VERSION = 1


def _progress(iterable, *, desc: str, total: int | None = None):
    """Use tqdm only in an interactive terminal; logs cover non-TTY runs."""
    if total == 0:
        return []
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )


def _config_hash(config: Config) -> str:
    """Hash of config values that affect indexed content.

    Changes to these values require re-indexing.
    """
    data = (
        f"{config.chunk_size}:"
        f"{config.chunk_overlap}:"
        f"{config.embedding_provider}:"
        f"{config.embedding_dimensions}:"
        f"{config.embedding_model}:"
        f"{config.embedding_base_url}:"
        f"{config.ocr_language}"
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class IndexResult:
    """Outcome of indexing a single document."""
    item_key: str
    title: str
    status: str          # "indexed", "failed", "empty", "skipped"
    reason: str = ""
    n_chunks: int = 0
    n_tables: int = 0
    quality_grade: str = ""  # A/B/C/D/F quality grade per document


class Indexer:
    """
    Orchestrates the full indexing pipeline.

    Pipeline: Zotero -> PDF -> Chunks -> Embeddings -> VectorStore
    """

    def __init__(self, config: Config):
        self.config = config
        self.zotero = ZoteroClient(config.zotero_data_dir)

        self.chunker = Chunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        # Use factory to create appropriate embedder based on config
        self.embedder = create_embedder(config)
        self.store = VectorStore(config.chroma_db_path, self.embedder)
        self.journal_ranker = JournalRanker()
        self._empty_docs_path = config.chroma_db_path / "empty_docs.json"
        self._config_hash_path = config.chroma_db_path / "config_hash.txt"
        if config.vision_enabled and config.anthropic_api_key:
            from .feature_extraction.vision_api import VisionAPI
            cost_log_path = config.chroma_db_path.parent / "vision_costs.json"
            self._vision_api = VisionAPI(
                api_key=config.anthropic_api_key,
                model=config.vision_model,
                cost_log_path=cost_log_path,
            )
        else:
            self._vision_api = None

    # ------------------------------------------------------------------
    # Empty-doc tracking (keyed by item_key -> pdf file hash)
    # ------------------------------------------------------------------

    def _load_empty_docs(self) -> dict[str, str]:
        """Load {item_key: pdf_hash} for docs that yielded no chunks."""
        if self._empty_docs_path.exists():
            return json.loads(self._empty_docs_path.read_text())
        return {}

    def _save_empty_docs(self, mapping: dict[str, str]) -> None:
        self._empty_docs_path.write_text(json.dumps(mapping, indent=2))

    @staticmethod
    def _pdf_hash(path: Path) -> str:
        """Fast hash of first 64 KiB of a PDF (enough to detect replacement)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(65536))
        return h.hexdigest()

    @property
    def _extraction_cache_dir(self) -> Path:
        return self.config.chroma_db_path.parent / "extraction-cache"

    @property
    def _embedding_cache_dir(self) -> Path:
        return self.config.chroma_db_path.parent / "embedding-cache"

    @staticmethod
    def _safe_cache_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"

    def _extraction_cache_meta(self, item: ZoteroItem, ocr_mode: str) -> dict:
        stat = item.pdf_path.stat()
        return {
            "version": _EXTRACTION_CACHE_VERSION,
            "item_key": item.item_key,
            "pdf_path": str(item.pdf_path),
            "pdf_hash": self._pdf_hash(item.pdf_path),
            "pdf_size": stat.st_size,
            "pdf_mtime_ns": stat.st_mtime_ns,
            "ocr_mode": ocr_mode,
            "ocr_language": self.config.ocr_language,
            "vision_enabled": bool(self._vision_api),
        }

    def _extraction_cache_path(self, meta: dict) -> Path:
        item_key = self._safe_cache_name(str(meta["item_key"]))
        suffix = self._safe_cache_name(
            f"{meta['ocr_mode']}-{meta['ocr_language']}-{meta['pdf_hash']}"
        )
        return self._extraction_cache_dir / f"{item_key}-{suffix}.pkl"

    def _load_extraction_cache(self, item: ZoteroItem, ocr_mode: str):
        if item.pdf_path is None or not item.pdf_path.exists():
            return None
        meta = self._extraction_cache_meta(item, ocr_mode)
        path = self._extraction_cache_path(meta)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            logger.warning(
                "Ignoring unreadable extraction cache for %s: %s: %s",
                item.item_key,
                type(exc).__name__,
                exc,
            )
            return None

        cached_meta = payload.get("meta") if isinstance(payload, dict) else None
        if cached_meta != meta:
            logger.info("Ignoring stale extraction cache for %s", item.item_key)
            return None
        extraction = payload.get("extraction")
        if extraction is None:
            return None
        logger.info("Using cached extraction for %s: %s", item.item_key, path.name)
        return extraction

    def _save_extraction_cache(self, item: ZoteroItem, ocr_mode: str, extraction) -> None:
        if item.pdf_path is None or not item.pdf_path.exists():
            return
        meta = self._extraction_cache_meta(item, ocr_mode)
        path = self._extraction_cache_path(meta)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp_path.open("wb") as f:
                pickle.dump({"meta": meta, "extraction": extraction}, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
            logger.debug("Saved extraction cache for %s: %s", item.item_key, path)
        except Exception as exc:
            logger.warning(
                "Could not save extraction cache for %s: %s: %s",
                item.item_key,
                type(exc).__name__,
                exc,
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _delete_extraction_cache_for_item(self, item: ZoteroItem, ocr_mode: str) -> None:
        if item.pdf_path is None or not item.pdf_path.exists():
            return
        meta = self._extraction_cache_meta(item, ocr_mode)
        path = self._extraction_cache_path(meta)
        if not path.exists():
            return
        try:
            path.unlink()
            logger.debug("Deleted extraction cache for %s: %s", item.item_key, path)
        except OSError as exc:
            logger.warning(
                "Could not delete extraction cache for %s: %s: %s",
                item.item_key,
                type(exc).__name__,
                exc,
            )

    def _embedding_cache_meta(self, item: ZoteroItem, ocr_mode: str) -> dict:
        return {
            "version": _EMBEDDING_CACHE_VERSION,
            "item_key": item.item_key,
            "pdf_hash": self._pdf_hash(item.pdf_path),
            "ocr_mode": ocr_mode,
            "ocr_language": self.config.ocr_language,
            "vision_enabled": bool(self._vision_api),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "embedding_dimensions": self.config.embedding_dimensions,
            "embedding_base_url": self.config.embedding_base_url,
        }

    def _embedding_cache_path(self, meta: dict) -> Path:
        item_key = self._safe_cache_name(str(meta["item_key"]))
        cache_key = hashlib.sha256(
            json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:24]
        return self._embedding_cache_dir / f"{item_key}-{cache_key}.pkl"

    @staticmethod
    def _diff_meta_keys(expected: dict, actual: dict) -> list[str]:
        keys = set(expected) | set(actual)
        return sorted(k for k in keys if expected.get(k) != actual.get(k))

    def _load_embedding_cache(self, item: ZoteroItem, ocr_mode: str) -> dict | None:
        if item.pdf_path is None or not item.pdf_path.exists():
            return None
        meta = self._embedding_cache_meta(item, ocr_mode)
        path = self._embedding_cache_path(meta)
        if not path.exists():
            pattern = f"{self._safe_cache_name(str(meta['item_key']))}-*.pkl"
            candidates = list(self._embedding_cache_dir.glob(pattern))
            if candidates:
                try:
                    with candidates[0].open("rb") as f:
                        candidate_payload = pickle.load(f)
                    candidate_meta = (
                        candidate_payload.get("meta")
                        if isinstance(candidate_payload, dict)
                        else None
                    )
                    if isinstance(candidate_meta, dict):
                        changed_keys = self._diff_meta_keys(meta, candidate_meta)
                        logger.info(
                            "Embedding cache exists but is stale for %s (changed: %s)",
                            item.item_key,
                            ", ".join(changed_keys) if changed_keys else "unknown",
                        )
                except Exception:
                    pass
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            logger.warning(
                "Ignoring unreadable embedding cache for %s: %s: %s",
                item.item_key,
                type(exc).__name__,
                exc,
            )
            return None

        if not isinstance(payload, dict):
            logger.info("Ignoring malformed embedding cache for %s", item.item_key)
            return None

        cached_meta = payload.get("meta")
        records = payload.get("records")
        summary = payload.get("summary")
        if not isinstance(cached_meta, dict) or not isinstance(records, dict):
            logger.info("Ignoring malformed embedding cache payload for %s", item.item_key)
            return None
        if cached_meta != meta:
            changed_keys = self._diff_meta_keys(meta, cached_meta)
            logger.info(
                "Ignoring stale embedding cache for %s (changed: %s)",
                item.item_key,
                ", ".join(changed_keys) if changed_keys else "unknown",
            )
            return None
        if summary is not None and not isinstance(summary, dict):
            logger.info("Ignoring embedding cache with invalid summary for %s", item.item_key)
            return None
        logger.info("Using cached embeddings for %s, skipping extraction", item.item_key)
        return payload

    def _save_embedding_cache(self, item: ZoteroItem, ocr_mode: str, payload: dict) -> bool:
        if item.pdf_path is None or not item.pdf_path.exists():
            return False
        meta = self._embedding_cache_meta(item, ocr_mode)
        path = self._embedding_cache_path(meta)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        cache_payload = {
            "meta": meta,
            "records": payload.get("records", {}),
            "summary": payload.get("summary", {}),
        }
        try:
            with tmp_path.open("wb") as f:
                pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
            logger.debug("Saved embedding cache for %s: %s", item.item_key, path)
            return True
        except Exception as exc:
            logger.warning(
                "Could not save embedding cache for %s: %s: %s",
                item.item_key,
                type(exc).__name__,
                exc,
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return False

    def _needs_reindex(self, item: ZoteroItem) -> tuple[bool, str]:
        """Check if a document needs (re)indexing based on PDF hash.

        Returns:
            (needs_reindex, reason) where reason is:
            - "new": Document not in index
            - "changed": PDF hash differs from stored hash
            - "no_hash": Document indexed without hash, needs reindex
            - "current": Document is up-to-date, no reindex needed
        """
        existing_meta = self.store.get_document_meta(item.item_key)
        if not existing_meta:
            return True, "new"

        stored_hash = existing_meta.get("pdf_hash")
        if not stored_hash:
            return True, "no_hash"

        current_hash = self._pdf_hash(item.pdf_path)
        if stored_hash != current_hash:
            return True, "changed"

        return False, "current"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def index_all(
        self,
        force_reindex: bool = False,
        limit: int | None = None,
        item_key: str | None = None,
        title_pattern: str | None = None,
        ocr_mode: str = "auto",
    ) -> dict:
        """
        Index all PDFs in Zotero library.

        Args:
            force_reindex: Delete and re-index matching items
            limit: Maximum number of items to index
            item_key: If provided, only index this specific Zotero item key
            title_pattern: If provided, only index items matching this regex pattern
            ocr_mode: OCR mode passed through to PDF extraction

        Returns:
            Dict with 'results' (list[IndexResult]) and summary counts.
        """
        if ocr_mode not in {"auto", "always", "off"}:
            raise ValueError("ocr_mode must be one of: auto, always, off")

        items = self.zotero.get_all_items_with_pdfs()
        items = [i for i in items if i.pdf_path and i.pdf_path.exists()]
        raw_item_count = len(items)
        unique_items: dict[str, ZoteroItem] = {}
        for item in items:
            unique_items.setdefault(item.item_key, item)
        items = list(unique_items.values())
        if len(items) != raw_item_count:
            logger.info(
                "Discovered %d PDF attachments for %d unique Zotero items "
                "(deduplicated %d duplicate attachments)",
                raw_item_count,
                len(items),
                raw_item_count - len(items),
            )
        else:
            logger.info(f"Discovered {len(items)} papers with PDFs in Zotero library")

        # Apply filters
        if item_key:
            items = [i for i in items if i.item_key == item_key]
            if not items:
                logger.error(f"No item found with key: {item_key}")
                return {"results": [], "indexed": 0, "failed": 0, "empty": 0, "skipped": 0, "already_indexed": 0}

        if title_pattern:
            import re
            pattern = re.compile(title_pattern, re.IGNORECASE)
            items = [i for i in items if pattern.search(i.title)]
            logger.info(f"Title filter: {len(items)} papers match '{title_pattern}'")

        if limit:
            items = items[:limit]
            logger.info(f"Limit applied: processing at most {limit} papers")

        if force_reindex:
            existing = self.store.get_indexed_doc_ids()
            item_keys = {i.item_key for i in items}
            for doc_id in existing & item_keys:
                self.store.delete_document(doc_id)
            indexed_ids = set()
            empty_docs: dict[str, str] = {}
        else:
            indexed_ids = self.store.get_indexed_doc_ids()
            empty_docs = self._load_empty_docs()

        # Check for config mismatch
        current_hash = _config_hash(self.config)
        stored_hash = None
        if self._config_hash_path.exists():
            stored_hash = self._config_hash_path.read_text().strip()

        if stored_hash and stored_hash != current_hash and not force_reindex:
            logger.warning(
                "Config has changed since last index (chunk_size, overlap, embedding, or section settings). "
                "Run with --force to re-index, otherwise results may be inconsistent."
            )

        results: list[IndexResult] = []
        to_index: list[ZoteroItem] = []
        reindex_reasons: dict[str, str] = {}

        for item in items:
            if item.item_key in indexed_ids:
                needs_reindex, reason = self._needs_reindex(item)
                if needs_reindex:
                    self.store.delete_document(item.item_key)
                    indexed_ids.discard(item.item_key)
                    reindex_reasons[item.item_key] = reason
                    logger.info(f"Reindexing {item.item_key}: {reason}")
                else:
                    continue

            if item.item_key in empty_docs:
                current_hash = self._pdf_hash(item.pdf_path)
                if current_hash == empty_docs[item.item_key]:
                    results.append(IndexResult(
                        item.item_key, item.title, "skipped",
                        reason="no extractable text (unchanged PDF)"))
                    continue
                else:
                    del empty_docs[item.item_key]
                    reindex_reasons[item.item_key] = "changed"

            to_index.append(item)

        reindex_count = len(reindex_reasons)
        n_skipped = sum(1 for r in results if r.status == "skipped")
        logger.info(
            f"Index plan: {len(to_index)} to index, "
            f"{reindex_count} to reindex (PDF changed), "
            f"{len(indexed_ids)} already indexed, "
            f"{n_skipped} skipped (empty/unchanged)"
        )
        if not to_index:
            logger.info("Nothing to index — all papers are up to date")

        quality_distribution: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        aggregated_extraction_stats = {
            "total_pages": 0,
            "text_pages": 0,
            "ocr_pages": 0,
            "empty_pages": 0,
        }

        # ---- Phase 0: Load embedding cache hits before extraction ----
        doc_embedded_cache_hits: dict[str, tuple[ZoteroItem, dict]] = {}
        to_extract: list[ZoteroItem] = []
        for item in to_index:
            cached_payload = self._load_embedding_cache(item, ocr_mode)
            if cached_payload is not None:
                doc_embedded_cache_hits[item.item_key] = (item, cached_payload)
                continue
            to_extract.append(item)

        # ---- Phase 1: Extract uncached documents (vision specs collected but deferred) ----
        figures_dir = self.config.chroma_db_path.parent / "figures"
        doc_extractions: dict[str, tuple[ZoteroItem, object]] = {}  # item_key -> (item, extraction)

        total_to_extract = len(to_extract)
        extraction_times: list[float] = []
        phase1_start = time.perf_counter()
        log_interval = 5  # log every N papers

        extraction_iterator = _progress(
            to_extract,
            desc="Extracting",
            total=total_to_extract,
        )
        for i, item in enumerate(extraction_iterator, 1):
            t0 = time.perf_counter()
            try:
                logger.debug(
                    f"Starting extraction {item.item_key}: "
                    f"title={item.title!r}, pdf={item.pdf_path}"
                )
                extraction = self._load_extraction_cache(item, ocr_mode)
                if extraction is None:
                    extraction = extract_document(
                        item.pdf_path,
                        write_images=True,
                        images_dir=figures_dir,
                        ocr_language=self.config.ocr_language,
                        ocr_mode=ocr_mode,
                        vision_api=self._vision_api,
                    )
                    self._save_extraction_cache(item, ocr_mode, extraction)
                doc_extractions[item.item_key] = (item, extraction)
            except Exception as e:
                logger.error(f"Failed to extract {item.item_key}: {type(e).__name__}: {e}")
                results.append(IndexResult(
                    item.item_key, item.title, "failed",
                    reason=f"{type(e).__name__}: {e}"))

            elapsed = time.perf_counter() - t0
            extraction_times.append(elapsed)

            if i % log_interval == 0 or i == total_to_extract:
                avg_time = sum(extraction_times) / len(extraction_times)
                remaining = total_to_extract - i
                eta_secs = avg_time * remaining
                if eta_secs >= 60:
                    eta_str = f"{eta_secs / 60:.1f}m"
                else:
                    eta_str = f"{eta_secs:.0f}s"
                logger.info(
                    f"Extraction: {i}/{total_to_extract} papers "
                    f"({avg_time:.1f}s avg, ETA {eta_str})"
                )

        phase1_elapsed = time.perf_counter() - phase1_start
        if total_to_extract > 0:
            logger.info(
                f"Extraction complete: {total_to_extract} papers in "
                f"{phase1_elapsed:.1f}s ({phase1_elapsed / total_to_extract:.1f}s avg)"
            )

        # ---- Phase 2: Resolve vision batch (one API call for all papers) ----
        if self._vision_api and doc_extractions:
            from .pdf_processor import resolve_pending_vision, PendingVisionWork
            pending_count = sum(
                len(v[1].pending_vision.specs)
                for v in doc_extractions.values()
                if v[1].pending_vision is not None and v[1].pending_vision.specs
            )
            pending_docs = sum(
                1 for v in doc_extractions.values()
                if v[1].pending_vision is not None and v[1].pending_vision.specs
            )
            if pending_count > 0:
                logger.info(
                    f"Vision: {pending_count} tables across {pending_docs} papers "
                    f"queued for Batch API (up to 3 waves, est. 10-30min per wave)"
                )
            phase2_start = time.perf_counter()
            resolve_pending_vision(
                {k: v[1] for k, v in doc_extractions.items()},
                self._vision_api,
            )
            phase2_elapsed = time.perf_counter() - phase2_start
            if pending_count > 0:
                logger.info(
                    f"Vision complete: {pending_count} tables in "
                    f"{phase2_elapsed / 60:.1f}min ({phase2_elapsed / max(pending_count, 1):.1f}s avg/table)"
                )

        # ---- Phase 3: Index each document (cached + extracted) ----
        total_to_index = len(doc_embedded_cache_hits) + len(doc_extractions)
        index_times: list[float] = []
        phase3_start = time.perf_counter()
        if total_to_index > 0:
            logger.info(f"Indexing: chunking and storing {total_to_index} papers")
            needs_embedding_service = bool(doc_extractions)
            if not needs_embedding_service and doc_embedded_cache_hits:
                logger.info(
                    "All %d papers served from embedding cache; skipping embedding health check",
                    len(doc_embedded_cache_hits),
                )
            try:
                if needs_embedding_service:
                    self.embedder.embed_query("indexing health check")
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                logger.error(
                    "Embedding endpoint is not ready before indexing phase: %s",
                    reason,
                )
                for item, _extraction in doc_extractions.values():
                    results.append(IndexResult(
                        item.item_key,
                        item.title,
                        "failed",
                        reason=f"embedding health check failed before indexing: {reason}",
                    ))
                self._save_empty_docs(empty_docs)
                return {
                    "results": results,
                    "indexed": sum(1 for r in results if r.status == "indexed"),
                    "failed": sum(1 for r in results if r.status == "failed"),
                    "empty": sum(1 for r in results if r.status == "empty"),
                    "skipped": sum(1 for r in results if r.status == "skipped"),
                    "already_indexed": len(indexed_ids),
                    "quality_distribution": quality_distribution,
                    "extraction_stats": aggregated_extraction_stats,
                }

        indexing_items: list[tuple[str, tuple[ZoteroItem, object, str]]] = []
        indexing_items.extend(
            (k, (v[0], v[1], "cached_embedding"))
            for k, v in doc_embedded_cache_hits.items()
        )
        indexing_items.extend(
            (k, (v[0], v[1], "extracted"))
            for k, v in doc_extractions.items()
        )
        iterator = _progress(
            indexing_items,
            desc="Indexing/embedding",
            total=total_to_index,
        )
        for idx, (item_key, (item, payload, source)) in enumerate(iterator, 1):
            t0 = time.perf_counter()
            try:
                if source == "cached_embedding":
                    n_chunks, n_tables, reason, extraction_stats, quality_grade = (
                        self._index_from_embedding_cache(item, payload)
                    )
                    cache_payload = payload
                else:
                    n_chunks, n_tables, reason, extraction_stats, quality_grade, cache_payload = (
                        self._index_extraction(item, payload)
                    )

                # Aggregate extraction stats
                for key in ["total_pages", "text_pages", "ocr_pages", "empty_pages"]:
                    aggregated_extraction_stats[key] += extraction_stats.get(key, 0)

                # Track quality distribution
                if quality_grade in quality_distribution:
                    quality_distribution[quality_grade] += 1

                if n_chunks > 0:
                    if source == "extracted" and cache_payload:
                        if self._save_embedding_cache(item, ocr_mode, cache_payload):
                            self._delete_extraction_cache_for_item(item, ocr_mode)
                    results.append(IndexResult(
                        item.item_key, item.title, "indexed",
                        n_chunks=n_chunks, n_tables=n_tables,
                        quality_grade=quality_grade))
                else:
                    empty_docs[item.item_key] = self._pdf_hash(item.pdf_path)
                    results.append(IndexResult(
                        item.item_key, item.title, "empty", reason=reason,
                        quality_grade=quality_grade))
                logger.debug(f"Completed {item.item_key}: {n_chunks} chunks, {n_tables} tables, quality {quality_grade}")
            except Exception as e:
                logger.error(f"Failed to index {item.item_key}: {type(e).__name__}: {e}")
                results.append(IndexResult(
                    item.item_key, item.title, "failed",
                    reason=f"{type(e).__name__}: {e}"))

            index_times.append(time.perf_counter() - t0)
            if idx % log_interval == 0 or idx == total_to_index:
                avg_t = sum(index_times) / len(index_times)
                remaining = total_to_index - idx
                eta_secs = avg_t * remaining
                eta_str = f"{eta_secs / 60:.1f}m" if eta_secs >= 60 else f"{eta_secs:.0f}s"
                logger.info(
                    f"Indexing: {idx}/{total_to_index} papers "
                    f"({avg_t:.1f}s avg, ETA {eta_str})"
                )

        phase3_elapsed = time.perf_counter() - phase3_start
        if total_to_index > 0:
            logger.info(
                f"Indexing complete: {total_to_index} papers in "
                f"{phase3_elapsed:.1f}s ({phase3_elapsed / total_to_index:.1f}s avg)"
            )

        self._save_empty_docs(empty_docs)

        counts = {
            "indexed": sum(1 for r in results if r.status == "indexed"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "empty": sum(1 for r in results if r.status == "empty"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "already_indexed": len(indexed_ids),
            "quality_distribution": quality_distribution,
            "extraction_stats": aggregated_extraction_stats,
        }

        # Save config hash after successful indexing
        if counts["indexed"] > 0 or counts["already_indexed"] > 0:
            self._config_hash_path.write_text(current_hash)

        return {"results": results, **counts}

    def _index_from_embedding_cache(self, item: ZoteroItem, cached_payload: dict) -> tuple[int, int, str, dict, str]:
        """Write precomputed cached embeddings into the vector store."""
        records = cached_payload.get("records", {})
        if not isinstance(records, dict):
            raise ValueError("embedding cache payload missing records")

        for key in ("chunks", "tables", "figures"):
            self.store.add_precomputed_records(records.get(key))

        summary = cached_payload.get("summary", {}) or {}
        n_chunks = len((records.get("chunks") or {}).get("ids", []))
        n_tables = len((records.get("tables") or {}).get("ids", []))
        extraction_stats = summary.get("extraction_stats", {}) or {}
        quality_grade = summary.get("quality_grade", "")
        return n_chunks, n_tables, "", extraction_stats, quality_grade

    def _index_document_detailed(self, item: ZoteroItem) -> tuple[int, int, str, dict, str]:
        """
        Extract and index a single document (includes vision resolution).

        For batch indexing use index_all() which batches vision across all docs.
        """
        if item.pdf_path is None or not item.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for {item.item_key}")

        figures_dir = self.config.chroma_db_path.parent / "figures"
        extraction = extract_document(
            item.pdf_path,
            write_images=True,
            images_dir=figures_dir,
            ocr_language=self.config.ocr_language,
            ocr_mode="auto",
            vision_api=self._vision_api,
        )

        # Resolve vision for this single document
        if extraction.pending_vision is not None and self._vision_api:
            from .pdf_processor import resolve_pending_vision
            resolve_pending_vision({item.item_key: extraction}, self._vision_api)

        n_chunks, n_tables, reason, extraction_stats, quality_grade, _cache_payload = (
            self._index_extraction(item, extraction)
        )
        return n_chunks, n_tables, reason, extraction_stats, quality_grade

    def _index_extraction(self, item: ZoteroItem, extraction) -> tuple[int, int, str, dict, str, dict | None]:
        """
        Index a pre-extracted document (vision already resolved).

        Returns:
            (n_chunks, n_tables, reason, extraction_stats, quality_grade, cache_payload)
        """
        if not extraction.pages:
            return 0, 0, "PDF has 0 pages (corrupt or unreadable)", extraction.stats, "F", None

        total_chars = sum(len(p.markdown) for p in extraction.pages)
        quality_grade = extraction.quality_grade

        logger.debug(
            f"  Extracted {len(extraction.pages)} pages, {total_chars} chars "
            f"(text: {extraction.stats['text_pages']}, "
            f"ocr: {extraction.stats['ocr_pages']}, "
            f"empty: {extraction.stats['empty_pages']}, "
            f"quality: {quality_grade})"
        )

        if total_chars == 0:
            return 0, 0, f"{len(extraction.pages)} pages but no text", extraction.stats, quality_grade, None

        # Chunk using the new interface
        chunks = self.chunker.chunk(
            extraction.full_markdown,
            extraction.pages,
            extraction.sections,
        )
        if not chunks:
            return (
                0,
                0,
                f"{len(extraction.pages)} pages, {total_chars} chars but no chunks created",
                extraction.stats,
                quality_grade,
                None,
            )
        logger.debug(f"  Created {len(chunks)} chunks")

        # Look up journal quartile
        journal_quartile = self.journal_ranker.lookup(item.publication)

        # Store text chunks
        doc_meta = {
            "title": item.title,
            "authors": item.authors,
            "year": item.year,
            "citation_key": item.citation_key,
            "publication": item.publication,
            "journal_quartile": journal_quartile or "",
            "doi": item.doi,
            "tags": item.tags,
            "collections": item.collections,
            "pdf_hash": self._pdf_hash(item.pdf_path),
            "quality_grade": quality_grade,
        }
        chunk_records = self.store.create_chunk_records(item.item_key, doc_meta, chunks)
        self.store.add_precomputed_records(chunk_records)

        # Build reference map for table/figure placement
        from ._reference_matcher import match_references
        ref_map = match_references(extraction.full_markdown, chunks, extraction.tables, extraction.figures)

        # Enrich tables/figures with reference context.
        # Only for real captions (Table N / Figure N), not synthetic ones.
        import re
        from .pdf_processor import SYNTHETIC_CAPTION_PREFIX
        from ._reference_matcher import get_reference_context
        _TAB_NUM_RE = re.compile(r"(?:Table|Tab\.?)\s+(\d+)", re.IGNORECASE)
        _FIG_NUM_RE = re.compile(r"(?:Figure|Fig\.?)\s+(\d+)", re.IGNORECASE)
        for table in extraction.tables:
            if table.artifact_type:
                continue  # skip layout artifacts
            if table.caption and not table.caption.startswith(SYNTHETIC_CAPTION_PREFIX):
                m = _TAB_NUM_RE.search(table.caption)
                if m:
                    ctx = get_reference_context(extraction.full_markdown, chunks, ref_map, "table", int(m.group(1)))
                    table.reference_context = ctx
        for fig in extraction.figures:
            if fig.caption and not fig.caption.startswith(SYNTHETIC_CAPTION_PREFIX):
                m = _FIG_NUM_RE.search(fig.caption)
                if m:
                    ctx = get_reference_context(extraction.full_markdown, chunks, ref_map, "figure", int(m.group(1)))
                    fig.reference_context = ctx

        # Store tables if enabled (skip layout artifacts)
        n_tables = 0
        real_tables = [t for t in extraction.tables if not t.artifact_type]
        n_artifacts = len(extraction.tables) - len(real_tables)
        table_records = None
        if real_tables:
            table_records = self.store.create_table_records(
                item.item_key,
                doc_meta,
                real_tables,
                ref_map=ref_map,
            )
            self.store.add_precomputed_records(table_records)
            n_tables = len(real_tables)
        if n_artifacts:
            logger.debug(f"  Skipped {n_artifacts} artifact table(s)")
        logger.debug(f"  Extracted {n_tables} tables")

        # Store figures if enabled
        n_figures = 0
        figure_records = None
        if extraction.figures:
            try:
                figure_records = self.store.create_figure_records(
                    item.item_key,
                    doc_meta,
                    extraction.figures,
                    ref_map=ref_map,
                )
                self.store.add_precomputed_records(figure_records)
                n_figures = len(extraction.figures)
                logger.debug(f"  Extracted {n_figures} figures")
            except Exception as e:
                logger.warning(f"Figure storage failed for {item.item_key}: {e}")

        logger.debug(f"Indexed {item.item_key}: {len(chunks)} chunks, {n_tables} tables, {n_figures} figures, quality {quality_grade}")
        cache_payload = {
            "records": {
                "chunks": chunk_records,
                "tables": table_records,
                "figures": figure_records,
            },
            "summary": {
                "n_chunks": len(chunks),
                "n_tables": n_tables,
                "n_figures": n_figures,
                "extraction_stats": extraction.stats,
                "quality_grade": quality_grade,
            },
        }
        return len(chunks), n_tables, "", extraction.stats, quality_grade, cache_payload

    def index_document(self, item: ZoteroItem) -> int:
        """Index a single document. Returns number of chunks created."""
        n_chunks, _n_tables, _reason, _stats, _quality = self._index_document_detailed(item)
        return n_chunks

    def reindex_document(self, item_key: str) -> int:
        """Re-index a specific document."""
        self.store.delete_document(item_key)
        item = self.zotero.get_item(item_key)
        if item:
            return self.index_document(item)
        return 0

    def get_stats(self) -> dict:
        """Get index statistics."""
        doc_ids = self.store.get_indexed_doc_ids()
        total_chunks = self.store.count()
        return {
            "total_documents": len(doc_ids),
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(total_chunks / len(doc_ids), 1) if doc_ids else 0,
        }

    def get_library_diagnostics(self) -> dict:
        """Delegate to ZoteroClient for library-wide diagnostics."""
        return self.zotero.get_library_diagnostics()

