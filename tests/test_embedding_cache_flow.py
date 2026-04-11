"""Tests for embedding-cache orchestration in Indexer.index_all()."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from deep_zotero.indexer import Indexer
from deep_zotero.models import ZoteroItem


def _make_item(tmp_path: Path, key: str = "DOC1") -> ZoteroItem:
    pdf_path = tmp_path / f"{key}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nembedding-cache-test\n%%EOF")
    return ZoteroItem(
        item_key=key,
        title=f"title-{key}",
        authors="Tester",
        year=2026,
        pdf_path=pdf_path,
        citation_key=f"CITE-{key}",
        publication="Test Journal",
    )


def _build_indexer(tmp_path: Path, item: ZoteroItem) -> Indexer:
    indexer = Indexer.__new__(Indexer)
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    indexer.config = SimpleNamespace(
        chroma_db_path=chroma_dir,
        ocr_language="eng",
        chunk_size=400,
        chunk_overlap=100,
        embedding_provider="local",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        embedding_base_url=None,
    )
    indexer.zotero = MagicMock()
    indexer.zotero.get_all_items_with_pdfs.return_value = [item]
    indexer.embedder = MagicMock()
    indexer.embedder.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    indexer.chunker = MagicMock()
    indexer.journal_ranker = MagicMock()
    indexer.journal_ranker.lookup = MagicMock(return_value="Q1")
    indexer._vision_api = None

    store = MagicMock()
    store.get_indexed_doc_ids = MagicMock(return_value=set())
    store.get_document_meta = MagicMock(return_value=None)
    store.delete_document = MagicMock()
    store.add_precomputed_records = MagicMock()
    indexer.store = store

    indexer._empty_docs_path = chroma_dir / "empty_docs.json"
    indexer._config_hash_path = chroma_dir / "config_hash.txt"
    return indexer


def _cached_payload(item: ZoteroItem) -> dict:
    return {
        "records": {
            "chunks": {
                "ids": [f"{item.item_key}_chunk_0000"],
                "documents": ["cached text"],
                "embeddings": [[0.11, 0.22, 0.33]],
                "metadatas": [{"doc_id": item.item_key, "chunk_type": "text"}],
            },
            "tables": None,
            "figures": None,
        },
        "summary": {
            "n_chunks": 1,
            "n_tables": 0,
            "n_figures": 0,
            "extraction_stats": {
                "total_pages": 1,
                "text_pages": 1,
                "ocr_pages": 0,
                "empty_pages": 0,
            },
            "quality_grade": "A",
        },
    }


def _index_ok_result(payload: dict) -> tuple[int, int, str, dict, str, dict]:
    return (
        1,
        0,
        "",
        {"total_pages": 1, "text_pages": 1, "ocr_pages": 0, "empty_pages": 0},
        "A",
        payload,
    )


def test_embedding_cache_hit_skips_extract_document(tmp_path: Path) -> None:
    item = _make_item(tmp_path, "HIT_SKIP_EXTRACT")
    indexer = _build_indexer(tmp_path, item)
    payload = _cached_payload(item)
    assert indexer._save_embedding_cache(item, "auto", payload)

    with patch("deep_zotero.indexer.extract_document") as mock_extract:
        result = indexer.index_all(ocr_mode="auto")

    assert mock_extract.call_count == 0
    assert result["indexed"] == 1
    assert result["failed"] == 0


def test_embedding_cache_hit_uses_precomputed_store_and_no_embed_calls(tmp_path: Path) -> None:
    item = _make_item(tmp_path, "HIT_PRECOMPUTED")
    indexer = _build_indexer(tmp_path, item)
    payload = _cached_payload(item)
    assert indexer._save_embedding_cache(item, "auto", payload)

    with patch("deep_zotero.indexer.extract_document"):
        indexer.index_all(ocr_mode="auto")

    # No extraction misses -> no embedding health-check call
    indexer.embedder.embed_query.assert_not_called()
    assert call(payload["records"]["chunks"]) in indexer.store.add_precomputed_records.call_args_list


def test_successful_index_saves_embedding_cache_and_deletes_extraction_cache(tmp_path: Path) -> None:
    item = _make_item(tmp_path, "MISS_SAVE_EMBED")
    indexer = _build_indexer(tmp_path, item)
    payload = _cached_payload(item)

    # Seed extraction cache first, then verify it is deleted after embedding cache save.
    indexer._save_extraction_cache(item, "auto", {"seed": True})
    extraction_path = indexer._extraction_cache_path(indexer._extraction_cache_meta(item, "auto"))
    assert extraction_path.exists()

    with patch("deep_zotero.indexer.extract_document", return_value={"dummy": True}):
        with patch.object(Indexer, "_index_extraction", return_value=_index_ok_result(payload)):
            result = indexer.index_all(ocr_mode="auto")

    assert result["indexed"] == 1
    embed_path = indexer._embedding_cache_path(indexer._embedding_cache_meta(item, "auto"))
    assert embed_path.exists()
    assert not extraction_path.exists()


def test_force_reindex_still_uses_embedding_cache_and_skips_extract(tmp_path: Path) -> None:
    item = _make_item(tmp_path, "FORCE_HIT")
    indexer = _build_indexer(tmp_path, item)
    payload = _cached_payload(item)
    assert indexer._save_embedding_cache(item, "auto", payload)
    indexer.store.get_indexed_doc_ids.return_value = {item.item_key}

    with patch("deep_zotero.indexer.extract_document") as mock_extract:
        result = indexer.index_all(force_reindex=True, ocr_mode="auto")

    assert mock_extract.call_count == 0
    indexer.store.delete_document.assert_called_once_with(item.item_key)
    assert result["indexed"] == 1


def test_embedding_cache_meta_mismatch_falls_back_to_extract(tmp_path: Path) -> None:
    item = _make_item(tmp_path, "META_MISMATCH")
    indexer = _build_indexer(tmp_path, item)
    payload = _cached_payload(item)
    assert indexer._save_embedding_cache(item, "auto", payload)

    with patch("deep_zotero.indexer.extract_document", return_value={"dummy": True}) as mock_extract:
        with patch.object(Indexer, "_index_extraction", return_value=_index_ok_result(payload)):
            result = indexer.index_all(ocr_mode="always")  # different mode -> cache meta mismatch

    assert mock_extract.call_count == 1
    indexer.embedder.embed_query.assert_called_once()
    assert result["indexed"] == 1


def test_index_failure_keeps_extraction_cache_file(tmp_path: Path) -> None:
    item = _make_item(tmp_path, "FAIL_KEEP_EXTRACT")
    indexer = _build_indexer(tmp_path, item)

    with patch("deep_zotero.indexer.extract_document", return_value={"dummy": True}):
        with patch.object(Indexer, "_index_extraction", side_effect=RuntimeError("boom")):
            result = indexer.index_all(ocr_mode="auto")

    assert result["failed"] == 1
    extraction_path = indexer._extraction_cache_path(indexer._extraction_cache_meta(item, "auto"))
    assert extraction_path.exists()
