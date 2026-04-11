"""Integration tests for structured metadata search.

These tests verify that author, tag, and collection filtering work correctly.
They use a mock VectorStore to avoid requiring a real ChromaDB instance.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from deep_zotero.models import ZoteroItem, Chunk, StoredChunk
from deep_zotero.vector_store import VectorStore
from deep_zotero.server import _build_chromadb_filters, _apply_text_filters, _has_text_filters


class TestChromaDBFilterBuilder:
    """Test _build_chromadb_filters function (year filters only)."""

    def test_no_filters_returns_none(self):
        """Empty filter set should return None."""
        result = _build_chromadb_filters()
        assert result is None

    def test_single_year_min_filter(self):
        """Single year_min filter."""
        result = _build_chromadb_filters(year_min=2020)
        assert result == {"year": {"$gte": 2020}}

    def test_single_year_max_filter(self):
        """Single year_max filter."""
        result = _build_chromadb_filters(year_max=2023)
        assert result == {"year": {"$lte": 2023}}

    def test_year_range_filter(self):
        """Year range creates $and condition."""
        result = _build_chromadb_filters(year_min=2020, year_max=2023)
        assert result == {
            "$and": [
                {"year": {"$gte": 2020}},
                {"year": {"$lte": 2023}},
            ]
        }


class TestTextFilterApplication:
    """Test _apply_text_filters function (post-retrieval filtering)."""

    def _make_result(self, doc_id, authors, tags, collections):
        """Helper to create a mock result with metadata."""
        return StoredChunk(
            id=f"{doc_id}_chunk_0000",
            text="Test content",
            metadata={
                "doc_id": doc_id,
                "authors": authors,
                "authors_lower": authors.lower(),
                "tags": tags,
                "tags_lower": tags.lower(),
                "collections": collections,
            }
        )

    def test_no_filters_returns_all(self):
        """No filters should return all results."""
        results = [
            self._make_result("doc1", "Smith", "HRV", "Thesis"),
            self._make_result("doc2", "Jones", "ECG", "Other"),
        ]
        filtered = _apply_text_filters(results)
        assert len(filtered) == 2

    def test_author_filter_case_insensitive(self):
        """Author filter should be case-insensitive."""
        results = [
            self._make_result("doc1", "Smith, John", "HRV", ""),
            self._make_result("doc2", "Jones, Alice", "ECG", ""),
        ]
        filtered = _apply_text_filters(results, author="SMITH")
        assert len(filtered) == 1
        assert filtered[0].metadata["doc_id"] == "doc1"

    def test_author_filter_substring(self):
        """Author filter should match substrings."""
        results = [
            self._make_result("doc1", "Smith, John; Jones, Alice", "", ""),
        ]
        filtered = _apply_text_filters(results, author="jones")
        assert len(filtered) == 1

    def test_tag_filter_case_insensitive(self):
        """Tag filter should be case-insensitive."""
        results = [
            self._make_result("doc1", "Author", "HRV; methodology", ""),
            self._make_result("doc2", "Author", "ECG; signal", ""),
        ]
        filtered = _apply_text_filters(results, tag="METHODOLOGY")
        assert len(filtered) == 1
        assert filtered[0].metadata["doc_id"] == "doc1"

    def test_collection_filter_substring(self):
        """Collection filter should match substrings."""
        results = [
            self._make_result("doc1", "Author", "", "Thesis Chapter 5; Background"),
            self._make_result("doc2", "Author", "", "Other Collection"),
        ]
        filtered = _apply_text_filters(results, collection="Chapter 5")
        assert len(filtered) == 1
        assert filtered[0].metadata["doc_id"] == "doc1"

    def test_combined_filters_and_logic(self):
        """Multiple filters should use AND logic."""
        results = [
            self._make_result("doc1", "Smith", "HRV", "Thesis"),  # All match
            self._make_result("doc2", "Smith", "ECG", "Thesis"),  # Wrong tag
            self._make_result("doc3", "Jones", "HRV", "Thesis"),  # Wrong author
        ]
        filtered = _apply_text_filters(results, author="smith", tag="hrv")
        assert len(filtered) == 1
        assert filtered[0].metadata["doc_id"] == "doc1"

    def test_no_match_returns_empty(self):
        """No matches should return empty list."""
        results = [
            self._make_result("doc1", "Smith", "HRV", "Thesis"),
        ]
        filtered = _apply_text_filters(results, author="nonexistent")
        assert len(filtered) == 0


class TestZoteroItemMetadataFields:
    """Test that ZoteroItem has the new metadata fields."""

    def test_zotero_item_has_doi_field(self):
        """ZoteroItem should have doi field."""
        item = ZoteroItem(
            item_key="ABC123",
            title="Test Paper",
            authors="Smith, J.",
            year=2020,
            pdf_path=None,
            doi="10.1234/test",
        )
        assert item.doi == "10.1234/test"

    def test_zotero_item_has_tags_field(self):
        """ZoteroItem should have tags field."""
        item = ZoteroItem(
            item_key="ABC123",
            title="Test Paper",
            authors="Smith, J.",
            year=2020,
            pdf_path=None,
            tags="HRV; methodology; review",
        )
        assert item.tags == "HRV; methodology; review"

    def test_zotero_item_has_collections_field(self):
        """ZoteroItem should have collections field."""
        item = ZoteroItem(
            item_key="ABC123",
            title="Test Paper",
            authors="Smith, J.",
            year=2020,
            pdf_path=None,
            collections="Thesis Chapter 5; Background Reading",
        )
        assert item.collections == "Thesis Chapter 5; Background Reading"

    def test_zotero_item_defaults(self):
        """New fields should have empty string defaults."""
        item = ZoteroItem(
            item_key="ABC123",
            title="Test Paper",
            authors="Smith, J.",
            year=2020,
            pdf_path=None,
        )
        assert item.doi == ""
        assert item.tags == ""
        assert item.collections == ""


class TestVectorStoreMetadata:
    """Test that VectorStore stores the new metadata fields."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder that returns fixed-size embeddings."""
        embedder = Mock()
        embedder.dimensions = 768  # Required for VectorStore dimension tracking
        embedder.embed = Mock(return_value=[[0.1] * 768])
        embedder.embed_query = Mock(return_value=[0.1] * 768)
        return embedder

    @pytest.fixture
    def temp_store(self, mock_embedder, tmp_path):
        """Create a temporary VectorStore for testing."""
        return VectorStore(tmp_path / "test_chroma", mock_embedder)

    def test_add_chunks_stores_new_metadata(self, temp_store):
        """add_chunks should store doi, tags, collections in metadata."""
        doc_meta = {
            "title": "Test Paper",
            "authors": "Smith, J.; Jones, A.",
            "year": 2020,
            "citation_key": "smith2020test",
            "publication": "Test Journal",
            "doi": "10.1234/test.2020",
            "tags": "HRV; methodology",
            "collections": "Thesis Chapter 5",
            "journal_quartile": "Q1",
        }
        chunks = [
            Chunk(
                text="This is test content.",
                chunk_index=0,
                page_num=1,
                char_start=0,
                char_end=21,
                section="introduction",
            )
        ]

        temp_store.add_chunks("test_doc_001", doc_meta, chunks)

        # Retrieve and verify
        results = temp_store.collection.get(
            ids=["test_doc_001_chunk_0000"],
            include=["metadatas"]
        )

        assert results["metadatas"], "Should have metadata"
        meta = results["metadatas"][0]

        # Verify new fields are stored
        assert meta["doi"] == "10.1234/test.2020", "DOI should be stored"
        assert meta["tags"] == "HRV; methodology", "Tags should be stored"
        assert meta["tags_lower"] == "hrv; methodology", "Lowercase tags should be stored"
        assert meta["collections"] == "Thesis Chapter 5", "Collections should be stored"

        # Verify lowercase author field for searching
        assert meta["authors_lower"] == "smith, j.; jones, a.", "Lowercase authors should be stored"

    def test_year_filter_works_in_chromadb(self, temp_store):
        """Year filters (ChromaDB-native) should work correctly."""
        # Add documents with different years
        for year in [2018, 2020, 2022]:
            doc_meta = {
                "title": f"Paper {year}",
                "authors": "Author",
                "year": year,
                "doi": "",
                "tags": "",
                "collections": "",
            }
            chunks = [
                Chunk(text=f"Content from {year}", chunk_index=0, page_num=1, char_start=0, char_end=20)
            ]
            temp_store.add_chunks(f"doc_{year}", doc_meta, chunks)

        # Search with year filter
        results = temp_store.search(
            query="content",
            top_k=10,
            filters={"year": {"$gte": 2020}}
        )

        assert len(results) == 2, f"Should find 2020 and 2022 papers, found {len(results)}"
        years = {r.metadata["year"] for r in results}
        assert 2018 not in years, "2018 should be filtered out"
        assert 2020 in years and 2022 in years

    def test_metadata_stored_correctly(self, temp_store):
        """Verify all new metadata fields are stored and retrievable."""
        doc_meta = {
            "title": "Full Metadata Test",
            "authors": "Smith, John; Jones, Alice",
            "year": 2021,
            "doi": "10.1234/test",
            "tags": "HRV; methodology",
            "collections": "Thesis Chapter 5",
        }
        chunks = [
            Chunk(text="Test content", chunk_index=0, page_num=1, char_start=0, char_end=12)
        ]
        temp_store.add_chunks("full_meta_doc", doc_meta, chunks)

        # Retrieve and verify
        results = temp_store.search(query="test", top_k=1)
        assert len(results) == 1

        meta = results[0].metadata
        assert meta["authors"] == "Smith, John; Jones, Alice"
        assert meta["authors_lower"] == "smith, john; jones, alice"
        assert meta["tags"] == "HRV; methodology"
        assert meta["tags_lower"] == "hrv; methodology"
        assert meta["collections"] == "Thesis Chapter 5"
        assert meta["doi"] == "10.1234/test"

    def test_text_filters_work_with_stored_data(self, temp_store):
        """Test that post-retrieval text filtering works with real stored data."""
        # Add multiple documents
        docs = [
            ("doc1", "Smith, John", "HRV; methodology", "Thesis"),
            ("doc2", "Jones, Alice", "ECG; processing", "Other"),
            ("doc3", "Smith, Jane", "HRV; validation", "Thesis"),
        ]

        for doc_id, authors, tags, collections in docs:
            doc_meta = {
                "title": f"Paper by {authors}",
                "authors": authors,
                "year": 2021,
                "doi": "",
                "tags": tags,
                "collections": collections,
            }
            chunks = [
                Chunk(text="Research content", chunk_index=0, page_num=1, char_start=0, char_end=16)
            ]
            temp_store.add_chunks(doc_id, doc_meta, chunks)

        # Get all results
        all_results = temp_store.search(query="research", top_k=10)
        assert len(all_results) == 3, "Should have all 3 documents"

        # Apply author filter
        smith_results = _apply_text_filters(all_results, author="smith")
        assert len(smith_results) == 2, "Should find 2 Smiths"

        # Apply tag filter
        hrv_results = _apply_text_filters(all_results, tag="hrv")
        assert len(hrv_results) == 2, "Should find 2 HRV papers"

        # Apply combined filters
        smith_hrv = _apply_text_filters(all_results, author="smith", tag="hrv")
        assert len(smith_hrv) == 2, "Both Smiths have HRV tag"

        # Apply collection filter
        thesis_results = _apply_text_filters(all_results, collection="Thesis")
        assert len(thesis_results) == 2, "Should find 2 Thesis papers"


class TestGetDocumentMeta:
    """Test the get_document_meta helper method."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = Mock()
        embedder.dimensions = 768  # Required for VectorStore dimension tracking
        # Return correct number of embeddings based on input
        embedder.embed = Mock(side_effect=lambda texts, **kw: [[0.1] * 768 for _ in texts])
        return embedder

    @pytest.fixture
    def temp_store(self, mock_embedder, tmp_path):
        return VectorStore(tmp_path / "test_chroma", mock_embedder)

    def test_get_document_meta_returns_first_chunk_metadata(self, temp_store):
        """get_document_meta should return metadata from first chunk."""
        doc_meta = {
            "title": "Test Paper",
            "authors": "Smith",
            "year": 2020,
            "doi": "10.1234/test",
            "tags": "tag1; tag2",
            "collections": "col1",
        }
        chunks = [
            Chunk(text="First chunk", chunk_index=0, page_num=1, char_start=0, char_end=11),
            Chunk(text="Second chunk", chunk_index=1, page_num=1, char_start=12, char_end=24),
        ]
        temp_store.add_chunks("test_doc", doc_meta, chunks)

        result = temp_store.get_document_meta("test_doc")

        assert result is not None, "Should return metadata"
        assert result["doc_title"] == "Test Paper"
        assert result["doi"] == "10.1234/test"
        assert result["tags"] == "tag1; tag2"

    def test_get_document_meta_returns_none_for_missing(self, temp_store):
        """get_document_meta should return None for non-existent doc."""
        result = temp_store.get_document_meta("nonexistent_doc")
        assert result is None


class TestServerToolsAcceptFilters:
    """Test that server tools accept the new filter parameters.

    FastMCP wraps functions so we check the tool's description instead
    of using inspect.signature directly.
    """

    def test_search_papers_has_filter_params_in_description(self):
        """search_papers tool description should mention filter params."""
        from deep_zotero.server import search_papers

        desc = getattr(search_papers, "description", None) or (search_papers.__doc__ or "")

        assert "author" in desc.lower(), "search_papers description should mention 'author'"
        assert "tag" in desc.lower(), "search_papers description should mention 'tag'"
        assert "collection" in desc.lower(), "search_papers description should mention 'collection'"

    def test_search_topic_has_filter_params_in_description(self):
        """search_topic tool description should mention filter params."""
        from deep_zotero.server import search_topic

        desc = getattr(search_topic, "description", None) or (search_topic.__doc__ or "")

        assert "author" in desc.lower(), "search_topic description should mention 'author'"
        assert "tag" in desc.lower(), "search_topic description should mention 'tag'"
        assert "collection" in desc.lower(), "search_topic description should mention 'collection'"

    def test_search_tables_has_filter_params_in_description(self):
        """search_tables tool description should mention filter params."""
        from deep_zotero.server import search_tables

        desc = getattr(search_tables, "description", None) or (search_tables.__doc__ or "")

        assert "author" in desc.lower(), "search_tables description should mention 'author'"
        assert "tag" in desc.lower(), "search_tables description should mention 'tag'"
        assert "collection" in desc.lower(), "search_tables description should mention 'collection'"

    def test_build_chromadb_filters_only_handles_years(self):
        """_build_chromadb_filters should only handle year filters."""
        from deep_zotero.server import _build_chromadb_filters

        # Year filters work
        result = _build_chromadb_filters(year_min=2020, year_max=2023)
        assert result is not None
        assert "year" in str(result)

        # No text filter parameters in signature
        import inspect
        sig = inspect.signature(_build_chromadb_filters)
        params = list(sig.parameters.keys())
        assert "author" not in params, "ChromaDB filters should not have author (handled by text filter)"
        assert "tag" not in params, "ChromaDB filters should not have tag"
        assert "collection" not in params, "ChromaDB filters should not have collection"

    def test_apply_text_filters_signature(self):
        """_apply_text_filters should have the expected signature."""
        from deep_zotero.server import _apply_text_filters
        import inspect

        sig = inspect.signature(_apply_text_filters)
        params = list(sig.parameters.keys())

        assert "results" in params, "_apply_text_filters should accept results"
        assert "author" in params, "_apply_text_filters should accept author"
        assert "tag" in params, "_apply_text_filters should accept tag"
        assert "collection" in params, "_apply_text_filters should accept collection"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
