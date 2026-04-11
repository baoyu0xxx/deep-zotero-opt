"""
Shared pytest fixtures for deep-zotero tests.

All fixtures that need to be shared across test modules should be defined here.
"""
from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PYTEST_TMP_ROOT = Path(
    os.environ.get("DEEP_ZOTERO_TEST_TMP", _PROJECT_ROOT / ".pytest-tmp")
)
_PYTEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
for _tmp_env in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_tmp_env] = str(_PYTEST_TMP_ROOT)

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "papers"
_PAPER_NAMES = ["noname1.pdf", "noname2.pdf", "noname3.pdf"]


# =============================================================================
# Path fixtures
# =============================================================================

@pytest.fixture
def real_papers_dir() -> Path:
    """Path to the directory containing real academic papers for testing."""
    path = _FIXTURES_DIR
    assert path.exists(), f"Real papers directory not found: {path}"
    return path


# =============================================================================
# Session-scoped extraction fixtures (run once, shared by all tests)
# =============================================================================

@pytest.fixture(scope="session")
def extracted_papers():
    """Extract all fixture PDFs once per session. Returns dict keyed by filename."""
    from deep_zotero.pdf_processor import extract_document

    return {
        name: extract_document(_FIXTURES_DIR / name)
        for name in _PAPER_NAMES
    }


@pytest.fixture(scope="session")
def chunked_papers(extracted_papers):
    """Chunk all fixture PDFs once per session. Returns dict keyed by filename."""
    from deep_zotero.chunker import Chunker

    chunker = Chunker(chunk_size=400, overlap=100)
    return {
        name: chunker.chunk(ex.full_markdown, ex.pages, ex.sections)
        for name, ex in extracted_papers.items()
    }


# =============================================================================
# Database fixtures
# =============================================================================

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Temporary directory for ChromaDB during tests."""
    db_path = tmp_path / "chroma"
    db_path.mkdir(parents=True, exist_ok=True)
    return db_path


@pytest.fixture
def mock_zotero_db(tmp_path: Path) -> Path:
    """Create a mock Zotero database with sample data for testing.

    This creates a minimal Zotero SQLite schema with the tables needed
    for full-text search testing.
    """
    db_path = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db_path)

    # Create minimal Zotero schema
    conn.executescript("""
        -- Core item tables
        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY,
            itemTypeID INTEGER,
            key TEXT UNIQUE,
            libraryID INTEGER,
            dateAdded TEXT,
            dateModified TEXT
        );

        -- Attachment relationship
        CREATE TABLE itemAttachments (
            itemID INTEGER PRIMARY KEY,
            parentItemID INTEGER,
            path TEXT,
            contentType TEXT,
            FOREIGN KEY (itemID) REFERENCES items(itemID),
            FOREIGN KEY (parentItemID) REFERENCES items(itemID)
        );

        -- Full-text search tables (Zotero's custom FTS)
        CREATE TABLE fulltextWords (
            wordID INTEGER PRIMARY KEY,
            word TEXT UNIQUE
        );

        CREATE TABLE fulltextItemWords (
            wordID INTEGER,
            itemID INTEGER,
            PRIMARY KEY (wordID, itemID)
        );

        CREATE TABLE fulltextItems (
            itemID INTEGER PRIMARY KEY,
            indexedPages INTEGER,
            totalPages INTEGER,
            indexedChars INTEGER,
            version INTEGER
        );

        -- Item metadata
        CREATE TABLE itemData (
            itemID INTEGER,
            fieldID INTEGER,
            valueID INTEGER,
            PRIMARY KEY (itemID, fieldID)
        );

        CREATE TABLE itemDataValues (
            valueID INTEGER PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE fields (
            fieldID INTEGER PRIMARY KEY,
            fieldName TEXT
        );

        -- Creators
        CREATE TABLE creators (
            creatorID INTEGER PRIMARY KEY,
            firstName TEXT,
            lastName TEXT
        );

        CREATE TABLE itemCreators (
            itemID INTEGER,
            creatorID INTEGER,
            orderIndex INTEGER
        );

        -- Insert some test data
        INSERT INTO items (itemID, key) VALUES (1, 'ABC12345');
        INSERT INTO items (itemID, key) VALUES (2, 'DEF67890');
        INSERT INTO items (itemID, key) VALUES (3, 'GHI11111');

        -- Attachment items
        INSERT INTO items (itemID, key) VALUES (101, 'ATT00001');
        INSERT INTO items (itemID, key) VALUES (102, 'ATT00002');

        INSERT INTO itemAttachments (itemID, parentItemID, path)
        VALUES (101, 1, 'storage:test1.pdf');
        INSERT INTO itemAttachments (itemID, parentItemID, path)
        VALUES (102, 2, 'storage:test2.pdf');

        -- Full-text words for testing boolean search
        INSERT INTO fulltextWords (wordID, word) VALUES (1, 'heart');
        INSERT INTO fulltextWords (wordID, word) VALUES (2, 'rate');
        INSERT INTO fulltextWords (wordID, word) VALUES (3, 'variability');
        INSERT INTO fulltextWords (wordID, word) VALUES (4, 'ecg');
        INSERT INTO fulltextWords (wordID, word) VALUES (5, 'electrode');

        -- Item 1 (ABC12345) contains: heart, rate, variability, ecg
        INSERT INTO fulltextItemWords (wordID, itemID) VALUES (1, 101);
        INSERT INTO fulltextItemWords (wordID, itemID) VALUES (2, 101);
        INSERT INTO fulltextItemWords (wordID, itemID) VALUES (3, 101);
        INSERT INTO fulltextItemWords (wordID, itemID) VALUES (4, 101);

        -- Item 2 (DEF67890) contains: heart, electrode
        INSERT INTO fulltextItemWords (wordID, itemID) VALUES (1, 102);
        INSERT INTO fulltextItemWords (wordID, itemID) VALUES (5, 102);

        INSERT INTO fulltextItems (itemID, indexedPages, totalPages) VALUES (101, 10, 10);
        INSERT INTO fulltextItems (itemID, indexedPages, totalPages) VALUES (102, 5, 5);
    """)

    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# Config fixtures
# =============================================================================

@pytest.fixture
def mock_config(temp_db_path: Path, tmp_path: Path):
    """Create a test configuration.

    Uses local embeddings to avoid needing API keys in tests.
    """
    from deep_zotero.config import Config

    # Create minimal Zotero directory structure for config validation
    zotero_dir = tmp_path / "zotero"
    zotero_dir.mkdir(exist_ok=True)
    (zotero_dir / "zotero.sqlite").touch()

    return Config(
        zotero_data_dir=zotero_dir,
        chroma_db_path=temp_db_path,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimensions=384,
        chunk_size=400,
        chunk_overlap=100,
        gemini_api_key=None,
        embedding_provider="local",
        embedding_timeout=120.0,
        embedding_max_retries=3,
        rerank_alpha=0.7,
        rerank_section_weights=None,
        rerank_journal_weights=None,
        rerank_enabled=True,
        oversample_multiplier=3,
        oversample_topic_factor=5,
        stats_sample_limit=10000,
        ocr_language="eng",
        openalex_email=None,
        vision_enabled=False,
        vision_model="claude-haiku-4-5-20251001",
        anthropic_api_key=None,
    )


# =============================================================================
# Journal ranker fixtures
# =============================================================================

@pytest.fixture
def test_scimago_csv(tmp_path: Path) -> Path:
    """Create a test SCImago CSV file with known journals."""
    csv_path = tmp_path / "test_scimago.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title_normalized", "quartile"])
        # Q1 journals
        writer.writerow(["nature", "Q1"])
        writer.writerow(["science", "Q1"])
        writer.writerow(["cell", "Q1"])
        writer.writerow(["the lancet", "Q1"])
        writer.writerow(["new england journal of medicine", "Q1"])
        writer.writerow(["ieee transactions on biomedical engineering", "Q1"])
        writer.writerow(["circulation", "Q1"])
        writer.writerow(["plos one", "Q1"])
        # Q2 journals
        writer.writerow(["journal of physiology", "Q2"])
        writer.writerow(["physiological measurement", "Q2"])
        writer.writerow(["medical engineering physics", "Q2"])
        # Q3 journals
        writer.writerow(["journal of medical systems", "Q3"])
        writer.writerow(["biomedical signal processing and control", "Q3"])
        # Q4 journals
        writer.writerow(["medical hypotheses", "Q4"])
        writer.writerow(["journal of low power electronics", "Q4"])
    return csv_path


@pytest.fixture
def test_overrides_csv(tmp_path: Path) -> Path:
    """Create a test journal overrides CSV file."""
    csv_path = tmp_path / "journal_overrides.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("# Manual corrections for fuzzy matching mistakes\n")
        f.write("# Format: input_title,correct_quartile\n")
        f.write("IEEE Transactions on Biomedical Circuits and Systems,Q1\n")
        f.write("Journal of Neural Engineering,Q1\n")
    return csv_path
