"""ChromaDB vector storage with chunk management."""
import logging
import re
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import TYPE_CHECKING
from .models import Chunk, StoredChunk
from .interfaces import EmbedderProtocol

if TYPE_CHECKING:
    from .models import ExtractedTable

logger = logging.getLogger(__name__)


def _ref_chunk_index(ref_map: dict, element_type: str, item) -> int:
    """Look up chunk_index from ref_map using caption number."""
    caption = getattr(item, 'caption', None)
    if caption:
        m = re.search(r"(\d+)", caption)
        if m:
            return ref_map.get((element_type, int(m.group(1))), -1)
    return -1


class EmbeddingDimensionMismatchError(Exception):
    """Raised when embedder dimensions don't match existing index."""


class VectorStore:
    """
    ChromaDB-backed vector store for document chunks.

    Handles:
    - Adding chunks with metadata
    - Semantic search with filters
    - Adjacent chunk retrieval for context expansion
    - Document-level operations (delete, list)
    """

    def __init__(self, db_path: Path, embedder: EmbedderProtocol):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get embedder dimensions
        embedder_dims = getattr(embedder, 'dimensions', None)

        # Check if collection exists and has data
        try:
            existing = self.client.get_collection("chunks")
            existing_count = existing.count()
            if existing_count > 0 and embedder_dims is not None:
                # Check stored dimension in metadata
                stored_dims = existing.metadata.get("embedding_dimensions")
                if stored_dims is not None and stored_dims != embedder_dims:
                    raise EmbeddingDimensionMismatchError(
                        f"Embedding dimension mismatch: index has {stored_dims} dimensions "
                        f"but current embedder uses {embedder_dims} dimensions. "
                        f"Delete the index and reindex with --force, or switch back to "
                        f"the original embedding provider.\n"
                        f"Index path: {db_path}"
                    )
        except (ValueError, chromadb.errors.NotFoundError):
            # Collection doesn't exist yet, that's fine
            pass

        # Create or get collection with dimension metadata
        metadata = {"hnsw:space": "cosine"}
        if embedder_dims is not None:
            metadata["embedding_dimensions"] = embedder_dims

        self.collection = self.client.get_or_create_collection(
            name="chunks",
            metadata=metadata
        )
        self.embedder = embedder

    @staticmethod
    def _validate_precomputed_embeddings(
        documents: list[str], embeddings: list[list[float]] | None
    ) -> list[list[float]]:
        if embeddings is None:
            raise ValueError("precomputed embeddings are required")
        if len(documents) != len(embeddings):
            raise ValueError(
                f"embedding count mismatch: {len(documents)} documents vs "
                f"{len(embeddings)} embeddings"
            )
        return embeddings

    def add_precomputed_records(self, records: dict | None) -> None:
        """Insert precomputed records into ChromaDB without embedding again."""
        if not records:
            return
        ids = records.get("ids") or []
        documents = records.get("documents") or []
        embeddings = records.get("embeddings") or []
        metadatas = records.get("metadatas") or []
        if not ids:
            return
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def create_chunk_records(
        self,
        doc_id: str,
        doc_meta: dict,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> dict | None:
        """Build text-chunk records for collection.add() and cache payloads."""
        if not chunks:
            return None

        ids = [f"{doc_id}_chunk_{c.chunk_index:04d}" for c in chunks]
        texts = [c.text for c in chunks]
        if embeddings is None:
            embeddings = self.embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT")
        embeddings = self._validate_precomputed_embeddings(texts, embeddings)

        metadatas = [
            {
                "doc_id": doc_id,
                "doc_title": doc_meta.get("title", ""),
                "authors": doc_meta.get("authors", ""),
                "authors_lower": doc_meta.get("authors", "").lower(),  # For case-insensitive search
                "year": doc_meta.get("year") or 0,
                "citation_key": doc_meta.get("citation_key", ""),
                "publication": doc_meta.get("publication", ""),
                "doi": doc_meta.get("doi", ""),
                "tags": doc_meta.get("tags", ""),
                "tags_lower": doc_meta.get("tags", "").lower(),  # For case-insensitive search
                "collections": doc_meta.get("collections", ""),
                "pdf_hash": doc_meta.get("pdf_hash", ""),  # Feature 6: for update detection
                "quality_grade": doc_meta.get("quality_grade", ""),  # Feature 11: extraction quality
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
                "total_chunks": len(chunks),
                "char_start": c.char_start,
                "char_end": c.char_end,
                "section": c.section,
                "section_confidence": c.section_confidence,
                "journal_quartile": doc_meta.get("journal_quartile", ""),
                "chunk_type": "text",
            }
            for c in chunks
        ]
        return {
            "ids": ids,
            "documents": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }

    def add_chunks(self, doc_id: str, doc_meta: dict, chunks: list[Chunk]) -> None:
        """
        Add all chunks for a document.

        Args:
            doc_id: Unique document identifier (Zotero item key)
            doc_meta: Document metadata (title, authors, year)
            chunks: List of Chunk objects to store
        """
        records = self.create_chunk_records(doc_id, doc_meta, chunks)
        self.add_precomputed_records(records)

    def add_precomputed_chunks(
        self,
        doc_id: str,
        doc_meta: dict,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add text chunks using caller-provided embeddings."""
        records = self.create_chunk_records(doc_id, doc_meta, chunks, embeddings=embeddings)
        self.add_precomputed_records(records)

    def create_table_records(
        self,
        doc_id: str,
        doc_meta: dict,
        tables: list["ExtractedTable"],
        ref_map: dict[tuple[str, int], int] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> dict | None:
        """Build table records for collection.add() and cache payloads."""
        if not tables:
            return None

        ids = [
            f"{doc_id}_table_{t.page_num:04d}_{t.table_index:02d}"
            for t in tables
        ]
        texts = [t.to_markdown() for t in tables]
        if embeddings is None:
            embeddings = self.embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT")
        embeddings = self._validate_precomputed_embeddings(texts, embeddings)

        metadatas = [
            {
                "doc_id": doc_id,
                "doc_title": doc_meta.get("title", ""),
                "authors": doc_meta.get("authors", ""),
                "authors_lower": doc_meta.get("authors", "").lower(),
                "year": doc_meta.get("year") or 0,
                "citation_key": doc_meta.get("citation_key", ""),
                "publication": doc_meta.get("publication", ""),
                "doi": doc_meta.get("doi", ""),
                "tags": doc_meta.get("tags", ""),
                "tags_lower": doc_meta.get("tags", "").lower(),
                "collections": doc_meta.get("collections", ""),
                "journal_quartile": doc_meta.get("journal_quartile", ""),
                "pdf_hash": doc_meta.get("pdf_hash", ""),  # Feature 6: for update detection
                "quality_grade": doc_meta.get("quality_grade", ""),  # Feature 11: extraction quality
                "page_num": t.page_num,
                "chunk_index": _ref_chunk_index(ref_map, "table", t) if ref_map else -1,
                "chunk_type": "table",
                "table_index": t.table_index,
                "table_caption": t.caption or "",  # None -> empty string for ChromaDB
                "table_num_rows": t.num_rows,
                "table_num_cols": t.num_cols,
                "reference_context": getattr(t, 'reference_context', '') or "",
                # Section detection doesn't apply to tables
                "section": "table",
                "section_confidence": 1.0,
            }
            for t in tables
        ]
        return {
            "ids": ids,
            "documents": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }

    def add_tables(
        self,
        doc_id: str,
        doc_meta: dict,
        tables: list["ExtractedTable"],
        ref_map: dict[tuple[str, int], int] | None = None,
    ) -> None:
        """
        Add table chunks for a document.

        Tables are stored as separate chunks with markdown representation.
        Chunk IDs use format: {doc_id}_table_{page:04d}_{table_idx:02d}

        Args:
            doc_id: Unique document identifier (Zotero item key)
            doc_meta: Document metadata (title, authors, year, etc.)
            tables: List of ExtractedTable objects to store
        """
        records = self.create_table_records(
            doc_id,
            doc_meta,
            tables,
            ref_map=ref_map,
        )
        self.add_precomputed_records(records)

    def add_precomputed_tables(
        self,
        doc_id: str,
        doc_meta: dict,
        tables: list["ExtractedTable"],
        embeddings: list[list[float]],
        ref_map: dict[tuple[str, int], int] | None = None,
    ) -> None:
        """Add tables using caller-provided embeddings."""
        records = self.create_table_records(
            doc_id,
            doc_meta,
            tables,
            ref_map=ref_map,
            embeddings=embeddings,
        )
        self.add_precomputed_records(records)

    def create_figure_records(
        self,
        doc_id: str,
        doc_meta: dict,
        figures: list,
        ref_map: dict[tuple[str, int], int] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> dict | None:
        """Build figure records for collection.add() and cache payloads."""
        if not figures:
            return None

        ids = []
        documents = []
        metadatas = []

        for fig in figures:
            chunk_id = f"{doc_id}_fig_{fig.page_num:03d}_{fig.figure_index:02d}"

            # Use caption for embedding, or fallback text for orphans
            text = fig.to_searchable_text()

            metadata = {
                "doc_id": doc_id,
                "doc_title": doc_meta.get("title", ""),
                "authors": doc_meta.get("authors", ""),
                "authors_lower": doc_meta.get("authors", "").lower(),
                "year": doc_meta.get("year") or 0,
                "citation_key": doc_meta.get("citation_key", ""),
                "publication": doc_meta.get("publication", ""),
                "doi": doc_meta.get("doi", ""),
                "tags": doc_meta.get("tags", ""),
                "tags_lower": doc_meta.get("tags", "").lower(),
                "collections": doc_meta.get("collections", ""),
                "journal_quartile": doc_meta.get("journal_quartile", ""),
                "pdf_hash": doc_meta.get("pdf_hash", ""),
                "quality_grade": doc_meta.get("quality_grade", ""),
                "chunk_type": "figure",
                "page_num": fig.page_num,
                "chunk_index": _ref_chunk_index(ref_map, "figure", fig) if ref_map else -1,
                "figure_index": fig.figure_index,
                "caption": fig.caption or "",  # Empty string for orphans
                "image_path": str(fig.image_path) if fig.image_path else "",
                "reference_context": getattr(fig, 'reference_context', '') or "",
                # Section detection doesn't apply to figures
                "section": "figure",
                "section_confidence": 1.0,
            }

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(metadata)

        if not ids:
            return None

        if embeddings is None:
            embeddings = self.embedder.embed(documents, task_type="RETRIEVAL_DOCUMENT")
        embeddings = self._validate_precomputed_embeddings(documents, embeddings)
        return {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }

    def add_figures(
        self,
        doc_id: str,
        doc_meta: dict,
        figures: list,
        ref_map: dict[tuple[str, int], int] | None = None,
    ) -> None:
        """Add figure chunks to the store.

        Figures are stored as separate chunks with caption as text.
        Chunk IDs use format: {doc_id}_fig_{page:03d}_{fig_idx:02d}

        Args:
            doc_id: Document ID (Zotero item key)
            doc_meta: Document-level metadata
            figures: List of ExtractedFigure objects
        """
        records = self.create_figure_records(
            doc_id,
            doc_meta,
            figures,
            ref_map=ref_map,
        )
        self.add_precomputed_records(records)

    def add_precomputed_figures(
        self,
        doc_id: str,
        doc_meta: dict,
        figures: list,
        embeddings: list[list[float]],
        ref_map: dict[tuple[str, int], int] | None = None,
    ) -> None:
        """Add figures using caller-provided embeddings."""
        records = self.create_figure_records(
            doc_id,
            doc_meta,
            figures,
            ref_map=ref_map,
            embeddings=embeddings,
        )
        self.add_precomputed_records(records)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[StoredChunk]:
        """
        Search for similar chunks.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional ChromaDB where clause

        Returns:
            List of StoredChunk objects sorted by similarity
        """
        # Use RETRIEVAL_QUERY task type for asymmetric search
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                chunks.append(StoredChunk(
                    id=chunk_id,
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1 - results['distances'][0][i]  # Convert distance to similarity
                ))
        return chunks

    def get_adjacent_chunks(
        self,
        doc_id: str,
        chunk_index: int,
        window: int = 2
    ) -> list[StoredChunk]:
        """
        Get chunks adjacent to a given chunk for context expansion.

        Args:
            doc_id: Document ID
            chunk_index: Center chunk index
            window: Number of chunks before/after to include

        Returns:
            List of chunks sorted by chunk_index
        """
        results = self.collection.get(
            where={
                "$and": [
                    {"doc_id": {"$eq": doc_id}},
                    {"chunk_index": {"$gte": chunk_index - window}},
                    {"chunk_index": {"$lte": chunk_index + window}}
                ]
            },
            include=["documents", "metadatas"]
        )

        chunks = []
        if results['ids']:
            for i, chunk_id in enumerate(results['ids']):
                chunks.append(StoredChunk(
                    id=chunk_id,
                    text=results['documents'][i],
                    metadata=results['metadatas'][i]
                ))

        return sorted(chunks, key=lambda c: c.metadata['chunk_index'])

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks for a document."""
        self.collection.delete(where={"doc_id": {"$eq": doc_id}})

    def get_indexed_doc_ids(self) -> set[str]:
        """Get set of all indexed document IDs.

        Memory-efficient: extracts doc_id from chunk IDs without loading metadata.
        Handles text chunks ({doc_id}_chunk_{index:04d}),
        table chunks ({doc_id}_table_{page:04d}_{table_idx:02d}),
        and figure chunks ({doc_id}_fig_{page:03d}_{fig_idx:02d}).
        """
        results = self.collection.get(include=[])  # IDs only, no documents/metadata
        if not results['ids']:
            return set()

        doc_ids = set()
        for chunk_id in results['ids']:
            # Handle text chunks, table chunks, and figure chunks
            if '_chunk_' in chunk_id:
                parts = chunk_id.rsplit('_chunk_', 1)
            elif '_table_' in chunk_id:
                parts = chunk_id.rsplit('_table_', 1)
            elif '_fig_' in chunk_id:
                parts = chunk_id.rsplit('_fig_', 1)
            else:
                continue
            if len(parts) == 2:
                doc_ids.add(parts[0])
        return doc_ids

    def count(self) -> int:
        """Return total number of chunks."""
        return self.collection.count()

    def get_document_meta(self, doc_id: str) -> dict | None:
        """Get metadata for a document's first chunk.

        Useful for checking stored metadata (e.g., pdf_hash) without
        loading full document content.

        Args:
            doc_id: Document ID to look up

        Returns:
            Metadata dict from first chunk, or None if not found
        """
        results = self.collection.get(
            where={"doc_id": {"$eq": doc_id}},
            limit=1,
            include=["metadatas"]
        )
        if results["metadatas"]:
            return results["metadatas"][0]
        return None
