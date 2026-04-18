"""Search with automatic context expansion."""
from dataclasses import replace

from .models import StoredChunk, RetrievalResult
from .interfaces import VectorStoreProtocol


class Retriever:
    """
    Semantic search with context expansion.

    Wraps VectorStore to provide:
    - Automatic context expansion around hits
    - Result formatting with full metadata
    """

    def __init__(self, vector_store: VectorStoreProtocol):
        self.store = vector_store

    @staticmethod
    def _stored_chunk_to_result(hit: StoredChunk) -> RetrievalResult:
        jq = hit.metadata.get("journal_quartile", "")
        journal_quartile = jq if jq else None
        return RetrievalResult(
            chunk_id=hit.id,
            text=hit.text,
            score=hit.score,
            doc_id=hit.metadata["doc_id"],
            doc_title=hit.metadata["doc_title"],
            authors=hit.metadata["authors"],
            year=hit.metadata["year"] or None,
            page_num=hit.metadata["page_num"],
            chunk_index=hit.metadata["chunk_index"],
            citation_key=hit.metadata.get("citation_key", ""),
            publication=hit.metadata.get("publication", ""),
            tags=hit.metadata.get("tags", ""),
            collections=hit.metadata.get("collections", ""),
            section=hit.metadata.get("section", "unknown"),
            section_confidence=hit.metadata.get("section_confidence", 1.0),
            journal_quartile=journal_quartile,
        )

    def search_base(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant chunks without loading adjacent context."""
        hits = self.store.search(query, top_k=top_k, filters=filters)
        return [self._stored_chunk_to_result(hit) for hit in hits]

    def expand_context(
        self,
        results: list[RetrievalResult],
        context_window: int = 1,
    ) -> list[RetrievalResult]:
        """Load adjacent chunks only for preselected results."""
        if context_window <= 0 or not results:
            return results

        centers = [(result.doc_id, result.chunk_index) for result in results]
        adjacent_by_center = self.store.get_adjacent_chunks_batch(centers, window=context_window)

        expanded = []
        for result in results:
            center_idx = result.chunk_index
            adjacent = adjacent_by_center.get((result.doc_id, center_idx), [])

            context_before = []
            context_after = []
            for adj in adjacent:
                adj_idx = adj.metadata["chunk_index"]
                if adj_idx < center_idx:
                    context_before.append(adj.text)
                elif adj_idx > center_idx:
                    context_after.append(adj.text)

            expanded.append(
                replace(
                    result,
                    context_before=context_before,
                    context_after=context_after,
                )
            )

        return expanded

    def search(
        self,
        query: str,
        top_k: int = 10,
        context_window: int = 1,
        filters: dict | None = None
    ) -> list[RetrievalResult]:
        """
        Search for relevant chunks and expand context.

        Args:
            query: Search query
            top_k: Number of results
            context_window: Chunks before/after to include (0-5)
            filters: Optional metadata filters

        Returns:
            List of RetrievalResult with expanded context
        """
        results = self.search_base(query, top_k=top_k, filters=filters)
        return self.expand_context(results, context_window=context_window)
