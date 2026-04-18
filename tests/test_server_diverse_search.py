"""Tests for the forthcoming `search_diverse_papers` MCP tool."""
from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

from deep_zotero.models import RetrievalResult
import deep_zotero.server as server


def _require_diverse_tool():
    tool = getattr(server, "search_diverse_papers", None)
    if tool is None:
        pytest.xfail("search_diverse_papers is not implemented yet")
    return tool


def _call_tool(tool, **kwargs):
    sig = inspect.signature(tool)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return tool(**kwargs)

    supported = {name: value for name, value in kwargs.items() if name in sig.parameters}
    return tool(**supported)


def _make_result(
    doc_id: str,
    chunk_index: int,
    text: str,
    score: float,
    *,
    section: str = "results",
    page_num: int = 1,
    journal_quartile: str | None = "Q1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
        text=text,
        score=score,
        doc_id=doc_id,
        doc_title=f"Title for {doc_id}",
        authors=f"Author {doc_id}",
        year=2024,
        page_num=page_num,
        chunk_index=chunk_index,
        citation_key=f"{doc_id}2024",
        publication="Test Journal",
        section=section,
        section_confidence=1.0,
        journal_quartile=journal_quartile,
    )


def _collect_leaf_dicts(node, inherited_doc_id=None):
    items = []
    if isinstance(node, dict):
        doc_id = node.get("doc_id", inherited_doc_id)
        has_nested_results = any(
            isinstance(value, (dict, list, tuple))
            for value in node.values()
        )
        looks_like_passage = any(
            key in node for key in (
                "chunk_index",
                "passage",
                "text",
                "best_passage",
                "relevance_score",
                "score",
                "avg_score",
            )
        )

        if looks_like_passage and not any(key in node for key in ("passages", "chunks", "results")):
            items.append({"doc_id": doc_id, **node})

        for value in node.values():
            if isinstance(value, (dict, list, tuple)):
                items.extend(_collect_leaf_dicts(value, doc_id))
    elif isinstance(node, (list, tuple)):
        for value in node:
            items.extend(_collect_leaf_dicts(value, inherited_doc_id))
    return items


def _collect_passage_records(node):
    records = []
    for item in _collect_leaf_dicts(node):
        if item.get("chunk_index") is not None or item.get("passage") or item.get("best_passage") or item.get("text"):
            records.append(item)
    return records


class FakeRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.search_base_calls = []
        self.expand_context_calls = []
        self.search_calls = []

    def search(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        self.search_calls.append({"args": args, "kwargs": kwargs})
        return list(self.results)

    def search_base(self, *args, **kwargs):
        self.search_base_calls.append({"args": args, "kwargs": kwargs})
        return list(self.results)

    def expand_context(self, results, context_window=1):
        self.expand_context_calls.append({
            "results_count": len(results),
            "context_window": context_window,
        })
        if context_window <= 0:
            return list(results)
        return [
            replace(
                r,
                context_before=[f"before-{r.doc_id}-{r.chunk_index}"],
                context_after=[f"after-{r.doc_id}-{r.chunk_index}"],
            )
            for r in results
        ]


class FakeReranker:
    def __init__(self):
        self.calls = []
        self.last_results = None

    def rerank(self, results, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        self.last_results = list(results)
        scored = [
            replace(result, composite_score=result.score if result.composite_score is None else result.composite_score)
            for result in results
        ]
        return sorted(scored, key=lambda result: result.composite_score or 0.0, reverse=True)


def _patch_search_stack(monkeypatch, results, *, rerank_enabled=True):
    retriever = FakeRetriever(results)
    reranker = FakeReranker()
    monkeypatch.setattr(server, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(server, "_get_reranker", lambda: reranker)
    monkeypatch.setattr(
        server,
        "_config",
        SimpleNamespace(
            rerank_enabled=rerank_enabled,
            oversample_multiplier=1,
            oversample_topic_factor=1,
        ),
    )
    return retriever, reranker


class TestSearchDiversePapersSurface:
    def test_surface_mentions_filter_vocabulary(self):
        tool = _require_diverse_tool()
        desc = getattr(tool, "description", "") or (tool.__doc__ or "")
        desc_lower = desc.lower()

        for term in ("author", "tag", "collection", "required_terms", "chunk_types", "section_weights", "journal_weights"):
            assert term in desc_lower, f"expected '{term}' in diverse-search tool surface"

        sig = inspect.signature(tool)
        params = set(sig.parameters)
        assert any(name in params for name in ("top_k", "num_papers", "limit")), "expected a paper/result limit parameter"
        assert any(
            name in params
            for name in ("passages_per_paper", "passages_per_doc", "max_passages_per_doc", "max_chunks_per_doc")
        ), "expected a per-doc cap parameter"


class TestSearchDiversePapersBehavior:
    def test_returns_multiple_papers_even_when_one_doc_dominates(self, monkeypatch):
        tool = _require_diverse_tool()
        results = [
            _make_result("doc-main", 10, "alpha beta gamma", 0.99),
            _make_result("doc-main", 11, "alpha beta delta", 0.985),
            _make_result("doc-main", 12, "alpha beta epsilon", 0.98),
            _make_result("doc-main", 13, "alpha beta zeta", 0.975),
            _make_result("doc-secondary", 2, "alpha beta from another paper", 0.91),
            _make_result("doc-tertiary", 4, "alpha beta elsewhere", 0.9),
        ]
        retriever, reranker = _patch_search_stack(monkeypatch, results)

        response = _call_tool(
            tool,
            query="alpha beta",
            top_k=4,
            num_papers=4,
            limit=4,
            passages_per_paper=2,
            passages_per_doc=2,
            max_passages_per_doc=2,
            max_chunks_per_doc=2,
        )

        assert retriever.search_base_calls, "expected the retriever to be queried"
        assert not retriever.search_calls, "candidate retrieval should use search_base without eager context expansion"
        assert reranker.calls, "expected reranking to remain enabled in this scenario"

        passage_records = _collect_passage_records(response)
        doc_ids = [record.get("doc_id") for record in passage_records if record.get("doc_id")]

        assert len(set(doc_ids)) >= 2, "one dominant document should not monopolize the output"
        assert "doc-secondary" in doc_ids
        assert "doc-tertiary" in doc_ids

    def test_caps_passages_per_doc_and_suppresses_nearby_chunks(self, monkeypatch):
        tool = _require_diverse_tool()
        results = [
            _make_result("doc-main", 3, "alpha beta core text", 0.99),
            _make_result("doc-main", 4, "alpha beta adjacent text", 0.988),
            _make_result("doc-main", 9, "alpha beta distant text", 0.986),
            _make_result("doc-main", 20, "alpha beta another distant text", 0.97),
            _make_result("doc-other", 1, "alpha beta from a second paper", 0.95),
        ]
        _patch_search_stack(monkeypatch, results)

        response = _call_tool(
            tool,
            query="alpha beta",
            top_k=5,
            num_papers=5,
            limit=5,
            passages_per_paper=2,
            passages_per_doc=2,
            max_passages_per_doc=2,
            max_chunks_per_doc=2,
        )

        passage_records = _collect_passage_records(response)
        main_chunks = [record.get("chunk_index") for record in passage_records if record.get("doc_id") == "doc-main" and record.get("chunk_index") is not None]

        assert len(main_chunks) <= 2, "per-doc passage cap should limit the dominant document"
        assert not ({3, 4} <= set(main_chunks)), "nearby chunks from the same document should be suppressed"

    def test_required_terms_filtering_happens_before_grouping(self, monkeypatch):
        tool = _require_diverse_tool()
        results = [
            _make_result("doc-main", 1, "heart rate variability drives the ranking", 0.99),
            _make_result("doc-main", 2, "heart only appears here", 0.98),
            _make_result("doc-main", 3, "rate only appears here", 0.97),
            _make_result("doc-no-match", 1, "unrelated text", 0.96),
        ]
        retriever, reranker = _patch_search_stack(monkeypatch, results)

        response = _call_tool(
            tool,
            query="heart rate",
            required_terms=["heart", "rate"],
            top_k=4,
            num_papers=4,
            limit=4,
        )

        assert retriever.search_base_calls, "expected the retriever to be queried"
        assert not retriever.search_calls, "candidate retrieval should use search_base without eager context expansion"
        assert reranker.calls, "expected reranking to run after filtering"
        assert reranker.last_results is not None
        assert len(reranker.last_results) == 1, "required_terms should trim the candidate set before document grouping"
        assert reranker.last_results[0].doc_id == "doc-main"
        assert "heart rate variability" in reranker.last_results[0].text.lower()

        passage_records = _collect_passage_records(response)
        assert any(record.get("doc_id") == "doc-main" for record in passage_records)
        assert all("doc-no-match" != record.get("doc_id") for record in passage_records)

    def test_disabled_rerank_falls_back_to_raw_scores(self, monkeypatch):
        tool = _require_diverse_tool()
        results = [
            _make_result("doc-main", 1, "highest raw score", 0.99),
            _make_result("doc-secondary", 1, "second", 0.9),
            _make_result("doc-tertiary", 1, "third", 0.8),
        ]
        retriever, reranker = _patch_search_stack(monkeypatch, results, rerank_enabled=False)

        response = _call_tool(
            tool,
            query="highest raw score",
            top_k=3,
            num_papers=3,
            limit=3,
        )

        assert retriever.search_base_calls, "expected the retriever to be queried"
        assert not retriever.search_calls, "candidate retrieval should use search_base without eager context expansion"
        assert not reranker.calls, "reranker should be bypassed when disabled"

        passage_records = _collect_passage_records(response)
        assert passage_records, "disabled rerank fallback should still return results"

        for record in passage_records:
            if "relevance_score" in record and "composite_score" in record and record["composite_score"] is not None:
                assert record["composite_score"] == record["relevance_score"]
            if "avg_score" in record and "avg_composite_score" in record and record["avg_composite_score"] is not None:
                assert record["avg_score"] == record["avg_composite_score"]

    def test_context_expansion_applies_only_to_final_selected_passages(self, monkeypatch):
        tool = _require_diverse_tool()
        results = [
            _make_result("doc-a", 1, "alpha", 0.99),
            _make_result("doc-a", 5, "alpha 2", 0.98),
            _make_result("doc-a", 9, "alpha 3", 0.97),
            _make_result("doc-b", 2, "beta", 0.96),
            _make_result("doc-b", 8, "beta 2", 0.95),
            _make_result("doc-c", 3, "gamma", 0.94),
        ]
        retriever, _ = _patch_search_stack(monkeypatch, results)

        response = _call_tool(
            tool,
            query="alpha beta gamma",
            top_k=2,
            passages_per_paper=2,
            context_window=2,
        )

        assert retriever.search_base_calls
        assert not retriever.search_calls
        assert len(retriever.expand_context_calls) == 1, "context expansion should run once on final selected passages"

        expanded_count = retriever.expand_context_calls[0]["results_count"]
        assert expanded_count <= 4, "expanded passages should be capped by top_k * passages_per_paper"

        returned_count = sum(len(paper.get("passages", [])) for paper in response.get("results", []))
        assert expanded_count == returned_count


def _require_topic_tool():
    tool = getattr(server, "search_topic", None)
    if tool is None:
        pytest.xfail("search_topic is not implemented yet")
    return tool


class TestSearchTopicContextExpansion:
    def test_expands_only_final_topk_lead_passages(self, monkeypatch):
        tool = _require_topic_tool()
        results = [
            _make_result("doc-a", 1, "alpha", 0.99),
            _make_result("doc-a", 2, "alpha second", 0.98),
            _make_result("doc-b", 1, "beta", 0.95),
            _make_result("doc-c", 1, "gamma", 0.90),
        ]
        retriever, reranker = _patch_search_stack(monkeypatch, results)

        response = _call_tool(
            tool,
            query="alpha beta gamma",
            top_k=2,
            context_window=2,
        )

        assert retriever.search_base_calls, "expected base retrieval call"
        assert not retriever.search_calls, "search_topic should avoid eager context expansion over candidates"
        assert reranker.calls, "reranking should still run before grouping"

        assert len(retriever.expand_context_calls) == 1, "lead passage context expansion should run once"
        assert retriever.expand_context_calls[0]["results_count"] == 2, "expand only final top_k lead passages"

        for paper in response.get("results", []):
            lead = paper.get("lead_passage", {})
            assert "context_before" in lead
            assert "context_after" in lead
