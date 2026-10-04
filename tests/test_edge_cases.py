"""Boundary and integration regressions beyond the starter grading tests."""

from types import SimpleNamespace

import numpy as np
from qdrant_client import QdrantClient

from src.m1_chunking import chunk_hierarchical, chunk_semantic, chunk_structure_aware
from src.m2_search import (
    DenseSearch,
    SearchResult,
    query_facets,
    reciprocal_rank_fusion,
)
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import EvalResult, failure_analysis
from src.pipeline import run_query


def test_hierarchy_bounds_long_paragraphs_and_ids():
    text = "word " * 400 + "x" * 300
    parents, children = chunk_hierarchical(text, 200, 80, {"source": "a.md"})
    other, _ = chunk_hierarchical(text, 200, 80, {"source": "b.md"})
    assert all(0 < len(p.text) <= 200 for p in parents)
    assert all(0 < len(c.text) <= 80 for c in children)
    assert {p.metadata["parent_id"] for p in parents}.isdisjoint(
        p.metadata["parent_id"] for p in other
    )
    assert all(
        c.parent_id in {p.metadata["parent_id"] for p in parents} for c in children
    )


def test_structure_does_not_split_fenced_headers_or_table():
    text = "# Title\n\n## Section\n```python\n# not a header\n```\n| A | B |\n|---|---|\n| 1 | 2 |"
    chunks = chunk_structure_aware(text)
    assert len(chunks) == 2
    assert "# not a header" in chunks[1].text
    assert "| 1 | 2 |" in chunks[1].text


def test_empty_chunks_do_not_load_models():
    assert chunk_semantic("") == []
    assert chunk_hierarchical("") == ([], [])
    assert chunk_structure_aware("") == []
    assert CrossEncoderReranker().rerank("q", []) == []


def test_rrf_exact_scores_and_no_double_count():
    first = SearchResult("A", 50, {}, "bm25")
    second = SearchResult("B", 0.9, {}, "dense")
    fused = reciprocal_rank_fusion([[first, first], [second, first]], k=60)
    assert fused[0].text == "A"
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-12


def test_dense_qdrant_roundtrip_with_deterministic_vectors():
    class Encoder:
        def encode(self, text, **kwargs):
            texts = [text] if isinstance(text, str) else text
            vectors = np.zeros((len(texts), 1024), dtype=np.float32)
            for i, item in enumerate(texts):
                vectors[i, 0 if "leave" in item else 1] = 1
            return vectors[0] if isinstance(text, str) else vectors

    dense = DenseSearch(client=QdrantClient(":memory:"))
    dense._encoder = Encoder()
    dense.index(
        [
            {"text": "leave policy", "metadata": {"source": "a.md"}},
            {"text": "VPN", "metadata": {"source": "b.md"}},
        ]
    )
    result = dense.search("leave", top_k=1)
    assert result[0].text == "leave policy"
    assert result[0].metadata["source"] == "a.md"
    assert result[0].method == "dense"
    dense.index([])
    assert dense.search("leave") == []


def test_failure_order_and_unscored_rows():
    low = EvalResult("low", "a", ["c"], "gt", 0.2, 0.4, 0.5, 0.1)
    high = EvalResult("high", "a", ["c"], "gt", 0.9, 0.8, 0.9, 0.8)
    unscored = EvalResult("unscored", "a", ["c"], "gt", None, None, None, None)
    failures = failure_analysis([high, unscored, low], bottom_n=1)
    assert failures[0]["question"] == "low"
    assert failures[0]["worst_metric"] == "context_recall"
    assert "Chunking/Retrieval" in failures[0]["error_tree"]


def test_parent_expansion_keeps_three_distinct_sources():
    results = [
        SearchResult("child1", 1, {"parent_id": "p1", "source": "a.md"}, "hybrid"),
        SearchResult("child2", 0.9, {"parent_id": "p1", "source": "a.md"}, "hybrid"),
        SearchResult("child3", 0.8, {"parent_id": "p2", "source": "b.md"}, "hybrid"),
    ]
    search = SimpleNamespace(
        search=lambda q: results, parents={"p1": "Full A", "p2": "Full B"}
    )
    reranker = SimpleNamespace(rerank=lambda *args, **kwargs: results)
    answer, contexts = run_query("q", search, reranker)
    assert len(contexts) == 2
    assert answer == contexts[0]
    assert "Full A" in contexts[0] and "Full B" in contexts[1]
    assert "child1" not in contexts[0]


def test_query_facets_keep_secondary_question():
    assert query_facets("nghỉ phép bao nhiêu ngày và lương trong khoảng nào?") == [
        "nghỉ phép bao nhiêu ngày",
        "lương trong khoảng nào?",
    ]
    assert query_facets("lương thử việc là bao nhiêu?") == [
        "lương thử việc là bao nhiêu?"
    ]
