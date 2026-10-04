"""Collect reproducible chunking, retrieval and latency evidence without LLM calls."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["RAG_OFFLINE"] = "1"

from qdrant_client import QdrantClient

from config import NAIVE_COLLECTION
from src.m1_chunking import chunk_hierarchical, compare_strategies, load_documents
from src.m2_search import BM25Search, DenseSearch, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker


def main():
    documents = load_documents()
    stats = compare_strategies(documents)
    chunks = []
    for document in documents:
        _, children = chunk_hierarchical(
            document["text"], metadata=document["metadata"]
        )
        chunks.extend(
            {"text": child.text, "metadata": child.metadata} for child in children
        )
    bm25 = BM25Search()
    bm25.index(chunks)
    probes = {}
    for label, query in {
        "lookup": "Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?",
        "version": "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
        "numeric": "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?",
    }.items():
        results = bm25.search(query, top_k=20)
        probes[label] = [
            {"source": r.metadata["source"], "score": r.score} for r in results[:3]
        ]
    dense = DenseSearch()
    dense_results = dense.search("nghỉ phép năm", top_k=20, collection=NAIVE_COLLECTION)
    fused = reciprocal_rank_fusion(
        [bm25.search("nghỉ phép năm", top_k=20), dense_results]
    )
    memory = DenseSearch(client=QdrantClient(":memory:"))
    memory._encoder = dense._get_encoder()
    memory.index(
        [
            {
                "text": "Nhân viên được nghỉ phép năm 15 ngày.",
                "metadata": {"source": "policy"},
            },
            {
                "text": "Cấu hình VPN để truy cập hệ thống.",
                "metadata": {"source": "it"},
            },
        ],
        collection="evidence_roundtrip",
    )
    memory_result = memory.search(
        "nghỉ phép năm", top_k=1, collection="evidence_roundtrip"
    )
    assert memory_result[0].metadata["source"] == "policy"
    docs = [
        {
            "text": "Nhân viên được nghỉ 15 ngày phép mỗi năm.",
            "score": 0.8,
            "metadata": {},
        },
        {"text": "Mật khẩu thay đổi mỗi 120 ngày.", "score": 0.7, "metadata": {}},
        {"text": "VPN dùng WireGuard.", "score": 0.6, "metadata": {}},
    ]
    latency = benchmark_reranker(
        CrossEncoderReranker(), "Nhân viên nghỉ phép bao nhiêu ngày?", docs, n_runs=3
    )
    evidence = {
        "loaded_documents": len(documents),
        "indexed_children": len(chunks),
        "chunking": stats,
        "bm25_probes": probes,
        "dense_backend": dense.backend,
        "dense_candidates": len(dense_results),
        "hybrid_candidates": len(fused),
        "memory_roundtrip_passed": True,
        "reranker_latency_ms": latency,
        "evaluation_note": "Local measurements; not RAGAS scores or Claude generation.",
    }
    path = ROOT / "reports" / "implementation_evidence.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
