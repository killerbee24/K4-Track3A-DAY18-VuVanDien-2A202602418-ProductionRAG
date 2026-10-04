"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run baseline and production RAG")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run retrieval with local answer/enrichment fallbacks; skip paid RAGAS",
    )
    args = parser.parse_args()
    if args.offline:
        os.environ["RAG_OFFLINE"] = "1"
    else:
        from src.llm import generate_text, safe_error, validate_config

        try:
            validate_config()
            generate_text("Chỉ trả lời OK.", "Kiểm tra quyền gọi model.", max_tokens=16)
        except Exception as error:
            raise SystemExit(
                f"LLM preflight failed: {safe_error(error)}. "
                "Sửa quyền MWAPI hoặc dùng --offline để kiểm tra local."
            ) from error
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline

    run_baseline()
    if not args.offline:
        with open("reports/naive_baseline_report.json", encoding="utf-8") as report_file:
            baseline_report = json.load(report_file)
        if baseline_report.get("evaluation_status") != "success":
            raise SystemExit(
                "Baseline RAGAS chưa hoàn chỉnh "
                f"(status={baseline_report.get('evaluation_status', 'unknown')}, "
                f"invalid_scores={baseline_report.get('invalid_scores', 0)}). "
                "Dừng trước 117 enrichment calls; kiểm tra/bổ sung quota MWAPI rồi chạy lại."
            )

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline

    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)

    # Ensure reports are located in reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            if (
                naive.get("evaluation_status") != "success"
                or prod.get("evaluation_status") != "success"
            ):
                print(f"  {m:<23} {'N/A':>8} {'N/A':>12} {'N/A':>8}")
                continue
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_[HọTên].md")
    print("  3. Chạy: python check_lab.py")


if __name__ == "__main__":
    main()
