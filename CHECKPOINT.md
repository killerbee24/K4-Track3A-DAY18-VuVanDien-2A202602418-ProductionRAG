# CHECKPOINT — Lab 18: Production RAG

> Cập nhật 04/10/2026 (GMT+7). `[x]` = có code/kết quả kiểm tra; mục chạy LLM thật chỉ được đánh dấu khi API thành công.
>
> Production đã chạy online thành công bằng `claude-haiku-4-5-20251001`: 20/20 câu, 80/80 metric value, `evaluation_status=success`. Để tiết kiệm, M5 dùng local fallback có ghi rõ trong report, tránh 117 API call enrichment. Baseline cũ vẫn partial và không dùng để tính delta. Reflection là bản tổng hợp bằng chứng, cần học viên xác nhận.

## 0. Mục tiêu hoàn thành

- [x] Hoàn thiện toàn bộ TODO bắt buộc trong 5 module.
- [x] Tất cả unit test trong `tests/` đều pass.
- [x] Pipeline offline và production online đều chạy end-to-end với exit code 0.
- [x] Sinh `reports/ragas_report.json` có đủ 20 mẫu và bốn metric.
- [x] Report RAGAS được chấm thật đủ 20 câu (`success`, `scored_questions=20`).
- [x] Hoàn thành failure analysis cho bottom-5 câu hỏi.
- [x] Viết reflection có bằng chứng tại `analysis/reflections/reflection_VuVanDien.md`.
- [ ] Học viên xác nhận trải nghiệm/action plan trong reflection trước khi nộp.
- [x] Chạy `python check_lab.py` trước khi nộp.

## 1. Chuẩn bị môi trường

- [x] Kiểm tra Python phiên bản 3.11 trở lên: `python --version`.
- [x] Tạo môi trường ảo: `python -m venv .venv`.
- [x] Kích hoạt môi trường ảo: `.venv\Scripts\Activate.ps1`.
- [x] Cài dependencies: `pip install -r requirements.txt`.
- [x] Tạo `.env` từ `.env.example`: `Copy-Item .env.example .env`.
- [x] Điền key MWAPI vào `LLM_API_KEY` trong `.env`.
- [x] Điền `LLM_MODEL` bằng Claude model ID được key cho phép.
- [x] Đặt `LLM_API_FORMAT=anthropic` và `LLM_BASE_URL=https://api.mwapi.dev`.
- [x] Chạy `python check_llm.py` để kiểm tra cấu hình.
- [x] Chạy `python check_llm.py --models` để lấy model ID nếu cần.
- [x] Request Claude thật thành công: `check_llm.py --test` trả `OK`, model `claude-opus-4-6`.
- [x] Học viên đã cho phép gửi tài liệu lab, câu hỏi và context tới MWAPI để chạy enrichment/RAGAS online.
- [x] Khởi động Qdrant: `docker compose up -d`.
- [x] Kiểm tra Qdrant đang chạy: `docker compose ps`.
- [x] Tải trước các model nếu cần:
  - [x] `all-MiniLM-L6-v2` cho semantic chunking.
  - [x] `BAAI/bge-m3` cho dense retrieval.
  - [x] `BAAI/bge-reranker-v2-m3` cho reranking.

## 2. Baseline

- [x] Chạy baseline online (57 basic chunks, 20 câu trả lời Claude); RAGAS chỉ hoàn thành một phần vì hết quota.
- [x] Xác nhận baseline không bị crash.
- [x] Xác nhận có `reports/naive_baseline_report.json` sau khi M2 và M4 đã hoạt động.
- [ ] Ghi lại bốn điểm baseline để so sánh với production:
  - [ ] Faithfulness: `0.4928 — kết quả partial, chưa dùng để nộp`
  - [ ] Answer Relevancy: `0.3740 — kết quả partial, chưa dùng để nộp`
  - [ ] Context Precision: `0.3917 — kết quả partial, chưa dùng để nộp`
  - [ ] Context Recall: `0.3750 — kết quả partial, chưa dùng để nộp`

## 3. M1 — Advanced Chunking

File: `src/m1_chunking.py`

### Semantic chunking

- [x] Implement `chunk_semantic()`.
- [x] Tách văn bản thành các câu hợp lệ và loại câu rỗng.
- [x] Sinh embedding cho từng câu bằng `all-MiniLM-L6-v2`.
- [x] Tính cosine similarity an toàn, tránh chia cho 0.
- [x] Tạo chunk mới khi similarity nhỏ hơn threshold.
- [x] Trả về `list[Chunk]`, không trả danh sách rỗng với input hợp lệ.
- [x] Metadata có `strategy="semantic"`.

### Hierarchical chunking

- [x] Implement `chunk_hierarchical()`.
- [x] Tạo parent chunks với kích thước mục tiêu khoảng 2.048 ký tự.
- [x] Tạo child chunks với kích thước mục tiêu khoảng 256 ký tự.
- [x] Mỗi parent có một `parent_id` duy nhất.
- [x] Mỗi child có `parent_id` trỏ tới parent tồn tại.
- [x] Metadata phân biệt `chunk_type="parent"` và `chunk_type="child"`.
- [x] Kích thước trung bình của child nhỏ hơn parent.
- [x] Kiểm tra trường hợp paragraph dài hơn giới hạn chunk.

### Structure-aware chunking

- [x] Implement `chunk_structure_aware()`.
- [x] Nhận diện header Markdown cấp 1–3.
- [x] Giữ header trong nội dung chunk.
- [x] Lưu section vào metadata.
- [x] Không tạo chunk rỗng.
- [x] Hạn chế cắt ngang bảng, danh sách hoặc code block.

### Kiểm tra M1

- [x] Chạy `pytest tests/test_m1.py -v`.
- [x] Chạy `compare_strategies()` qua `scripts/collect_evidence.py`: basic 57, semantic 208, hierarchical 117, structure 107.
- [x] Ghi số liệu basic/semantic/hierarchical/structure để dùng trong reflection.

## 4. M2 — Hybrid Search

File: `src/m2_search.py`

### Vietnamese segmentation và BM25

- [x] Implement `segment_vietnamese()` bằng `underthesea.word_tokenize()`.
- [x] Thay `_` bằng khoảng trắng theo yêu cầu của scaffold.
- [x] Implement `BM25Search.index()`.
- [x] Lưu documents và corpus tokens đúng thứ tự.
- [x] Khởi tạo `BM25Okapi`.
- [x] Implement `BM25Search.search()`.
- [x] Loại kết quả có score không liên quan (`score <= 0`).
- [x] Kết quả BM25 có `method="bm25"`.
- [x] Query “nghỉ phép năm” đưa tài liệu nghỉ phép lên đầu.

### Dense search và Qdrant

- [x] Implement `DenseSearch.index()`.
- [x] Tạo/recreate collection với vector size 1.024 và cosine distance.
- [x] Embed toàn bộ chunk bằng `BAAI/bge-m3`.
- [x] Upsert text và metadata vào Qdrant.
- [x] Implement `DenseSearch.search()` bằng `query_points()`.
- [x] Kết quả dense có `method="dense"`.
- [x] Kiểm tra cả Qdrant Docker và fallback in-memory.

### Reciprocal Rank Fusion

- [x] Implement `reciprocal_rank_fusion()`.
- [x] Gộp kết quả dựa trên nội dung tài liệu, tránh bản ghi trùng.
- [x] Áp dụng công thức `1 / (k + rank + 1)`.
- [x] Sắp xếp RRF score giảm dần.
- [x] Kết quả hợp nhất có `method="hybrid"`.
- [x] Chỉ trả tối đa `top_k` kết quả.

### Kiểm tra M2

- [x] Chạy `pytest tests/test_m2.py -v`.
- [x] Thử thủ công truy vấn lookup, version và câu hỏi có con số.
- [x] Kiểm tra top-20 BM25 và top-20 dense được hợp nhất đúng.

## 5. M3 — Cross-encoder Reranking

File: `src/m3_rerank.py`

- [x] Implement `CrossEncoderReranker._load_model()`.
- [x] Dùng `sentence_transformers.CrossEncoder`.
- [x] Chỉ load model một lần và cache trong `self._model`.
- [x] Implement `rerank()`.
- [x] Trả `[]` khi documents rỗng.
- [x] Tạo đúng các cặp `(query, document)`.
- [x] Xử lý cả output score scalar và array/list.
- [x] Sắp xếp theo `rerank_score` giảm dần.
- [x] Giữ lại `original_score` và metadata.
- [x] Trả tối đa top 3 kết quả theo cấu hình.
- [x] Chạy `pytest tests/test_m3.py -v`.
- [x] Đo latency bằng `benchmark_reranker()` và ghi kết quả:
  - [x] Average: `314.85 ms`
  - [x] Minimum: `278.55 ms`
  - [x] Maximum: `382.21 ms`
- [x] Optional: triển khai `FlashrankReranker`; chưa chạy model/benchmark tùy chọn.

## 6. M4 — RAGAS Evaluation và Failure Analysis

File: `src/m4_eval.py`

### RAGAS

- [x] Implement `evaluate_ragas()` với Claude qua MWAPI và embeddings local.
- [x] Tạo `Dataset` với `question`, `answer`, `contexts`, `ground_truth`.
- [x] Chạy đủ bốn metric:
  - [x] Faithfulness.
  - [x] Answer Relevancy.
  - [x] Context Precision.
  - [x] Context Recall.
- [x] Tạo `EvalResult` cho từng câu hỏi.
- [x] Tính aggregate score cho từng metric.
- [x] Trả đủ các key kể cả khi RAGAS lỗi.
- [x] Bọc lỗi API/RAGAS bằng `try/except`, không để pipeline crash.

### Failure analysis

- [x] Implement `failure_analysis()`.
- [x] Tính điểm trung bình của bốn metric cho mỗi câu.
- [x] Xác định metric thấp nhất của từng câu.
- [x] Sắp xếp câu hỏi từ tệ nhất đến tốt nhất.
- [x] Trả bottom-N với `diagnosis` và `suggested_fix`.
- [x] Ánh xạ đúng lỗi retrieval, reranking, context hoặc generation.
- [x] Chạy `pytest tests/test_m4.py -v`.

## 7. M5 — Enrichment Pipeline

File: `src/m5_enrichment.py`

### Các kỹ thuật riêng lẻ

- [x] Implement `summarize_chunk()` qua LLM cấu hình chung.
- [x] Có fallback extractive khi không có API key hoặc API lỗi.
- [x] Implement `generate_hypothesis_questions()` qua LLM cấu hình chung.
- [x] Câu hỏi sinh ra là `list[str]`, không vượt `n_questions`.
- [x] Implement `contextual_prepend()` qua LLM cấu hình chung.
- [x] Luôn giữ nguyên nội dung chunk gốc trong kết quả.
- [x] Implement `extract_metadata()` qua LLM cấu hình chung.
- [x] Metadata trả về là dict và có fallback hợp lệ.

### Combined mode

- [x] Implement `_enrich_single_call()` qua LLM cấu hình chung.
- [x] Combined mode dùng một lời gọi SDK/chunk, kiểm tra bằng HTTP mock; retry transport có thể thêm request.
- [x] Parse JSON an toàn khi LLM trả Markdown code fence hoặc JSON lỗi.
- [x] Trả đủ `summary`, `questions`, `context`, `metadata`.
- [x] Có fallback khi thiếu API key hoặc API lỗi.
- [x] HTTP mock xác nhận `enriched_text` khác `original_text`; lượt production cuối dùng local fallback để tiết kiệm quota.
- [x] Index contextual text; giữ summary/HyQA riêng, chưa ghép vào context/index.

### Kiểm tra M5

- [x] Unit tests M5 pass; production local enrichment xử lý đủ 117/117 chunk.
- [x] Kiểm tra chế độ không có API key.
- [x] Kiểm tra chế độ có API key: kết nối và generation/RAGAS thành công; chủ động không gọi Claude enrichment.
- [x] Combined mode đã implement và test; điểm bonus phụ thuộc reviewer.

## 8. Chạy pipeline hoàn chỉnh

- [x] Chạy toàn bộ tests: `pytest tests/ -v`.
- [x] Xác nhận không còn TODO bắt buộc:

```powershell
(Select-String -Path src/*.py -Pattern "# TODO:").Count
```

- [x] Chạy production: `python src/pipeline.py --offline`, đủ 20 câu.
- [x] Chạy `python main.py --offline`; số liệu RAGAS/Delta chưa có.
- [x] Chạy production online tiết kiệm: `python src/pipeline.py --local-enrichment`, đủ 20 câu và RAGAS success.
- [x] Xác nhận pipeline không dùng danh sách chunk rỗng.
- [x] Xác nhận hybrid search trả kết quả cho cả 20 câu hỏi.
- [x] Xác nhận reranker trả tối đa ba context.
- [ ] Xác nhận câu trả lời chỉ dựa trên context.
- [x] Xác nhận `reports/ragas_report.json` được sinh ra.

## 9. Kết quả RAGAS

Production online được chấm đủ bằng Claude Haiku. Baseline là lượt partial cũ nên không tính delta; điều này tránh tiêu thêm khoảng 80 lượt evaluator chỉ để tạo bảng so sánh.

| Metric            | Naive Baseline | Production | Delta | Đạt mục tiêu? |
| ----------------- | -------------: | ---------: | ----: | ------------- |
| Faithfulness | N/A (partial) | 0.9025 | N/A | [x] |
| Answer Relevancy | N/A (partial) | 0.8962 | N/A | [x] |
| Context Precision | N/A (partial) | 0.9000 | N/A | [x] |
| Context Recall | N/A (partial) | 0.8833 | N/A | [x] |

- [x] Có ít nhất ba metric đạt 0.70 để lấy trọn điểm RAGAS.
- [x] Bonus: Faithfulness đạt ít nhất 0.85.
- [x] Bonus: Cả bốn metric đạt ít nhất 0.75.
- [x] Score đều trên 0,88; bottom-5 vẫn được kiểm tra theo chunking → retrieval → reranking → prompt.

## 10. Failure Analysis

File: `analysis/failure_analysis.md`

- [x] Điền họ tên học viên.
- [x] Điền bảng production; baseline/delta ghi N/A vì baseline partial.
- [x] Chọn đúng bottom-5 từ kết quả per-question.
- [x] Với mỗi failure, điền question, expected, got và worst metric.
- [x] Viết Error Tree cho từng failure.
- [x] Xác định root cause cụ thể, không chỉ ghi “kết quả sai”.
- [x] Đưa ra suggested fix có thể triển khai.
- [x] Hoàn thành một case study để presentation.
- [x] Nêu việc sẽ tối ưu nếu có thêm một giờ.

## 11. Reflection cá nhân

Tạo file: `analysis/reflections/reflection_VuVanDien.md`

- [x] Điền họ tên và ngày hoàn thành.
- [x] Mapping đủ năm lecture concepts vào module và hàm cụ thể.
- [x] Có số liệu per-document trong `reports/implementation_evidence.json`.
- [x] Có nhận xét thật về BM25, dense và RRF.
- [x] Có latency reranker.
- [x] Xác định context recall là metric thấp nhất và giải thích lệch intent/version ground truth.
- [x] Mô tả exact error message đã gặp.
- [x] Mô tả quá trình debug và cách giải quyết.
- [x] Nêu kiến thức còn thiếu và cách bổ sung.
- [x] Viết action plan cho project cá nhân.
- [x] Action plan có chunking, search, reranking, evaluation và enrichment.
- [x] Có timeline triển khai theo tuần.

## 12. Kiểm tra trước khi nộp

- [x] `pytest tests/ -v` pass 100%.
- [x] `python main.py --offline` exit code 0.
- [x] `python src/pipeline.py --local-enrichment` online thành công và được chấm thật.
- [x] Chạy `python check_lab.py`: 67/67 tests pass, 0 TODO, 0 lỗi.
- [x] `python check_lab.py` báo “Bài lab sẵn sàng để nộp”.
- [x] `reports/ragas_report.json` có `aggregate` và `num_questions`.
- [x] `analysis/failure_analysis.md` đã có 5 case thực tế, không còn placeholder.
- [x] Thay 5 case thủ công bằng bottom-5 RAGAS thật.
- [x] Có reflection cá nhân, không nộp mỗi file template.
- [x] Không commit `.env`, API key, cache model hoặc dữ liệu bí mật.
- [x] Kiểm tra `git status` trước khi commit.
- [x] Repository đặt đúng tên:
      `K4-Track3A-DAY18-VuVanDien-2A202602418-ProductionRAG`.
- [ ] Push repository lên GitHub ở chế độ Public.
- [ ] Mở lại GitHub và xác nhận đầy đủ source, report và analysis.
- [ ] Nộp link repository lên VLearn LMS/Codelab trước hạn.

## 13. Ghi chú tiến độ

- Code M1–M5 đã triển khai, 0 TODO.
- Regression cuối: 67/67 tests pass (xem `reports/validation_report.json`).
- Qdrant Docker đang chạy; dense roundtrip in-memory cũng pass.
- Corpus: 26 tài liệu có text (25 Markdown + 1 PDF); 2 PDF scan được bỏ qua.
- Chunks per-document: basic 57, semantic 208, hierarchical 117 children/26 parents, structure 107.
- Production cuối chạy offline 159,3 giây; context câu Senior có cả nghỉ phép v2024 và bảng lương.
- Latency query cuối: retrieval 262,95 ms, rerank 5.345,11 ms, tổng 5.608,06 ms.
- Benchmark reranker 3 docs warm: avg 314,85 ms, min 278,55 ms, max 382,21 ms.
- Production report có 20 mẫu, 80/80 metric value, `success`; model Claude Haiku và `local_fallback` enrichment được ghi trong JSON.
- RAGAS production: faithfulness 0,9025; relevancy 0,8962; precision 0,9000; recall 0,8833.
- Baseline partial được giữ để audit, không dùng làm score/delta cuối.
- Chưa commit/push GitHub hoặc nộp LMS.

| Thời điểm | Hạng mục | Kết quả | Bước tiếp theo |
|---|---|---|---|
| 04/10/2026 | Implementation + tests | Hoàn thiện M1–M5, tests local pass | Xem validation_report cuối |
| 04/10/2026 | Pipeline offline | Baseline/production đủ 20 câu, JSON report và latency | Chạy online |
| 04/10/2026 | Phân tích/reflection | 5 case thủ công + reflection có số liệu | RAGAS thật → bottom-5 → học viên review |
| 04/10/2026 | Retry Claude | Request thật trả OK; học viên cho phép full online | Chạy baseline/RAGAS |
| 04/10/2026 | Online baseline | 20 câu trả lời Claude; 41/80 metric value hoàn thành | Bổ sung quota vì `API_KEY_QUOTA_EXHAUSTED` |
| 04/10/2026 | Online production tiết kiệm | Haiku + local enrichment, 20/20 câu, RAGAS 80/80 success | Cập nhật bottom-5 và chạy check cuối |
