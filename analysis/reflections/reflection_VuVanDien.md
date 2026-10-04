# Reflection — Lab 18: Production RAG

Học viên: Vũ Văn Diện — 2A202602418

Khóa: K4, Track 3A

Ngày cập nhật: 04/10/2026 (GMT+7)

Tài liệu này tổng hợp số liệu và lỗi quan sát được trong phiên triển khai với
Codex. Học viên cần đọc lại và xác nhận phần trải nghiệm/action plan trước khi nộp.
Production đã tạo đủ 20 câu trả lời và được RAGAS chấm đủ 80/80 metric value bằng
`claude-haiku-4-5-20251001`. Report có `evaluation_status=success`; bottom-5 được
phân tích từ score thật. Baseline cũ vẫn partial và không được dùng để tính delta.

## 1. Lecture mapping

| Concept | Module/hàm | Quan sát và phân tích |
|---|---|---|
| Semantic chunking | M1 / `chunk_semantic()` | Encode câu bằng MiniLM, tách khi cosine nhỏ hơn 0,85. So sánh được lưu ở `reports/implementation_evidence.json`. Threshold cao tạo nhiều đoạn nhỏ; model không chuyên tiếng Việt và header ngắn là hai điểm cần đo thêm. |
| Hierarchical retrieval | M1 / `chunk_hierarchical()`, `pipeline.run_query()` | Child tối đa 256 ký tự, parent tối đa 2.048 ký tự; corpus thực tế tạo 117 child. Hash source/text giúp parent ID không trùng giữa tài liệu. Retrieve/rerank child rồi trả parent đầy đủ để giữ bảng và điều kiện chính sách. |
| BM25 + dense fusion | M2 / `segment_vietnamese()`, `reciprocal_rank_fusion()` | BM25 nhận lookup kết hôn và mua sắm, nhưng trong probe nghỉ phép năm còn xếp chính sách 2023 cao. Hybrid kết hợp bge-m3 và lọc superseded để khắc phục xung đột phiên bản. RRF gộp thứ hạng, không cộng trực tiếp hai thang điểm. |
| Cross-encoder reranking | M3 / `CrossEncoderReranker.rerank()` | Rerank tối đa 20 candidate, chọn tối đa 3 parent khác nhau. Lần chạy đầu có rerank trung bình 5.057,43 ms/query; benchmark 3 tài liệu warm trung bình 289,19 ms. Dedupe parent giúp tránh cả ba context cùng một tài liệu. |
| RAGAS 4 metrics | M4 / `evaluate_ragas()`, `failure_analysis()` | Production đạt faithfulness 0,9025; answer relevancy 0,8962; context precision 0,9000; context recall 0,8833. Recall thấp nhất do hai ground truth yêu cầu đối chiếu bản cũ trong khi retrieval đúng chủ ý lọc superseded. Metric lỗi được giữ là `null`, không biến thành điểm 0 giả. |
| Contextual embeddings/enrichment | M5 / `_enrich_single_call()`, `enrich_chunks()` | Combined mode hỗ trợ một SDK call/chunk và parse/fallback an toàn. Lượt chấm cuối dùng `local_fallback` để tiết kiệm 117 API call: prepend nguồn có tính xác định, giữ source/parent ID và không thêm dữ kiện sinh. Report ghi rõ mode này để không nhầm với Claude enrichment. |

So sánh chunking chạy từng tài liệu riêng biệt để không nối chính sách/phiên bản
khác nhau. Nguồn số liệu cuối cùng là JSON evidence; không dùng số chunk nhỏ hơn
làm tiêu chí duy nhất để kết luận chiến lược tốt hơn.

Kết quả per-document: basic 57 chunks (trung bình 366 ký tự), semantic 208
(99 ký tự), hierarchical 117 children (178 ký tự, max 256), structure-aware 107
(194 ký tự). Basic vẫn có đoạn 565 ký tự vì baseline không chẻ paragraph dài;
hierarchical xử lý trường hợp này bằng splitter có giới hạn rõ ràng.

## 2. Khó khăn và debug

### Quyền API của gateway

Exact response từ request thật:

```text
Error code: 403 - {'code': 'GROUP_NOT_ALLOWED', 'message': 'API Key 所属专属分组不再允许当前用户使用'}
```

`check_llm.py` xác nhận đã điền key/model; `/v1/models` trả danh sách model nhưng
`/v1/messages` từng bị từ chối. Sau khi quyền được khôi phục, `check_llm.py --test`
trả `OK` và baseline sinh được 20 câu trả lời. Điều này phân biệt cấu hình client
hợp lệ với quyền gọi model thực tế; không cần thay bằng một OpenAI key giả.

### Hạn mức API trong lượt RAGAS online

Exact response khi evaluator đang chạy:

```text
Error code: 429 - {'code': 'API_KEY_QUOTA_EXHAUSTED', 'message': 'API key 额度已用完'}
```

RAGAS hoàn thành 41/80 metric value rồi thiếu quota. Pipeline được dừng khi bước
production enrichment bắt đầu để tránh hơn 100 request chắc chắn lỗi. Quan sát này
cũng phát hiện report partial cũ đổi `NaN` thành 0; evaluator đã được sửa để giữ
giá trị lỗi là `null`, ghi `scored_metric_values` và chỉ đếm câu có đủ bốn metric.
Sau khi bổ sung quota, chuyển từ Opus sang Haiku và dùng enrichment local; production
hoàn thành 80/80 metric value mà không phát sinh 117 lời gọi enrichment.

### Docker chưa khởi động

```text
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

Khởi động Docker Desktop, chạy lại `docker compose up -d`, xác nhận `docker compose
ps`. Sau đó dense search thực tế kết nối Qdrant Docker và roundtrip in-memory cũng
được kiểm tra để biết fallback không che lỗi implementation.

### Context thiếu ý phụ trong câu hỏi multi-hop

Lần chạy đầu của câu Senior 9 năm trả ba parent liên quan nghỉ phép nhưng thiếu
bảng lương. Nguyên nhân: embedding/rerank full query ưu tiên chủ đề nghỉ phép.
Đã bổ sung `query_facets()` tách các ý quanh “và”, tìm kiếm từng ý và dành candidate
cho ý phụ trước khi dedupe parent. Production trả đúng 18 ngày và lương 20–35 triệu;
faithfulness/recall của câu đều 1,0. Tuy nhiên context precision bằng 0 vì parent thứ
ba về nghỉ không lương bị thêm vào, cho thấy recall đã được sửa nhưng cần early-stop.

### Kết quả kiểm thử không được gọi API trả phí

Thêm fixture `offline_llm` để unit test không đọc key thật rồi gọi Claude. Adapter
được kiểm tra bằng `httpx.MockTransport` cho cả chuẩn Anthropic và OpenAI-compatible;
kiểm tra header, endpoint, model, parsing và lỗi fallback. Test retrieval/reranker
dùng model local thật, trừ roundtrip deterministic riêng dùng vector kiểm soát được.

### Kiến thức cần bổ sung

Tiếp tục học ảnh hưởng threshold với tiếng Việt, query decomposition và coverage
multi-hop, calibration score cross-encoder, chuẩn hóa bộ ground truth cho câu hỏi
số học, và cách giới hạn concurrency/retry của RAGAS để kiểm soát chi phí gateway.

## 3. Action plan đề xuất cho project ProductionRAG

Project được chọn là trợ lý tra cứu chính sách nội bộ trong repository hiện tại;
đây là kế hoạch đề xuất, chưa phải mô tả một project khác của học viên.

Hiện trạng: 25 Markdown + 1 PDF có text được nạp; 2 PDF scan cần OCR. Pipeline đã
có hybrid retrieval, parent expansion, reranking, contextual enrichment và evaluator
Claude. Bottleneck quan sát được là reranker CPU. Production đã có answer/RAGAS đầy
đủ; metric thấp nhất là context recall 0,8833. Baseline vẫn partial và được giữ lại
chỉ để audit, không tiêu thêm quota để tạo delta hình thức.

1. Chunking: dùng hierarchical cho chính sách dài, giữ source/version và parent;
   thử structure-aware cho các bảng và so sánh bằng cùng test set.
2. Search: giữ BM25 + dense + RRF, duy trì filter phiên bản và bổ sung test multi-hop.
3. Reranking: giữ bge-reranker làm mốc chất lượng, benchmark FlashRank/GPU với cùng
   candidate set; không giảm top-k trước khi đo recall.
4. Evaluation: dùng report production hiện tại làm mốc regression (cả bốn metric
   trên 0,88); tách câu hỏi chính sách hiện hành khỏi câu so sánh phiên bản.
5. Enrichment: dùng combined mode, cân nhắc cache theo hash chunk/model/prompt sau
   khi có dữ liệu latency/cost thật; chỉ index HyQA khi chứng minh cải thiện recall.

Timeline cho vòng cải tiến tiếp theo:

- Tuần 1: sửa bottom-5 theo context threshold/topic filter; thêm adversarial queries,
  rà ground truth version và câu tạm ứng (chưa nêu rõ quy ước pro-rata).
- Tuần 2: benchmark candidate count/reranker, thử OCR hai PDF scan, tối ưu cache
  enrichment và lập ngưỡng regression cho score/latency trước khi mở rộng corpus.

Ưu tiên nếu có thêm một giờ: giảm context thừa ở câu Senior/mua sắm/phân loại lương,
sau đó đo lại precision và recall trên đúng 20 câu trước khi tối ưu latency.
