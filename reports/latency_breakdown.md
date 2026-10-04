# Latency breakdown — Lab 18

Nguồn đo: `latency_report.json`, `implementation_evidence.json` và console của lượt
production thành công. Hệ thống dùng `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`,
Qdrant Docker và `claude-haiku-4-5-20251001`. Enrichment dùng local fallback để
tránh 117 API call; generation và RAGAS là request Claude thật.

| Bước | Thời gian |
|---|---:|
| Nạp tài liệu + chunking | 254,94 ms |
| Enrichment local (117 chunks) | 3,00 ms |
| BM25 + dense indexing | 49.247,80 ms |
| Load reranker | 8.246,66 ms |
| Retrieval trung bình / query | 249,70 ms |
| Rerank trung bình / query | 5.548,59 ms |
| Claude generation trung bình / query | 2.486,39 ms |
| Tổng query trung bình | 8.284,69 ms |
| Tổng query min / max | 5.939,35 / 15.197,84 ms |
| RAGAS 80 metric tasks | 197,7 giây |
| Toàn bộ production | 421,2 giây |

Reranking vẫn chiếm phần lớn latency truy vấn local; generation Haiku đứng thứ hai.
Query multi-hop có thể chạy thêm rerank cho facet phụ nên tạo phần đuôi max 15,2
giây. RAGAS không nằm trong `total_ms` từng query và được ghi riêng. Các số liệu là
một lượt chạy trên máy local, chưa phải A/B benchmark được kiểm soát.

Benchmark warm trên 3 tài liệu mẫu, 3 lượt, đã loại lần load/warm-up:
average 289,19 ms, min 263,33 ms, max 322,74 ms trong lần đo đầu tiên.
Số liệu mới nhất được cập nhật tại `implementation_evidence.json`; không so trực
tiếp benchmark 3 tài liệu với query thực tế có tối đa 20 candidate.

Lần benchmark cuối: average 314,85 ms, min 278,55 ms, max 382,21 ms.

Hướng tối ưu tiếp theo: đo CPU/GPU, batch size và top candidate theo cùng test set;
chỉ giảm top-k nếu context recall của câu hỏi nhiều ý vẫn được giữ. FlashRank đã có
implementation tùy chọn nhưng chưa được benchmark nên chưa kết luận tốc độ/chất lượng.
