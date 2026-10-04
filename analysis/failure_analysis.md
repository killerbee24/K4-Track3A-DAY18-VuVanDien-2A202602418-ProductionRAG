# Failure Analysis — Lab 18: Production RAG

Học viên: Vũ Văn Diện — 2A202602418

Khóa: K4, Track 3A

Ngày cập nhật: 04/10/2026 (GMT+7)

## Trạng thái và bằng chứng

Baseline đã sinh đủ 20 câu trả lời bằng Claude nhưng chỉ chấm được 41/80 metric
value trước lỗi `429 API_KEY_QUOTA_EXHAUSTED`; report baseline vì vậy vẫn `partial`.
Sau khi bổ sung quota, production chạy thành công bằng Claude Haiku, Qdrant Docker,
BM25 + bge-m3 + cross-encoder và được chấm đủ 80/80 metric value. M5 dùng
`local_fallback` có ghi trong report để tiết kiệm 117 API call enrichment.

Nguồn kiểm tra: `reports/naive_baseline_report.json`, `reports/ragas_report.json`,
`reports/latency_report.json`, `reports/implementation_evidence.json`.

## RAGAS Scores

Production được chấm đủ 20/20 câu (80/80 metric value), `evaluation_status=success`.
Model generation/judge là `claude-haiku-4-5-20251001`; enrichment dùng
`local_fallback` để tiết kiệm 117 API call. Baseline là lượt partial cũ nên không
tính delta để tránh so sánh sai.

| Metric | Naive Baseline | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | N/A (partial) | 0,9025 | N/A |
| Answer Relevancy | N/A (partial) | 0,8962 | N/A |
| Context Precision | N/A (partial) | 0,9000 | N/A |
| Context Recall | N/A (partial) | 0,8833 | N/A |

## Bottom-5 RAGAS thật

### 1. Senior 9 năm — average 0,7305

- Question: Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép
  năm và lương trong khoảng nào?
- Expected/Got: đều là 18 ngày phép và lương 20–35 triệu/tháng.
- Scores: faithfulness 1,0; relevancy 0,9220; precision 0,0; recall 1,0.
- Context: `nghi_phep_nam_v2024.md`, `bang_luong_2024.md`,
  `nghi_phep_khong_luong.md`.
- Error Tree: Output đúng → Context đủ hai ý → Có context thứ ba không liên quan →
  Retrieval/Reranking precision.
- Root cause: query facets đã giữ đúng hai nguồn multi-hop nhưng bước fill top-3
  thêm chính sách nghỉ không lương không cần thiết.
- Suggested fix: với query nhiều ý, dừng sau khi mỗi facet đã có một parent đủ mạnh;
  chỉ thêm context thứ ba khi vượt ngưỡng rerank.

### 2. Mua thiết bị 55 triệu — average 0,7825

- Expected/Got: CEO phê duyệt vì đơn hàng trên 50 triệu; câu trả lời còn nêu yêu cầu
  ba báo giá và xác nhận cấu hình CNTT.
- Scores: faithfulness 0,75; relevancy 0,8800; precision 0,5; recall 1,0.
- Context: `tam_ung.md`, `mua_sam.md`, `chi_phi_expense.md`.
- Error Tree: Output đúng → Nguồn đúng ở vị trí 2 → Hai nguồn tài chính khác chủ đề
  vẫn lọt top-3 → Retrieval/Reranking precision.
- Root cause: từ khóa “phê duyệt” và số tiền xuất hiện trong nhiều quy trình tài chính.
- Suggested fix: thêm metadata topic `procurement` và ưu tiên source có thực thể
  “mua/thiết bị”; bỏ context sau khi nguồn mua sắm đã trả đủ ngưỡng tiền.

### 3. Phân loại thông tin lương — average 0,8063

- Expected/Got: Bí mật, cấp 3; mã hóa khi truyền và giới hạn need-to-know.
- Scores: faithfulness 0,8; relevancy 0,9251; precision 0,5; recall 1,0.
- Context: `phan_loai_du_lieu.md`, `ky_luong.md`, `bang_luong_2024.md`.
- Error Tree: Output đúng → Hai nguồn đầu đủ cho phép nối suy luận → Bảng lương thứ
  ba không bổ sung quy tắc phân loại → Retrieval/Reranking precision.
- Root cause: parent expansion giữ bảng lương vì cùng thực thể “lương”, dù câu hỏi
  cần chính sách bảo mật chứ không cần số tiền.
- Suggested fix: áp dụng intent/category filter `data_classification` và giới hạn hai
  parent khi đã có cả bằng chứng “lương là Bí mật” và định nghĩa cấp 3.

### 4. Thâm niên cộng phép — average 0,8175

- Expected: chính sách 2024 là mỗi 3 năm; ground truth còn đối chiếu bản 2023 là
  mỗi 5 năm. Got: trả đúng chính sách hiện hành nhưng không nhắc bản cũ.
- Scores: faithfulness 1,0; relevancy 0,7699; precision 1,0; recall 0,5.
- Context: bản 2024 cùng hai chính sách nghỉ khác; không có bản 2023.
- Error Tree: Output hiện hành đúng → Context không chứa chi tiết lịch sử trong
  ground truth → Version retrieval/evaluation alignment.
- Root cause: filter superseded hoạt động đúng cho câu hỏi hiện hành, trong khi
  ground truth lại yêu cầu thêm đối chiếu lịch sử không được hỏi trực tiếp.
- Suggested fix: hoặc rút ground truth về chính sách hiện hành, hoặc đánh dấu câu
  version-comparison để retrieve cả hai phiên bản trong cùng family.

### 5. Bắt buộc MFA — average 0,8195

- Expected: v2 bắt buộc MFA; ground truth còn nói v1 không yêu cầu. Got: trả đúng
  v2 và các phương thức MFA, không nhắc v1.
- Scores: faithfulness 1,0; relevancy 0,7779; precision 1,0; recall 0,5.
- Context: `mat_khau_v2.md`, `lam_viec_tu_xa.md`, `mua_sam.md`; không có v1.
- Error Tree: Output hiện hành đúng → Thiếu đối chiếu v1 trong context → Version
  retrieval/evaluation alignment.
- Root cause: giống case 4, filter hiện hành và ground truth lịch sử chưa cùng mục tiêu.
- Suggested fix: tạo metadata `question_requires_history`; chỉ mở lại tài liệu
  superseded khi câu hỏi/ground truth thực sự yêu cầu so sánh phiên bản.

## Case study cho presentation

Chọn câu Senior 9 năm. Quan sát ban đầu cho thấy child→parent dedupe là cần thiết
nhưng chưa đủ: ba parent khác nhau vẫn có thể đều thuộc chủ đề nghỉ phép. Đã tách
ý quanh “và”, bổ sung candidate từ phần lương và dành một vị trí cho ý phụ. Lần
chạy production xác nhận top context có cả chính sách nghỉ phép 2024 và bảng lương
2024; answer đạt faithfulness/recall 1,0 nhưng context precision bằng 0 vì parent
thứ ba về nghỉ không lương không cần thiết.

Error Tree walkthrough:

1. Output production đủ hai ý? Có: 18 ngày và 20–35 triệu/tháng.
2. Context đủ hai nguồn? Có, nhờ `query_facets()` và reserve candidate.
3. Vì sao vẫn là worst case? Context thứ ba không liên quan kéo precision xuống 0.
4. Bước tiếp theo: đặt rerank threshold/early-stop sau khi mỗi facet đã có nguồn.

## Nếu có thêm một giờ

Ưu tiên sửa context precision cho ba case đầu bằng threshold theo rerank score và
topic metadata. Sau đó tách test hiện hành khỏi test so sánh phiên bản để ground
truth khớp intent, rồi đo lại trade-off giữa top-k, recall và latency. Baseline chỉ
chấm lại khi còn ngân sách; không tiêu thêm quota chỉ để tạo delta hình thức.
