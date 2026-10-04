# Lab 18: Production RAG Pipeline

**K4-Track3A · Ngày 18 · Production RAG**
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency       | Bắt buộc? | Dùng cho                                     |
| ---------------- | --------- | -------------------------------------------- |
| Docker (Qdrant)  | ✅ Có     | M2 Dense Search                              |
| Python 3.11+     | ✅ Có     | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| Key MWAPI + Claude model ID | ⚠️ M4+M5 | Generation, RAGAS eval và Enrichment; không cần key OpenAI |

**Pre-download models** (tránh timeout trong lab):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**

```bash
git clone <repo-url>
cd K4-Track3A-DAY18-VuVanDien-2A202602418-ProductionRAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
git clone <repo-url>
cd K4-Track3A-DAY18-VuVanDien-2A202602418-ProductionRAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```

_(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)_

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**

```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Chỉ tạo khi chưa có .env; điền key MWAPI và model
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**

```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Chỉ tạo khi chưa có .env; điền key MWAPI và model
python naive_baseline.py                # Khởi tạo baseline
```

_(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)_

## Dùng Claude qua MWAPI

Sửa `.env` hiện có, giữ lại key của bạn. Không cần tạo tài khoản hoặc key OpenAI:

```dotenv
LLM_API_KEY=YOUR_MWAPI_KEY
LLM_API_FORMAT=anthropic
LLM_BASE_URL=https://api.mwapi.dev
LLM_MODEL=YOUR_ALLOWED_CLAUDE_MODEL_ID
LLM_TIMEOUT=60
EVAL_EMBEDDING_MODEL=BAAI/bge-m3
```

Thay hai giá trị `YOUR_...` bằng key và model ID thật trong dashboard MWAPI.
Code cũng đọc key từ `MWAPI_API_KEY`, `ANTHROPIC_API_KEY` hoặc `OPENAI_API_KEY`
nếu `LLM_API_KEY` chưa được điền, để tương thích `.env` cũ.
Giá trị model không được mặc định vì quyền truy cập phụ thuộc key của bạn.

```powershell
python check_llm.py             # Kiểm tra cấu hình, không gọi API
python check_llm.py --models    # Gọi GET /v1/models để lấy model ID
python check_llm.py --test      # Một request ngắn có tính phí để thử kết nối
```

Nếu gateway không cho liệt kê model, lấy ID từ dashboard. Chuẩn mặc định là
Anthropic Messages (`/v1/messages`). Chỉ khi tài khoản hỗ trợ OpenAI Chat
Completions, đổi `LLM_API_FORMAT=openai` và
`LLM_BASE_URL=https://api.mwapi.dev/v1`; key và model vẫn là của MWAPI.

Baseline, production và enrichment dùng chung `src/llm.py`. RAGAS được truyền
Claude evaluator và `HuggingFaceEmbeddings` local; không gọi OpenAI Embeddings.
Lần đầu dùng embeddings cần tải model. Khi thiếu cấu hình hoặc API lỗi, generation
trả context đầu tiên, enrichment dùng fallback local. RAGAS ghi trạng thái
`skipped`/`failed` cùng điểm 0; các điểm này không phải kết quả đánh giá thật.

M1–M5 đã được triển khai. Hierarchical retrieval lấy child để tìm kiếm và mở rộng
lại parent cho LLM; các parent trùng bị loại bỏ để giữ đủ nguồn cho multi-hop.
Truy vấn hiện hành lọc chính sách superseded, truy vấn có mốc năm/phiên bản vẫn
có thể truy xuất lịch sử. M5 mặc định index contextual text; summary và HyQA được
lưu để phân tích, chưa ghép vào index nhằm tránh thêm nội dung suy diễn vào context.

Có thể chạy `python main.py --offline` khi key chưa được cấp quyền: BM25, dense,
reranker và Qdrant vẫn chạy thật; Claude/RAGAS được bỏ qua. Report giữ đủ 20 mẫu
Q&A/context với metric từng câu là `null`, `scored_questions=0`. Đây là kiểm tra
luồng local, chưa đủ điều kiện nộp RAGAS. `python check_lab.py` sẽ trả exit code 1
nếu report chưa được đánh giá thật hoặc tests/TODO/deliverables còn thiếu.
Lệnh online kiểm tra quyền LLM bằng một request ngắn trước khi chạy toàn bộ, tránh
lặp lại lỗi quyền trên từng chunk. `python scripts/collect_evidence.py` ghi số liệu
chunking/retrieval/benchmark vào `reports/implementation_evidence.json`.

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python src/pipeline.py --local-enrichment # Chỉ production; tiết kiệm 117 API calls M5
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

`--local-enrichment` vẫn chạy M5 cho toàn bộ chunk bằng contextual fallback có tính
xác định, nhưng chỉ dùng Claude cho answer generation và RAGAS. Report ghi
`enrichment_mode=local_fallback` để kết quả minh bạch và tái lập được.

## Cấu trúc repo

```
K4-Track3A-DAY18-VuVanDien-2A202602418-ProductionRAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ 5 modules đã implement, không còn TODO bắt buộc
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động                                  |
| ---------- | ------------------------------------------ |
| 10 phút    | Setup môi trường + chạy`naive_baseline.py` |
| 90 phút    | Implement M1 → M2 → M3 → M4 → M5           |
| 20 phút    | Chạy pipeline + RAGAS + failure analysis   |
| 30 phút    | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**`K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`_(Ví dụ: `K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)_
- **Hạn chót nộp bài:** **23h59 ngày diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).
