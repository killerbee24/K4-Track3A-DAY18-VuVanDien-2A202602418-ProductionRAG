"""Inspect LLM configuration; --models lists IDs, --test makes one paid call."""

import argparse
import sys

from config import LLM_API_FORMAT, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT
from src.llm import api_base_url, generate_text, safe_error, validate_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="List models allowed by the key")
    parser.add_argument("--test", action="store_true", help="Make one short, billable LLM call")
    args = parser.parse_args()
    print(f"API format: {LLM_API_FORMAT}")
    print(f"Base URL: {api_base_url()}")
    print(f"Model: {LLM_MODEL or '(chưa điền LLM_MODEL)'}")
    print(f"API key: {'đã điền' if LLM_API_KEY and LLM_API_KEY != 'sk-...' else 'chưa điền'}")
    try:
        if args.models:
            if not LLM_API_KEY or LLM_API_KEY == "sk-...":
                raise ValueError("Điền LLM_API_KEY trong .env trước khi liệt kê model.")
            if LLM_API_FORMAT not in {"anthropic", "openai"}:
                raise ValueError("LLM_API_FORMAT phải là anthropic hoặc openai.")
            import httpx
            root = api_base_url().removesuffix("/v1")
            headers = {"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01"}
            if LLM_API_FORMAT == "openai":
                headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
            response = httpx.get(f"{root}/v1/models", headers=headers, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            models = response.json().get("data", [])
            if not models:
                raise ValueError("Gateway không trả model ID. Lấy ID từ dashboard MWAPI.")
            for model in models:
                print(model["id"])
        if args.test:
            validate_config()
            print(generate_text("Chỉ trả lời đúng một từ: OK.", "Kiểm tra kết nối.", max_tokens=16))
        if not args.models and not args.test:
            validate_config()
            print("Cấu hình đầy đủ. Chưa gửi request API.")
        return 0
    except Exception as error:  # noqa: BLE001 - CLI must sanitize provider errors.
        print(f"Lỗi: {safe_error(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
