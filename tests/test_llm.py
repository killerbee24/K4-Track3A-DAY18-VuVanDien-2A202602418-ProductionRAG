"""MWAPI integration checks: mocked HTTP, no real keys or billable requests."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from src import llm, m4_eval, m5_enrichment, pipeline
import naive_baseline


@pytest.fixture(autouse=True)
def fake_config(monkeypatch, offline_llm):
    monkeypatch.setattr(llm, "LLM_API_KEY", "fake-test-key")
    monkeypatch.setattr(llm, "LLM_MODEL", "claude-test-model")
    monkeypatch.setattr(llm, "LLM_API_FORMAT", "anthropic")
    monkeypatch.setattr(llm, "LLM_BASE_URL", "https://api.mwapi.dev")
    cached_client = llm.get_client
    cached_client.cache_clear()
    yield
    cached_client.cache_clear()


@pytest.mark.parametrize("api_format,base,expected", [
    ("anthropic", "https://api.mwapi.dev/v1/", "https://api.mwapi.dev"),
    ("anthropic", "https://api.mwapi.dev", "https://api.mwapi.dev"),
    ("openai", "https://api.mwapi.dev", "https://api.mwapi.dev/v1"),
    ("openai", "https://api.mwapi.dev/v1", "https://api.mwapi.dev/v1"),
])
def test_base_url(monkeypatch, api_format, base, expected):
    monkeypatch.setattr(llm, "LLM_API_FORMAT", api_format)
    monkeypatch.setattr(llm, "LLM_BASE_URL", base)
    assert llm.api_base_url() == expected


def test_anthropic_http_request(monkeypatch):
    from anthropic import Anthropic
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant",
            "model": "claude-test-model", "content": [
                {"type": "text", "text": "Đúng "}, {"type": "text", "text": "context."},
            ], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = Anthropic(api_key="fake-test-key", base_url=llm.api_base_url(),
                           http_client=transport, max_retries=0)
        monkeypatch.setattr(llm, "get_client", lambda: client)
        assert llm.generate_text("System instruction", "Question", max_tokens=32) == "Đúng context."
    request = seen[0]
    assert str(request.url) == "https://api.mwapi.dev/v1/messages"
    assert request.headers["x-api-key"] == "fake-test-key"
    payload = json.loads(request.content)
    assert payload["system"] == "System instruction"
    assert payload["model"] == "claude-test-model"
    assert payload["max_tokens"] == 32
    assert payload["messages"] == [{"role": "user", "content": "Question"}]


def test_openai_http_request(monkeypatch):
    from openai import OpenAI
    monkeypatch.setattr(llm, "LLM_API_FORMAT", "openai")
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={
            "id": "chat_test", "object": "chat.completion", "created": 0,
            "model": "claude-test-model", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "OK"},
            }],
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(api_key="fake-test-key", base_url=llm.api_base_url(),
                        http_client=transport, max_retries=0)
        monkeypatch.setattr(llm, "get_client", lambda: client)
        assert llm.generate_text("System", "Question") == "OK"
    request = seen[0]
    assert str(request.url) == "https://api.mwapi.dev/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(request.content)["messages"][0]["role"] == "system"


def test_missing_model_does_not_call_sdk(monkeypatch):
    monkeypatch.setattr(llm, "LLM_MODEL", "")
    assert not llm.is_configured()
    with pytest.raises(ValueError, match="LLM_MODEL"):
        llm.generate_text("system", "question")


def test_client_factory_uses_gateway_and_cache():
    with patch("anthropic.Anthropic") as factory:
        first = llm.get_client()
        assert llm.get_client() is first
    factory.assert_called_once_with(
        api_key="fake-test-key", base_url="https://api.mwapi.dev",
        timeout=llm.LLM_TIMEOUT, max_retries=1,
    )


def test_invalid_format_and_secret_redaction(monkeypatch):
    monkeypatch.setattr(llm, "LLM_API_FORMAT", "invalid")
    with pytest.raises(ValueError, match="LLM_API_FORMAT"):
        llm.validate_config()
    assert "fake-test-key" not in llm.safe_error(ValueError("bad key fake-test-key"))


def test_eval_models_use_gateway_and_local_embeddings():
    with patch("langchain_community.embeddings.HuggingFaceEmbeddings") as embedder:
        judge, embeddings = llm.get_eval_models()
    assert judge.model == "claude-test-model"
    assert judge.anthropic_api_url == "https://api.mwapi.dev"
    assert judge.anthropic_api_key.get_secret_value() == "fake-test-key"
    assert embeddings is embedder.return_value
    assert embedder.call_args.kwargs["model_name"] == llm.EVAL_EMBEDDING_MODEL


def test_openai_eval_models_still_use_local_embeddings(monkeypatch):
    monkeypatch.setattr(llm, "LLM_API_FORMAT", "openai")
    with patch("langchain_community.embeddings.HuggingFaceEmbeddings") as embedder:
        judge, embeddings = llm.get_eval_models()
    assert judge.model_name == "claude-test-model"
    assert judge.openai_api_base == "https://api.mwapi.dev/v1"
    assert embeddings is embedder.return_value


def test_combined_one_call_preserves_source(monkeypatch):
    response = '```json\n' + json.dumps({
        "summary": "Short", "questions": ["How many?"], "context": "Policy",
        "metadata": {"source": "invented.md", "parent_id": "invented"},
    }) + '\n```'
    generate = Mock(return_value=response)
    monkeypatch.setattr(m5_enrichment, "generate_text", generate)
    result = m5_enrichment.enrich_chunks([
        {"text": "Original text", "metadata": {"source": "real.md", "parent_id": "parent_1"}},
    ])[0]
    assert generate.call_count == 1
    assert result.original_text == "Original text"
    assert result.enriched_text == "Policy\n\nOriginal text"
    assert result.auto_metadata["source"] == "real.md"
    assert result.auto_metadata["parent_id"] == "parent_1"


@pytest.mark.parametrize("response", ["not JSON", '{"questions": null, "metadata": null}', "[]"])
def test_combined_malformed_response_falls_back(monkeypatch, response):
    monkeypatch.setattr(m5_enrichment, "generate_text", Mock(return_value=response))
    result = m5_enrichment.enrich_chunks([{"text": "Original text", "metadata": {}}])[0]
    assert "Original text" in result.enriched_text
    assert isinstance(result.hypothesis_questions, list)
    assert isinstance(result.auto_metadata, dict)


def test_no_key_enrichment_offline(monkeypatch):
    monkeypatch.setattr(llm, "LLM_API_KEY", "")
    generate = Mock(side_effect=AssertionError("Must not call API"))
    monkeypatch.setattr(m5_enrichment, "generate_text", generate)
    result = m5_enrichment.enrich_chunks([{"text": "Original text.", "metadata": {}}])[0]
    assert result.enriched_text != result.original_text
    generate.assert_not_called()


def test_ragas_explicit_models_and_nan_handling(monkeypatch):
    rows = [{"question": "Q", "answer": "A", "contexts": ["C"], "ground_truth": "GT",
             "faithfulness": 0.8, "answer_relevancy": float("nan"),
             "context_precision": 0.7, "context_recall": 0.9}]
    evaluator = Mock(return_value=SimpleNamespace(
        to_pandas=lambda: SimpleNamespace(iterrows=lambda: enumerate(rows)),
    ))
    modules = {
        "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_dict=lambda d: d)),
        "ragas": SimpleNamespace(evaluate=evaluator),
        "ragas.metrics": SimpleNamespace(**{m: Mock() for m in m4_eval.METRICS}),
        "ragas.run_config": SimpleNamespace(RunConfig=Mock()),
    }
    monkeypatch.setattr(m4_eval, "get_eval_models", lambda: ("claude-judge", "local-embeddings"))
    with patch.dict(sys.modules, modules):
        result = m4_eval.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert evaluator.call_args.kwargs["llm"] == "claude-judge"
    assert evaluator.call_args.kwargs["embeddings"] == "local-embeddings"
    assert result["faithfulness"] == 0.8
    assert result["answer_relevancy"] == 0.0
    assert result["evaluation_status"] == "partial"
    assert len(result["per_question"]) == 1


def test_ragas_missing_configuration_is_skipped(monkeypatch):
    monkeypatch.setattr(llm, "LLM_MODEL", "")
    result = m4_eval.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert result["evaluation_status"] == "skipped"
    assert all(result[m] == 0 for m in m4_eval.METRICS)


def test_ragas_errors_do_not_expose_key():
    with patch.dict(sys.modules, {"datasets": SimpleNamespace(
        Dataset=SimpleNamespace(from_dict=Mock(side_effect=ValueError("bad key fake-test-key"))),
    ), "ragas": SimpleNamespace(evaluate=Mock()),
        "ragas.metrics": SimpleNamespace(**{m: Mock() for m in m4_eval.METRICS}),
        "ragas.run_config": SimpleNamespace(RunConfig=Mock())}):
        result = m4_eval.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert result["evaluation_status"] == "failed"
    assert "fake-test-key" not in result["evaluation_error"]


def test_report_keeps_question_data_and_evaluation_status(tmp_path):
    row = m4_eval.EvalResult("Q", "A", ["C"], "GT", 0.8, 0.7, 0.9, 0.6)
    path = tmp_path / "report.json"
    results = {**{m: getattr(row, m) for m in m4_eval.METRICS},
               "per_question": [row], "evaluation_status": "success", "invalid_scores": 0}
    m4_eval.save_report(results, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1
    assert report["per_question"][0]["answer"] == "A"
    assert report["evaluation_status"] == "success"
    assert set(report["aggregate"]) == set(m4_eval.METRICS)


def test_pipeline_generation_and_api_failure(monkeypatch):
    search = SimpleNamespace(search=lambda query: [SimpleNamespace(text="Context", score=1, metadata={})])
    reranker = SimpleNamespace(rerank=lambda *args, **kwargs: [])
    generate = Mock(return_value="Claude answer")
    monkeypatch.setattr(pipeline, "generate_text", generate)
    assert pipeline.run_query("Q", search, reranker) == ("Claude answer", ["Context"])
    generate.side_effect = RuntimeError("Gateway error")
    assert pipeline.run_query("Q", search, reranker) == ("Context", ["Context"])


def test_baseline_uses_shared_generation(monkeypatch):
    monkeypatch.setattr(naive_baseline, "load_documents", lambda: [
        {"text": "Context", "metadata": {"source": "policy.md"}},
    ])
    search = Mock()
    search.search.return_value = [SimpleNamespace(text="Context")]
    monkeypatch.setattr(naive_baseline, "DenseSearch", lambda: search)
    monkeypatch.setattr(naive_baseline, "load_test_set", lambda: [{"question": "Q", "ground_truth": "GT"}])
    generate = Mock(return_value="Claude answer")
    evaluate = Mock(return_value={m: 0.8 for m in m4_eval.METRICS})
    monkeypatch.setattr(naive_baseline, "generate_text", generate)
    monkeypatch.setattr(naive_baseline, "evaluate_ragas", evaluate)
    monkeypatch.setattr(naive_baseline, "save_report", Mock())
    monkeypatch.setattr(naive_baseline.os, "makedirs", Mock())
    naive_baseline.main()
    assert generate.call_count == 1
    evaluate.assert_called_once_with(["Q"], ["Claude answer"], [["Context"]], ["GT"])
