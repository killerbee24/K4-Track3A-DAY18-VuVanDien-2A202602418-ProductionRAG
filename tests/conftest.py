"""Unit tests run offline; paid Claude calls belong in check_llm.py --test."""

import pytest


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "LLM_API_KEY", "")
    monkeypatch.setattr(llm, "LLM_MODEL", "")
