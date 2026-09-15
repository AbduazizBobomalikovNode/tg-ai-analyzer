"""phase4/B — embedding modelini almashtirgandan keyingi qayta indekslash.

Bu yerda **tarmoq ham, DB ham yo'q**: CLI navbatga nima qo'yishini va ish
konfiguratsiyani qanday o'qishini tekshiramiz. Haqiqiy embedding yo'li
`tests/test_llm_adapters.py` da (soxta klient bilan).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app import main as M
from app.config import get_settings
from app.worker import settings as WS
from app.worker import tasks as T


class _Queue:
    """`app.worker.queue` o'rnini bosadi — Redis'siz."""

    def __init__(self, job_id: str | None = "job-1") -> None:
        self.calls: list[dict[str, Any]] = []
        self.job_id = job_id
        self.closed = False

    async def enqueue(
        self,
        function: str,
        *args: Any,
        job_id: str | None = None,
        defer_by: float | None = None,
        **kwargs: Any,
    ) -> str | None:
        self.calls.append({"function": function, "args": args, "kwargs": kwargs, "job_id": job_id})
        return self.job_id

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> _Queue:
    q = _Queue()
    monkeypatch.setattr("app.worker.queue.enqueue", q.enqueue)
    monkeypatch.setattr("app.worker.queue.close_queue", q.close)
    return q


def test_reembed_cli_enqueues_with_explicit_model(queue: _Queue) -> None:
    rc = asyncio.run(M._reembed(["--model", "gemini-embedding-2", "--account", "3"]))
    assert rc == 0
    call = queue.calls[0]
    assert call["function"] == "reembed_all"
    assert call["args"] == (3,)
    assert call["kwargs"]["model"] == "gemini-embedding-2"
    # deterministik job id — bir akkauntga parallel re-embed yo'q
    assert call["job_id"] == "reembed:acc:3"
    assert queue.closed


def test_reembed_cli_defaults_to_configured_model(queue: _Queue) -> None:
    rc = asyncio.run(M._reembed([]))
    assert rc == 0
    call = queue.calls[0]
    assert call["args"] == (None,)
    assert call["kwargs"]["model"] == get_settings().embed_model == "gemini-embedding-001"
    assert call["job_id"] == T.REEMBED_JOB_ID


def test_reembed_cli_reports_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """ARQ bir xil job_id ni qaytarmaydi — CLI buni xato sifatida ko'rsatadi."""
    q = _Queue(job_id=None)
    monkeypatch.setattr("app.worker.queue.enqueue", q.enqueue)
    monkeypatch.setattr("app.worker.queue.close_queue", q.close)
    assert asyncio.run(M._reembed([])) == 1


def test_reembed_cli_rejects_unknown_argument(queue: _Queue) -> None:
    assert asyncio.run(M._reembed(["--everything"])) == 2
    assert queue.calls == []


def test_reembed_task_is_registered_and_deletes_nothing() -> None:
    assert T.reembed_all in WS.WorkerSettings.functions
    # Kafolat: bu ish hech qachon o'chirmaydi (eski vektorlar joyida qoladi).
    import inspect

    src = inspect.getsource(T.reembed_all)
    assert "delete" not in src.lower()


def test_reembed_skips_when_embedding_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_ENABLED", "false")
    get_settings.cache_clear()
    try:
        assert asyncio.run(T.reembed_all({}, None, "gemini-embedding-2")) == {"skipped": "disabled"}
    finally:
        monkeypatch.delenv("EMBED_ENABLED", raising=False)
        get_settings.cache_clear()
