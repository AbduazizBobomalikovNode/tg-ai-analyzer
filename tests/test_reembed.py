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


def test_reembed_task_is_registered() -> None:
    assert T.reembed_all in WS.WorkerSettings.functions


async def test_reembed_writes_a_new_model_row_without_touching_the_old_one() -> None:
    """Kafolat: qayta embed **qo'shadi**, o'chirmaydi/almashtirmaydi.

    Ilgari bu yerda `"delete" not in inspect.getsource(...)` turardi — u
    noto'g'ri narsani o'lchardi: matnda "delete" yo'qligi ustidan yozish
    (`UPDATE`/`ON CONFLICT DO UPDATE`) yoki `UniqueViolation` bilan qulashdan
    **himoya qilmaydi**. Endi haqiqiy SQL tekshiriladi.
    """
    from sqlalchemy.dialects import postgresql

    captured: list[Any] = []

    class _Session:
        async def execute(self, stmt: Any) -> None:
            captured.append(stmt)

    await T._upsert_message_vectors(_Session(), [(7, [0.1, 0.2])], "gemini-embedding-2")

    (stmt,) = captured
    sql = str(stmt.compile(dialect=postgresql.dialect())).upper()
    assert sql.startswith("INSERT INTO MESSAGE_EMBEDDINGS")
    # Kalit `(message_id, model)` — eski model qatori boshqa kalitda, tegilmaydi.
    assert "ON CONFLICT (MESSAGE_ID, MODEL) DO NOTHING" in sql
    assert "DELETE" not in sql and "DO UPDATE" not in sql


def test_reembed_skips_when_embedding_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_ENABLED", "false")
    get_settings.cache_clear()
    try:
        assert asyncio.run(T.reembed_all({}, None, "gemini-embedding-2")) == {"skipped": "disabled"}
    finally:
        monkeypatch.delenv("EMBED_ENABLED", raising=False)
        get_settings.cache_clear()
