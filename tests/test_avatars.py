"""Chat profil rasmi (ui/R1a) — `services/avatars.py`.

Qo'riqlaydi:
  * fayl **faqat** `avatar_photo_id` o'zgarganda yuklanadi (RPC tejash);
  * rasmi yo'q chatda maydonlar tozalanadi va fayl o'chadi;
  * 512 KB dan katta yoki noma'lum formatdagi javob rad etiladi;
  * `FloodWaitError` — retry YO'Q, matn `chats.avatar_error` ga tushadi;
  * bo'sh `history` bucket'ida umuman yuklanmaydi;
  * **"media saqlanmaydi"**: fayl yuklaydigan yagona modul shu.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import avatars as AV

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "app"


# ─── soxta DB va klient ──────────────────────────────────────────────────────


class _Row(SimpleNamespace):
    pass


def _chat_row(**kw: Any) -> _Row:
    base = {
        "id": 1,
        "account_id": 5,
        "tg_peer_id": 777,
        "access_hash": 42,
        "type": "channel",
        "title": "Kanal",
        "about": None,
        "participants_count": None,
        "avatar_photo_id": None,
        "avatar_mime": None,
        "avatar_updated_at": None,
        "avatar_error": None,
    }
    base.update(kw)
    return _Row(**base)


class _FakeSession:
    def __init__(self, rows: dict[int, _Row]) -> None:
        self.rows = rows

    async def get(self, model: Any, ident: int) -> Any:
        return self.rows.get(int(ident))

    def expunge(self, obj: Any) -> None: ...


class _FakeScope:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *exc: Any) -> None: ...


class _FakeClient:
    def __init__(self, data: Any = b"\xff\xd8\xff" + b"x" * 100, raise_: Any = None) -> None:
        self.data = data
        self.raise_ = raise_
        self.calls = 0

    async def download_profile_photo(self, entity: Any, file: Any = None) -> Any:
        self.calls += 1
        if self.raise_ is not None:
            raise self.raise_
        return self.data


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Soxta DB + pool + sozlamalar; `history` bucket ochiq."""
    from app.mtproto import pool as pool_mod

    rows: dict[int, _Row] = {}
    session = _FakeSession(rows)
    monkeypatch.setattr(AV, "session_scope", lambda: _FakeScope(session))
    monkeypatch.setattr(
        AV, "get_settings", lambda: SimpleNamespace(data_dir=str(tmp_path), avatars_enabled=True)
    )
    monkeypatch.setattr(AV, "_guard", lambda account_id: None)

    client = _FakeClient()
    entity: dict[str, Any] = {"ent": SimpleNamespace(photo=SimpleNamespace(photo_id=999))}

    async def fake_entity(account_id: int, peer_id: int, **kw: Any) -> Any:
        return entity["ent"]

    async def fake_get(account_id: int) -> Any:
        return client

    monkeypatch.setattr(pool_mod.pool, "entity", fake_entity)
    monkeypatch.setattr(pool_mod.pool, "get", fake_get)
    return SimpleNamespace(rows=rows, client=client, entity=entity, tmp=tmp_path)


# ─── yuklash mantiqi ─────────────────────────────────────────────────────────


async def test_downloads_when_photo_id_is_new(env: Any) -> None:
    env.rows[1] = _chat_row(avatar_photo_id=999)
    assert await AV.refresh_avatar(5, 1) is True
    row = env.rows[1]
    assert env.client.calls == 1
    assert row.avatar_photo_id == 999
    assert row.avatar_mime == "image/jpeg"
    assert row.avatar_error is None
    assert AV.path_for(1, "image/jpeg").exists()


async def test_skips_when_photo_id_unchanged(env: Any) -> None:
    env.rows[1] = _chat_row(avatar_photo_id=999)
    assert await AV.refresh_avatar(5, 1) is True
    assert env.client.calls == 1
    # ikkinchi chaqiruv: id o'zgarmagan, fayl joyida → yuklash YO'Q
    assert await AV.refresh_avatar(5, 1) is False
    assert env.client.calls == 1

    # rasm almashdi → qayta yuklanadi
    env.entity["ent"] = SimpleNamespace(photo=SimpleNamespace(photo_id=1000))
    assert await AV.refresh_avatar(5, 1) is True
    assert env.client.calls == 2
    assert env.rows[1].avatar_photo_id == 1000


async def test_no_photo_clears_fields_and_file(env: Any) -> None:
    env.rows[1] = _chat_row(avatar_photo_id=999)
    await AV.refresh_avatar(5, 1)
    path = AV.path_for(1, "image/jpeg")
    assert path.exists()

    env.entity["ent"] = SimpleNamespace(photo=None)
    assert await AV.refresh_avatar(5, 1) is False
    row = env.rows[1]
    assert row.avatar_photo_id is None and row.avatar_mime is None
    assert row.avatar_updated_at is not None  # navbatga qaytmaydi
    assert not path.exists()


async def test_oversize_rejected(env: Any) -> None:
    env.rows[1] = _chat_row(avatar_photo_id=999)
    env.client.data = b"\xff\xd8\xff" + b"x" * (AV.MAX_AVATAR_BYTES + 1)
    assert await AV.refresh_avatar(5, 1) is False
    assert "too_large" in (env.rows[1].avatar_error or "")
    assert not AV.path_for(1, "image/jpeg").exists()


async def test_unknown_format_rejected(env: Any) -> None:
    env.rows[1] = _chat_row(avatar_photo_id=999)
    env.client.data = b"GIF89a" + b"x" * 50
    assert await AV.refresh_avatar(5, 1) is False
    assert env.rows[1].avatar_error == "unknown_format"


async def test_flood_wait_is_not_retried(env: Any) -> None:
    from telethon import errors as tg_errors

    env.rows[1] = _chat_row(avatar_photo_id=999)
    env.client.raise_ = tg_errors.FloodWaitError(request=None)
    env.client.raise_.seconds = 42
    assert await AV.refresh_avatar(5, 1) is False
    assert env.client.calls == 1  # bitta urinish — retry yo'q
    assert "flood_wait" in (env.rows[1].avatar_error or "")


async def test_bucket_guard_blocks_download(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AV, "_guard", lambda account_id: "error: rate_limited:history")
    env.rows[1] = _chat_row(avatar_photo_id=999)
    assert await AV.refresh_avatar(5, 1) is False
    assert env.client.calls == 0
    assert "rate_limited:history" in (env.rows[1].avatar_error or "")


async def test_disabled_by_settings(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        AV, "get_settings", lambda: SimpleNamespace(data_dir=str(env.tmp), avatars_enabled=False)
    )
    env.rows[1] = _chat_row(avatar_photo_id=999)
    assert await AV.refresh_avatar(5, 1) is False
    assert env.client.calls == 0


async def test_foreign_account_is_ignored(env: Any) -> None:
    env.rows[1] = _chat_row(account_id=6, avatar_photo_id=999)
    assert await AV.refresh_avatar(5, 1) is False
    assert env.client.calls == 0


def test_photo_id_of_handles_empty_photo() -> None:
    assert AV.photo_id_of(SimpleNamespace(photo=None)) is None
    assert AV.photo_id_of(SimpleNamespace(photo=SimpleNamespace(photo_id=0))) is None
    assert AV.photo_id_of(SimpleNamespace(photo=SimpleNamespace(photo_id=7))) == 7


# ─── peer turiga qarab `GetFull*` ────────────────────────────────────────────


async def test_full_info_dispatches_by_input_peer_type() -> None:
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    seen: list[Any] = []

    class _Client:
        async def __call__(self, req: Any) -> Any:
            seen.append(req)
            return SimpleNamespace(
                full_chat=SimpleNamespace(about="x"),
                full_user=SimpleNamespace(about="x", id=1),
                chats=[],
                users=[],
            )

    client = _Client()
    await AV.full_info(client, tlt.InputPeerChannel(channel_id=1, access_hash=2))
    await AV.full_info(client, tlt.InputPeerChat(chat_id=3))
    await AV.full_info(client, tlt.InputPeerUser(user_id=4, access_hash=5))
    assert isinstance(seen[0], tlf.channels.GetFullChannelRequest)
    assert isinstance(seen[1], tlf.messages.GetFullChatRequest)
    assert isinstance(seen[2], tlf.users.GetFullUserRequest)

    with pytest.raises(TypeError):
        await AV.full_info(client, object())


# ─── invariant: faqat shu modul fayl yuklaydi ────────────────────────────────

_DOWNLOAD_RE = re.compile(r"\b(download_media|download_file|GetFileRequest)\b")
#: `avatars.py` — yagona yuklovchi; `allowlist.py` — registr (chaqiruvchi emas).
_ALLOWED_FILES = {SRC / "services" / "avatars.py", SRC / "mtproto" / "allowlist.py"}


def test_no_other_module_downloads_files() -> None:
    """ "Media saqlanmaydi" — `upload.GetFile` yo'li faqat profil rasmi uchun.

    Bu test sinsa: kimdir ilovaga fayl yuklab olish yo'lini qo'shgan. Avval
    CLAUDE.md ("Kontent") va migratsiya 0017 izohini o'qing — bu siyosat
    o'zgarishi user qaroriga bog'liq.
    """
    leaked: list[str] = []
    for path in SRC.rglob("*.py"):
        if path in _ALLOWED_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for m in _DOWNLOAD_RE.finditer(text):
            leaked.append(f"{path.relative_to(SRC)}: {m.group(0)}")
    assert not leaked, "fayl yuklash `services/avatars.py` dan tashqarida: " + ", ".join(leaked)


def test_upload_getfile_is_read_only() -> None:
    from app.mtproto.allowlist import DENIED, READ, is_write

    assert "upload.GetFileRequest" in READ
    assert not is_write("upload.GetFileRequest")
    assert "upload.GetFileRequest" not in DENIED
