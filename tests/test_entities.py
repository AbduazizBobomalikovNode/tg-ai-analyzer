"""Entity chiqarish va indekslash (phase4/A) — regex, normalizatsiya, idempotentlik.

Qo'riqlanadigan asosiy invariant: **telefon raqami va email hech qachon
chiqarilmaydi**. `phone_like` turi yo'q, email esa skanerdan oldin matndan
olib tashlanadi — aks holda `ali@example.com` dagi `@example` username,
`example.com` esa domen bo'lib indeksga tushardi.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.services import entities as E


def _values(text: str, tg: Any = None) -> list[tuple[str, str]]:
    return [(e.kind, e.value) for e in E.extract_entities(text, tg)]


# ─── username / hashtag / cashtag ────────────────────────────────────────────


def test_usernames_respect_telegram_shape() -> None:
    got = _values("@nurbek va @Ali_2000 salom, @ali (qisqa) va @" + "x" * 40)
    assert (E.KIND_USERNAME, "@nurbek") in got
    assert (E.KIND_USERNAME, "@ali_2000") in got  # lowercase
    assert (E.KIND_USERNAME, "@ali") not in got  # 5 belgidan qisqa
    assert all(len(v) <= E.MAX_VALUE_CHARS for _, v in got)


def test_username_raw_is_kept_alongside_normalized_value() -> None:
    (ent,) = E.extract_entities("@Nurbek keldi")
    assert (ent.kind, ent.value, ent.value_raw) == (E.KIND_USERNAME, "@nurbek", "@Nurbek")


def test_hashtags_accept_cyrillic_and_fold_to_one_value() -> None:
    lat = _values("reliz #Deploy bugun")
    cyr = _values("релиз #деплой bugun")
    assert lat == [(E.KIND_HASHTAG, "#deploy")]
    assert cyr == [(E.KIND_HASHTAG, "#deploy")]  # uz_fold kirillni lotinga tushiradi


def test_hashtag_needs_a_leading_letter() -> None:
    assert _values("#2026 yil va #1") == []
    assert _values("#uz2026") == [(E.KIND_HASHTAG, "#uz2026")]


def test_cashtag_is_uppercase_only() -> None:
    assert _values("$BTC o'sdi, $btc emas") == [(E.KIND_CASHTAG, "$btc")]


# ─── URL / domen / t.me ──────────────────────────────────────────────────────


def test_url_variants_collapse_to_registrable_domain() -> None:
    assert _values("https://WWW.Example.com/a/b?x=1") == [(E.KIND_URL_DOMAIN, "example.com")]
    assert _values("http://sub.example.com") == [(E.KIND_URL_DOMAIN, "example.com")]
    assert _values("www.example.com yaxshi") == [(E.KIND_URL_DOMAIN, "example.com")]
    # ikki bo'g'inli public suffix — uch yorliq saqlanadi
    assert _values("https://shop.olcha.com.uz/x") == [(E.KIND_URL_DOMAIN, "olcha.com.uz")]


def test_trailing_punctuation_is_not_part_of_the_domain() -> None:
    assert _values("qarang https://example.com.") == [(E.KIND_URL_DOMAIN, "example.com")]


def test_url_fragment_does_not_become_a_hashtag() -> None:
    assert _values("https://example.com/a#deploy") == [(E.KIND_URL_DOMAIN, "example.com")]


def test_tme_forms_keep_the_path_as_value() -> None:
    assert _values("https://t.me/Nurbek") == [(E.KIND_TME_LINK, "nurbek")]
    assert _values("t.me/c/12345/678") == [(E.KIND_TME_LINK, "c/12345/678")]
    assert _values("https://telegram.me/kanal/42") == [(E.KIND_TME_LINK, "kanal/42")]
    # t.me havolasi domen sifatida **ikkinchi marta** yozilmaydi
    assert all(k != E.KIND_URL_DOMAIN for k, _ in _values("https://t.me/kanal"))


def test_bare_tme_host_without_path_is_ignored() -> None:
    assert _values("https://t.me/") == []


# ─── hech qachon: telefon va email ───────────────────────────────────────────


def test_phone_numbers_are_never_extracted() -> None:
    text = "tel: +998 90 123 45 67, yana +998901234567 va 90-123-45-67"
    assert E.extract_entities(text) == []
    assert "phone_like" not in E.KINDS


def test_emails_are_never_extracted_and_leak_neither_user_nor_domain() -> None:
    got = _values("yozing: Ali.Vali+tag@Example.co.uk yoki boshqa")
    assert got == []


def test_email_next_to_a_real_entity_does_not_poison_it() -> None:
    got = _values("ali@example.com va @nurbek #deploy")
    assert got == [(E.KIND_USERNAME, "@nurbek"), (E.KIND_HASHTAG, "#deploy")]


# ─── mention_name — faqat Telegram entity'laridan ────────────────────────────


def test_mention_name_comes_only_from_telegram_entities() -> None:
    text = "Oʻtkir Rahimov keldi"
    ent = type("MessageEntityMentionName", (), {})()
    ent.offset, ent.length, ent.user_id = 0, 14, 7  # type: ignore[attr-defined]
    got = E.extract_entities(text, [ent])
    assert [(e.kind, e.value) for e in got] == [(E.KIND_MENTION_NAME, "otkir rahimov")]
    assert got[0].value_raw == "Oʻtkir Rahimov"  # xom shakl saqlanadi
    # entity'siz aynan shu matndan hech qanday ism **taxmin qilinmaydi**
    assert E.extract_entities(text) == []


def test_mention_name_uses_utf16_offsets() -> None:
    text = "🙂 Alisher keldi"  # emoji — 2 ta UTF-16 birligi
    ent = type("MessageEntityMentionName", (), {})()
    ent.offset, ent.length = 3, 7  # type: ignore[attr-defined]
    got = E.extract_entities(text, [ent])
    assert [(e.kind, e.value) for e in got] == [(E.KIND_MENTION_NAME, "alisher")]


def test_other_telegram_entity_types_are_ignored() -> None:
    ent = type("MessageEntityBold", (), {})()
    ent.offset, ent.length = 0, 4  # type: ignore[attr-defined]
    assert E.extract_entities("bold matn", [ent]) == []


def test_uz_apostrophes_in_names_fold_to_one_value() -> None:
    values = []
    for spelling in ("Oʻtkir", "O'tkir", "O‘tkir"):
        ent = type("MessageEntityMentionName", (), {})()
        ent.offset, ent.length = 0, len(spelling)  # type: ignore[attr-defined]
        values.append(E.extract_entities(spelling, [ent])[0].value)
    assert set(values) == {"otkir"}


# ─── umumiy xulq ─────────────────────────────────────────────────────────────


def test_duplicates_collapse_and_cap_holds() -> None:
    assert _values("@nurbek @NURBEK") == [(E.KIND_USERNAME, "@nurbek")]
    spam = " ".join(f"#tag{i}" for i in range(200))
    assert len(E.extract_entities(spam)) == E.MAX_ENTITIES_PER_MESSAGE


def test_empty_input_does_nothing() -> None:
    assert E.extract_entities("") == []
    assert E.extract_entities("   salom dunyo   ") == []


def test_kinds_match_the_migration_check_list() -> None:
    """DB `CHECK` va koddagi ro'yxat ajralib ketmasin (`phone_like` ikkalasida ham yo'q)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "migrations" / "versions" / "0012_node_entities.py"
    spec = importlib.util.spec_from_file_location("m0012", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.KINDS) == set(E.KINDS)
    assert "phone_like" not in module.KINDS
    assert module.down_revision == "0011" and module.revision == "0012"


def test_registrable_domain_edge_cases() -> None:
    assert E.registrable_domain("WWW.Sub.Example.com:8443") == "example.com"
    assert E.registrable_domain("example.com.") == "example.com"
    assert E.registrable_domain("localhost") == "localhost"


# ─── planner satri → (kind, value) ───────────────────────────────────────────


def test_parse_entity_query_recognizes_shapes() -> None:
    assert E.parse_entity_query("@Nurbek") == (E.KIND_USERNAME, "@nurbek")
    assert E.parse_entity_query("#Деплой") == (E.KIND_HASHTAG, "#deploy")
    assert E.parse_entity_query("$BTC") == (E.KIND_CASHTAG, "$btc")
    assert E.parse_entity_query("https://t.me/c/12/34") == (E.KIND_TME_LINK, "c/12/34")
    assert E.parse_entity_query("www.Example.com") == (E.KIND_URL_DOMAIN, "example.com")


def test_parse_entity_query_rejects_free_text_and_emails() -> None:
    assert E.parse_entity_query("to'lov muddati") is None
    assert E.parse_entity_query("ali@example.com") is None
    assert E.parse_entity_query("") is None
    assert E.parse_entity_query("@ali") is None  # username shakli emas


def test_resolve_entities_dedupes_and_drops_unknown() -> None:
    pairs = E.resolve_entities(["@Nurbek", "@nurbek", "nima gap", "#deploy"])
    assert pairs == [(E.KIND_USERNAME, "@nurbek"), (E.KIND_HASHTAG, "#deploy")]


# ─── indekslash (soxta sessiya) ──────────────────────────────────────────────


class _FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    """Navbatdagi natijalarni qaytaradi va bajarilgan SQL matnini saqlaydi."""

    def __init__(self, *results: list[Any]) -> None:
        self.queue = list(results)
        self.sql: list[str] = []

    async def execute(self, stmt: Any, *a: Any, **kw: Any) -> _FakeResult:
        self.sql.append(str(stmt))
        return _FakeResult(self.queue.pop(0) if self.queue else [])


NOW = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)


async def test_index_message_entities_inserts_with_on_conflict_do_nothing() -> None:
    session = _FakeSession([])  # mavjud qatorlar yo'q
    n = await E.index_message_entities(
        session,  # type: ignore[arg-type]
        account_id=7,
        chat_id=3,
        message_id=101,
        text="@nurbek #deploy https://example.com",
        seen_at=NOW,
    )
    assert n == 3
    assert "node_entities" in session.sql[0]  # mavjudlarini o'qish
    insert_sql = session.sql[-1]
    assert "INSERT INTO node_entities" in insert_sql
    assert "ON CONFLICT" in insert_sql and "DO NOTHING" in insert_sql


async def test_index_is_idempotent_when_everything_already_exists() -> None:
    existing = [
        (1, 101, E.KIND_USERNAME, "@nurbek"),
        (2, 101, E.KIND_HASHTAG, "#deploy"),
    ]
    session = _FakeSession(existing)
    n = await E.index_message_entities(
        session,  # type: ignore[arg-type]
        account_id=7,
        chat_id=3,
        message_id=101,
        text="@nurbek #deploy",
    )
    assert n == 0
    assert len(session.sql) == 1  # faqat SELECT — na DELETE, na INSERT


async def test_edited_message_loses_its_stale_entities() -> None:
    existing = [(9, 101, E.KIND_USERNAME, "@eskiuser")]
    session = _FakeSession(existing)
    n = await E.index_message_entities(
        session,  # type: ignore[arg-type]
        account_id=7,
        chat_id=3,
        message_id=101,
        text="@yangiuser",
    )
    assert n == 1
    assert "DELETE FROM node_entities" in session.sql[1]
    assert "INSERT INTO node_entities" in session.sql[2]


async def test_message_without_entities_does_no_insert() -> None:
    session = _FakeSession([])
    assert (
        await E.index_message_entities(
            session,  # type: ignore[arg-type]
            account_id=7,
            chat_id=3,
            message_id=101,
            text="oddiy matn, telefon +998901234567",
        )
        == 0
    )
    assert all("INSERT" not in s for s in session.sql)


async def test_empty_batch_touches_no_sql() -> None:
    session = _FakeSession()
    assert await E.index_messages_entities(session, []) == 0  # type: ignore[arg-type]
    assert session.sql == []


# ─── F3/L9: `t.me/+<raqam>` — telefon raqami, taklif havolasi emas ───────────


def test_phone_shaped_tme_link_is_dropped() -> None:
    """Raqamli `t.me/+…` indeksga tushmaydi — bu telefon raqami.

    Telegram raqam bo'yicha chatni aynan shu ko'rinishda ochadi. Indeksda
    qolsa raqam qidiriladigan bo'lardi va "telefon raqami hech qachon
    qaytarilmaydi" qoidasi yon eshikdan buzilardi.
    """
    for raw in (
        "https://t.me/+998901234567",
        "t.me/998901234567",
        "https://telegram.me/+12025550143",
    ):
        assert [e for e in E.extract_entities(raw) if e.kind == E.KIND_TME_LINK] == []
        assert E.parse_entity_query(raw) is None


def test_invite_links_with_letters_are_kept() -> None:
    """Harfli taklif hash'i — bu haqiqiy havola, indeksda qoladi."""
    (ent,) = [
        e for e in E.extract_entities("https://t.me/+AbC12dEf34") if e.kind == E.KIND_TME_LINK
    ]
    assert ent.value == "+abc12def34"
    assert E.parse_entity_query("t.me/joinchat/AbC12dEf34") is not None


def test_normal_tme_username_link_still_works() -> None:
    (ent,) = [e for e in E.extract_entities("https://t.me/nurbek") if e.kind == E.KIND_TME_LINK]
    assert ent.value == "nurbek"


def test_extract_entities_order_matches_the_docstring() -> None:
    """L8: tartib — **tur bo'yicha** (URL → username → hashtag), matn bo'yicha emas."""
    values = [e.value for e in E.extract_entities("@nurbek #Deploy https://www.example.com/a")]
    assert values == ["example.com", "@nurbek", "#deploy"]
