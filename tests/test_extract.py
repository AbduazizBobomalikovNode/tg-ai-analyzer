"""`extract` ko'nikmasi (`services/extract.py`) — deterministik ajratish.

Qo'riqlaydi:
  * **telefon raqami hech qachon chiqmaydi** — na blokda, na faktda, na
    summary'da, na dalilda; faqat soni (`ex_phone_n`);
  * havola (+ domen), manzil, narx, sana/vaqt, fayl, mention/heshteg naqshlari;
  * narx telefon deb yutilmaydi; e-mail mention emas;
  * fayl nomi saqlanmaydi — faqat `document` sanog'i;
  * sitata tokeni blokda buzilmaydi va (oyna yo'lida) dalilga qo'shiladi.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from app.services import deep_search as DS
from app.services import extract as EX
from app.services import output_formats as OF
from app.services import search as S
from app.services import skills as SK

PHONES = (
    "+998 90 123 45 67",
    "+998901234567",
    "998901234567",
    "(90) 123-45-67",
    "90 123 45 67",
    "901234567",
    "+7 912 345-67-89",
    "+1 (555) 123 4567",
)
#: Chiqishda hech qachon bo'lmasligi kerak bo'lgan raqam ketma-ketliklari.
_DIGITS = re.compile(r"\d")


def _only_digits(s: str) -> str:
    return "".join(_DIGITS.findall(s))


@pytest.mark.parametrize("phone", PHONES)
def test_strip_phones_masks_every_format(phone: str) -> None:
    text = f"Aloqa: {phone}, kechqurun."
    assert EX.count_phones(text) == 1
    out = EX.strip_phones(text)
    assert EX.PHONE_MASK in out and _only_digits(phone) not in _only_digits(out)


def test_money_and_dates_are_not_phones() -> None:
    for text in (
        "Narxi 900 000 so'm",
        "150000000 so'm",
        "Sana: 20.09.2026",
        "2026-09-20 14:30",
        "$1200",
        "Summa: 2 500 000 сум",
    ):
        assert EX.count_phones(text) == 0, text
        assert EX.strip_phones(text) == text


def test_links_with_domain_and_phone_inside_url() -> None:
    clean = EX.strip_phones(
        "Batafsil: https://Example.com/a?b=1. Kanal: t.me/ijara_uz, www.olx.uz/d/123) "
        "va wa.me/998901234567"
    )
    got = EX.find_links(clean)
    assert ("https://Example.com/a?b=1", "example.com") in got
    assert ("https://t.me/ijara_uz", "t.me") in got
    assert ("https://www.olx.uz/d/123", "olx.uz") in got
    assert all("998901234567" not in u for u, _ in got)


def test_addresses_label_and_pattern() -> None:
    got = EX.find_addresses(
        "Manzil: Chilonzor tumani, 5-mavze, 12-uy\n"
        "Ofis Yunusobod tumani, Amir Temur ko'chasi 15-uy da joylashgan. Tel keyin.\n"
        "Адрес: ул. Навои, дом 5"
    )
    values = [v for v, _ in got]
    assert "Chilonzor tumani, 5-mavze, 12-uy" in values
    assert any("Amir Temur ko'chasi" in v for v in values)
    assert "ул. Навои, дом 5" in values
    assert not EX.find_addresses("Bugun havo yaxshi, uyga ketdim.")


def test_prices_dates_files_mentions() -> None:
    prices = EX.find_prices("Narxi 150 000 so'm, ijara $400, depozit 2.5 mln so'm. Цена: 1 200")
    values = {v: c for v, c in prices}
    assert values.get("150 000 so'm") == "UZS" and values.get("$400") == "USD"
    assert values.get("2.5 mln so'm") == "UZS"
    assert values.get("1 200") == ""  # yorliq ("Цена:") qiymatdan olib tashlanadi
    dates = [v for v, _ in EX.find_dates("20.09.2026 da, 2026-10-01 gacha, 15 sentyabr 14:30")]
    assert dates == ["20.09.2026", "2026-10-01", "15 sentyabr", "14:30"]
    files = EX.find_files("Hisobot: hisobot_2026.pdf va https://x.uz/doc.pdf")
    assert files == [("hisobot_2026.pdf", "pdf")]
    mentions = [v for v, _ in EX.find_mentions("@ijara_bot va #ijara #c12m34 a@b.com @ab")]
    assert mentions == ["@ijara_bot", "#ijara"]


def test_normalize_kinds() -> None:
    assert EX.normalize_kinds("links") == ["links"]
    assert EX.normalize_kinds("prices, addresses") == ["addresses", "prices"]
    assert EX.normalize_kinds("all") == list(EX.KINDS)
    assert EX.normalize_kinds("", "guruhdagi havolalarni chiqar") == ["links"]
    assert EX.normalize_kinds("", "e'lonlardagi narx va manzillar") == ["addresses", "prices"]
    assert EX.normalize_kinds("", "ro'yxatini sanab chiq") == list(EX.KINDS)  # "sana" emas


def _rows() -> list[EX.Row]:
    now = datetime(2026, 9, 20, tzinfo=UTC)
    return [
        EX.Row(1, 12, 101, now, "Ijara 400$ Manzil: Chilonzor 5-mavze. Tel: +998 90 123 45 67"),
        EX.Row(2, 12, 102, now, "https://olx.uz/a va https://olx.uz/a ; 901234567 ga qo'ng'iroq"),
        EX.Row(3, 12, 103, now, "", "document"),
        EX.Row(4, 12, 104, now, "Yana https://olx.uz/a @ijara_uz"),
    ]


def test_extract_counts_dedupes_and_never_outputs_phones() -> None:
    ex = EX.extract(_rows(), list(EX.KINDS))
    assert ex.scanned == 4 and ex.phones == 2 and ex.documents == 1
    [link] = ex.items["links"]
    assert link.value == "https://olx.uz/a" and link.count == 3 and link.tg_msg_id == 102
    block = EX.render_block(ex, list(EX.KINDS), "uz")
    assert "#c12m101" in block and "​" not in block
    assert "fayl nomi saqlanmaydi" in block and "Telefon raqamlari: 2 ta" in block
    for phone in ("901234567", "998901234567"):
        assert phone not in _only_digits(block)
    facts = {f.name: f.value for f in EX.facts_for(ex, list(EX.KINDS))}
    assert facts["ex_phone_n"] == 2 and facts["ex_links_n"] == 1 and facts["ex_documents_n"] == 1
    assert facts["ex_addresses_n"] == 1 and facts["ex_prices_n"] == 1


def _ctx(question: str, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "raw_question": question,
        "locale": "uz",
        "plan": DS.DeepPlan(),
        "selected": [],
        "chats_by_id": {12: SimpleNamespace(id=12, title="Ijara", type="group", link_prefix="")},
        "claims": [],
        "ranked": [],
        "taint": DS._Taint(),
    }
    base.update(kw)
    ns = SimpleNamespace(**base)
    ns.add_evidence = lambda hits, cap=400: SK.RunState.add_evidence(ns, hits, cap=cap)  # type: ignore[arg-type]
    return ns


async def test_run_extract_from_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        S.Hit(r.message_id, r.tg_msg_id, 1.0, "fts", chat_id=r.chat_id, text=r.text)
        for r in _rows()
    ]
    ctx = _ctx("telefon raqamlari va havolalarni chiqar", ranked=list(hits))
    res = await EX.run_extract(ctx, {"kind": "all"})
    assert res.summary["source"] == "evidence" and len(ctx.ranked) == 4  # yangi dalil yo'q
    everything = "\n".join(
        [*ctx.claims, *(f.render() for f in res.facts), str(res.summary), *res.notes]
    )
    assert "901234567" not in _only_digits(everything)
    assert any("phone numbers are never output" in n for n in res.notes)
    [block] = OF.extract_blocks(ctx.claims)
    assert "https://olx.uz/a" in block


async def test_run_extract_window_adds_cited_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows()

    async def fake_rows(ctx: Any, chat_ids: Any, since: Any, until: Any) -> list[EX.Row]:
        assert list(chat_ids) == [12]
        return rows

    monkeypatch.setattr(EX, "_window_rows", fake_rows)
    ctx = _ctx("bu hafta guruhdagi havolalar", selected=[12])
    res = await EX.run_extract(ctx, {})
    assert res.summary == {"source": "window", "messages": 4, "phones": 2, "links": 1}
    assert [h.tg_msg_id for h in ctx.ranked] == [102]  # jadvalda keltirilgan xabar
    assert ctx.ranked[0].source == "extract"
    assert "901234567" not in _only_digits(ctx.ranked[0].text)
    assert any("last 7 days" in n for n in res.notes)


def test_extract_is_registered_quarantined() -> None:
    skill = SK.SKILLS["extract"]
    assert skill.quarantined and skill.min_mode == "fast" and skill.cost_usd == 0.0
    assert set(skill.args) == {"kind", "chat_id"}
    assert "Telefon" in skill.description_uz and "Phone" in skill.description_en
