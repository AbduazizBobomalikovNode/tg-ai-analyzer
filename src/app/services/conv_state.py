"""Suhbatning ishchi holati — `conversations.state` (migratsiya 0024).

Muammo: "Otabek nima dedi?" → javob; keyin "u yana nima yozdi?". Planner
tarixni ko'radi, lekin "u" ni har turn'da **qaytadan** aniqlaydi (ba'zan
`clarify`, ba'zan butun korpus bo'ylab qidiruv), oyna ("kecha") esa ikkinchi
savolda umuman yo'q. Holat bu bilimni **kodda** saqlaydi:

* `resolved_entities` — `{"Otabek": {"chat_id": 3893, "title": "…", "kind": "private"}}`:
  savoldagi nom bilan to'liq mos kelgan chat maslahati (`agent.chat_hints`)
  yoki muvaffaqiyatli tool chaqiruvidagi chat — **va** shu turn'da
  haqiqatan ishlatilgan bo'lsa;
* `selected_chats`, `last_window` (`since/until/kind/tz`), `last_result`
  (`skill`, `evidence_refs` ≤60, raqamli `facts`), `last_answer_kind`,
  `topic` (planner qayta yozgan savol), `language` (uz/ru/en — user matnidan).

Qoidalar (`apply_to_plan`, planner **ustidan**, kodda):

1. "u / o'sha / shu odam / shu guruh / он / him …" (`REFERENT_RE`) va
   holatda oxirgi odam/chat bor → `plan.chats` = o'sha chat, hech narsa
   qayta aniqlanmaydi (tanlangan UI chatlari bundan ustun).
2. Follow-up (referent yoki follow-up intenti) va savolda yangi oyna yo'q →
   `last_window` meros qoladi.
3. `correction` / `context_switch` — tegishli qismlar tozalanadi
   (`update_after_turn`), shu turn'da meros ham yo'q.

Holat **model yozmaydi**: `update_after_turn` sof funksiya, saqlash
yugurish oxirida (`orchestrator._save_state`). Xabar matni hech qachon
kirmaydi — saqlashdan oldin `_Taint.state_leak`, planner promptiga
qo'shishdan oldin `_Taint.assert_state`. Chat sarlavhalari begona matn
bo'lgani uchun blok `<untrusted_data kind="conversation_state">` ichida.
"""

# Olmosh va til naqshlari uch tilda (kirill ham) — mahsulot matni, ASCII o'xshashligi emas.
# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.logging import get_logger
from app.services import textnorm as TN
from app.services import untrusted as U

log = get_logger(__name__)

#: Holat hajmi qopqoqlari — planner promptiga ixcham blok bo'lib tushadi.
MAX_ENTITIES = 8
MAX_SELECTED = 10
MAX_REFS = 60
MAX_FACTS = 20
MAX_TOPIC_CHARS = 300
STATE_VERSION = 1

#: Olmosh / ko'rsatish so'zi — "o'sha odam", "shu guruh", "он", "him".
#: Ataylab tor: "bu chat" (tanlangan chat) bu yerda emas — u orkestratorning
#: 6-qoidasi (tanlangan kontekst chati).
REFERENT_RE = re.compile(
    r"(?<![\w'])(?:u|uni|unga|uning|undan|unda|ular|ularga|ularni|ularning)(?![\w'])"
    r"|(?<![\w'])(?:o'sha|osha|ushbu|shu)\s+(?:odam\w*|kishi\w*|inson\w*|yigit\w*|qiz\w*"
    r"|guruh\w*|gruppa\w*|chat\w*|kanal\w*|vaqt\w*|kun\w*|payt\w*|davr\w*)"
    r"|(?<![\w'])o'sha(?![\w'])"
    r"|(?<!\w)(?:он|она|его|ему|ей|её|ее|него|нему|ней|они|их|им)(?!\w)"
    r"|(?<!\w)(?:тот|та|той|того|том|этот|эта|этого|этой|этом)\s+(?:человек\w*|чат\w*|групп\w*"
    r"|канал\w*|врем\w*|день|дня|период\w*)"
    r"|\b(?:he|she|him|her|his|they|them|their)\b"
    r"|\b(?:that|the\s+same)\s+(?:person|guy|chat|group|channel|time|day|period)\b",
    re.I,
)
#: Faqat vaqtga ishora ("o'sha vaqt", "тот день") — oyna merosi uchun.
TIME_REFERENT_RE = re.compile(
    r"(?:o'sha|osha|shu)\s+(?:vaqt|kun|payt|davr)\w*"
    r"|(?:тот|та|того|том|этот|это)\s+(?:врем|день|дня|период)\w*"
    r"|\b(?:that|the\s+same)\s+(?:time|day|period)\b",
    re.I,
)

#: Holat tozalanadigan follow-up intentlari.
CLEARING_FOLLOW_UPS = ("correction", "context_switch")
#: Oynani meros qiladigan follow-up intentlari (referentsiz ham).
CARRY_FOLLOW_UPS = ("deeper", "repeat", "reformat", "explain", "about_prior")

_CYR_RE = re.compile(r"[а-яё]", re.I)
_UZ_CYR_RE = re.compile(r"[ўқғҳ]", re.I)
_UZ_LAT_RE = re.compile(
    r"[og]['‘’ʻʼ`]|\b(?:men|sen|biz|siz|nima\w*|qanday|qaysi|qachon|kim\w*|necha\w*|qancha"
    r"|bilan|uchun|haqida|bugun|kecha|hafta\w*|yil\w*|oy\w*|xabar\w*|yoz\w*|dedi|degan"
    r"|bormi|yo'q|emas|ham|va|yoki|guruh\w*|kanal\w*)\b",
    re.I,
)
_EN_RE = re.compile(
    r"\b(?:what|who|when|how|which|did|does|the|and|messages?|wrote|write|chat|group|last|this)\b",
    re.I,
)
_WORD_RE = re.compile(r"[\w'‘’ʻʼ`]+", re.U)


# ─── holat ───────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ConvState:
    """`conversations.state` ning tipli ko'rinishi. Bo'sh holat — hamma maydon sukutda."""

    resolved_entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Eng oxirgi aniqlangan (yoki ishlatilgan) entity kaliti — referent shu.
    last_entity: str = ""
    selected_chats: list[int] = field(default_factory=list)
    last_window: dict[str, Any] | None = None
    last_result: dict[str, Any] = field(default_factory=dict)
    last_answer_kind: str = ""
    topic: str = ""
    language: str = ""
    turns: int = 0
    #: Nom to'qnashuvi bo'yicha so'ralgan `clarify`: `{"key": "Otabek", "chat_ids": [..]}`.
    #: Keyingi turn shu ro'yxatdan **bitta** chatga qaratilsa — `resolved_entities[key]`.
    pending_name: dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not (
            self.resolved_entities
            or self.selected_chats
            or self.last_window
            or self.last_result
            or self.topic
            or self.pending_name
        )

    def last_entity_ref(self) -> dict[str, Any] | None:
        if self.last_entity and self.last_entity in self.resolved_entities:
            return self.resolved_entities[self.last_entity]
        if self.resolved_entities:
            return list(self.resolved_entities.values())[-1]
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "v": STATE_VERSION,
            "resolved_entities": self.resolved_entities,
            "last_entity": self.last_entity,
            "selected_chats": list(self.selected_chats),
            "last_window": self.last_window,
            "last_result": self.last_result,
            "last_answer_kind": self.last_answer_kind,
            "topic": self.topic,
            "language": self.language,
            "turns": self.turns,
            "pending_name": self.pending_name,
        }


def _int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int) and v > 0:
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _num(v: Any) -> float | int | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v


def from_json(raw: Any) -> ConvState:
    """DB qatori → `ConvState`. Buzuq/eski maydonlar jimgina sukutga tushadi."""
    st = ConvState()
    if not isinstance(raw, Mapping):
        return st
    ents = raw.get("resolved_entities")
    if isinstance(ents, Mapping):
        for key, val in list(ents.items())[-MAX_ENTITIES:]:
            if not isinstance(val, Mapping):
                continue
            cid = _int(val.get("chat_id"))
            if cid is None:
                continue
            st.resolved_entities[U.attr(str(key), max_chars=64)] = {
                "chat_id": cid,
                "title": U.attr(str(val.get("title") or ""), max_chars=120),
                "kind": U.attr(str(val.get("kind") or ""), max_chars=16),
            }
    last = raw.get("last_entity")
    st.last_entity = U.attr(str(last), max_chars=64) if isinstance(last, str) else ""
    sel = raw.get("selected_chats")
    if isinstance(sel, list):
        st.selected_chats = [c for c in (_int(x) for x in sel) if c][:MAX_SELECTED]
    win = raw.get("last_window")
    if isinstance(win, Mapping) and isinstance(win.get("until"), str):
        st.last_window = {
            "since": win.get("since") if isinstance(win.get("since"), str) else None,
            "until": win["until"],
            "kind": str(win.get("kind") or "")[:24],
            "tz": str(win.get("tz") or "")[:48],
        }
    res = raw.get("last_result")
    if isinstance(res, Mapping):
        refs = [
            {"chat_id": c, "tg_msg_id": m}
            for r in (res.get("evidence_refs") or [])
            if isinstance(r, Mapping)
            and (c := _int(r.get("chat_id")))
            and (m := _int(r.get("tg_msg_id")))
        ][:MAX_REFS]
        facts_raw = res.get("facts")
        facts: dict[str, float | int] = {}
        if isinstance(facts_raw, Mapping):
            for k, v in list(facts_raw.items())[:MAX_FACTS]:
                n = _num(v)
                if n is not None:
                    facts[str(k)[:40]] = n
        st.last_result = {
            "skill": str(res.get("skill") or "")[:32],
            "steps": [str(s)[:32] for s in (res.get("steps") or []) if isinstance(s, str)][:6],
            "evidence_refs": refs,
            "facts": facts,
        }
    st.last_answer_kind = str(raw.get("last_answer_kind") or "")[:16]
    st.topic = U.neutralize(str(raw.get("topic") or ""))[:MAX_TOPIC_CHARS]
    lang = raw.get("language")
    st.language = lang if lang in ("uz", "ru", "en") else ""
    turns = raw.get("turns")
    st.turns = turns if isinstance(turns, int) and turns >= 0 else 0
    pending = raw.get("pending_name")
    if isinstance(pending, Mapping) and isinstance(pending.get("key"), str):
        ids = [c for c in (_int(x) for x in (pending.get("chat_ids") or [])) if c][:MAX_ENTITIES]
        if len(ids) >= 2:
            st.pending_name = {"key": U.attr(pending["key"], max_chars=64), "chat_ids": ids}
    return st


def name_words(text: str) -> list[str]:
    from app.services import agent as AG

    return [
        w
        for w in AG._hint_words(text)
        if len(w) >= AG.CHAT_HINT_MIN_CHARS and w not in AG._HINT_STOPWORDS
    ]


def resolve_name(state: ConvState | None, text: str) -> dict[str, Any] | None:
    """Matndagi nom (yoki olmosh) → holatdagi aniqlangan entity (yoki `None`).

    "Otabekka yubor" + holatda `Otabek → 110` → `{"chat_id": 110, …}`; "unga
    yubor" → oxirgi entity. Moslik `agent.chat_hints` bilan bir mezon
    (`_word_forms` + `_similar` ≥ 0.8): kalitning **har** so'zi matnda bo'lsin.
    """
    found = resolve_entity(state, text)
    return found[1] if found else None


def resolve_entity(state: ConvState | None, text: str) -> tuple[str, dict[str, Any]] | None:
    """`resolve_name` — lekin `(kalit, entity)` bilan."""
    if state is None or not state.resolved_entities:
        return None
    from app.services import agent as AG

    forms: set[str] = set()
    for word in name_words(text):
        forms |= AG._word_forms(word)
    if forms:
        for key, ent in reversed(list(state.resolved_entities.items())):
            words = name_words(key)
            if words and all(
                max(AG._similar(f, w) for f in forms) >= AG.CHAT_HINT_MIN_RATIO for w in words
            ):
                return key, ent
    if has_referent(text):
        ref = state.last_entity_ref()
        if ref is not None:
            return state.last_entity or "", ref
    return None


def name_collision(hints: Sequence[Any], question: str) -> tuple[str, list[Any]] | None:
    """Bitta nomga **teng** mos kelgan ≥2 shaxsiy chat — `(kalit, maslahatlar)` yoki `None`.

    "Otabek nima dedi?" + "Otabek Kursdosh" (0.5), "Otabek Xonadosh" (0.5) →
    to'qnashuv. Aniq mos (`score` eng yuqori va yolg'iz: "Otabek" = 1.0) —
    to'qnashuv emas. Guruh/kanallar hisobga olinmaydi: "Samarqand guruhi" bir
    necha chatga tegishli bo'lishi odatiy, planner hammasini oladi.
    """
    groups: dict[str, list[Any]] = {}
    for h in hints:
        if str(getattr(h, "kind", "") or "") != "private":
            continue
        groups.setdefault(entity_key(question, str(h.title)), []).append(h)

    def strength(h: Any) -> tuple[int, float]:
        # Nechta sarlavha so'zi mos kelgani, keyin ulushi: "Otabek Kursdoshga" da
        # "Otabek Kursdosh" (2 so'z) "Otabek" (1 so'z) dan aniqroq — to'qnashuv emas.
        words = len(name_words(str(h.title))) or 1
        return round(float(h.score) * words), float(h.score)

    for key, items in groups.items():
        if len(items) < 2:
            continue
        top = max(strength(h) for h in items)
        tied = [h for h in items if strength(h) == top]
        if len(tied) >= 2:
            return key, tied
    return None


# ─── sof yordamchilar ────────────────────────────────────────────────────────


def detect_language(text: str, *, previous: str = "") -> str:
    """Foydalanuvchi matnidan til — `uz | ru | en`. Aniq bo'lmasa oldingisi (yoki `uz`)."""
    body = text or ""
    if _CYR_RE.search(body):
        return "uz" if _UZ_CYR_RE.search(body) else "ru"
    uz = len(_UZ_LAT_RE.findall(body))
    en = len(_EN_RE.findall(body))
    if uz > en and uz:
        return "uz"
    if en > uz and en:
        return "en"
    return previous or "uz"


def has_referent(question: str) -> bool:
    return bool(REFERENT_RE.search(TN.strip_invisible(question or "")))


def is_follow_up(question: str, intent_follow_up: str) -> bool:
    """Oldingi turn'ga bog'liq savolmi — referent yoki follow-up intenti."""
    if intent_follow_up in CLEARING_FOLLOW_UPS:
        return False
    return has_referent(question) or intent_follow_up in CARRY_FOLLOW_UPS


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def window_of(state: ConvState) -> tuple[datetime | None, datetime, str, str] | None:
    """`last_window` → `(since, until, kind, tz)` (buzuq bo'lsa `None`)."""
    win = state.last_window
    if not win:
        return None
    until = _parse_dt(win.get("until"))
    if until is None:
        return None
    return _parse_dt(win.get("since")), until, str(win.get("kind") or ""), str(win.get("tz") or "")


def entity_key(question: str, title: str) -> str:
    """Savolda sarlavhaga mos kelgan so'z (qo'shimchasiz) — holat kaliti.

    "Otabekning xabarlari" + sarlavha "Otabek Karimov" → "Otabek". Moslik
    `agent.chat_hints` bilan bir mezon (`_word_forms` + `_similar` ≥ 0.8).
    Topilmasa sarlavhaning o'zi.
    """
    from app.services import agent as AG

    t_words = [w for w in AG._hint_words(title) if len(w) >= AG.CHAT_HINT_MIN_CHARS]
    for raw in _WORD_RE.findall(question or ""):
        folded = TN.uz_fold(raw).lower()
        if len(folded) < AG.CHAT_HINT_MIN_CHARS or folded in AG._HINT_STOPWORDS:
            continue
        for form in sorted(AG._word_forms(folded), key=len):
            if t_words and max(AG._similar(form, w) for w in t_words) >= AG.CHAT_HINT_MIN_RATIO:
                base = raw[: len(form)] if len(folded) == len(raw) else raw
                return U.attr(base, max_chars=64)
    return U.attr(title, max_chars=64)


@dataclass(slots=True)
class Applied:
    """`apply_to_plan` natijasi — nima o'zgardi (log / `plan_notes` uchun)."""

    chats: list[int] | None = None
    window: bool = False
    entity: str = ""
    notes: list[str] = field(default_factory=list)


def apply_to_plan(
    plan: Any,
    state: ConvState,
    *,
    question: str,
    selected: Sequence[int] = (),
    known_chats: Iterable[int] = (),
) -> Applied:
    """Holatni rejaga qo'yadi — **planner ustidan**, lekin faqat aniq holatlarda.

    `plan` — `deep_search.DeepPlan` (import halqasi bo'lmasin deb `Any`).
    """
    out = Applied()
    if state.empty:
        return out
    follow = str(getattr(plan, "intent_follow_up", "none") or "none")
    if follow in CLEARING_FOLLOW_UPS:
        out.notes.append(f"state: {follow} — nothing carried over")
        return out
    known = set(known_chats)
    text = f"{question}\n{getattr(plan, 'standalone_question', '') or ''}"
    referent = has_referent(question)
    # 1. referent → oxirgi odam/chat (tanlangan UI chatlari ustun).
    ref = state.last_entity_ref() if referent else None
    if ref and not selected and ref["chat_id"] in known:
        plan.chats = [int(ref["chat_id"])]
        out.chats = list(plan.chats)
        out.entity = state.last_entity or next(iter(state.resolved_entities), "")
        out.notes.append(f"state: referent → {out.entity} [id:{ref['chat_id']}]")
    # 2. follow-up + savolda yangi oyna yo'q → oxirgi oyna meros.
    carry = referent or follow in CARRY_FOLLOW_UPS or bool(TIME_REFERENT_RE.search(text))
    if carry and not getattr(plan, "window_from_question", False):
        win = window_of(state)
        if win is not None:
            since, until, kind, tz = win
            plan.since, plan.until = since, until
            plan.window_from_question = True
            plan.window_kind = kind or "carried"
            if tz:
                plan.window_tz = tz
            out.window = True
            out.notes.append(f"state: window carried over ({kind or 'window'})")
    return out


def _numeric_facts(facts: Iterable[Any]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for f in facts:
        name = getattr(f, "name", "")
        value = getattr(f, "value", None)
        n = _num(value)
        if name and n is not None:
            out[str(name)[:40]] = n
        if len(out) >= MAX_FACTS:
            break
    return out


def update_after_turn(
    prev: ConvState,
    *,
    question: str,
    plan: Any,
    steps: Sequence[str],
    facts: Iterable[Any] = (),
    evidence_refs: Sequence[Mapping[str, int]] = (),
    hints: Sequence[Any] = (),
    chat_titles: Mapping[int, str] | None = None,
    tool_calls: Sequence[Mapping[str, Any]] = (),
    selected: Sequence[int] = (),
    answer_kind: str = "",
    terminal: str = "",
    pending: Mapping[str, Any] | None = None,
) -> ConvState:
    """Turn'dan keyingi holat — **sof** funksiya (model qatnashmaydi).

    `hints` — `agent.ChatHint` lar (savoldagi nom → chat, deterministik);
    faqat to'liq mos kelgani (`score == 1`) va shu turn'da ishlatilgan
    chatga tegishlisi entity bo'ladi. `tool_calls` — agent jurnali: `ok`
    bo'lgan chaqiruvning `chat` argumenti ham entity (sarlavha bilan).
    `terminal` — `clarify`/`answer_direct`/`write_propose`: bunday turn
    natija/oynani o'zgartirmaydi.

    `pending` — shu turn nom to'qnashuvi bo'yicha `clarify` so'radi
    (`{"key", "chat_ids"}`): holatda eslab qolinadi. Keyingi turn reja shu
    chatlardan **bittasiga** qaratilsa, u `resolved_entities[key]` bo'ladi —
    "Otabek" keyingi savollarda (va `send_result` da) o'sha chat.
    """
    follow = str(getattr(plan, "intent_follow_up", "none") or "none")
    st = from_json(prev.to_json())  # nusxa
    pending_prev = dict(st.pending_name)
    st.pending_name = {}
    if follow == "context_switch":
        st = ConvState(language=prev.language, turns=prev.turns)
    elif follow == "correction":
        st.last_window = None
        st.last_result = {}
        st.topic = ""
    st.turns += 1
    st.language = detect_language(question, previous=st.language)
    titles = dict(chat_titles or {})
    plan_chats = list(getattr(plan, "chats", None) or [])
    picked = (
        int(plan_chats[0])
        if pending_prev and len(plan_chats) == 1 and plan_chats[0] in pending_prev["chat_ids"]
        else None
    )

    def resolve_pending() -> None:
        if picked is None:
            return
        key = str(pending_prev["key"])
        st.resolved_entities.pop(key, None)
        st.resolved_entities[key] = {
            "chat_id": picked,
            "title": U.attr(titles.get(picked, ""), max_chars=120),
            "kind": "private",
        }
        st.last_entity = key

    if terminal:
        resolve_pending()
        if pending and isinstance(pending.get("key"), str):
            ids = [c for c in (_int(x) for x in (pending.get("chat_ids") or [])) if c]
            if len(ids) >= 2:
                st.pending_name = {
                    "key": U.attr(pending["key"], max_chars=64),
                    "chat_ids": ids[:MAX_ENTITIES],
                }
        return st
    seen = {int(r["chat_id"]) for r in evidence_refs if r.get("chat_id")}
    # ── entity'lar: chat maslahati + muvaffaqiyatli tool chaqiruvi ──────────
    # Maslahat faqat shu turn'da **ishlatilgan** chatga bo'lsa entity bo'ladi:
    # reja o'sha chatga qaratilgan yoki dalil o'sha chatdan kelgan. Turn'ga
    # bitta (eng yaxshi) maslahat — ikkinchi darajali moslik referentni
    # o'g'irlamasin.
    new_key = ""
    for h in hints:
        cid = int(h.chat_id)
        if not (cid in plan_chats or (not plan_chats and cid in seen)):
            continue
        key = entity_key(question, str(h.title))
        st.resolved_entities.pop(key, None)
        st.resolved_entities[key] = {
            "chat_id": cid,
            "title": U.attr(str(h.title), max_chars=120),
            "kind": U.attr(str(getattr(h, "kind", "") or ""), max_chars=16),
        }
        new_key = key
        break
    for call in tool_calls:
        if not call.get("ok"):
            continue
        args = call.get("args") if isinstance(call.get("args"), Mapping) else {}
        tool_chat = _int(args.get("chat")) if args else None
        if tool_chat is None or tool_chat not in titles:
            continue
        key = entity_key(question, titles[tool_chat])
        if key not in st.resolved_entities:
            st.resolved_entities[key] = {
                "chat_id": tool_chat,
                "title": U.attr(titles[tool_chat], max_chars=120),
                "kind": "",
            }
            new_key = new_key or key
    if not new_key and len(plan_chats) == 1:
        # Referent turn'i: odam o'zgarmadi — oxirgi entity o'sha chatda qoladi.
        for key, val in st.resolved_entities.items():
            if val.get("chat_id") == plan_chats[0]:
                new_key = key
    if new_key:
        st.last_entity = new_key
    resolve_pending()
    while len(st.resolved_entities) > MAX_ENTITIES:
        st.resolved_entities.pop(next(iter(st.resolved_entities)))
    st.selected_chats = [int(c) for c in (selected or plan_chats)][:MAX_SELECTED]
    since = getattr(plan, "since", None)
    until = getattr(plan, "until", None)
    if getattr(plan, "window_from_question", False) and until is not None:
        st.last_window = {
            "since": since.isoformat() if since else None,
            "until": until.isoformat(),
            "kind": str(getattr(plan, "window_kind", "") or ""),
            "tz": str(getattr(plan, "window_tz", "") or ""),
        }
    main = [s for s in steps if s not in ("compute",)] or list(steps)
    refs = [
        {"chat_id": int(r["chat_id"]), "tg_msg_id": int(r["tg_msg_id"])}
        for r in evidence_refs
        if r.get("chat_id") and r.get("tg_msg_id")
    ][:MAX_REFS]
    st.last_result = {
        "skill": main[-1] if main else "",
        "steps": [str(s) for s in steps][:6],
        "evidence_refs": refs or list(st.last_result.get("evidence_refs") or [])
        if follow in ("reformat", "explain", "about_prior")
        else refs,
        "facts": _numeric_facts(facts),
    }
    st.last_answer_kind = answer_kind[:16]
    topic = str(getattr(plan, "standalone_question", "") or question)
    st.topic = U.neutralize(topic)[:MAX_TOPIC_CHARS]
    return st


def strip_text(state: ConvState) -> ConvState:
    """Holatning **matnsiz** nusxasi: sarlavhalar va `topic` bo'sh, qolgani (id,
    oyna, havolalar, raqamlar) o'zgarmaydi.

    `_save_state` zaxira yo'li: taint qo'riqchisi matnli maydonda shubha
    topsa, holat butunlay tashlanmaydi — aks holda `last_result.evidence_refs`
    eski turn'da qolib, "shuni ruscha yoz" / "bu qaysi chatdan edi?" eski
    dalilni o'qirdi (daily-30 t6/t7).
    """
    st = from_json(state.to_json())
    st.resolved_entities = {key: {**val, "title": ""} for key, val in st.resolved_entities.items()}
    st.topic = ""
    return st


def render_block(state: ConvState) -> str:
    """Planner uchun ixcham holat bloki — JSON, konvert ichida (sarlavhalar begona).

    Xabar matni yo'q: faqat id, sarlavha, raqam, planner/foydalanuvchi matni.
    """
    if state.empty:
        return "conversation_state: none"
    compact: dict[str, Any] = {}
    if state.resolved_entities:
        compact["resolved_entities"] = state.resolved_entities
        if state.last_entity:
            compact["last_entity"] = state.last_entity
    if state.selected_chats:
        compact["selected_chats"] = state.selected_chats
    if state.last_window:
        compact["last_window"] = state.last_window
    if state.last_result:
        res = dict(state.last_result)
        refs = res.pop("evidence_refs", [])
        res["evidence_refs"] = len(refs)
        compact["last_result"] = res
    if state.last_answer_kind:
        compact["last_answer_kind"] = state.last_answer_kind
    if state.topic:
        compact["topic"] = state.topic
    if state.language:
        compact["language"] = state.language
    if state.pending_name:
        compact["pending_name"] = state.pending_name
    body = U.neutralize(json.dumps(compact, ensure_ascii=False, default=str))
    return (
        f'<untrusted_data kind="conversation_state">\n{body}\n</untrusted_data>\n'
        "conversation_state: kept by code across turns. Resolve 'u/o'sha/shu odam/shu guruh/"
        "o'sha vaqt/он/him/that chat' from resolved_entities.last_entity and last_window "
        "before anything else — do not clarify what the state already answers. "
        "pending_name: the user was just asked which of these chat ids they meant by that "
        "name — their reply picks ONE of them (target only that chat id)."
    )


# ─── DB ──────────────────────────────────────────────────────────────────────


async def load(session_factory: Any, conversation_id: int) -> ConvState:
    """Holatni o'qiydi. Xato bo'lsa bo'sh holat — holat yordamchi, javobdan muhim emas."""
    from app.db.models import Conversation

    try:
        async with session_factory() as session:
            conv = await session.get(Conversation, conversation_id)
            return from_json(conv.state if conv is not None else None)
    except Exception as exc:
        log.warning("conv_state.load_failed", conversation_id=conversation_id, error=str(exc)[:200])
        return ConvState()


async def save(session_factory: Any, conversation_id: int, state: ConvState) -> bool:
    from app.db.models import Conversation

    try:
        async with session_factory() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv is None:
                return False
            conv.state = state.to_json()
            await session.flush()
        return True
    except Exception as exc:
        log.warning("conv_state.save_failed", conversation_id=conversation_id, error=str(exc)[:200])
        return False


__all__ = [
    "REFERENT_RE",
    "Applied",
    "ConvState",
    "apply_to_plan",
    "detect_language",
    "entity_key",
    "from_json",
    "has_referent",
    "is_follow_up",
    "load",
    "name_collision",
    "name_words",
    "render_block",
    "resolve_entity",
    "resolve_name",
    "save",
    "strip_text",
    "update_after_turn",
    "window_of",
]
