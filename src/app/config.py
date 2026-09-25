"""Markazlashgan konfiguratsiya. Hamma env o'qish shu yerdan o'tadi."""

from __future__ import annotations

import base64
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WriteMode(StrEnum):
    READ_ONLY = "read_only"
    WRITE_WITH_CONFIRM = "write_with_confirm"
    AUTONOMOUS = "autonomous"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ─── Bot API ───
    bot_token: str
    control_bot_id: int
    allowed_user_ids: str = ""

    # ─── MTProto ───
    tg_api_id: int
    tg_api_hash: str
    tg_device_model: str = "Desktop"
    tg_system_version: str = "Windows 10"
    tg_app_version: str = "5.7.1"

    # ─── Xavfsizlik ───
    master_key_b64: str
    webapp_base_url: str = "https://example.com"

    # ─── Web UI ───
    web_session_ttl_hours: int = Field(default=24 * 14, ge=1, le=24 * 90)
    # Auth flow: telefon → kod → 2FA. Bitta oqim shuncha daqiqa yashaydi.
    web_auth_flow_ttl_min: int = Field(default=10, ge=2, le=30)
    # Bir IP'dan 10 daqiqada nechta login boshlash mumkin (brute-force/spam)
    web_auth_rate_per_ip: int = Field(default=5, ge=1, le=100)
    # AI chatga kontekst sifatida beriladigan oxirgi xabarlar limiti
    # >200 faqat sinxronlangan (DB) chatlar uchun — jonli GetHistory 200 bilan cheklanadi
    web_context_max_messages: int = Field(default=1000, ge=10, le=5000)
    web_context_default_messages: int = Field(default=50, ge=5, le=1000)
    # Har javobdan keyin arzon model bilan avtomatik baho (relevance/usefulness/grounded)
    web_auto_eval: bool = True
    # Baholanadigan javoblar ulushi (0.0-1.0) — xarajatni kamaytirish uchun sampling
    web_auto_eval_sample: float = Field(default=1.0, ge=0.0, le=1.0)
    # Agent (tool) sikli chegaralari — xarajat qopqog'i
    agent_max_iterations: int = Field(default=5, ge=1, le=12)
    agent_tool_result_tokens: int = Field(default=12_000, ge=1_000, le=60_000)
    # Bitta yugurishda **bajarilgan** (to'silmagan) tool chaqiruvlari qopqog'i.
    # O'lchov (`evals/runs/conv-20260917-144833.md`): iteratsiya shifti yolg'iz
    # o'zi yetmaydi — model har raundda 4 tadan chaqiradi va bitta savol 6-11
    # bajarilgan chaqiruvga ($0.11-$0.40) cho'ziladi. Qopqoqqa yetganda
    # "bor narsa bilan javob ber" turn'i qo'shiladi va tool'lar o'chiriladi.
    agent_max_executed_calls: int = Field(default=6, ge=1, le=40)
    # Yugurish narxi qopqog'i (USD, `llm/pricing.estimate_cost` taxmini bo'yicha,
    # har iteratsiyadan keyin tekshiriladi). 0 = o'chiq.
    agent_max_cost_usd: float = Field(default=0.15, ge=0.0, le=10.0)
    # 7-bosqich: fayllar (generatsiya rasmlar) va auto-reply
    data_dir: str = "data"
    image_gen_enabled: bool = True
    image_max_per_day: int = Field(default=30, ge=0, le=1000)  # foydalanuvchi bo'yicha
    autoreply_enabled: bool = True
    autoreply_context_messages: int = Field(default=15, ge=3, le=50)
    # Yozish amallari (6-bosqich): taklif muddati va akkaunt bo'yicha soatlik limit
    write_proposal_ttl_hours: int = Field(default=24, ge=1, le=168)
    write_rate_per_hour: int = Field(default=20, ge=1, le=500)
    # Telegram server-side qidiruv (agent tool'lari). Har chaqiruv real MTProto
    # so'rovi — akkaunt bo'yicha daqiqalik shift. 0 = tool'lar o'chiq.
    tg_search_rate_per_minute: int = Field(default=10, ge=0, le=120)
    # Embedding indeksi (vektor qidiruv). Gemini embed — arzon, lekin o'chirsa bo'ladi
    embed_enabled: bool = True
    embed_batch_per_run: int = Field(default=500, ge=50, le=5000)

    # ─── Infra ───
    database_url: str
    redis_url: str

    # ─── LLM: umumiy ───
    # Standart provider: auto | claude | gemini | deepseek.
    # `auto` — Claude kredensiali (Claude Code / Anthropic login) topilsa
    # `claude`, aks holda `gemini`. Vazifa darajasida `llm_task_*` bilan
    # bekor qilinadi.
    llm_provider: str = "auto"
    # Tanlangan provider vazifani bajara olmasa (masalan DeepSeek'da embedding
    # yo'q) — shu providerga o'tiladi. Gemini'dan boshqasini qo'yish tavsiya
    # etilmaydi: faqat u to'liq imkoniyatli.
    llm_fallback_provider: str = "gemini"
    # Fon vazifalari (worker digest'lari, hafta tugunlari, briefing, judge)
    # Gemini **Flex** tier'ida yuradi: narx yarmi, evaziga kechikish 1-15 daq
    # va 429/503 da server zaxirasi yo'q (adapter bir marta standart tier'da
    # qayta uradi). Interaktiv yo'l har doim standart. false = hammasi standart.
    llm_flex_background: bool = True

    # Vazifa → `provider:model` spec. Bo'sh = llm_provider + standart model.
    llm_task_route: str = ""
    llm_task_search: str = ""
    llm_task_tools: str = ""
    llm_task_deep: str = ""
    llm_task_embed: str = ""
    llm_task_image: str = ""

    # ─── Claude / Anthropic ───
    # Kredensial ustuvorligi (birinchi topilgani ishlatiladi):
    #   1. ANTHROPIC_API_KEY            — oddiy API kalit
    #   2. CLAUDE_CODE_OAUTH_TOKEN      — `claude setup-token` chiqargan token
    #      (Claude Code obunasi), yoki ANTHROPIC_AUTH_TOKEN
    #   3. diskdagi profil              — `ant auth login` / Claude Code login
    #      (~/.config/anthropic yoki ANTHROPIC_CONFIG_DIR)
    # Hech biri yo'q → `auto` rejimda Gemini ishlatiladi.
    anthropic_api_key: str = ""
    anthropic_auth_token: str = ""
    claude_code_oauth_token: str = ""
    anthropic_base_url: str = ""  # bo'sh = api.anthropic.com
    claude_model_router: str = "claude-haiku-4-5"  # arzon: intent, digest, judge
    claude_model_fast: str = "claude-sonnet-5"  # search/tools — sifat/narx muvozanati
    claude_model_deep: str = "claude-opus-5"  # "chuqur tahlil" tugmasi

    # ─── Claude Code CLI (`claude_code` provider) ───
    # `claude -p` subprocess — Claude Code'ning o'z login sessiyasidan (Keychain /
    # ~/.claude) foydalanadi, token nusxalash shart emas. `auto` tartibida API
    # kredensiali topilmasa va CLI login qilingan bo'lsa tanlanadi.
    claude_code_auto: bool = True  # False = auto rejimda CLI hisobga olinmaydi
    claude_code_bin: str = ""  # bo'sh = PATH'dan `claude`
    claude_code_timeout: int = Field(default=300, ge=10, le=3600)  # soniya, bitta so'rov

    # ─── Gemini ───
    # 2026-09: 2.5 avlod suhbat modellari (`flash-lite`/`flash`/`pro`) yangi
    # mijozlarga yopiq (404) — defaultlar 3.x avlodga o'tkazilgan
    # (`app.llm.router.RETIRING`, `docs/llm.md`). Rasm/embedding modeli
    # o'zgarmagan (o'lchov/moslik talab qiladi, avval mahsus o'tish yo'li bilan).
    gemini_api_key: str = ""
    gemini_model_router: str = "gemini-3.5-flash-lite"
    gemini_model_fast: str = "gemini-3.5-flash"
    gemini_model_deep: str = "gemini-3.1-pro-preview"
    gemini_model_image: str = "gemini-2.5-flash-image"
    gemini_model_embed: str = "gemini-embedding-001"
    embed_dim: int = 768
    # Fikrlash (thinking) byudjeti — token. Fikrlash tokenlari `max_tokens`
    # byudjetidan yeyiladi, ya'ni cheklanmagan holda uzun javob jumla
    # o'rtasida kesilardi (jonli: `gemini-3.1-pro-preview`, sintez).
    # `gemini-3*` da bu "low"/"high" darajasiga aylanadi, `gemini-2.5*` da
    # token soni bo'lib ketadi. 0 = sozlamani umuman yubormaslik (SDK sukuti).
    gemini_thinking_budget: int = Field(default=1024, ge=0, le=32_768)
    # `gemini-3*` uchun **vazifa bo'yicha** fikrlash darajasi
    # (`GEMINI_THINKING_LEVEL_<TASK>`): "" = sozlamani yubormaslik (model
    # sukuti). ROUTE (filtr, map, judge, digest) — `minimal`, flash-lite'ning
    # o'z sukuti: ilgari hammaga `low` berilib, arzon vazifada fikrlash
    # **oshirilardi** (docs/cost-research.md, B3-7). `minimal` ni
    # qo'llamaydigan model (3.7/3.8-flash, 3.1-pro) uchun adapter `low` ga
    # tushiradi. `GEMINI_THINKING_BUDGET=0` hammasini o'chiradi.
    gemini_thinking_level_route: Literal["", "minimal", "low", "medium", "high"] = "minimal"
    gemini_thinking_level_search: Literal["", "minimal", "low", "medium", "high"] = "low"
    gemini_thinking_level_tools: Literal["", "minimal", "low", "medium", "high"] = "low"
    gemini_thinking_level_deep: Literal["", "minimal", "low", "medium", "high"] = "low"
    # Deep sintezi **dayjest** shaklida (`synth_budget(...).kind == "digest"`:
    # oyna / sanoq javobi) — chaqiruv bo'yicha fikrlash darajasi
    # (`LLM().chat(..., thinking=...)`). Jonli: sanoq sintezi 2,617 fikrlash
    # tokeni yedi (sintez narxining yarmi), holbuki javob — tayyor raqam va
    # da'volarni so'zga aylantirish. "" = o'chiq (vazifa darajasi qoladi).
    # `minimal` ni qo'llamaydigan model (3.7/3.8-flash, 3.1-pro) uchun adapter
    # `low` ga tushiradi.
    deep_digest_thinking: Literal["", "minimal", "low", "medium", "high"] = "minimal"

    # ─── DeepSeek (OpenAI-mos API) ───
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_chat: str = "deepseek-chat"  # function calling BOR
    deepseek_model_reasoner: str = "deepseek-reasoner"  # function calling YO'Q

    # ─── Kuzatuv va limitlar (8-bosqich) ───
    app_version: str = "0.8.0"
    sentry_dsn: str = ""  # bo'sh = o'chiq; `pip install -e ".[monitoring]"`
    sentry_traces_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    metrics_token: str = ""  # bo'sh = /metrics ochiq (faqat ichki tarmoqda qoldiring!)
    # Foydalanuvchi bo'yicha kunlik LLM byudjeti (0 = cheksiz)
    llm_daily_token_budget: int = Field(default=2_000_000, ge=0)
    llm_daily_cost_budget_usd: float = Field(default=5.0, ge=0.0)
    # Foydalanuvchi o'zgartirgan limit (Sozlamalar → Byudjet, `user_budgets`)
    # va bugungi qo'shimcha uchun **qattiq shift** (0 = shiftsiz). Berilsa
    # override + boost shu qiymatga qisiladi va UI shiftni ko'rsatadi.
    llm_daily_token_budget_max: int = Field(default=0, ge=0)
    llm_daily_cost_budget_usd_max: float = Field(default=0.0, ge=0.0)
    # Chat so'rovlari: foydalanuvchi bo'yicha daqiqasiga
    chat_rate_per_minute: int = Field(default=20, ge=1, le=600)

    # ─── Rejim ───
    env: str = "dev"
    log_level: str = "INFO"
    default_locale: str = "uz"
    max_accounts: int = Field(default=3, ge=1, le=10)
    default_write_mode: WriteMode = WriteMode.WRITE_WITH_CONFIRM

    # --- phase1/A2 ---
    # So'rov kengaytirish: bitta Task.ROUTE chaqiruvi + Redis kesh (arxitektura v2, 6-bo'lim).
    search_expand_enabled: bool = True
    search_expand_ttl_s: int = Field(default=24 * 3600, ge=60)
    # Rerank: "llm" (listwise, Task.ROUTE) yoki "none" (kirish tartibi saqlanadi).
    # Tashqi provider (vertex/cohere) — 13-bo'limdagi ochiq savol, hozircha yo'q.
    rerank_provider: str = "llm"
    rerank_top_n: int = Field(default=40, ge=1, le=200)

    # --- phase1/B ---
    # MTProto metod sinflari bo'yicha shift (`app/mtproto/ratelimit.py`).
    # Bitta umumiy kvota o'rniga to'rtta: arzon tarix o'qish qimmat global
    # qidiruv bilan bir xil chegarada turmasin. `TG_SEARCH_RATE_PER_MINUTE`
    # eskirgan alias bo'lib qoladi: sukutdan farq qilsa `search` ni boshqaradi,
    # 0 esa serverga boradigan barcha tool'larni o'chiradi.
    tg_rate_history_per_min: int = Field(default=40, ge=0, le=600)
    tg_rate_search_per_min: int = Field(default=12, ge=0, le=600)
    tg_rate_resolve_per_min: int = Field(default=4, ge=0, le=600)
    tg_rate_resolve_per_day: int = Field(default=60, ge=0, le=100_000)
    tg_rate_discover_per_min: int = Field(default=3, ge=0, le=600)
    tg_rate_discover_per_day: int = Field(default=30, ge=0, le=100_000)
    tg_rate_contacts_per_min: int = Field(default=3, ge=0, le=600)
    tg_rate_contacts_per_day: int = Field(default=30, ge=0, le=100_000)

    # --- phase2/C ---
    # Real-time updates listener (`app/worker/updates.py`, arxitektura v2 §8.3).
    # FAQAT worker jarayonida ishlaydi — api'da hech qachon (bitta MTProto
    # sessiya = update oqimining bitta iste'molchisi).
    tg_updates_listener: bool = False
    # catch_up (`updates.GetDifference`) davri, daqiqa; ±20% jitter bilan.
    tg_updates_catchup_minutes: int = Field(default=30, ge=1, le=1440)
    # Listener'ning `{peer → chats.id}` keshi (soniya). Noma'lum peer bu
    # muddat ichida qayta so'ralmaydi — listener dialog tortmaydi.
    tg_updates_chat_cache_s: int = Field(default=60, ge=5, le=3600)

    # --- phase2/B ---
    # Deep search (`services/deep_search.py`). Byudjetlar **kodda** turadi
    # (`MAX_ROUNDS`, `MAX_CANDIDATES`, `DEEP_RESULT_TOKENS`) — bu yerda faqat
    # operatsion tugmalar: rejimni o'chirish, devor vaqtini uzaytirish va
    # foydalanuvchi bo'yicha shift (bitta deep so'rov ≈ 10x oddiy so'rov narxi).
    deep_search_enabled: bool = True
    deep_soft_wall_s: float = Field(default=0, ge=0, le=600)  # 0 = deep_search.SOFT_WALL_S
    deep_hard_wall_s: float = Field(default=0, ge=0, le=900)  # 0 = deep_search.HARD_WALL_S
    deep_rate_per_minute: int = Field(default=3, ge=0, le=60)

    # --- phase2/A ---
    # Ierarxik retrieval indeksi (`retrieval_nodes`): epizod → kun → hafta.
    # Segmentatsiya qopqoqlari **kodda** (`services/episodes.py`: 45 daqiqa,
    # 40 xabar, 6000 belgi) — ular o'lchov bilan qo'yilgan, env bilan emas.
    # Bu yerda faqat operatsion tugmalar.
    node_index_enabled: bool = True
    # Bir cron yugurishida embed qilinadigan tugunlar (0 = `embed_batch_per_run`).
    node_embed_batch_per_run: int = Field(default=0, ge=0, le=5000)
    # Hafta tugunlari (level 3) — bitta `Task.ROUTE` chaqiruvi, chat bo'yicha.
    week_nodes_enabled: bool = True

    # --- phase3/B ---
    # Shaxsiy kunlik briefing (L3) — `services/briefing.py`, arxitektura v2, 9-bo'lim.
    # Akkaunt darajasidagi sozlamalar (`tz`, `quiet_*`, `briefing_hour`,
    # `briefing_chats`) DB'da (migratsiya 0010); bu yerda faqat global tugmalar.
    briefing_enabled: bool = True  # False = cron'lar umuman ishlamaydi
    # Chain-of-Density raundlari (TL;DR + mavzular ustida, arzon `Task.ROUTE`).
    # 0 = o'chiq. Har raund = bitta qo'shimcha LLM chaqiruvi.
    briefing_cod_rounds: int = Field(default=2, ge=0, le=4)
    # To'liq ishlov ko'radigan chatlar soni; qolgani bir qatordan (9-bo'lim).
    briefing_top_chats: int = Field(default=3, ge=1, le=10)
    # Bitta briefingga kiradigan chatlar (xarajat qopqog'i).
    briefing_max_chats: int = Field(default=20, ge=1, le=100)

    # --- phase3/C ---
    # Alertlar (`services/alerts.py`, migratsiya 0011). Qoidaning o'zi DB'da —
    # bu yerda faqat global tugmalar va xarajat qopqoqlari.
    alerts_enabled: bool = True  # False = skaner ham, yetkazish ham ishlamaydi
    # Bitta skaner yugurishida bitta qoida ko'radigan xabarlar soni.
    # Oshirish = uzoq tranzaksiya; kamaytirish = kursor sekinroq siljiydi.
    alerts_scan_limit: int = Field(default=500, ge=10, le=5000)
    # Akkauntga qo'yish mumkin bo'lgan qoidalar soni. Semantik qoida har
    # skanerlashda pgvector bo'ylab yuguradi — cheksiz ro'yxat DB'ni yeydi.
    alerts_max_rules_per_account: int = Field(default=20, ge=1, le=200)

    # --- phase3/A ---
    # Deep search'ning Telegram **server** pog'onalari (arxitektura v2 §8,
    # `services/deep_search.escalate` + `services/tg_search.py`). Lokal indeks
    # birinchi: server rung'lari faqat qamrov yetmaganda ishga tushadi.
    deep_server_rungs_enabled: bool = True
    # Lokal fan-out shundan kam hit bersa — server pog'onasiga chiqiladi.
    deep_server_min_local_hits: int = Field(default=8, ge=0, le=500)
    # Bitta deep run'da **jami** server RPC chaqiruvlari qopqog'i. Bu narx
    # qopqog'i emas, ban riskini kamaytirish: har chaqiruv real MTProto so'rovi
    # va `MethodBuckets` kvotasini yeydi.
    deep_server_max_calls: int = Field(default=6, ge=0, le=40)

    # --- phase4/A ---
    # Entity indeksi (`node_entities`, migratsiya 0012) — regex bilan, LLM'siz.
    # O'chirilsa ingest entity yozmaydi va backfill cron'i darhol chiqadi;
    # `entity_search` esa bo'sh jadval ustida jimgina bo'sh natija beradi.
    entities_enabled: bool = True
    # Backfill: bitta ish nechta xabar ko'radi va bitta SELECT nechta oladi.
    # Kursor Redis'da (`entities:backfill:<account_id>`) — `messages.id` bo'yicha.
    entities_backfill_limit: int = Field(default=20_000, ge=100, le=1_000_000)
    entities_backfill_batch: int = Field(default=500, ge=50, le=5_000)

    # --- phase4/B ---
    # Vazifa → **model** override (provider-agnostik). `llm_task_*` dan farqi:
    # u `provider:model` juftini beradi, bu esa faqat model nomini almashtiradi
    # — provider o'sha-o'sha (router qaysi providerni tanlagan bo'lsa).
    # Model provider imkoniyatiga mos kelmasa router **aniq xato** tashlaydi
    # (jimgina boshqa modelga o'tmaydi): noto'g'ri nom deploy paytida ko'rinsin.
    # Voris modelga o'tish yo'li ham shu (masalan `gemini-2.5-*` 2026-09'da
    # yangi mijozlarga yopildi → `LLM_MODEL_ROUTE=<voris>`, `app.llm.router.RETIRING`).
    llm_model_route: str = ""
    llm_model_search: str = ""
    llm_model_tools: str = ""
    llm_model_deep: str = ""
    llm_model_embed: str = ""
    llm_model_image: str = ""  # Gemini rasm modeli (avval override yo'q edi)

    # Embedding modeli/o'lchami uchun provider-neytral nomlar. Bo'sh/0 =
    # eski maydonlar (`gemini_model_embed`, `embed_dim`) — yagona haqiqat
    # manbai bitta bo'lib qoladi, ikkita raqam bir-biriga zid bo'lmaydi.
    # `EMBED_DIMS` Gemini `output_dimensionality` — **DB ustuni** shu o'lchamda
    # qurilgan (`Vector(embed_dim)`), o'zgartirish migratsiya talab qiladi.
    embed_model_name: str = Field(default="", validation_alias="EMBED_MODEL")
    embed_dims_value: int = Field(default=0, ge=0, le=3072, validation_alias="EMBED_DIMS")

    # --- phase4/C ---
    # Indekslash embeddinglari uchun Gemini Batch API. Narx sinxron yo'lning
    # **yarmi**, evaziga javob soatlar ichida (SLA 24 soat). Indekslash fon
    # ishi — kechikish muhim emas; qidiruv tomoni (`RETRIEVAL_QUERY`) esa
    # hech qachon batch bo'lmaydi, u user kutib turgan yo'l.
    #
    # Default `sync` — ataylab. Batch rejimi yangi xabarning qidiruvda
    # ko'rinishini soatlarga kechiktiradi; buni user bilib yoqishi kerak.
    embed_mode: str = Field(default="sync", pattern="^(sync|batch)$")
    # Bitta jobdagi elementlar soni. Inline so'rov `asyncBatchEmbedContent`
    # tanasiga sig'ishi kerak, shuning uchun belgilar bo'yicha ikkinchi
    # qopqoq ham bor (quyida) — ikkalasidan qaysi biri avval to'lsa.
    embed_batch_max_items: int = Field(default=2000, ge=1, le=20000)
    # Inline so'rov tanasi uchun xavfsiz chegara (~8 MB matn). Gemini inline
    # batch so'rovini 20 MB bilan cheklaydi; 4000 belgilik xabarlarda 2000
    # element ≈ 8 MB, ya'ni zaxira bor.
    embed_batch_max_chars: int = Field(default=8_000_000, ge=10_000)
    # Bir job necha marta qayta yuborilishi mumkin. 2 dan keyin
    # `failed_permanent` — cheksiz qayta yuborish = cheksiz to'lov.
    embed_batch_max_attempts: int = Field(default=2, ge=1, le=5)
    # `poll_embed_batches` cron'i (5 daqiqa). O'chirilsa tugagan joblar
    # o'qilmay qoladi va elementlar abadiy "band" bo'lib turardi — shuning
    # uchun o'chirish faqat batch rejimi ham o'chganda mantiqiy.
    embed_batch_poll_enabled: bool = True
    # Bir poll yugurishida o'qiladigan tugagan joblar soni (xotira qopqog'i:
    # har jobda 2000 ta 768 o'lchamli vektor).
    # 2026-09-18: 4 → 20. 2.6M xabarlik korpusda bir cron yugurishi 4 ta jobni
    # (≈ 8k vektor) o'qisa, submit tezligi poll tezligidan oshib ketardi va
    # tugagan joblar navbatda to'planib qolardi. 20 x 2000 = 40k vektor/poll.
    embed_batch_poll_limit: int = Field(default=20, ge=1, le=50)

    # --- embed/throughput ---
    # Bir `embed_content` chaqiruvida N matn yuborilganda ba'zi modellar
    # **bitta** vektor qaytaradi (`gemini-embedding-2`, SDK `t_contents` ro'yxatni
    # bitta hujjatga yig'adi — `app.llm.gemini` izohiga qarang). Adapter buni
    # aniqlab har matnni alohida yuboradi; shu yerdagi son — parallel so'rovlar
    # qopqog'i (semafor). Kattalashtirish 429 (`RESOURCE_EXHAUSTED`) xavfi.
    embed_single_concurrency: int = Field(default=8, ge=1, le=64)
    # `EMBED_MODE=batch` da bitta cron yugurishi (15 daqiqa) nechta job
    # yuborishi mumkin. Har job `EMBED_BATCH_MAX_ITEMS` gacha element oladi,
    # ya'ni 20 x 2000 = 40k element/yugurish. Default 1 — eski xulq; backfill
    # paytida vaqtincha ko'tariladi (`docs/llm.md`).
    embed_batch_jobs_per_run: int = Field(default=1, ge=1, le=50)
    # `EMBED_MODE=sync` da bir yugurishda embed qilinadigan xabarlar.
    # 0 = `EMBED_BATCH_PER_RUN` (eski xulq — yagona haqiqat manbai bitta).
    embed_sync_per_run: int = Field(default=0, ge=0, le=50_000)

    # --- hardening ---
    # Ishonchli reverse-proxy tarmoqlari (CIDR ro'yxati, vergul bilan).
    # **Bo'sh (default) = `X-Forwarded-For` ga hech qachon ishonilmaydi** —
    # mijoz IP'si TCP peer'dan olinadi. Proxy ortida ishlatsangiz shu yerga
    # proxy tarmog'ini yozing (masalan `10.0.0.0/8,172.18.0.0/16`), aks holda
    # istalgan mijoz shift kalitini (`client_ip`) soxtalashtira oladi.
    trusted_proxies: str = ""
    # Worker jarayonida Prometheus HTTP eksporti (`/metrics`). 0 = o'chiq.
    # `api` dagi `/metrics` faqat api jarayonining metrikalarini beradi —
    # cron/ingest hisoblagichlari shu portdan chiqadi.
    worker_metrics_port: int = Field(default=0, ge=0, le=65535)
    # Bitta snapshot yugurishida bitta chatdan olinadigan postlar qopqog'i.
    # Usiz katta kanal butun soatlik oynani yeb, qolgan kanallarni ochlikda
    # qoldiradi (snapshot eng eski `captured_at` dan boshlab oladi).
    snapshot_max_messages_per_run: int = Field(default=5000, ge=100, le=100_000)

    # --- ui/R1a ---
    # Chat profil rasmlari (`services/avatars.py`, `DATA_DIR/avatars`).
    # "Media saqlanmaydi" invariantiga **chegaralangan istisno**: faqat profil
    # rasmi, xabar mediasi hech qachon. `false` — hech qanday fayl yuklanmaydi
    # (UI inisiallarni chizadi); qolgan profil ma'lumoti (`about`, belgilar)
    # baribir yangilanadi.
    avatars_enabled: bool = True

    # --- ui/R1b ---
    # Mini App `initData` imzosining maksimal yoshi (sekund). Telegram
    # hujjatida tavsiya — 24 soat; qisqartirilsa Mini App'ni uzoq ochiq
    # qoldirgan foydalanuvchi qayta kirishga majbur bo'ladi.
    miniapp_init_data_max_age: int = Field(default=86_400, ge=60, le=7 * 86_400)

    @property
    def trusted_proxy_networks(self) -> tuple[Any, ...]:
        """`TRUSTED_PROXIES` → `ip_network` ro'yxati. Noto'g'ri yozuv e'tiborsiz."""
        import ipaddress

        nets: list[Any] = []
        for part in self.trusted_proxies.split(","):
            raw = part.strip()
            if not raw:
                continue
            try:
                nets.append(ipaddress.ip_network(raw, strict=False))
            except ValueError:
                continue
        return tuple(nets)

    @field_validator("master_key_b64")
    @classmethod
    def _check_master_key(cls, v: str) -> str:
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("MASTER_KEY_B64 base64 emas") from exc
        if len(raw) != 32:
            raise ValueError("MASTER_KEY_B64 aynan 32 bayt bo'lishi kerak (AES-256)")
        return v

    @property
    def master_key(self) -> bytes:
        return base64.b64decode(self.master_key_b64)

    @property
    def web_secure_cookies(self) -> bool:
        """HTTPS'da bo'lsak `Secure` cookie. Lokal http://localhost'da o'chiq."""
        return self.webapp_base_url.lower().startswith("https://")

    @property
    def allowed_users(self) -> frozenset[int]:
        """Bo'sh bo'lsa — hamma ruxsat (faqat dev uchun)."""
        raw = self.allowed_user_ids.strip()
        if not raw:
            return frozenset()
        return frozenset(int(x) for x in raw.split(",") if x.strip())

    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    # --- phase4/B ---
    @property
    def embed_model(self) -> str:
        """Konfiguratsiyadagi embedding modeli (`EMBED_MODEL` yoki eski nom).

        Bu — **default**; `LLM_MODEL_EMBED` override berilgan bo'lsa haqiqiy
        model `app.llm.embedding.active_embed_model()` dan olinadi.
        """
        return (self.embed_model_name or self.gemini_model_embed).strip()

    @property
    def embed_dims(self) -> int:
        """Embedding o'lchami (`EMBED_DIMS`); 0 bo'lsa DB ustuni bilan bir xil."""
        return self.embed_dims_value or self.embed_dim


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def embed_dims_mismatch(s: Settings | None = None) -> str:
    """`EMBED_DIMS` ↔ DB ustuni mosligini tekshiradi; mos bo'lsa bo'sh satr (L10).

    `EMBED_DIMS` provider'ga `output_dimensionality` bo'lib ketadi, DB ustuni
    esa `Vector(embed_dim)` / `HALFVEC(embed_dim)` bilan **migratsiya vaqtida**
    qurilgan. Ikkalasi ajralsa har INSERT `expected N dimensions, not M` bilan
    yiqiladi — ya'ni indekslash to'liq to'xtaydi, lekin buni faqat cron
    loglaridan bilib olish mumkin edi. Shuning uchun tekshiruv startupda.

    O'lchamni haqiqatan o'zgartirish ustun turini almashtiradigan **migratsiya**
    talab qiladi (va butun korpusni qayta embed qilishni): `EMBED_DIMS` yolg'iz
    o'zi hech narsani ko'chirmaydi.
    """
    s = s or get_settings()
    if s.embed_dims == s.embed_dim:
        return ""
    return (
        f"EMBED_DIMS={s.embed_dims}, DB ustuni esa {s.embed_dim} o'lchamda "
        "(`Vector(embed_dim)`) — har embedding INSERT'i xato beradi. "
        "O'lchamni o'zgartirish ustun turini almashtiradigan migratsiya + "
        "to'liq qayta embed talab qiladi."
    )
