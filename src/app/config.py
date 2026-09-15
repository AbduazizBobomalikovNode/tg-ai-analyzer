"""Markazlashgan konfiguratsiya. Hamma env o'qish shu yerdan o'tadi."""

from __future__ import annotations

import base64
from enum import StrEnum
from functools import lru_cache

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
    gemini_api_key: str = ""
    gemini_model_router: str = "gemini-2.5-flash-lite"
    gemini_model_fast: str = "gemini-2.5-flash"
    gemini_model_deep: str = "gemini-2.5-pro"
    gemini_model_image: str = "gemini-2.5-flash-image"
    gemini_model_embed: str = "gemini-embedding-001"
    embed_dim: int = 768

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
