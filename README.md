# tg-ai-analyzer

![Python](https://img.shields.io/badge/python-3.12-blue)
![Telethon](https://img.shields.io/badge/Telethon-MTProto-2CA5E0?logo=telegram&logoColor=white)
![aiogram](https://img.shields.io/badge/aiogram-3.x-2CA5E0)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791?logo=postgresql&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Claude%20%7C%20Gemini%20%7C%20DeepSeek-8E44AD)
![Status](https://img.shields.io/badge/status-WIP%20(stage%200)-orange)

Telegram chatlarini AI orqali tahlil qiluvchi agent. MTProto (akkaunt nomidan
o'qish/yozish) + Bot API (foydalanuvchi interfeysi) + **Claude / Gemini / DeepSeek** (LLM).

LLM uchun alohida API kalit shart emas: **Claude Code obunangiz** bo'lsa
(`claude setup-token`) shu token bilan ishlaydi; xohlasangiz Gemini yoki
DeepSeek API kalitini ulaysiz.

## Texnologiyalar

| Qatlam | Stack |
|---|---|
| Til | Python 3.12, asyncio |
| Telegram | [Telethon](https://github.com/LonamiWebs/Telethon) (MTProto, akkaunt) + [aiogram 3](https://github.com/aiogram/aiogram) (Bot API, UI) |
| LLM | Claude (`anthropic` SDK yoki `claude` CLI sessiyasi) + Gemini (`google-genai`) + DeepSeek (OpenAI-mos) — `Task` bo'yicha router |
| Ma'lumot | PostgreSQL 16 + pgvector, SQLAlchemy 2 (async) + Alembic |
| Queue | Redis + ARQ (sync, snapshot, embedding, rollup worker'lari) |
| Web | FastAPI — Mini App auth backend |
| Infra | Docker Compose, structlog, ruff + mypy, pytest |

## Nima qiladi

- minglab xabar orasidan kerakligini topadi (FTS + semantik hybrid search)
- kanal statistikasi: ko'rishlar, reaksiyalar, forward — bugun / hafta / oy / yil / butun davr
- chat xulosasi, keyingi post uchun g'oya va senariy
- post yozish, tahrirlash, forward, copy, pin, rejalashtirish
- internetdan rasm qidirish yoki Gemini bilan generatsiya
- javoblarni Telegram HTML formatida chiroyli qaytarish

## Nima qilmaydi — texnik jihatdan

| Kafolat | Qanday ta'minlangan |
|---|---|
| **Xabar o'chira olmaydi** | `app/mtproto/allowlist.py` — deny-by-default RPC allowlist. Barcha `Delete*` metodlari `DENIED` da. Agent tool registry'da delete tool umuman yo'q |
| **Boshqaruv botiga yoza olmaydi** | `app/mtproto/guard.py` — peer deny-list. Bu peer inline chat tanlash ro'yxatidan ham filtrlanadi |
| **Logout / ban / profil o'zgartira olmaydi** | Barchasi `DENIED` da, hatto login oynasi ichida ham |
| **Ruxsatsiz yoza olmaydi** | Default `write_with_confirm` — har yozish user tasdig'ini kutadi |
| **Audit yo'qolmaydi** | `agent_actions` jadvali append-only (Postgres trigger) |

Bu kafolatlar prompt'ga emas, kodga o'rnatilgan. `tests/test_allowlist.py` va
`tests/test_peer_guard.py` ularni har run'da tekshiradi.

## Talablar

- Docker + Docker Compose (yoki lokal Python 3.12 + Postgres 16/pgvector + Redis)
- Telegram **bot token** — [@BotFather](https://t.me/BotFather)
- Telegram **API ID / hash** — [my.telegram.org](https://my.telegram.org)
- **Gemini API key** — majburiy (embedding va rasm faqat Gemini'da)
- Chat/agent uchun bittasi: **Claude Code obunasi** (`claude setup-token`) yoki
  `ANTHROPIC_API_KEY`, yoki Gemini/DeepSeek kaliti — Claude topilsa u default

## Ishga tushirish

```bash
cp .env.example .env
# .env ni to'ldiring. MASTER_KEY_B64 uchun:
python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"

make up          # postgres + redis + migrate + bot + api + worker
make logs
```

Tekshiruv:

```bash
docker compose run --rm bot python -m app.main check
```

## Web UI — telefon orqali ulash va AI chat

`make up` dan keyin **http://localhost:8080** (prod'da `WEBAPP_BASE_URL`, HTTPS):

1. **`/login`** — telefon raqami → Telegram kod → (bo'lsa) 2FA parol. Kod faqat
   shu HTTPS sahifada kiritiladi — Telegram chatga yozilgan kod bekor bo'ladi.
   Session shifrlangan holda saqlanadi, brauzerga imzolangan cookie beriladi.
2. **`/chat`** — AI chat. Chapda akkaunt va Telegram dialoglar; chatni tanlasangiz
   oxirgi N ta xabar (slider) `<untrusted_data>` konvertida modelga kontekst
   bo'lib boradi. Savol: "bu hafta nima muhokama qilindi?", "eng ko'p ko'rilgan
   post?", "keyingi post uchun 3 g'oya". Suhbatlar saqlanadi (`conversations`).
   Til: `?lang=uz|ru|en`.

Bu rejimda agent **hech narsa yozmaydi/o'chirmaydi** — faqat o'qish
(`messages.GetDialogs/GetHistory`), tool'lar 5–6-bosqichda.

API: `POST /api/auth/phone|code|password`, `GET /api/me`,
`GET /api/accounts/{id}/dialogs`, `POST /api/conversations/{id}/messages`.
O'zgartiruvchi so'rovlar `X-Requested-With: fetch` header'ini talab qiladi (CSRF).
`api` servisi **bitta uvicorn worker** — auth oqimlari jarayon xotirasida.

## Konfiguratsiya

Barcha sozlamalar `.env` orqali (`app/config.py::get_settings()`). To'liq ro'yxat
va izohlar — [`.env.example`](.env.example). Asosiylari:

| O'zgaruvchi | Vazifasi |
|---|---|
| `BOT_TOKEN`, `CONTROL_BOT_ID` | Boshqaruv boti. Bot ID avtomatik peer deny-list'ga tushadi |
| `ALLOWED_USER_IDS` | Botdan foydalana oladigan Telegram user ID'lar |
| `TG_API_ID`, `TG_API_HASH` | MTProto kredensiallar |
| `TG_DEVICE_MODEL`, `TG_SYSTEM_VERSION`, `TG_APP_VERSION` | Login device profili — **o'zgartirmang** (ban riski) |
| `MASTER_KEY_B64` | Session'larni shifrlash uchun master key. Yo'qolsa session'lar tiklanmaydi |
| `WEBAPP_BASE_URL` | Mini App uchun HTTPS URL (login kodi faqat shu orqali) |
| `DATABASE_URL`, `REDIS_URL` | Postgres (asyncpg) va Redis |
| `LLM_PROVIDER`, `LLM_FALLBACK_PROVIDER` | `auto` (default) \| `claude` \| `claude_code` \| `gemini` \| `deepseek`. `auto` = API kredensiali → `claude`; `claude` CLI login → `claude_code`; aks holda `gemini` |
| `CLAUDE_CODE_AUTO`, `CLAUDE_CODE_BIN` | `claude_code` provider: CLI'ni auto'da hisobga olish / binar yo'li |
| `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN` | Claude kredensiali (birinchisi ustun). Ikkalasi ham bo'lmasa `~/.config/anthropic` profili (`ant auth login`) qidiriladi |
| `LLM_TASK_*` | Vazifa darajasida provider/model override (`route/search/tools/deep/embed/image`) |
| `GEMINI_API_KEY`, `DEEPSEEK_API_KEY` | Provider kalitlari |
| `DEFAULT_WRITE_MODE` | `read_only` \| `write_with_confirm` (default) \| `autonomous` |
| `WEB_SESSION_TTL_HOURS`, `WEB_AUTH_FLOW_TTL_MIN`, `WEB_AUTH_RATE_PER_IP` | Web cookie muddati, login oqimi TTL, IP bo'yicha limit |
| `WEB_CONTEXT_DEFAULT_MESSAGES`, `WEB_CONTEXT_MAX_MESSAGES` | AI chat kontekstiga beriladigan oxirgi xabarlar soni |
| `MAX_ACCOUNTS` | Ulanadigan akkauntlar limiti |

### LLM router

Ilova kodi provider SDK'siga to'g'ridan-to'g'ri murojaat qilmaydi — faqat
`app.llm.LLM` fasadi orqali, model tanlash `Task` darajasida:

| `Task` | Claude (`claude` / `claude_code`) | Gemini | DeepSeek |
|---|---|---|---|
| `ROUTE` | `claude-haiku-4-5` | `gemini-2.5-flash-lite` | `deepseek-chat` |
| `SEARCH` | `claude-opus-5` | `gemini-2.5-flash` | `deepseek-chat` |
| `TOOLS` | `claude-opus-5` (`claude_code`: — fallback) | `gemini-2.5-flash` | `deepseek-chat` |
| `DEEP` | `claude-opus-5` | `gemini-2.5-pro` | `deepseek-reasoner` |
| `EMBED` | — (Gemini'ga fallback) | `gemini-embedding-001` | — (Gemini'ga fallback) |
| `IMAGE` | — (Gemini'ga fallback) | `gemini-2.5-flash-image` | — (Gemini'ga fallback) |

### Claude Code sessiyasidan foydalanish (`claude_code`)

Mashinada `claude` CLI o'rnatilgan va `/login` qilingan bo'lsa — **hech narsa
sozlash shart emas**: `LLM_PROVIDER=auto` uni topadi va `claude -p` orqali
Claude Code'ning o'z sessiyasidan foydalanadi (Keychain / `~/.claude`).

- Built-in tool'lar o'chiq (`--tools ""`), settings/hooks/CLAUDE.md yuklanmaydi,
  sessiya diskka yozilmaydi — bu faqat LLM chaqiruvi.
- Cheklov: **function calling yo'q** → `Task.TOOLS` (agent sikli) avtomat
  Gemini'ga tushadi. Chat, qidiruv sintezi, chuqur tahlil, JSON javoblar — Claude'da.
- Docker'da CLI yo'q — konteynerda quyidagi token yo'li ishlatiladi.

### Claude Code tokeni bilan ishlash (`claude`, Docker uchun)

```bash
claude setup-token          # brauzerda login → sk-ant-oat01-... chiqadi
# .env:
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
LLM_PROVIDER=auto           # (default) — Claude topilgani uchun u tanlanadi
```

`docker compose run --rm bot python -m app.main check` → `check.llm_provider`
qatorida `effective=claude claude_auth=CLAUDE_CODE_OAUTH_TOKEN` ko'rinishi kerak.
Token secret'i log'ga tushmaydi. Claude'da embedding/rasm yo'q — bular
avtomat Gemini'ga ketadi, shuning uchun `GEMINI_API_KEY` baribir kerak.

## Lokal ishlab chiqish

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test
make lint
```

## Struktura

```
src/app/
  config.py          env → Settings (yagona manba)
  logging.py         structlog + sirlarni maskalash
  crypto/            envelope encryption (session string'lar uchun)
  mtproto/
    allowlist.py     ⚠️ RPC deny-by-default — o'chirish imkonsizligi shu yerda
    guard.py         ⚠️ peer deny-list — boshqaruv botiga yozib bo'lmaydi
    client.py        guardrail o'rnatilgan Telethon klient
  llm/
    base.py          tiplar, Capability, retry
    claude.py        anthropic SDK adapter (API key / Claude Code token / profil)
    claude_code.py   `claude -p` adapter — Claude Code login sessiyasi, tool'siz
    gemini.py        google-genai adapter (chat, tools, embed, image)
    deepseek.py      OpenAI-mos adapter (chat, tools)
    router.py        Task → provider/model + auto tanlov + capability fallback
  db/                SQLAlchemy modellar
  bot/               aiogram — handler, middleware
  services/
    auth_flow.py     telefon → kod → 2FA holat mashinasi (auth_window ichida)
    session_store.py session'ni envelope bilan DB'ga yozish/o'qish
    accounts.py      login natijasini User/Account'ga bog'lash
    chat_service.py  AI chat: kontekst <untrusted_data> + LLM fasad + tarix
  mtproto/pool.py    ulangan akkauntlar klient pool'i (faqat o'qish)
  web/               FastAPI — /login, /chat sahifalari + /api (cookie auth, CSRF)
  worker/            ARQ — sync, snapshot, embedding, rollup
  i18n/              uz / ru / en
migrations/          Alembic
```

## Bosqichlar

To'liq reja, qarorlar va texnik tahlil: **[PLAN.md](PLAN.md)**

| # | Bosqich | Holat |
|---|---|---|
| 0 | Skeleton, DB, guardrail, i18n, LLM qatlami (Claude+Gemini+DeepSeek) | ✅ |
| 1 | Auth: telefon → kod → 2FA (web), multi-account, session encryption; QR login | 🟡 web tayyor, QR ⬜ |
| 2 | Ingestion: history sync, metric snapshot | ⬜ |
| 3 | Hybrid search + inline chat picker | ⬜ |
| 4 | Statistika: rollup, vaqt oynalari | ⬜ |
| 5 | Agent v1 (read-only) | ⬜ |
| 6 | Write actions + tasdiqlash UI | ⬜ |
| 7 | Kontent: post, rasm, scheduling, auto-reply | ⬜ |
| 8 | Polish: monitoring, limitlar | ⬜ |

## Ogohlantirish

Userbot avtomatizatsiyasi Telegram ToS chegarasida. Akkaunt cheklanishi
mumkin. Login qilinadigan akkauntlar uchun `TG_DEVICE_*` qiymatlarini
o'zgartirmang va flood-wait'ga har doim rioya qiling.
