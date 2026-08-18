# tg-ai-analyzer

![Python](https://img.shields.io/badge/python-3.12-blue)
![Telethon](https://img.shields.io/badge/Telethon-MTProto-2CA5E0?logo=telegram&logoColor=white)
![aiogram](https://img.shields.io/badge/aiogram-3.x-2CA5E0)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791?logo=postgresql&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Gemini%20%7C%20DeepSeek-8E44AD)
![Status](https://img.shields.io/badge/status-WIP%20(stage%200)-orange)

Telegram chatlarini AI orqali tahlil qiluvchi agent. MTProto (akkaunt nomidan
o'qish/yozish) + Bot API (foydalanuvchi interfeysi) + **Gemini / DeepSeek** (LLM).

## Texnologiyalar

| Qatlam | Stack |
|---|---|
| Til | Python 3.12, asyncio |
| Telegram | [Telethon](https://github.com/LonamiWebs/Telethon) (MTProto, akkaunt) + [aiogram 3](https://github.com/aiogram/aiogram) (Bot API, UI) |
| LLM | Gemini (`google-genai`) + DeepSeek (OpenAI-mos) — `Task` bo'yicha router |
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
- DeepSeek API key — ixtiyoriy

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
| `LLM_PROVIDER`, `LLM_FALLBACK_PROVIDER` | `gemini` \| `deepseek` |
| `LLM_TASK_*` | Vazifa darajasida provider/model override (`route/search/tools/deep/embed/image`) |
| `GEMINI_API_KEY`, `DEEPSEEK_API_KEY` | Provider kalitlari |
| `DEFAULT_WRITE_MODE` | `read_only` \| `write_with_confirm` (default) \| `autonomous` |
| `MAX_ACCOUNTS` | Ulanadigan akkauntlar limiti |

### LLM router

Ilova kodi provider SDK'siga to'g'ridan-to'g'ri murojaat qilmaydi — faqat
`app.llm.LLM` fasadi orqali, model tanlash `Task` darajasida:

| `Task` | Gemini | DeepSeek |
|---|---|---|
| `ROUTE` | `gemini-2.5-flash-lite` | `deepseek-chat` |
| `SEARCH` / `TOOLS` | `gemini-2.5-flash` | `deepseek-chat` |
| `DEEP` | `gemini-2.5-pro` | `deepseek-reasoner` |
| `EMBED` | `gemini-embedding-001` | — (Gemini'ga fallback) |
| `IMAGE` | `gemini-2.5-flash-image` | — (Gemini'ga fallback) |

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
    gemini.py        google-genai adapter (chat, tools, embed, image)
    deepseek.py      OpenAI-mos adapter (chat, tools)
    router.py        Task → provider/model + capability fallback
  db/                SQLAlchemy modellar
  bot/               aiogram — handler, middleware
  web/               FastAPI — Mini App auth backend
  worker/            ARQ — sync, snapshot, embedding, rollup
  i18n/              uz / ru / en
migrations/          Alembic
```

## Bosqichlar

To'liq reja, qarorlar va texnik tahlil: **[PLAN.md](PLAN.md)**

| # | Bosqich | Holat |
|---|---|---|
| 0 | Skeleton, DB, guardrail, i18n, LLM qatlami (Gemini+DeepSeek) | ✅ |
| 1 | Auth: QR login + Mini App fallback, multi-account | ⬜ |
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
