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

**Sahifa:** [abduazizbobomalikovnode.github.io/tg-ai-analyzer/](https://abduazizbobomalikovnode.github.io/tg-ai-analyzer/)

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

## Ingestion (2-bosqich)

Login'dan keyin `sync_account` avtomat navbatga tushadi (ARQ worker):

1. `refresh_chats` — dialoglar → `chats` (tur, admin, a'zolar soni).
2. `sync_chat` — har chat tarixi **yangidan eskiga**, 100 talik batch, har batch
   DB'ga darhol (`INSERT … ON CONFLICT`), progress `chats.synced_total /
   total_estimate`. Keyingi ishlar `synced_max_id` dan yangilarini,
   `synced_min_id` dan orqaga davom ettiradi. Media saqlanmaydi.
3. **Snapshot cron** (har soat, `xx:05`) — kanal postlari views/forwards/
   reactions → `message_metric_snapshots`: <24 soat har soat, 1-7 kun har 6
   soat, 7-90 kun kuniga bir marta. **O'chirmang** — "bu hafta +N ko'rish"
   faqat shu jadvaldan.
4. `incremental_sync_all` — har 10 daqiqa yangi xabarlar.

Ban riski: batch'lar orasida pauza, FloodWait → ish to'xtaydi va `retry_after`
bilan qayta navbat (retry-storm yo'q); **ramp-up** — akkaunt 24 soatdan yosh
bo'lsa ≤ 1000 xabar × 20 chat. UI: dialog ro'yxatida progress, "⟳" tugmasi,
kontekst slider 200+ bo'lsa DB'dan.

## Qidiruv va token tejash (3-bosqich)

`services/search.py` — kontekst endi "oxirgi N xabar" emas, **strategiya**:

| Strategiya | Nima qiladi | Qachon (`auto`) |
|---|---|---|
| `search` | FTS (`simple` + prefix) + `pg_trgm` + pgvector (Gemini embedding) → RRF; topilgan xabar ± qo'shnisi + oxirgi 15 | savolda kalit so'z bor |
| `window` | savoldagi vaqt oynasi (bugun/hafta/oy/yil); byudjetdan katta bo'lsa **map-reduce**: arzon `Task.ROUTE` model bo'laklarni digest qiladi, asosiy model digest'lar ustida javob beradi | "bu hafta…", "за месяц…" |
| `recent` | oxirgi N (byudjetgacha) | boshqa hollarda / xulosa so'rovi |

Token tejash: xabar matni siqiladi (URL → domen, whitespace, 700 belgi),
kontekst byudjeti 6k (deep: 14k) token, tarixdagi eski turn'lar 1500 belgigacha,
oxirgi 12 turn; Claude'da uzun system prompt `cache_control` bilan keshlanadi.
Embedding cron (`embed_messages`, 15 daq) `EMBED_ENABLED` bilan boshqariladi.

### Javob formati

AI javoblari **Markdown** — `web/static/md.js` (o'zimizniki, xavfsiz: avval escape,
keyin teglar; havola faqat http(s)/mailto): sarlavha, ro'yxat (ichma-ich), kod,
blockquote, **GFM jadval** (kichik ekranda gorizontal scroll), **mermaid**
diagrammalar (```mermaid — pie, xychart-beta, flowchart; `vendor/mermaid.min.js`
faqat kerak bo'lganda yuklanadi, CDN yo'q). Interfeys 320px'dan boshlab moslashadi
(sidebar drawer, jadval/diagramma scroll, dashboard grid).

## Agent rejimi (read-only tool'lar) va prompt'lar

`services/prompts.py` — barcha system prompt'lar bitta joyda (yadro + chat/agent
varianti, digest, judge, router). System bayt darajasida barqaror (kesh uchun);
sana/til/pinned chat kabi o'zgaruvchilar user turn'ida (`runtime_note`).

`services/tools.py` — **faqat o'qish** tool'lari (DB ustidan, MTProto'ga tegmaydi):

| Tool | Nima uchun |
|---|---|
| `list_chats` | qaysi chat aniq bo'lmasa (bir marta) |
| `search_messages` | "top / qayerda / kim aytdi / X haqida" — hybrid qidiruv, #id + t.me havola |
| `get_recent_messages` | "nima yangilik" |
| `get_message_context` | bitta xabar atrofi (±N) — tekshirish |
| `get_window_digest` | hafta/oy xulosasi — **kunlik digest keshi** (`chat_digests`, xom xabar emas) |
| `get_chat_stats` | son/ko'rish/reaksiya/forward, o'sish (snapshot), top postlar, kun-soat, oldingi davr bilan taqqos — qo'lda sanamaydi |

`services/agent.py` — sikl: ≤ `AGENT_MAX_ITERATIONS` LLM chaqiruv, iteratsiyada ≤ 4
tool, tool natijalari umumiy `AGENT_TOOL_RESULT_TOKENS` byudjeti, oxirida majburiy
tool'siz javob; har chaqiruv `agent_actions` (append-only) ga. Yozish/o'chirish
tool'i registry'da yo'q. Rejim: `auto` (sinxron chat bor + savol ma'lumot haqida →
agent; aniq strategiya/salomlashish → direct), `agent`, `direct` — UI'da tanlanadi.

Xarajat qoidalari: arzon model (`ROUTER`) — digest/judge/intent; `FAST` (default
Sonnet 5) — chat/agent; `DEEP` (Opus 5) faqat tugma bilan; embedding faqat ≥ 20 belgi;
auto-baho sampling (`WEB_AUTO_EVAL_SAMPLE`); kunlik digest kechasi oldindan hisoblanadi
(`build_daily_digests`), savolda qayta hisoblanmaydi.

## Yozish amallari va tasdiqlash (6-bosqich)

`services/write_tools.py` — `send_message`, `edit_message`, `forward_message`,
`pin_message` (delete yo'q va bo'lmaydi). Agent ularni chaqirsa **hech narsa
yuborilmaydi**: `agent_actions` ga `proposed` yozuv tushadi, chatda karta chiqadi:
matn, manzil chat, muddat → **✅ Tasdiqlash va yuborish** / **❌ Rad etish**.
Faqat tasdiqda `services/actions.execute_action` MTProto'ga boradi.

Har chat uchun rejim (topbar'da ✍️): `read_only` (default yangi chatlar uchun
`DEFAULT_WRITE_MODE`), `write_with_confirm`, `autonomous` — oxirgisi faqat aniq
tanlanganda, **akkauntga bitta chat**, baribir audit'ga tushadi.

Himoya qatlamlari: egalik (`agent_runs.user_id`), `WRITE_PROPOSAL_TTL_HOURS`,
`WRITE_RATE_PER_HOUR`, `assert_writable()` taklifda ham, bajarishda ham
(boshqaruv boti / 777000 / @replies → `blocked`), `GuardedTelegramClient` allowlist,
`agent_actions` append-only trigger. API: `GET /api/actions?status=proposed`,
`POST /api/actions/{id}/confirm|reject`, `PATCH …/chats/{id}/write_mode`.

## Kontent (7-bosqich): rasm, scheduling, auto-reply

* **Rasm** — agent `generate_image(prompt, style)` (Gemini image, `Task.IMAGE`) →
  `data/images/<uuid>.png` + `generated_images`, chatda ko'rinadi
  (`/api/images/{id}`, faqat egasiga); `send_message(image_id, text=caption)`
  taklifi tasdiqlanganda `send_file` (SendMedia — allowlist). Kunlik limit
  `IMAGE_MAX_PER_DAY`.
* **Scheduling** — `send_message(schedule_at=ISO)` → Telegram server-side
  rejalashtirilgan post (tasdiq bilan); topbar ⏰ — rejalashtirilganlar ro'yxati
  (`GetScheduledHistory`, READ); `list_scheduled_messages` tool.
* **Auto-reply** — topbar 🤖: per-chat qoida (trigger: savol / mention / kalit
  so'z / hamma; ko'rsatma; soatiga limit; jim soatlar). Worker har 5 daqiqada
  yangi kiruvchi xabarlarga LLM javob yozadi (`SKIP` mumkin) va **`send_message`
  taklifi** (`reply_to`) yaratadi — ✅ tasdiq; chat `autonomous` bo'lsa darhol.
  Yozish yo'li 6-bosqichdagi bilan bir xil (guard, rate limit, audit).

## Kuzatuv, limitlar, CI (8-bosqich)

* **Metrikalar** — `GET /metrics` (Prometheus): `tgai_http_*`, `tgai_llm_requests/
  tokens/cost/latency`, `tgai_tool_calls_total`, `tgai_write_actions_total`,
  `tgai_sync_messages_total`, `tgai_floodwait_total`, `tgai_snapshot_posts_total`.
  `METRICS_TOKEN` bilan himoyalang. Worker cron'lari Redis'ga heartbeat yozadi —
  dashboard "Tizim" kartasi (DB/Redis ping, so'nggi cron'lar, bugungi byudjet),
  `GET /api/stats/system`.
* **Sentry** — `SENTRY_DSN` + `pip install -e ".[monitoring]"`; sirlar
  `before_send` da maskalanadi.
* **Limitlar** — foydalanuvchi bo'yicha kunlik `LLM_DAILY_TOKEN_BUDGET` /
  `LLM_DAILY_COST_BUDGET_USD` (429 `chat.err.budget`), `CHAT_RATE_PER_MINUTE`;
  avvalgilar: `WRITE_RATE_PER_HOUR`, `IMAGE_MAX_PER_DAY`, agent byudjetlari,
  auth IP limiti.
* **CI** — `.github/workflows/ci.yml`: ruff, mypy (toza), pytest, invariant
  testlari + delete-tool grep, Docker build. Lokal: `make ci`.

## Dashboard va sifat baholash

`/dashboard` — so'rovlar, tokenlar (in/out), taxminiy xarajat (`llm/pricing.py`),
javob vaqti (median/o'rtacha), qoniqish (👍/👎), auto-baho, model/strategiya/chat
kesimlari, ingestion holati, "ko'rib chiqish kerak" ro'yxati. Kunlik SVG chart'lar
(kutubxonasiz), jadval ko'rinishi, light/dark.

Baholash: har javob ostida 👍/👎 (`POST …/messages/{id}/rate`); javob yuborilgach
fon rejimida arzon model (`Task.ROUTE`) **relevance / usefulness (1-5) / grounded**
baholaydi (`WEB_AUTO_EVAL`), kontekst qayta yuborilmaydi — faqat savol+javob.

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
    pricing.py       taxminiy narx jadvali (dashboard xarajati)
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
    chat_service.py  AI chat: strategiyali kontekst <untrusted_data> + LLM fasad + tarix
    search.py        FTS/trgm/vektor qidiruv, RRF, token byudjeti, map-reduce
    evaluation.py    👍/👎 + LLM-judge auto-baho (arzon model, fon)
    prompts.py       barcha system prompt'lar (chat/agent/digest/judge/router)
    tools.py         read-only agent tool'lari (registry + JSON schema)
    write_tools.py   yozish tool'lari: taklif (proposed) — yuborilmaydi
    actions.py       tasdiqlash/rad/bajarish, rate limit, TTL, per-chat rejim
    images.py        AI rasm generatsiya + fayl/meta + egalik
    autoreply.py     auto-reply qoidalari va worker mantiqi (taklif yaratadi)
    agent.py         tool sikli: byudjet, audit, majburiy yakun
    digests.py       kunlik digest keshi (chat_digests)
    analytics.py     chat statistikasi (LLM'siz)
    stats.py         dashboard agregatlari
    ingestion.py     chat registry, tarix sync, snapshot (faqat o'qish)
  mtproto/pool.py    ulangan akkauntlar klient pool'i (faqat o'qish)
  worker/tasks.py    ARQ ishlari: sync_account/sync_chat/snapshot_metrics/incremental
  web/               FastAPI — /login, /chat sahifalari + /api (cookie auth, CSRF)
  worker/            ARQ — sync, snapshot, embedding, rollup
  i18n/              uz / ru / en
migrations/          Alembic
```

## Hujjatlar

To'liq hujjatlar — [`docs/`](docs/README.md): arxitektura, xavfsizlik, ma'lumot
modeli, API, LLM/prompt/xarajat, ingestion, agent va yozish amallari, deploy,
development.

## Bosqichlar

To'liq reja, qarorlar va texnik tahlil: **[PLAN.md](PLAN.md)**

| # | Bosqich | Holat |
|---|---|---|
| 0 | Skeleton, DB, guardrail, i18n, LLM qatlami (Claude+Gemini+DeepSeek) | ✅ |
| 1 | Auth: telefon → kod → 2FA (web), multi-account, session encryption; QR login | 🟡 web tayyor, QR ⬜ |
| 2 | Ingestion: chat registry, history sync (ramp-up, FloodWait), snapshot cron, incremental | ✅ (real-time listener ⬜) |
| 3 | Hybrid search (FTS+trgm+pgvector, RRF), strategiyali kontekst, map-reduce, embedding cron | ✅ (inline picker ⬜) |
| 4 | Statistika: `analytics.chat_stats` (davr, o'sish, top, kun/soat) — tool orqali | 🟡 rollup ⬜ |
| 5 | Agent v1 (read-only tool'lar, audit, byudjet, prompt'lar markazda) | ✅ |
| 6 | Write actions: taklif → tasdiq → bajarish, per-chat rejim, guard/rate/TTL, audit | ✅ |
| 7 | Kontent: rasm generatsiya, scheduling (Telegram-side), auto-reply qoidalari | ✅ (internet rasm qidiruv ⬜) |
| 8 | Polish: Prometheus + Sentry, heartbeat/tizim holati, kunlik byudjet, CI | ✅ (yuk testi ⬜) |

## Ogohlantirish

Userbot avtomatizatsiyasi Telegram ToS chegarasida. Akkaunt cheklanishi
mumkin. Login qilinadigan akkauntlar uchun `TG_DEVICE_*` qiymatlarini
o'zgartirmang va flood-wait'ga har doim rioya qiling.
