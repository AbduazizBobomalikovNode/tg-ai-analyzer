# CLAUDE.md — tg-ai-analyzer

Telegram chat tahlilchi AI agent. Python 3.12, Telethon (MTProto) + aiogram
(Bot API) + Claude/Gemini/DeepSeek + Postgres/pgvector.

## Buyruqlar

```bash
make up        # docker compose (postgres, redis, migrate, bot, api, worker)
make test      # pytest
make lint      # ruff + mypy
make fmt       # ruff format + fix
make revision m="xxx"   # alembic autogenerate
docker compose run --rm bot python -m app.main check   # config/ulanish tekshiruvi
```

Lokal: `python3.12 -m venv .venv && pip install -e ".[dev]"`, `PYTHONPATH=src`.

## Buzib bo'lmaydigan invariantlar

Bu to'rttasi loyihaning mavjudlik sharti. Har qanday o'zgarish ulardan birini
zaiflashtirsa — **avval user'dan so'rang**.

1. **`src/app/mtproto/allowlist.py` — deny-by-default.** Yangi TL metodni
   `ALLOWED` ga qo'shishdan oldin: u destruktivmi? `DENIED` ga hech qachon
   tegilmaydi (faqat o'sadi). Agent uchun delete tool *yaratilmaydi*.
2. **`src/app/mtproto/guard.py` — peer deny-list.** Boshqaruv boti, 777000 va
   `@replies` ga yozib bo'lmaydi. Har `send/edit/forward/pin` oldidan
   `assert_writable()`.
3. **Chat kontenti — ishonchsiz ma'lumot.** Hech qachon prompt sifatida
   qo'shilmaydi, faqat `<untrusted_data>` envelope ichida. Prompt injection
   real xavf: agent begona odamlar yozgan matnni o'qiydi.
4. **Default `write_with_confirm`.** Yozish amali user tasdig'isiz
   bajarilmaydi. `autonomous` faqat user aniq yoqsa va faqat bitta chat uchun.

Bularni `tests/test_allowlist.py`, `tests/test_peer_guard.py` qo'riqlaydi —
bu testlarni o'chirish/yumshatish taqiqlanadi.

## Web UI (`app/web`)

- Auth: `services/auth_flow.py` — telefon → kod → 2FA, har oqim o'z Telethon
  klienti bilan **jarayon xotirasida** → `api` bitta uvicorn worker. Barcha auth
  TL metodlari `auth_window()` ichida. Kod/parol faqat `/api/auth/*` (HTTPS).
- Web sessiya: imzolangan cookie (`web/security.py`, kalit master key'dan HMAC),
  o'zgartiruvchi API'lar `X-Requested-With: fetch` talab qiladi. Bu himoyalarni
  olib tashlamang.
- AI chat: `services/chat_service.py` — Telegram xabarlar **faqat**
  `render_context()` konvertida (`<untrusted_data>`), tool yo'q (5-bosqichgacha).
  `mtproto/pool.py` faqat o'qish; `get_sender()` emas, `m.sender` (allowlist).
- Suhbatlar: `conversations` / `conversation_messages` (migratsiya 0002).

## Ingestion (`services/ingestion.py`, `worker/tasks.py`)

- Faqat o'qish (`GetDialogs`, `GetHistory`, `GetMessages`). `WAIT_BETWEEN_BATCHES`,
  ramp-up (`limits_for`) va FloodWait → `SyncPaused` → `defer_by` requeue —
  bularni "tezlashtirish" uchun olib tashlamang (4.2-band).
- Snapshot cron (`snapshot_metrics`, har soat) — o'chirilsa vaqt qatori bo'shlig'i
  abadiy. Tier'lar `snapshot_tiers()`.
- `synced_min_id == 1` — tarix boshiga yetilgan sentinel; `sync_state`
  running/idle/done/failed. Job id'lar deterministik (`sync:acc:<id>`) — bir
  akkauntga parallel sync yo'q.
- `api` va `worker` bir xil session'ni ishlatadi (ikkalasi ham `pool`) — bir
  host/egress IP'da muammo yo'q; `AuthKeyDuplicatedError` ko'rsangiz api'dagi jonli
  o'qishni DB'ga ko'chiring (`chat_service.fetch_context`).

## Qidiruv / kontekst / sifat

- `services/search.select_context` — yagona kontekst tanlash nuqtasi
  (recent/search/window/auto, token byudjeti). Yangi "promptga N xabar tashlash"
  yo'llarini qo'shmang — shu orqali o'ting. Katta oyna → `compact_window`
  (map = `Task.ROUTE`, arzon). Digest'lar ham `<untrusted_data kind="digests">`.
- Auto-baho (`services/evaluation`) faqat savol+javob ko'radi, kontekst emas —
  token uchun. Judge natijasi tavsiya, "haqiqat" emas; dashboardda foydalanuvchi
  bahosi bilan yonma-yon.
- `llm/pricing.py` — taxmin; yangi model qo'shsangiz jadvalni yangilang.

## Frontend

- `web/static/md.js` — AI javoblari uchun **yagona** Markdown renderer. Xavfsizlik
  "by construction": matn avval escape, teglar keyin; `innerHTML`ga faqat shu
  chiqishi kiradi. Tashqi md/DOMPurify kutubxona qo'shmasdan shu yerni kengaytiring.
- Mermaid `securityLevel: "strict"`, lazy (`vendor/mermaid.min.js`, MIT). CDN yo'q.

## Agent / tool'lar (5-bosqich)

- `services/tools.READ_TOOLS` — **faqat o'qish**. Yozish tool'i bu registry'ga
  qo'shilmaydi; 6-bosqichda alohida registry + `assert_writable()` + tasdiq.
  Delete tool'i hech qachon (invariant 1). Har natija `tools.envelope()` bilan.
- `services/prompts.py` — prompt'larni shu yerda o'zgartiring; system'ga sana/ID
  qo'shmang (kesh buziladi) — `runtime_note` user turn'iga.
- Byudjetlar (`AGENT_MAX_ITERATIONS`, `AGENT_TOOL_RESULT_TOKENS`, tool ichidagi
  `MAX_RESULT_CHARS`) — "javob to'liqroq bo'lsin" deb olib tashlamang; avval
  digest/stats tool'i orqali ixchamlang.
- Digest keshi (`chat_digests`) `msg_count` bo'yicha invalidatsiya — xabarlar
  qayta sync bo'lsa avtomat qayta hisoblanadi.

## Yozish amallari (6-bosqich)

- Yozish yo'li bitta: `write_tools.propose_or_execute` → `agent_actions(proposed)`
  → `actions.confirm_action` → `execute_action` → MTProto. To'g'ridan-to'g'ri
  `client.send_message` chaqiruvi boshqa joyda bo'lmasin.
- `execute_action` ichidagi `assert_writable`, rate limit, TTL, `read_only`
  tekshiruvi — olib tashlanmaydi. `autonomous` — akkauntga bitta chat
  (`set_chat_write_mode` tekshiradi), default emas.
- `agent_actions` append-only: faqat status/confirmed_at/result_msg_id/error.
  Yangi maydon kerak bo'lsa — yangi jadval, trigger'ni yumshatmang.
- Delete tool'i — hech qachon; `WRITE_TOOL_NAMES` frozenset, testlar qo'riqlaydi.

## Kontent (7-bosqich)

- Rasm fayllari `DATA_DIR/images` — faqat `services/images.py` yozadi/o'qiydi,
  berish `/api/images/{id}` egalik bilan; `path_for_send` akkaunt egasini tekshiradi.
- Auto-reply hech qachon o'zi yubormaydi — `write_tools.propose_or_execute`
  orqali (chat `autonomous` bo'lsa o'sha yerda bajariladi). Yangi "tezkor yo'l"
  qo'shmang.
- Scheduling — Telegram server-side (`schedule=`); o'z cron'imiz bilan post
  yuborish yo'q (`scheduled_jobs` hozircha bo'sh).

## Kuzatuv va limitlar (8-bosqich)

- Metrikalar `app/observability.py` — nomlar barqaror (`tgai_*`), yangi
  metrikani shu yerga qo'shing; heartbeat cron oxirida (`heartbeat(name)`).
- Kunlik byudjet (`services/limits.py`) chat so'rovidan **oldin** tekshiriladi;
  agent/digest/judge chaqiruvlari ham `conversation_messages` tokenlariga kiradi.
- CI invariant qadamini (`tests/test_allowlist.py`, `test_peer_guard.py`,
  `test_write_actions.py`, delete-grep) yumshatmang.

## Sirlar

- Session string, master key, 2FA parol, OTP kod **hech qachon log'ga tushmaydi**
  (`app/logging.py::_redact`).
- Session'lar envelope encryption bilan saqlanadi (`app/crypto/envelope.py`):
  per-account DEK → master key bilan wrap. Master key faqat `.env` da.
- Login kodi **faqat Mini App (HTTPS)** orqali qabul qilinadi. Chat orqali
  hech qachon — Telegram chat'ga yuborilgan kodni bekor qiladi.
- QR login (`auth.ExportLoginTokenRequest`) — asosiy yo'l, kod umuman kerak emas.

## Ban riskini kamaytirish

- `TG_DEVICE_MODEL` / `TG_SYSTEM_VERSION` / `TG_APP_VERSION` — o'zgarmas.
- `FloodWaitError` da har doim kutiladi, retry-storm yo'q.
- Yangi akkaunt: birinchi sutkada cheklangan sync.
- `max_jobs = 4` — parallellikni oshirmang.

## Ma'lumot modeli — nozik nuqtalar

- **Media saqlanmaydi.** Faqat `messages.media_type` belgisi. Talab bo'yicha
  to'liq ignore — fayl yuklab olinmaydi.
- **`message_metric_snapshots` — views/reactions vaqt qatori.** Telegram
  tarixiy qiymat bermaydi. "Bu hafta qancha ko'rish qo'shildi" faqat shu
  jadvaldan hisoblanadi. Snapshot cron'ini o'chirmang — bo'shliq abadiy qoladi.
- `messages.views` — oxirgi qiymat (tez filtr uchun), tarix emas.
- **`agent_actions` append-only** — Postgres trigger UPDATE/DELETE ni to'sadi.
  Faqat `status`, `confirmed_at`, `result_msg_id`, `error` o'zgaradi.

## LLM qatlami — Claude + Gemini + DeepSeek

Ilova kodi **hech qachon** provider SDK'siga to'g'ridan-to'g'ri murojaat
qilmaydi. Faqat `app.llm.LLM` fasadi orqali:

```python
from app.llm import LLM, Msg, Task
result = await LLM().chat(Task.SEARCH, [Msg.user("...")], json_mode=True)
```

Model tanlash **vazifa darajasida** (`Task`), model nomi darajasida emas.
Yangi kodda model nomini qattiq yozmang — `Task` ni bering, router hal qiladi.

| `Task` | Claude | Gemini | DeepSeek |
|---|---|---|---|
| `ROUTE` | `claude-haiku-4-5` | `gemini-2.5-flash-lite` | `deepseek-chat` |
| `SEARCH` | `claude-opus-5` | `gemini-2.5-flash` | `deepseek-chat` |
| `TOOLS` | `claude-opus-5` | `gemini-2.5-flash` | `deepseek-chat` |
| `DEEP` | `claude-opus-5` | `gemini-2.5-pro` | `deepseek-reasoner` |
| `EMBED` | ❌ yo'q | `gemini-embedding-001` | ❌ yo'q |
| `IMAGE` | ❌ yo'q | `gemini-2.5-flash-image` | ❌ yo'q |

**Provider tanlash (`LLM_PROVIDER`):** default `auto`
(`router.auto_provider`, jarayonda bir marta): (1) API kredensiali bo'lsa
`claude` — ustuvorlik `ANTHROPIC_API_KEY` → `CLAUDE_CODE_OAUTH_TOKEN`
(`claude setup-token`) / `ANTHROPIC_AUTH_TOKEN` → profil `~/.config/anthropic`;
(2) `claude` CLI o'rnatilgan va login qilingan bo'lsa `claude_code`
(`claude -p` subprocess, Keychain sessiyasi, `CLAUDE_CODE_AUTO=false` bilan
o'chiriladi); (3) aks holda `gemini`. Secret log'ga tushmaydi, faqat manba nomi.

`claude_code` xavfsizlik bayroqlari (`claude_code.build_argv`) o'zgarmas:
`--tools ""` (built-in tool'lar o'chiq), `--setting-sources ""`,
`--no-session-persistence`, neytral cwd. Bu bayroqlarni olib tashlash = server
fayl tizimini LLM'ga ochish. `capabilities()` da TOOLS yo'q — rost.

**Imkoniyat tuzoqlari:**
- Claude'da ham, DeepSeek'da ham **embedding va rasm generatsiya yo'q** →
  router avtomat Gemini'ga tushiradi. `GEMINI_API_KEY` har doim kerak.
- Claude Opus 5 / Sonnet 5 / Opus 4.7+ da `temperature` **rad etiladi** —
  adapter uni yubormaydi (`_NO_SAMPLING_PREFIXES`). Haiku 4.5 da qoladi.
- Claude'da alohida "json mode" yo'q — `json_mode=True` system ko'rsatma +
  fence tozalash orqali. `stop_reason=refusal` xato emas: bo'sh matn +
  `finish_reason="refusal"`.
- `claude_code` da **function calling yo'q** (CLI tool'larni o'z siklida
  bajaradi, `tool_use` qaytarmaydi) → `Task.TOOLS` fallback'ga tushadi.
  Ko'p-turnli suhbat transkript matni sifatida beriladi; `temperature`/
  `max_tokens` e'tiborsiz. Docker'da CLI yo'q — token yo'li.
- `deepseek-reasoner` da **function calling yo'q** → `Task.TOOLS` unga
  berilsa, fallback ishlaydi.
- DeepSeek kontekst **64K** (Gemini 1M) → "butun tarixni promptga tashlash"
  ishlamaydi, retrieval majburiy.

Yangi provider qo'shish: `BaseProvider` dan meros oling, `capabilities()` ni
rostgo'ylik bilan e'lon qiling (yo'q imkoniyatni bor deb ko'rsatmang —
router shunga ishonadi), `router.PROVIDERS` va `default_model()` ga qo'shing.

Uzun chat tarixi bir necha savolda ishlatilsa — Gemini **context caching**
yoqing.

## Konvensiyalar

- Bot javoblari **HTML `parse_mode`**, MarkdownV2 emas (escaping do'zaxi).
- Xabar 4096 belgidan uzun bo'lsa — bo'lish yoki fayl/Mini App sahifasi.
- i18n: `uz` / `ru` / `en`, `app/i18n/locales/*.json`. Yangi matn — uchchalasiga.
- Barcha env o'qish faqat `app/config.py::get_settings()` orqali.
- Log: `structlog`, event nomi `modul.hodisa` ko'rinishida (`mtproto.blocked`).
