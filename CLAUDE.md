# CLAUDE.md — tg-ai-analyzer

Telegram chat tahlilchi AI agent. Python 3.12, Telethon (MTProto) + aiogram
(Bot API) + Gemini + Postgres/pgvector.

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

## LLM qatlami — Gemini + DeepSeek

Ilova kodi **hech qachon** provider SDK'siga to'g'ridan-to'g'ri murojaat
qilmaydi. Faqat `app.llm.LLM` fasadi orqali:

```python
from app.llm import LLM, Msg, Task
result = await LLM().chat(Task.SEARCH, [Msg.user("...")], json_mode=True)
```

Model tanlash **vazifa darajasida** (`Task`), model nomi darajasida emas.
Yangi kodda model nomini qattiq yozmang — `Task` ni bering, router hal qiladi.

| `Task` | Gemini | DeepSeek |
|---|---|---|
| `ROUTE` | `gemini-2.5-flash-lite` | `deepseek-chat` |
| `SEARCH` | `gemini-2.5-flash` | `deepseek-chat` |
| `TOOLS` | `gemini-2.5-flash` | `deepseek-chat` |
| `DEEP` | `gemini-2.5-pro` | `deepseek-reasoner` |
| `EMBED` | `gemini-embedding-001` | ❌ yo'q |
| `IMAGE` | `gemini-2.5-flash-image` | ❌ yo'q |

**Imkoniyat tuzoqlari:**
- DeepSeek'da **embedding ham, rasm generatsiya ham yo'q** → router avtomat
  Gemini'ga tushiradi. `GEMINI_API_KEY` har doim kerak.
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
