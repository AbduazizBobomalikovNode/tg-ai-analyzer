# Ishlab chiqish

## Muhit
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=src
make test        # pytest (DB/Telegram/tarmoqsiz — 200+ test)
make lint        # ruff + mypy
make fmt
```
Testlar mashinadagi Claude kredensiallarini chetlab o'tadi (`ANTHROPIC_CONFIG_DIR`,
`CLAUDE_CODE_BIN` soxta). `tests/test_md_renderer.py` node bo'lsa ishlaydi.

## Konvensiyalar
* Env faqat `app/config.py::get_settings()` orqali.
* Log — structlog, `modul.hodisa`; sirlar `_redact` orqali.
* i18n — `uz/ru/en`, yangi matn uchchalasiga (`app/i18n/locales`); web string'lar
  `web.*`, xatolar `<domen>.err.*`.
* Bot javoblari HTML parse_mode; 4096 belgidan uzun — bo'lish/fayl.
* Yangi LLM chaqiruvi — faqat `LLM().chat(Task.X, …)`, model nomi qattiq yozilmaydi.
* Yangi kontekst yo'li — `search.select_context` orqali; yangi tool —
  `tools.READ_TOOLS` (o'qish) yoki `write_tools` (taklif); yozish yo'li bitta.
* Prompt o'zgarishi — `services/prompts.py`; system'ga o'zgaruvchan narsa qo'shmang.
* Migratsiya: `make revision m="..."` (autogenerate), `0001…0005` ketma-ket.
* Frontend — vanilla JS/CSS, tashqi CDN yo'q; markdown `md.js` (xavfsiz),
  mermaid vendored.

## Test qatlamlari
| Fayl | Nima |
|---|---|
| test_allowlist, test_peer_guard, test_envelope | invariantlar, crypto |
| test_llm_router, test_llm_adapters, test_llm_claude, test_llm_claude_code | provider'lar, auto tanlov |
| test_auth_flow, test_web | login FSM, cookie/CSRF, API |
| test_ingestion, test_search_quality, test_md_renderer | sync mapping, qidiruv/byudjet, markdown |
| test_agent, test_write_actions | tool registry, sikl, taklif/tasdiq/bajarish |

## Bosqichlar (keyingi)
7 — kontent: rasm generatsiya (Gemini image), rejalashtirilgan post UI, auto-reply;
8 — polish: Sentry/metrikalar, rollup jadvali, QR login, real-time listener,
CI (GitHub Actions), yuk testi.
