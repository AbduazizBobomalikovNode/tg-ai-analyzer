# Deploy va ekspluatatsiya

## Talablar
Docker + Compose; `my.telegram.org` (`TG_API_ID/HASH`); @BotFather bot token va
bot ID (`CONTROL_BOT_ID` — peer deny-list shunga tayanadi); `GEMINI_API_KEY`
(embedding/rasm majburiy); LLM: Claude (API key / Claude Code token) yoki Gemini /
DeepSeek; HTTPS domen (`WEBAPP_BASE_URL`) — login kodi shu orqali.

## Birinchi ishga tushirish
```bash
cp .env.example .env
python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"  # MASTER_KEY_B64
# .env: BOT_TOKEN, CONTROL_BOT_ID, TG_API_ID/HASH, MASTER_KEY_B64, GEMINI_API_KEY,
#       LLM: ANTHROPIC_API_KEY yoki CLAUDE_CODE_OAUTH_TOKEN (`claude setup-token`)
make up          # postgres, redis, migrate (alembic upgrade head), bot, api, worker
docker compose run --rm bot python -m app.main check   # config, LLM marshruti, DB, Redis
make logs
```
Web: `http://localhost:8080` → `/login` → telefon → kod → `/chat`.

## Servislar
| Servis | Buyruq | Eslatma |
|---|---|---|
| api | `uvicorn app.web.main:app` | **bitta worker** (auth oqimlari xotirada); reverse-proxy ortida `X-Forwarded-Proto` hurmat qilinadi (Secure cookie) |
| worker | `arq app.worker.settings.WorkerSettings` | `max_jobs=4`; cron'lar: snapshot xx:05, incremental /10, embed /15, digest 02:30 |
| bot | `python -m app.main bot` | boshqaruv boti (skelet) |
| migrate | `alembic upgrade head` | har deploy'da |

## Env (asosiylari)
`LLM_PROVIDER=auto`, `CLAUDE_MODEL_ROUTER/FAST/DEEP`, `WEB_*` (cookie TTL, auth
rate, kontekst limitlari, auto-eval, sample), `AGENT_MAX_ITERATIONS`,
`AGENT_TOOL_RESULT_TOKENS`, `WRITE_PROPOSAL_TTL_HOURS`, `WRITE_RATE_PER_HOUR`,
`EMBED_ENABLED`, `DEFAULT_WRITE_MODE`, `MAX_ACCOUNTS`, `ALLOWED_USER_IDS`.
To'liq ro'yxat va izohlar — `.env.example`.

## Kuzatuv (8-bosqich)
* `GET /metrics` — Prometheus (`METRICS_TOKEN` bilan yoping). Nomlar `tgai_*`
  (`app/observability.py`). Grafana uchun asosiy panellar: LLM tokenlar/xarajat
  (provider/model), latency p50/p95, tool chaqiruvlar, yozish amallari statusi,
  FloodWait soni, HTTP 5xx.
* Worker heartbeat — Redis `tgai:hb:<cron>`; `GET /api/stats/system` va dashboard
  "Tizim" kartasi. Cron 2× oraliqdan uzoq jim bo'lsa — worker yiqilgan.
* Sentry — `SENTRY_DSN` (`[monitoring]` extra); komponent tegi `api`/`worker`.

## Kuzatuv (loglar)
* Log'lar structlog, hodisa nomlari `modul.hodisa`: `ingest.sync_done`,
  `worker.snapshot.done`, `agent.done`, `chat.answer`, `write.proposed/executed/
  blocked`, `llm.fallback`, `mtproto.blocked`.
* Dashboard `/dashboard`: token/xarajat/latency/sifat, ingestion holati, amallar.
* Muhim signallar: `mtproto.blocked` (guardrail ishladi — kim/nima), FloodWait
  (`worker.sync_account.paused`), `pool.err.session_revoked` (akkaunt qayta login).

## Backup / rotatsiya
Postgres — odatiy dump. `MASTER_KEY_B64` — secret manager'da; yo'qolsa
session'lar tiklanmaydi. Rotatsiya: `crypto.envelope.rewrap_dek` bilan DEK'lar
qayta wrap (skript kerak bo'lsa 8-bosqichda).

## Ma'lum cheklovlar
`api` va `worker` bir xil session'ni ishlatadi (bir host/egress IP'da muammo yo'q);
`AuthKeyDuplicatedError` ko'rsangiz api'dagi jonli o'qishni DB'ga ko'chiring.
`claude_code` provider'da tool yo'q — agent Gemini'ga tushadi. Docker'da `claude`
CLI yo'q → `CLAUDE_CODE_OAUTH_TOKEN`.
