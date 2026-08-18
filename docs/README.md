# tg-ai-analyzer — hujjatlar

Telegram chatlarini AI orqali tahlil qiluvchi agent: MTProto (akkaunt nomidan
o'qish/yozish) + Web UI (telefon orqali login, AI chat, dashboard) + Claude /
Gemini / DeepSeek + Postgres/pgvector + ARQ worker.

| Hujjat | Nima haqida |
|---|---|
| [architecture.md](architecture.md) | Komponentlar, ma'lumot oqimi, servislar xaritasi |
| [security.md](security.md) | Buzib bo'lmaydigan invariantlar, himoya qatlamlari, sirlar |
| [data-model.md](data-model.md) | Jadvallar, migratsiyalar, nozik nuqtalar |
| [api.md](api.md) | HTTP API (auth, chat, telegram, actions, stats) |
| [llm.md](llm.md) | Provider'lar, `Task` router, prompt'lar, token/xarajat strategiyasi |
| [ingestion.md](ingestion.md) | Chat registry, tarix sync, snapshot, digest, embedding cron'lari |
| [agent.md](agent.md) | Read-only tool'lar, agent sikli, yozish amallari va tasdiqlash |
| [deployment.md](deployment.md) | Env, Docker Compose, tekshiruv, ekspluatatsiya |
| [development.md](development.md) | Lokal muhit, testlar, konvensiyalar, bosqichlar |

Qisqa boshlash — [../README.md](../README.md). Claude Code uchun qoidalar —
[../CLAUDE.md](../CLAUDE.md). Asl reja va qarorlar — [../PLAN.md](../PLAN.md).
