# LLM qatlami

Ilova kodi provider SDK'siga to'g'ridan-to'g'ri murojaat qilmaydi — faqat
`app.llm.LLM` fasadi, model tanlash **vazifa** (`Task`) darajasida.

```python
from app.llm import LLM, Msg, Task
res = await LLM().chat(Task.SEARCH, [Msg.system(...), Msg.user(...)], json_mode=True)
vec = await LLM().embed(["..."])
```

## Provider'lar va `auto`

`LLM_PROVIDER=auto` (default): API kredensiali bo'lsa `claude` (ustuvorlik
`ANTHROPIC_API_KEY` → `CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token`) →
`~/.config/anthropic` profil) → `claude` CLI login bo'lsa `claude_code`
(`claude -p`, tool'siz) → aks holda `gemini`. Aniq `gemini`/`deepseek` — eski
xatti-harakat.

| `Task` | Claude | Gemini | DeepSeek | Nimada ishlatiladi |
|---|---|---|---|---|
| ROUTE | haiku-4-5 | 2.5-flash-lite | deepseek-chat | intent, kunlik digest, judge |
| SEARCH | sonnet-5 (config) | 2.5-flash | deepseek-chat | direct chat |
| TOOLS | sonnet-5 (config) | 2.5-flash | deepseek-chat | agent sikli (`claude_code` da yo'q → fallback) |
| DEEP | opus-5 | 2.5-pro | deepseek-reasoner (tool'siz) | "chuqur tahlil" tugmasi |
| EMBED | — | gemini-embedding-001 | — | vektor indeks (`GEMINI_API_KEY` majburiy) |
| IMAGE | — | 2.5-flash-image | — | 7-bosqich |

Imkoniyat yetishmasa router `LLM_FALLBACK_PROVIDER` (gemini) ga o'tadi va log
yozadi; hech kim jimgina ishlamay qolmaydi.

## Prompt'lar

`services/prompts.py` — yagona manba: `_CORE` (rol, ishonch qoidasi, javob
shakli, chegaralar) → `CHAT_SYSTEM_PROMPT` (oldindan kontekst) va
`AGENT_SYSTEM_PROMPT` (tool'lar, yozish qoidalari), `MAP_DIGEST_PROMPT`,
`JUDGE_PROMPT`, `ROUTER_PROMPT`. System **bayt darajasida barqaror** — Claude
`cache_control`, Gemini implicit kesh. Sana / til / pinned chat — `runtime_note`
user turn'ida.

## Token va xarajat strategiyasi

| Qayer | Nima |
|---|---|
| Kontekst | `search.select_context`: `search` (FTS+trgm+pgvector, RRF, ±1 qo'shni + oxirgi 15), `window` (vaqt oynasi; katta bo'lsa **kunlik digest keshi**), `recent`; byudjet 6k / deep 14k token; matn siqish (URL→domen, 700 belgi) |
| Tarix | 12 turn × 1500 belgi |
| Agent | ≤5 iteratsiya, ≤4 tool/iteratsiya, natijalar ≤12k token, tool natijasi ≤9k belgi, majburiy yakun |
| Digest | kunlik, arzon model, kechasi oldindan, `msg_count` invalidatsiya |
| Embedding | faqat ≥20 belgi, batch 100, cron 15 daq |
| Judge | savol+javob (kontekst emas), `WEB_AUTO_EVAL_SAMPLE` |
| Modellar | ROUTER arzon; FAST Sonnet 5 (default); Opus faqat tugma bilan; `claude_code` = obuna, $0 |
| Kesh | Claude system ≥3000 belgi → `cache_control`; DeepSeek avtomatik disk-cache |

Xarajat bahosi — `llm/pricing.py` (ro'yxat narxi, taxmin). Har javobda
`tokens_in/out`, `latency_ms`, `cost_usd`; dashboard model/strategiya kesimida.
