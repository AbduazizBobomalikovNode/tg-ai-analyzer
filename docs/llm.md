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

## Model override va embedding modelini almashtirish (phase4/B)

`LLM_TASK_*` provider bilan modelni birga beradi; `LLM_MODEL_*` esa **faqat
modelni** almashtiradi, providerni router qanday tanlagan bo'lsa shundayligicha
qoldiradi:

| Env | Nima |
|---|---|
| `LLM_MODEL_ROUTE` / `_SEARCH` / `_TOOLS` / `_DEEP` / `_EMBED` | shu `Task` uchun model nomi |
| `EMBED_MODEL` | embedding modeli (eski nom: `GEMINI_MODEL_EMBED`) |
| `EMBED_DIMS` | Gemini `output_dimensionality`; DB ustuni bilan bir xil bo'lishi shart |

`EMBED_MODEL` filtri uchun joriy model nomi — `llm.embed_model_name()`,
ustuvorlik `resolve(Task.EMBED)` bilan **bir xil**: `LLM_TASK_EMBED` ning
model qismi → `LLM_MODEL_EMBED` → `EMBED_MODEL`.

`EMBED_DIMS` ni o'zgartirish yolg'iz o'zi hech narsani ko'chirmaydi: DB ustuni
`Vector(embed_dim)` / `HALFVEC(embed_dim)` bilan migratsiya vaqtida qurilgan,
mos kelmasa **har** embedding INSERT'i yiqiladi. Shuning uchun
`python -m app.main check` buni tekshiradi (api/worker startupda esa
ogohlantirish). Haqiqiy o'zgartirish = ustun turini almashtiradigan migratsiya
+ to'liq qayta embed.

Model provider imkoniyatiga mos kelmasa (`LLM_MODEL_TOOLS` ga tool'siz model)
router **aniq xato** tashlaydi — jimgina eski modelga qaytmaydi. `gemini-2.5-flash-lite`
2026-10-16 da eskiradi: 45 kun qolganda startupda `llm.model_retiring`
ogohlantirishi chiqadi, vorisga o'tish esa `LLM_MODEL_ROUTE` orqali (default
o'zgarmaydi — model almashtirish sifatni tebratadi, avval o'lchang).

`gemini-embedding-001` da kesilgan (`output_dimensionality < 3072`) vektor
birlik uzunlikda emas — adapter uni **mijoz tomonida** L2 normallashtiradi
(`gemini.NORMALIZE_CLIENT_SIDE`). `gemini-embedding-2` o'zi normallashtiradi.

Modelni almashtirish: `EMBED_MODEL` ni yangilang va

```bash
docker compose run --rm bot python -m app.main reembed --model gemini-embedding-2
```

Bu `worker.reembed_all` ni navbatga qo'yadi. Ish **hech narsa o'chirmaydi**:
yangi model qatorlari eskilari bilan yonma-yon yoziladi va qidiruv joriy
modelni filtrlaydi, shuning uchun to'xtatib qo'yilgan re-embed ham indeksni
buzmaydi; orqaga qaytish — `EMBED_MODEL` ni eski qiymatga qaytarish.
Jarayon loglarda: `worker.reembed.progress` / `worker.reembed.done`.

"Yonma-yon" **sxema darajasida** (migratsiya `0016`):
`message_embeddings` PK = `(message_id, model)`, `retrieval_node_embeddings`
PK = `(node_id, model)`. Ilgari kalit bir ustunli edi, ya'ni bu va'da bajarib
bo'lmas edi: yozuvchilar "joriy model bo'yicha yetishmayapti" deb tanlab,
o'sha `message_id` ustida `UniqueViolation` bilan qulardi, qidiruv esa yangi
modelni filtrlab bo'sh qaytarardi. Barcha yozuvchilar
(`worker.embed_messages`, `worker.reembed_all`, `nodes.embed_nodes`,
`embed_batches._write_vectors`) `ON CONFLICT (id, model) DO NOTHING` bilan
yozadi. Downgrade (`0016` → `0015`) har id uchun **eng yangi** qatorni
qoldirib qolganini o'chiradi — teskari yo'lda ma'lumot yo'qoladi.
