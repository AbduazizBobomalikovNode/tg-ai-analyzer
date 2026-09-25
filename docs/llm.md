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
| ROUTE | haiku-4-5 | 3.5-flash-lite | deepseek-chat | intent, kunlik digest, judge |
| SEARCH | sonnet-5 (config) | 3.5-flash | deepseek-chat | direct chat |
| TOOLS | sonnet-5 (config) | 3.5-flash | deepseek-chat | agent sikli (`claude_code` da yo'q → fallback) |
| DEEP | opus-5 | 3.1-pro-preview | deepseek-reasoner (tool'siz) | "chuqur tahlil" tugmasi |
| EMBED | — | gemini-embedding-001 | — | vektor indeks (`GEMINI_API_KEY` majburiy) |
| IMAGE | — | 2.5-flash-image | — | 7-bosqich |

2026-09: Gemini 2.5 avlod suhbat modellari (`flash-lite`/`flash`/`pro`) yangi
API kalitlariga **yopiq** (404 "no longer available to new users") — defaultlar
shuning uchun 3.x ga o'tkazilgan (`app.config.Settings.gemini_model_*`).
Embedding va rasm modeli o'zgarmagan: ular alohida o'tish yo'liga ega
(`EMBED_MODEL` / `reembed`, `LLM_MODEL_IMAGE`) va sifat/moslik o'lchovi talab
qiladi — shuning uchun avtomatik ko'chirilmagan.

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

### Gemini fikrlash darajasi — vazifa bo'yicha

Gemini 3.x'da fikrlash o'chmaydi, faqat darajasi bor (`thinking_level`), va
fikrlash tokenlari **output narxida** to'lanadi (`usage_metadata.thoughts_token_count`,
`conversation_messages.tokens_thinking`). Ilgari hamma vazifaga `low` berilardi —
flash-lite'ning o'z sukuti esa `minimal`, ya'ni arzon ROUTE chaqiruvlarida
(filtr, map, judge, digest) fikrlash **oshirilardi**. Endi har vazifa o'z
darajasini oladi (`router.thinking_level` → `gemini.thinking_config`):

| Env | Default | Kim ishlatadi |
|---|---|---|
| `GEMINI_THINKING_LEVEL_ROUTE` | `minimal` | filtr, map, judge, digest, kun/hafta tugunlari |
| `GEMINI_THINKING_LEVEL_SEARCH` | `low` | reduce, window_stats, direct javob, chat |
| `GEMINI_THINKING_LEVEL_TOOLS` | `low` | agent sikli |
| `GEMINI_THINKING_LEVEL_DEEP` | `low` | deep sintez (pro) |

Qiymatlar: `minimal|low|medium|high`, yoki bo'sh (`""`) — daraja umuman
yuborilmaydi (model sukuti, masalan 3.1-pro'da `high`). `minimal` ni
qo'llamaydigan modellarga (3.7/3.8-flash, 3.1-pro, `*-latest` aliaslari) adapter
`low` yuboradi — 400 o'rniga. `GEMINI_THINKING_BUDGET=0` hammasini o'chiradi
(konfiguratsiya yuborilmaydi); 2.5 avlodda byudjet (token soni) ishlatiladi,
daraja emas. Daraja almashtirish sifatni tebratadi — `make eval-run` ×3 bilan.

## Model override va embedding modelini almashtirish (phase4/B)

`LLM_TASK_*` provider bilan modelni birga beradi; `LLM_MODEL_*` esa **faqat
modelni** almashtiradi, providerni router qanday tanlagan bo'lsa shundayligicha
qoldiradi:

| Env | Nima |
|---|---|
| `LLM_MODEL_ROUTE` / `_SEARCH` / `_TOOLS` / `_DEEP` / `_EMBED` / `_IMAGE` | shu `Task` uchun model nomi |
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
router **aniq xato** tashlaydi — jimgina eski modelga qaytmaydi.
`gemini-2.5-flash-lite`/`-flash`/`-pro` 2026-09'da yangi mijozlarga yopilgan
(`app.llm.router.RETIRING`) — istalgan `Task` shu modelga tushsa (default
orqali yoki `LLM_MODEL_*` override bilan) startupda `llm.model_retiring`
ogohlantirishi chiqadi (bir marta, jarayon davomida). Voris modelga o'tish
`LLM_MODEL_*` orqali (default endi 3.x — qayta o'zgartirilmaydi, keyingi
almashtirish ham avval o'lchov, keyin override bilan).

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

## Embedding o'tkazuvchanligi — 2.6M xabarni backfill qilish

Indeksda bo'shliq bo'lsa (masalan `EMBED_MODEL` almashtirilgan yoki tarix
endigina sinxronlangan) default cron tezligi yetmaydi. Ikki yo'l bor, ikkalasi
ham bir xil yozuvchidan o'tadi (`ON CONFLICT (message_id, model) DO NOTHING`),
shuning uchun ularni aralashtirsa ham bo'ladi.

| | sync (`EMBED_MODE=sync`) | batch (`EMBED_MODE=batch`) |
|---|---|---|
| Bir yugurishda | `EMBED_SYNC_PER_RUN` (0 → `EMBED_BATCH_PER_RUN` = 500) | `EMBED_BATCH_JOBS_PER_RUN` × `EMBED_BATCH_MAX_ITEMS` |
| Cron | har 15 daqiqa | submit har 15 daqiqa, o'qish har 5 daqiqa (`poll_embed_batches`) |
| Kechikish | soniyalar | soatlar (SLA 24 soat) |
| Narx (`gemini-embedding-001`) | $0.15/M token | $0.075/M token |

**Sync bilan:** 500/15 daqiqa ≈ 48k/kun — 2.6M uchun ~54 kun. `EMBED_SYNC_PER_RUN`
ni ko'tarish mumkin (masalan 5000 ≈ 480k/kun), lekin har yugurish `EMBED_BATCH`
(100) lik chunk'larda **ketma-ket** so'rov qiladi va cron timeout'iga (1200 s)
urilishi mumkin.

**Batch bilan (tavsiya etiladi):** `EMBED_BATCH_JOBS_PER_RUN=20` va
`EMBED_BATCH_MAX_ITEMS=2000` → bir submit yugurishida 40k element, sutkasiga
96 yugurish ≈ **3.8M/kun**, ya'ni 2.6M bir kunga sig'adi. O'qish tomoni ham mos
bo'lishi kerak: `EMBED_BATCH_POLL_LIMIT=20` (default) × 12 yugurish/soat =
240 job/soat — submit tezligidan yuqori, navbat to'planmaydi.

Narx tartibi: 2.6M xabar × ~4000 belgi qopqog'i ostida o'rtacha ~40 belgi
(≈ 10 token) → ~26M token ≈ **$2**; uzunroq korpusda (o'rtacha ~160 belgi)
~104M token ≈ **$8**. Sinxron yo'lda shu raqamlar ikki barobar.

Backfill tugagach `EMBED_BATCH_JOBS_PER_RUN` ni 1 ga qaytaring — aks holda har
cron yugurishi bo'sh tanlov uchun 20 ta so'rov qiladi (zararsiz, lekin behuda).

### Model ro'yxatni bitta hujjat deb qabul qilsa

2026-09-18 da jonli tekshirilgan: `embed_content(contents=[3 ta satr])`
`gemini-embedding-001` da 3 ta vektor, `gemini-embedding-2` da esa **bitta**
qaytaradi. Sabab SDK ichida: mldev yo'lida model nomida `gemini-embedding-2`
bo'lsa `contents` `t_contents()` dan o'tadi va satrlar ro'yxati **bitta**
`UserContent(parts=[…])` ga yig'iladi.

Adapter (`app/llm/gemini.py::embed`) buni o'zi tuzatadi — uch pog'ona:

1. `contents=[str, …]` (eski, arzon yo'l);
2. kam vektor qaytsa — `contents=[Content(parts=[Part(text=…)]), …]`, ya'ni
   har matn alohida hujjat, **hamon bitta** so'rov;
3. u ham yetmasa — har matn uchun alohida so'rov,
   `EMBED_SINGLE_CONCURRENCY` (default 8) semafori bilan.

Pastga tushish `llm.embed_single_only` bilan loglanadi va tanlov **model
bo'yicha jarayon davomida** keshlanadi, ya'ni keyingi batch'lar to'g'ri
yo'ldan boradi. Qat'iy moslik tekshiruvi chaqiruvchida qoladi
(`worker.embed_misaligned`, `nodes.embed_misaligned`): adapter tuzata olmasa
ham noto'g'ri vektor **hech qachon** yozilmaydi.
