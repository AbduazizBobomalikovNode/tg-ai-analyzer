# LLM xarajati — profil, kamchiliklar, muqobil yo'llar

Sana: 2026-09-25. Davr: oxirgi 7 kun (2026-09-18 … 2026-09-25). Manba: jonli DB
(`conversation_messages`, `scan_runs`, `agent_runs`, `embed_batches`,
`chat_digests`), kod (`deep_search.py`, `scale_search.py`, `window_search.py`,
`agent.py`, `llm/gemini.py`, `llm/pricing.py`) va Google hujjatlari (C bo'limi).
Faqat o'qish; LLM chaqirilmadi. Narxlar — `pricing.py` taxmini (Gemini 3.x).

## Xulosa (bir paragraf)

7 kunda qayd etilgan LLM xarajati **$4.82** (192 assistant turn, 2026-09-25 ~09:40 UTC dagi kesim; parallel eval davom etayotgani uchun keyingi so'rovlarda son o'sadi), unga
qo'shimcha **qayd etilmagan** $0.23 (kunlik digest'lar, worker) va embedding
batch'lari **$4.14** (asosan `gemini-embedding-2` ga backfill) — jami ≈ **$9.2**.
Xarajatning **87% i eval/QA suhbatlaridan** ($4.18), haqiqiy foydalanuvchi
turn'lari atigi $0.66. Uch eng katta "cho'kma": (1) agent yo'li
`gemini-3.5-flash` da — **56%** (input-og'ir: 1.66M in / 21k out), (2) deep
sintez `gemini-3.1-pro-preview` da — ≈ **24%**, (3) embedding backfill (epizod
tugunlari xabar matnini qayta embed qiladi). Eng tez yutuq — `TOOLS`/`SEARCH`
ni `gemini-3.8-flash` ga o'tkazish (**≈ −$1.4 / −29%**, faqat `.env`
override, eval bilan), keyin evallarni Flex tierda yurgizish (**≈ −$2.1**)
va dalil satrlarini ixchamlash + dedup (**≈ −$0.4**).

---

## A. Real xarajat profili (7 kun)

### A1. Yo'l bo'yicha

| Yo'l | n | tokens_in p50 / p90 | tokens_out p50 / p90 | $ p50 / p90 | Jami $ | Ulush |
|---|---:|---:|---:|---:|---:|---:|
| agent (`auto`, Task.TOOLS, 3.5-flash) | 53 | 24.1k / 65.8k | 346 / 787 | 0.039 / 0.101 | 2.687 | **55.8%** |
| deep kaskad (`deep`, topic) | 28 | 54.4k / 102.9k | 1168 / 5249 | 0.038 / 0.085 | 1.247 | 25.9% |
| window (`deep`) | 8 | 7.2k / 89.9k | 992 / 1346 | 0.019 / 0.076 | 0.262 | 5.4% |
| deep kaskad (`exhaustive`) | 3 | 25.3k / 162.4k | 2330 / 12271 | 0.028 / 0.129 | 0.205 | 4.3% |
| window_stats (Task.SEARCH) | 11 | 9.1k / 10.5k | 771 / 986 | 0.015 / 0.023 | 0.173 | 3.6% |
| window (`exhaustive`) | 3 | 21.9k / 93.7k | 3634 / 10933 | 0.036 / 0.102 | 0.160 | 3.3% |
| estimate-only / cancel | 19 | 3.8k / 4.4k | 220 / 250 | 0.0017 / 0.0019 | 0.032 | 0.7% |
| judge (ROUTE) | 61 | 631 / 1087 | 45 / 54 | 0.0003 / 0.0004 | 0.020 | 0.4% |
| window (`fast`) | 1 | 7.2k | 744 | 0.014 | 0.014 | 0.3% |
| deep direct (`answer_direct`) | 3 | 925 | 77 / 436 | 0.0017 / 0.0046 | 0.009 | 0.2% |
| chat (Task.SEARCH) | 2 | 1.0k | 114 / 140 | 0.003 | 0.005 | 0.1% |
| **Jami (qayd etilgan)** | 192 | | | | **4.82** | 100% |
| digest'lar (worker, ROUTE) — *qayd etilmagan* | 88 | Σ 423k | Σ 39k | | ≈0.23 | — |
| embedding batch (messages) | 332 batch / 664k satr | | | | 1.95 | — |
| embedding batch (nodes, epizod) | 87 batch / 46k tugun | | | | 2.19 | — |

Fan-out (`fast`) yo'li amalda ishlatilmayapti — sukut `scale_mode="deep"`
(kaskad). Briefing 7 kunda 0 marta qurilgan.

### A2. Eng qimmat 10 turn

| # | conv | Nima | Yo'l / model | in / out | $ |
|---|---|---|---|---:|---:|
| 1 | 178 | qa2-e-exhaustive | exhaustive kaskad, pro | 162k / 12.3k | 0.154 |
| 2 | 197 | [eval] "Aniq oy — iyul" | agent, 3.5-flash, 5 iter | 94k / 454 | 0.145 |
| 3 | 174 | [eval] econ-01 takroriy savol | agent, `stop=iterations` 7 iter | 80k / 679 | 0.126 |
| 4 | 207 | econ-01 (qayta) | agent, `iterations` | 77k / 435 | 0.119 |
| 5 | 238 | "Bu hafta nima muhokama qilindi?" | window exhaustive, pro | 94k / 10.9k | 0.118 |
| 6 | 182 | [eval] "Kelasi haftada nima yozaman" | agent, `iterations` | 73k / 378 | 0.113 |
| 7 | 214 | econ-01 | agent, `iterations` | 70k / 353 | 0.108 |
| 8 | 211 | econ-01 | agent, `iterations` | 66k / 396 | 0.102 |
| 9 | 174 | econ-01 | agent, 5 iter | 59k / 1002 | 0.098 |
| 10 | 177 | qa2-b IT ta'lim | deep kaskad, pro | 89k / 8.6k | 0.096 |

10 dan 7 tasi — agent; ulardan 5 tasi `stop_reason=iterations` (takroriy
savol evali). `iterations` bilan tugagan 7 turn o'rtacha **$0.103**,
`answered` 46 turn — **$0.043** (2.6 iteratsiya).

### A3. Input vs output, kesh, byudjet

- **Agent**: input $2.49 (**93%**), output $0.19. Deep conv 238: input ≈ 70%.
  Umumiy profil input-og'ir (≈ 80–85%) — ya'ni **kiritish tokenini
  qisqartirish** va **kesh** asosiy dastak.
- **Kesh**: `cached_tokens` faqat log'ga yoziladi (`llm.chat cached=`), DB'ga
  emas. Joriy api log'ida 114 chaqiruvdan **bittasida** `cached=2032`, qolgani
  0. `estimate_cost` kesh chegirmasini hisobga olmaydi.
- **Thinking tokenlari hisobda yo'q**: `gemini._parse_chat` `tokens_out =
  candidates_token_count` — `thoughts_token_count` olinmaydi, Google esa
  thinking'ni output narxida hisoblaydi. Demak **$4.82 — pastki chegara**;
  adapter barcha `gemini-3*` ga `thinking_level="low"` beradi, flash-lite'ning
  sukuti esa `minimal` — ROUTE chaqiruvlarida biz fikrlashni **oshiryapmiz**.
- **Log'da token soni ko'rinmaydi**: `app/logging.scrub` `token` kalit
  bo'lagi bo'yicha `tokens_in`/`tokens_out` ni `***REDACTED***` qiladi —
  bosqich bo'yicha xarajatni log'dan tiklab bo'lmaydi.
- **Kunlik byudjet** (`LLM_DAILY_COST_BUDGET_USD=10`, `LLM_DAILY_TOKEN_BUDGET=4M`):
  eng og'ir kun (09-18) $2.78 (28%) va **2.09M token (52%)** — token limiti
  narx limitidan oldin tugaydi (pro va flash-lite tokeni bir xil sanaladi).
  Embedding va digest xarajati kunlik byudjetga **kirmaydi**.
- `scan_runs`: 66 yugurish, skan (ROUTE) jami $0.85; `relevance_cache`
  11,819 qator, 4.2k kesh hit — filtr keshi ishlayapti.

## B. Tokenlar qayerga ketadi

### B1. Statik prefikslar (o'lchangan, `CHARS_PER_TOKEN=3.5`)

| Blok | ≈ token | Har chaqiruvda qayta yuboriladimi |
|---|---:|---|
| `AGENT_SYSTEM_PROMPT` | 3,970 | ha, har iteratsiyada |
| 19 ta READ tool sxemasi | 4,390 | ha, har iteratsiyada |
| `DEEP_PLAN_PROMPT` | 3,575 | ha (+ chat indeksi user turn'ida) |
| `DEEP_SYNTH_PROMPT` | 2,010 | ha |
| `SCALE_FILTER_PROMPT` / `SCALE_MAP_PROMPT` / `SCALE_REDUCE_PROMPT` | 704 / 666 / 612 | har partiya/bo'lakda |

Agent iteratsiyasi o'rtacha ≈ 10k token; shundan **≈ 8.4k (80%+) — o'zgarmas
prefiks**. Gemini implicit keshi 3.x Flash/Pro'da **≥ 4,096 token** prefiksda
ishlaydi — agent prefiksi bu chegaradan o'tadi, plan (3.6k) va synth (2k)
system prompt'lari esa yo'q.

### B2. Conv 238 ("Bu hafta nima muhokama qilindi?", $0.118) — bosqichlar

`context.scan` + narx tenglamalaridan tiklangan (log'da tokenlar maskalangan):

| Bosqich | Model | ≈ in / out | ≈ $ | Ulush |
|---|---|---:|---:|---:|
| plan (+ 2 ta oldingi estimate-only turn'dagi qayta plan) | flash-lite | 4.3k / 0.2k (×3) | 0.002 (+0.004) | 2% |
| map, 16 bo'lak × ≤80 xabar | flash-lite | ≈ 56k / **≈ 9.8k** | 0.041 | 35% |
| synth: 150 xom satr + da'volar + sanoq bloki | **3.1-pro** | ≈ 33k / ≈ 0.9k (+ thinking, qayd yo'q) | ≈ 0.077 | 65% |

Ya'ni "1000 xom xabar pro'ga" emas: pro'ga **150 xom satr** + 97 da'vo + sanoq
tushadi; qolgan 640 dalil faqat map'da ko'riladi, **16 bo'lak qopqog'i**
(`WINDOW_MAX_CHUNKS`) tufayli eng katta 16 (chat, yuboruvchi) guruhidan
tashqarisi map'ga ham yetmaydi. Estimate $0.1137 — haqiqatga yaqin (+4%).

### B3. Isrof ro'yxati (jiddiylik bo'yicha)

1. **Agent qimmat modelda.** `Task.TOOLS` = `gemini-3.5-flash` ($1.50/$9.00),
   holbuki `gemini-3.8-flash` ($0.75/$3.75, 2026-12-31 gacha) — "eng aqlli
   Flash" va ikki barobar arzon. `SEARCH` (window_stats, reduce, direct) ham
   shu modelda.
2. **Agent prefiksi har iteratsiyada to'liq narxda** (kesh hit'i
   o'lchanmagan, `cached_tokens` saqlanmaydi). Takroriy savolda agent 6–7
   iteratsiya aylanadi: `repeat_guard` server tool'larini to'sadi, lekin
   model to'silgan chaqiruvlar bilan iteratsiyani yeyaveradi.
3. **Dalil satri ortiqcha metama'lumotli.** Har satr:
   `#c3819m7775 [<chat sarlavhasi 35 belgi> | 2026-09-21 08:12 | <kim>] matn`.
   Haftalik oynada xabar matni o'rtacha **118 belgi**, sarlavha+sana+id+kim
   ≈ **85–90 belgi** → satrning ≈ **42%** i takrorlanuvchi meta.
4. **Oyna yo'lida dedup yo'q.** 09-20…09-25 oynada 10,066 matnli xabardan
   faqat 6,829 tasi noyob (md5) — **32% aynan takror** (vakansiya
   kanallari kross-post). Kaskadda `dedup_fast` bor, `run_window` da yo'q.
5. **Map chiqishi verbose**: map chiqishi (≈ 600 token/bo'lak, $2.50/M)
   map narxining ≈ 60% ini tashkil qiladi (`MAP_MAX_CLAIMS=12`, JSON).
6. **Sintez har doim pro'da** (`Task.DEEP`), niyatdan qat'i nazar — "haftalik
   dayjest" ham, "solishtir/trend" ham.
7. **Thinking darajasi hamma vazifaga bir xil `low`**; flash-lite (ROUTE)
   sukuti `minimal`. Thinking tokenlari hisobda yo'q.
8. **Qayta plan**: exhaustive tasdiq oqimida plan har chaqiruvda qayta
   yuradi (conv 238: 3 marta). Arzon ($0.002), lekin bepul tuzatiladi.
9. **Qayd etilmagan xarajat**: `digests._summarize` (worker) va week-node
   chaqiruvlari `conversation_messages` ga yozilmaydi → dashboard va kunlik
   byudjetdan tashqarida; embedding ham shunday.
10. **Epizod embedding'i xabar matnini qayta embed qiladi**: 46.5k level-1
    tugun, o'rtacha 2,821 belgi (≈ 480 token/tugun) → $2.19 / 7 kun; xabarlar
    esa allaqachon alohida embed qilingan.
11. Judge ($0.02) va `render_history` (≤ 6 × 400 belgi) — ahamiyatsiz.

## C. Muqobil yo'llar (bizning stekka nisbatan)

Tejash — 7 kunlik profil ($4.82 LLM + $4.14 embed) bo'yicha taxmin.
Effort: S ≤ 1 kun, M 2–4 kun, L > 1 hafta.

| # | Yondashuv | Tejash (7 kun) | Effort | Sifat xavfi | Invariant |
|---|---|---:|:-:|---|---|
| C1 | `LLM_MODEL_TOOLS`/`LLM_MODEL_SEARCH=gemini-3.8-flash` | ≈ $1.41 (29%) | S | o'rta — tool-calling xulqi, eval ×3 | yo'q |
| C2 | Evallar (va judge/digest/week-node) **Flex** tierda (`service_tier="flex"`, −50%) | ≈ $2.1 eval + $0.12 fon | M | yo'q (latency 1–15 daq) | yo'q |
| C3 | Implicit kesh hit'ini o'lchash + prefiksni barqaror tutish; `cached`/`thoughts` tokenini DB'ga yozish va narxga kiritish | 0 → $0.8–1.6 (hit'ga bog'liq) | S | yo'q | yo'q |
| C4 | Agent: `repeat_guard` to'sgach tool'larni o'chirib "javob ber" turn'i | ≈ $0.42 (9%) | S | past | yo'q (qopqoq qattiqlashadi) |
| C5 | Ixcham dalil formati: chat bo'yicha guruh sarlavhasi (`## <chat>`), satrda `#N 09-21 08:12 Ali: matn`, token → qisqa indeks (validatsiya xaritasi kodda) | ≈ $0.30 (6%) | M | past (sitata validatsiyasi saqlanadi) | 3 — bitta konvert saqlanadi |
| C6 | Oyna yo'lida `dedup_fast` + "×N kanalda" belgisi | ≈ $0.05 profilda; "hafta" savolida −30% map | S | past (sanoq `count(*)` dan) | yo'q |
| C7 | Sintezni niyat bo'yicha tiering: `summary`/window → 3.8-flash, `compare`/`trend`/exhaustive → pro | ≈ $0.70 (15%) | S | **o'rta–yuqori**, eval ×3 majburiy | yo'q |
| C8 | ROUTE → `gemini-3.1-flash-lite` ($0.25/$1.50) + `thinking_level=minimal` | ≈ $0.33 (7%) | S | filtr recall'i — `calibrate_filter.py` | yo'q |
| C9 | Map chiqishini qisqartirish (`MAP_MAX_CLAIMS` 12→8, qisqa JSON kalitlari) | ≈ $0.14 (3%) | S | past | yo'q |
| C10 | Epizod vektori = a'zo xabar vektorlarining normallashgan o'rtachasi (SQL, bepul) yoki ≤ 1,500 belgi | embed nodes ≈ −$2 (backfill); keyin doimiy −90% | M | o'rta — tugun qidiruvi recall'ini o'lchash | yo'q |
| C11 | Deep **javob keshi**: kalit `(account, uz_fold(standalone_question), oyna, max(msg id) oynada, mode)`, TTL 1 soat | 5–10% (takroriy savollar) | M | past (korpus fingerprint'i eskirishni ushlaydi) | yo'q, lekin Telegram matni — suhbat/akkauntdan chiqmasin |
| C12 | "Nima muhokama qilindi" uchun kun tugunlari/`chat_digests` ni map o'rniga ishlatish (RAPTOR uslubi) | map −70…100% qayta so'ralgan oynada | L | o'rta — sitata level 0 ga qaytishi kerak | 3 — digestlar **o'sha** evidence konvertiga |
| C13 | Baho UX: `deep` da `est_total_usd > $0.05` bo'lsa kartani ko'rsatish, `> $0.10` — tasdiq | bilvosita | S | yo'q | 4 ga mos (default "so'ra") |
| C14 | LLMLingua-2 uslubidagi model-kompressor | 10–20% input | L | o'rta, torch bog'liqligi | yo'q — **tavsiya etilmaydi**, C5+C6 arzonroq |
| C15 | Batch API (chat) — interaktiv yo'l uchun | — | — | 24 soatgacha | — **mos emas**, faqat embedding (bor) va evallar |

Izohlar:

- **C1/C7 narxi vaqtinchalik**: 3.8-flash narxi 2027-01-01 dan $1.50/$7.50 ga
  ko'tariladi — o'shanda C1 tejami faqat output'da qoladi (≈ −17%). Model
  almashtirishni `LLM_MODEL_*` bilan qiling (kodda nom yo'q), `pricing.py`
  da qator bor.
- **C3**: 3.x Flash/Pro implicit keshi minimal 4,096 token, chegirma ≈ 90%
  (3.5-flash cache narxi $0.15 vs $1.50). Agent prefiksi (8.4k) mos keladi;
  plan prefiksini mos qilish uchun chat indeksini savoldan **oldin** qo'yish
  kerak (hozir history → savol → indeks). Explicit kesh saqlash to'lovi bor
  (pro: $4.50/M/soat) — bizning trafikda implicit yetarli.
- **C5** "bitta konvert" va `neutralize` ni o'zgartirmaydi — faqat satr
  shakli; `validate_citations` qisqa id → `c…m…` xaritasi bilan ishlaydi.
- **C12** CLAUDE.md dagi "Hafta tuguni faqat kun tugunlaridan" qoidasiga
  mos, lekin prod'da hozir 195 ta kun tuguni bor — qamrov yetarli emas;
  avval kun tugunlarini qurish qamrovini oshirish kerak.

## D. Ustuvor reja (top 10, tez yutuqlar birinchi)

| # | Harakat | Kutilgan tejash (7 kun) | Effort |
|---|---|---:|:-:|
| 1 | **O'lchov**: `cached_tokens` va `thoughts_token_count` ni `context` ga yozish, narxga thinking'ni qo'shish; `tokens_in/out/cached` ni `_KEY_ALLOW` ga; digest/week-node xarajatini yashirin "Digests" suhbatiga (judge uslubi) | 0 (lekin haqiqiy raqam; hozir past baholanadi) | S |
| 2 | `LLM_MODEL_TOOLS` / `LLM_MODEL_SEARCH` = `gemini-3.8-flash` (eval ×3 bilan) | **$1.41 (29%)** | S |
| 3 | Evallar runner'i va fon vazifalari (digest, week-node, judge) uchun Flex tier | **≈ $2.2** (eval xarajatining yarmi) | M |
| 4 | Agent `repeat_guard` → darhol javob turn'i; `iterations` stop'ni kamaytirish | $0.42 (9%) | S |
| 5 | ROUTE → `gemini-3.1-flash-lite` + `thinking_level=minimal` (filtr recall jadvali bilan) | $0.33 (7%) | S |
| 6 | Ixcham dalil formati (C5) + oyna yo'lida dedup (C6) | $0.35 (7%) | M |
| 7 | Sintez tiering (C7) — `summary`/window → flash; eval ×3 | $0.70 (15%) | S |
| 8 | Map chiqishi qisqartirish (C9) | $0.14 (3%) | S |
| 9 | Epizod embedding → centroid/qisqa matn (C10) | embedding'da −$2 / hafta backfill paytida | M |
| 10 | Deep javob keshi (C11) + `deep` uchun baho kartasi (C13) | 5–10% | M |

1–8 birga (bir-birini qisman qoplaydi): qayd etilgan LLM profilida
**≈ 45–55% (≈ $2.2–2.6)**, evallar Flex'ga o'tsa umumiy hisob ≈ **60%+**.

### Nima QILINMASIN

- O'lchov bilan qo'yilgan qopqoqlarni olib tashlamang/bo'shatmang:
  `MAX_CALLS_PER_TOOL`, `seen_calls`, `AGENT_MAX_EXECUTED_CALLS`,
  `AGENT_MAX_COST_USD`, `COST_CEILING`, `BUCKET_MAX_DEEP=200`,
  `SYNTH_EVIDENCE_LINES`, `DEEP_RESULT_TOKENS`. Qattiqlashtirish — mumkin,
  bo'shatish — faqat qayta o'lchov (≥ 3 marta) bilan.
- Chat indeksini yoki sanani **system** prompt'ga ko'chirmang (kesh prefiksi).
- `plan`/`reflect` ga xabar matnini "kontekst uchun" bermang (taint guard).
- Dayjest/digestlar uchun ikkinchi `<untrusted_data>` konverti ochmang.
- `SYNTH_MAX_TOKENS` ni pasaytirib chiqishni "tejamang" — kesilgan javob
  regressiyasi qaytadi; buning o'rniga niyat bo'yicha uzunlik ko'rsatmasi.
- Interaktiv yo'lga Batch/Flex qo'ymang (1–15 daqiqa / 24 soat).
- Embedding modelini `reembed` siz almashtirmang; `EMBED_DIMS` ni moslang.
- Sync'dagi `WAIT_BETWEEN_BATCHES`/ramp-up ga tegmang — bu LLM emas, ban xavfi.

## O'lchov natijasi (2026-09-25)

### Dayjest sintezida `minimal` fikrlash (B3-7, qisman)

`LLM().chat(..., thinking=...)` — chaqiruv bo'yicha override; `deep_search`
uni faqat `synth_budget(...).kind == "digest"` da beradi
(`DEEP_DIGEST_THINKING`, default `minimal`). Jonli, deploy'dan oldin/keyin
(`conversation_messages`, oyna yo'li, 8 soat):

| Yo'l | Model | n | thinking o'rtacha | out o'rtacha | $ o'rtacha |
|---|---|---:|---:|---:|---:|
| window_stats, oldin | 3.5-flash | 4 | 1,276 | 3,825 | 0.0739 |
| window_stats, keyin | 3.5-flash | 10 | **0** | 1,113 | **0.0172** |
| window dayjest (pro), keyin | 3.1-pro | 4 | 1,167 | 736 | 0.0285 |

Sanoq sintezida fikrlash yo'qoldi. Pro'dagi oyna dayjestiga ta'sir **yo'q**:
3.1-pro `minimal` ni rad etadi, adapter `low` yuboradi, bu esa `DEEP`
vazifasining o'z darajasi (keyingi qadam — C7, dayjestni flash'ga tiering).

### Model mavjudligi (loyiha kaliti, `client.models.list()` + adapter orqali tool chaqiruvi)

| Model | Ro'yxatda | Tool call + `thought_signature` (2 turn) | `minimal` | Qaytgan nom | `pricing.py` |
|---|:-:|:-:|:-:|---|---|
| `gemini-3.8-flash` | ha | ha (bytes sig, turn 2 STOP) | yo'q → adapter `low` | `gemini-3.8-flash` | bor ($0.75/$3.75) |
| `gemini-3.7-flash` | ha | — | yo'q (hujjat) | — | bor |
| `gemini-3.6-flash` | ha | — | — | — | bor |
| `gemini-flash-latest` | ha | ha | yo'q → `low` | `gemini-flash-latest` | bor |
| `gemini-3.5-flash` (joriy) | ha | ha | ha | `gemini-3.5-flash` | bor ($1.50/$9.00) |

Gemini adapteri har chat modeliga `TOOLS` beradi (`_TOOL_CAPABLE` —
faqat DeepSeek'da), ya'ni `LLM_MODEL_TOOLS=gemini-3.8-flash` jimgina
fallback'siz qabul qilinadi — kod o'zgarishi kerak emas. Bir xil
2 turnli sinovda turn-2 narxi: 3.8-flash $0.000146, 3.5-flash $0.000378.

### A/B (daily P1, 18 senariy) — **tugallanmadi**

| Konfiguratsiya | Yugurish | Pass (det+judge) | det | judge | $ | degraded | auto `stop_reason` | auto tool median |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| baseline (3.5-flash) #1 | 1 | **15/18** | 17/18 | 21/24 | 0.710 | 0 | answered ×4 | 0 |
| baseline #2 | — | yaroqsiz | | | 0.582 | | | |
| candidate (3.8-flash) | 0 | — | | | | | | |

Baseline #2 ning o'rtasida (daily-30, 6-turn'dan) har so'rov
`chat.err.budget / tokens` bilan qaytdi: user 1 ning **24 soatlik sirpanuvchi**
token byudjeti (`LLM_DAILY_TOKEN_BUDGET=4M`) tugadi — 4.00M token / $4.12,
shundan 09:00–13:00 UTC da ≈ 3.3M (parallel eval/QA trafik). Byudjet
xavfsizlik qopqog'i, uni o'lchov uchun ko'tarilmadi; to'liq qaytishi ≈ 20
soat. Baseline #1 failed'lar: daily-04 (judge: oyna matnda yo'q), daily-16
(exhaustive tasdiqsiz — bo'sh javob, judge), daily-45 (`must_propose_action`).

**Qaror: almashtirilmadi**, `.env` o'zgarmagan (`LLM_MODEL_SEARCH/TOOLS=
gemini-3.5-flash`). Qoida (kandidat pass ≥ baseline − 1, safety/multiturn'da
takroriy flip yo'q, narx ≥ 25% past) 3×3 o'lchovsiz qo'llanmaydi. Izoh: daily
P1 to'plami asosan deep turn'lar (20 deep / 4 auto, auto'da 0 tool
chaqiruvi) — uning narxini pro sintez va flash-lite skan belgilaydi, TOOLS
modeli emas. 3.8-flash qarori uchun agent-og'ir to'plam (`economy`,
`multiturn`, `people`) ham qo'shilishi kerak; yugurishni byudjet bo'shagan
kunda, boshqa eval trafigisiz qiling (6 yugurish ≈ 3M token — 4M limitga
zo'rg'a sig'adi).

### Orkestrator planner A/B (`ORCHESTRATOR_PLAN_TASK`, 2026-09-25)

Planner (`orchestrator.plan_orchestrated`) endi `ORCHESTRATOR_PLAN_TASK=route|search`
bilan tanlanadi (sukut `search`). `orch.plan` log'i `plan_task` ni, yaroqsiz
JSON esa `orch.plan_invalid` ni yozadi (`planner_ok=False`).

**Uchta o'lchov** (hammasi `gemini-3.5-flash` = SEARCH, `3.5-flash-lite` = ROUTE):

| # | Nima | Pass | $ jami | Planner $ | Yaroqsiz JSON |
|---|---|---:|---:|---:|---:|
| 1 | daily P1 (18 senariy, 20 deep), `search` — run 180642 | 16/18 | 0.744 | — | — |
| 2 | daily P1 takror, **amalda `search`** (run 183514, pastga qarang) | 17/18 | 0.759 | 0.146 (18 chaqiruv) | 0 |
| 3 | planner-only replay, o'sha 17 deep turn, `route` | — | — | 0.047 (17) | 0 |

- #2 `route` bo'lishi kerak edi, lekin chaqiruv joyi hali `Task.SEARCH` edi
  (`orch.plan` `plan_task=route` deb yozdi, `llm.chat` esa `task=search`).
  U **shovqin o'lchovi** bo'lib qoldi: bir xil konfiguratsiyada 3 ta flip
  (daily-03 ✗→✓, daily-16 ✗→✓, daily-45 ✓→✗ `must_propose_action`) va
  +2% narx. Ya'ni bitta yugurishdagi 1–3 flip — tebranish, signal emas.
- #3: `route` planner 17 turn'dan **15** tasida jonli `search` rejasi bilan
  **aynan bir xil** qadamlarni tanladi (farq — 369 suhbatining 2 ta
  follow-up turn'i: `reuse_prior` o'rniga `search_local`; xuddi shu replay
  `search` bilan ham 14/17 berdi — tarixni kesish replay'ning o'z
  cheklovi). Yaroqsiz JSON 0, kechikish 1.1–3.0 s (search 1.6–3.2 s).
  Planner narxi −68% ($0.146 → $0.047 tokenlar bo'yicha).
- **Lekin jami narxga ta'siri shiftlangan**: planner run #2 ning 19% i
  ($0.146 / $0.759). `route` bilan tejash ≈ $0.10 / 20 deep turn = **≈13%**.
  Planner bepul bo'lsa ham qoidaning "≥30% arzonroq" sharti bajarilmaydi —
  narxni sintez (`Task.DEEP`/pro) va skan belgilaydi.

**Qaror: sukut `search` qoladi.** Qoida (pass ≥ 16/18, safety/multiturn flip
yo'q, **jami** narx ≥30% past) — narx sharti printsipial yetib bo'lmaydi.
To'liq `route` e2e yugurishi $1.2 kunlik o'lchov qopqog'i tufayli
qilinmadi (#2 xatosi $0.76 ni yedi). `route` — jonli sinalgan, JSON
barqaror, tejamli ikkinchi variant: agar keyinroq planner ulushi o'ssa
(masalan sintez arzonlashsa), bitta `.env` qatori bilan yoqiladi — avval
to'liq daily P1 ×3.

## Manbalar

- Gemini API pricing — https://ai.google.dev/gemini-api/docs/pricing (yangilangan 2026-09-24): 3.8/3.7/3.6-flash $0.75/$3.75 (2026-12-31 gacha, keyin $1.50/$7.50), 3.5-flash $1.50/$9.00, 3.5-flash-lite $0.30/$2.50, 3.1-flash-lite $0.25/$1.50, 3.1-pro-preview $2/$12 (≤200k), cache $0.15 (3.5-flash), pro kesh saqlash $4.50/M/soat; "output price includes thinking tokens".
- Context caching — https://ai.google.dev/gemini-api/docs/caching (2026-09-02) va https://ai.google.dev/gemini-api/docs/generate-content/caching (2026-09-11): implicit kesh 2.5+ da sukut bo'yicha, 3.x Flash/Pro minimal 4,096 token, katta umumiy kontentni boshiga qo'yish tavsiyasi; explicit TTL sukuti 1 soat.
- Flex inference — https://ai.google.dev/gemini-api/docs/flex-inference (2026-09-23): `service_tier="flex"`, −50%, 1–15 daqiqa, 503/429 da server fallback yo'q.
- Batch mode — https://ai.google.dev/gemini-api/docs/batch-mode (2026-09-17): −50%, 24 soat maqsad, kesh bilan ishlaydi.
- Thinking — https://ai.google.dev/gemini-api/docs/thinking (2026-09-23): `thinking_level` minimal/low/medium/high; sukut: 3.8/3.7/3.6-flash `medium`, 3.5-flash-lite `minimal`, 3.1-pro `high`; thinking output narxida.
- RAPTOR (ierarxik xulosa daraxti) — Sarthi va b., ICLR 2024, https://arxiv.org/abs/2401.18059 (2024-01).
- LLMLingua-2 (prompt kompressiyasi) — Pan va b., ACL Findings 2024, https://arxiv.org/abs/2403.12968 (2024-03).
- GPTCache (semantik javob keshi) — https://github.com/zilliztech/GPTCache.

## Ilova — SQL

```sql
-- yo'l klassifikatsiyasi (A1)
WITH t AS (
  SELECT CASE
    WHEN task='tools' THEN 'agent' WHEN task='judge' THEN 'judge'
    WHEN task='search' THEN 'chat(search)'
    WHEN context->>'status' IN ('needs_confirm','cancelled') THEN 'estimate-only'
    WHEN context->>'direct'='true' THEN 'direct'
    WHEN context->>'retrieval_kind'='window' AND context->>'window_stats'='true' THEN 'window_stats'
    WHEN context->>'retrieval_kind'='window' THEN 'window('||(context->>'scale_mode')||')'
    WHEN context->>'scale_mode' IN ('deep','exhaustive') THEN 'cascade('||(context->>'scale_mode')||')'
    ELSE 'fanout' END p, tokens_in, tokens_out, cost_usd
  FROM conversation_messages
  WHERE created_at > now()-interval '7 days' AND role='assistant')
SELECT p, count(*),
  percentile_disc(0.5) WITHIN GROUP (ORDER BY tokens_in),
  percentile_disc(0.9) WITHIN GROUP (ORDER BY tokens_in),
  percentile_disc(0.5) WITHIN GROUP (ORDER BY tokens_out),
  percentile_disc(0.9) WITHIN GROUP (ORDER BY tokens_out),
  percentile_cont(0.5) WITHIN GROUP (ORDER BY cost_usd),
  percentile_cont(0.9) WITHIN GROUP (ORDER BY cost_usd),
  sum(cost_usd), 100*sum(cost_usd)/sum(sum(cost_usd)) OVER ()
FROM t GROUP BY p ORDER BY 9 DESC;

-- eval vs real
SELECT (c.title LIKE '[eval]%' OR c.title LIKE 'qa%' OR c.title LIKE 'live%') AS eval,
       sum(m.cost_usd)
FROM conversation_messages m JOIN conversations c ON c.id=m.conversation_id
WHERE m.created_at > now()-interval '7 days' GROUP BY 1;

-- agent stop sabablari
SELECT context->>'stop_reason', count(*), avg((context->>'iterations')::int),
       avg(tokens_in), avg(cost_usd)
FROM conversation_messages WHERE task='tools' AND created_at > now()-interval '7 days'
GROUP BY 1;

-- skan audit
SELECT mode, stopped_by, count(*), sum(cost_usd), avg(est_cost_usd), avg(candidates),
       avg(kept), sum(cache_hits)
FROM scan_runs WHERE created_at > now()-interval '7 days' GROUP BY 1,2;

-- embedding va digest (qayd etilmagan) xarajat
SELECT submitted_at::date, kind, count(*), sum(cost_usd), sum(jsonb_array_length(item_ids))
FROM embed_batches GROUP BY 1,2 ORDER BY 1,2;
SELECT count(*), sum(tokens_in), sum(tokens_out) FROM chat_digests
WHERE created_at > now()-interval '7 days';

-- kunlik byudjet ishlatilishi
SELECT m.created_at::date, c.user_id, sum(m.cost_usd), sum(m.tokens_in+m.tokens_out)
FROM conversation_messages m JOIN conversations c ON c.id=m.conversation_id
WHERE m.created_at > now()-interval '7 days' GROUP BY 1,2 ORDER BY 1;

-- oynadagi takror (conv 238 oynasi)
SELECT count(*), count(DISTINCT md5(m.text)), avg(length(c.title)),
       avg(least(length(m.text),400))
FROM messages m JOIN chats c ON c.id=m.chat_id
WHERE c.account_id=2 AND m.published_at >= '2026-09-20T19:00:00Z'
  AND m.published_at < '2026-09-25T09:28:00Z' AND coalesce(m.text,'')<>''
  AND m.deleted_at IS NULL;
```
