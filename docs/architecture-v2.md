# Arxitektura v2 — "burchakdan burchakka" qidiruv agenti

> Status: **dizayn hujjati** (lead tomonidan tasdiqlangan qarorlar). Joriy holat —
> [architecture.md](architecture.md). Bu hujjat 1–3-bosqichlarda nima quriladi va
> **nima uchun** aynan shunday quriladi, degan savolga javob beradi.
> Ochiq qolgan, user qarori kutilayotgan bandlar — 13-bo'limda.

## 1. Maqsad va tamoyillar

Maqsad: foydalanuvchi bitta savol yozadi ("Nurbek bilan to'lov haqida nima
kelishgandik?"), tizim esa **hamma sinxronlangan chatlar** bo'ylab qidiradi,
kerak bo'lsa Telegram serveridan sinxronlanmagan tarixni ham so'raydi,
o'xshash xabarlarni bir joyga yig'adi, tahlil qiladi va **sitatali** javob
beradi — ostida **halol qamrov hisoboti** bilan: qaysi chatlar ko'rildi,
qaysilarida bo'shliq bor. Digestlar ham xuddi shu indeks ustida ishlaydi:
bitta retrieval poydevori — ikki mahsulot.

Tamoyillar:

1. **To'rt invariant o'zgarmaydi.** `mtproto/allowlist.py` deny-by-default,
   `mtproto/guard.py` peer deny-list, chat kontenti faqat `<untrusted_data>`
   konvertida, default `write_with_confirm`. v2 faqat **READ** metodlar qo'shadi;
   bitta ham yozish yoki o'chirish metodi qo'shilmaydi.
2. **Halollik > taassurot.** "Hamma xabarlarni ko'rdim" degan gap hech qachon
   aytilmaydi. Qamrov kodda hisoblanadi, LLM'ga yozdirilmaydi.
3. **Deterministik ish — koddan, sub'ektiv ish — modeldan.** Segmentatsiya,
   dedup, RRF, qamrov statistikasi — sof kod. Model faqat tushunish,
   yorliqlash, sintez qiladi.
4. **Dual-LLM.** Rejalashtiruvchi va reflekt bosqichlari begona matnni
   **ko'rmaydi**; faqat sintezator ko'radi va unda tool yo'q. Prompt injection
   bilan boshqarilishi mumkin bo'lgan bosqich — harakat qila olmaydigan bosqich.
5. **Narx o'lchov bilan qo'yiladi.** `CLAUDE.md` dagi mavjud qopqoqlar
   (`MAX_CALLS_PER_TOOL = 3`, `seen_calls`, `runtime_note` ichidagi chat indeksi)
   kabi, v2 byudjetlari ham kodda qattiq turadi.

## 2. Umumiy oqim

```
                    ┌──────────────────────────── savol (web UI, deep=on/off)
                    ▼
        ┌───────────────────────┐   untrusted matn KO'RMAYDI
        │ PLAN (Task.ROUTE)     │   → intent, window, chats, subqueries 3–8,
        │ deep_search.plan()    │     entities, level, use_hyde
        └──────────┬────────────┘
                   │ asyncio.gather
   ┌───────────────┼────────────────┬──────────────┬─────────────────┐
   ▼               ▼                ▼              ▼                 ▼
 tsv_uz          tsv_ru        trigram          vector          retrieval_nodes
 (uz-lat+cyr)   (ru)       strict_word_sim   (1 embedding)    (episode/day/week)
   └───────────────┴────────────────┴──────────────┴─────────────────┘
                   │  + (miss bo'lsa) Telegram rung'lari: messages.Search,
                   │    SearchGlobal, SearchResultsCalendar …  ← token bucket
                   ▼
          ┌─────────────────┐   w/(k+rank), k=60   ┌──────────────────────┐
          │ search.rrf()    │─────────────────────►│ cluster.py           │
          └─────────────────┘                      │ sha1 → simhash≤3 →   │
                                                   │ cosine≥0.93 → agglo  │
                                                   └──────────┬───────────┘
                                                              ▼
                                                   ┌──────────────────────┐
                                                   │ rerank.py (ROUTE)    │
                                                   │ listwise, ≤60 nomzod │
                                                   └──────────┬───────────┘
                    ┌─────────────────────────────────────────┤
                    ▼                                         │
        ┌───────────────────────┐  faqat yorliq+statistika    │
        │ REFLECT (Task.ROUTE)  │  (matn EMAS)                │
        └──────────┬────────────┘                             │
                   │ ≤1 qo'shimcha raund (marginal-yield <10%)│
                   ▼                                          ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ SYNTHESIZE (Task.DEEP) — yagona untrusted ko'ruvchi, TOOLSIZ │
        └──────────┬───────────────────────────────────────────────────┘
                   ▼
        SSE /api/deep/stream: plan · search · cluster · rerank · token ·
                              coverage · done(narx)
```

Digest yo'li ham shu indeksdan oziqlanadi: L1 (kun) → `retrieval_nodes` level 2 →
L2 (hafta) → L3 (shaxsiy briefing).

## 3. Ma'lumot modeli o'zgarishlari

Joriy sxema `0001`–`0006` (+ `0007_agent_run_error` — 0-bosqich). Quyidagilar
1-bosqich migratsiyasida (raqami `0008`, 13-bo'limga qarang) va 2-bosqichda.

### 3.1 `messages` kengaytmasi (1-bosqich)

```sql
ALTER TABLE messages
  ADD COLUMN lang             varchar(8),       -- uz | ru | en | other
  ADD COLUMN lang_conf         real,
  ADD COLUMN fwd_from_peer_id  bigint,
  ADD COLUMN fwd_from_msg_id   bigint,
  ADD COLUMN fwd_date          timestamptz,
  ADD COLUMN top_msg_id        bigint,          -- forum topic / thread
  ADD COLUMN deleted_at        timestamptz,     -- soft-delete (2-bosqich)
  ADD COLUMN edited_at_seen    timestamptz;     -- mavjud edited_at bilan solishtiriladi

-- generated columns: inline to_tsvector so'rovda QURILMAYDI
ALTER TABLE messages
  ADD COLUMN text_fold text GENERATED ALWAYS AS (uz_fold(text)) STORED,
  ADD COLUMN tsv_uz tsvector GENERATED ALWAYS AS
      (to_tsvector('simple', uz_fold(text))) STORED,
  ADD COLUMN tsv_ru tsvector GENERATED ALWAYS AS
      (to_tsvector('russian', coalesce(text, ''))) STORED;

CREATE INDEX ix_messages_tsv_uz ON messages USING gin (tsv_uz);
CREATE INDEX ix_messages_tsv_ru ON messages USING gin (tsv_ru);
CREATE INDEX ix_messages_fold_trgm ON messages USING gin (text_fold gin_trgm_ops);
CREATE INDEX ix_messages_top_msg ON messages (chat_id, top_msg_id)
  WHERE top_msg_id IS NOT NULL;
```

Eski `ix_messages_fts` (`to_tsvector('simple', text)`) va `ix_messages_trgm`
migratsiyada **o'chiriladi** — ular yangi generated ustunlar bilan
takrorlanadi. `search.fts_search` ichidagi inline `func.to_tsvector` yo'q
bo'ladi (hozir indeks ifodasi bilan aynan mos kelmasligi mumkin — sekin skan
sababi).

### 3.2 `chats`

```sql
ALTER TABLE chats ADD COLUMN link_prefix varchar(64);  -- 't.me/c/<id>/' | 't.me/<username>/'
```

`access_hash` allaqachon bor va 0-bosqichda `pool.input_peer` /
`pool.input_peer_from_row` da ishlatilmoqda — v2 buni o'zgartirmaydi.

### 3.3 `retrieval_nodes` (2-bosqich)

```sql
CREATE TABLE retrieval_nodes (
  id           bigserial PRIMARY KEY,
  account_id   int  NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  chat_id      int  NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  level        smallint NOT NULL,      -- 0 msg | 1 episode | 2 day | 3 week | 4 month
  starts_at    timestamptz NOT NULL,
  ends_at      timestamptz NOT NULL,
  msg_lo       bigint NOT NULL,        -- tg_msg_id oralig'i (inclusive)
  msg_hi       bigint NOT NULL,
  msg_count    int NOT NULL DEFAULT 0,
  speakers     bigint[] NOT NULL DEFAULT '{}',
  text         text NOT NULL DEFAULT '',
  ctx_prefix   text NOT NULL DEFAULT '',
  content_hash bytea NOT NULL,         -- sha1(normalized text)
  simhash      bigint NOT NULL,        -- 64-bit, 3-char shingle
  structured   jsonb,
  tsv          tsvector GENERATED ALWAYS AS
                 (to_tsvector('simple', uz_fold(text))) STORED,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (chat_id, level, msg_lo, msg_hi)
);
CREATE INDEX ix_nodes_chat_level_time ON retrieval_nodes (chat_id, level, starts_at);
CREATE INDEX ix_nodes_tsv ON retrieval_nodes USING gin (tsv);
CREATE INDEX ix_nodes_simhash ON retrieval_nodes (simhash);

CREATE TABLE retrieval_node_embeddings (
  node_id bigint PRIMARY KEY REFERENCES retrieval_nodes(id) ON DELETE CASCADE,
  model   varchar(64) NOT NULL,
  vector  halfvec(768) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_node_emb_hnsw ON retrieval_node_embeddings
  USING hnsw (vector halfvec_cosine_ops);
```

**Level 0 ko'chirilmaydi.** Xabar darajasidagi vektorlar mavjud
`message_embeddings` da qoladi; level 0 tugunlari faqat mantiqiy ko'rsatkich —
dublikat 4 million qator yaratilmaydi. `message_embeddings.vector` ni
`vector(768)` → `halfvec(768)` ga o'tkazish va `model` ustunini indeksga
kiritish ham 2-bosqichda (xotira ~2×, aniqlik yo'qotishi o'lchovda sezilmadi).

### 3.4 `chat_digests` kengaytmasi (1-bosqich)

```sql
ALTER TABLE chat_digests
  ADD COLUMN level           smallint NOT NULL DEFAULT 1,  -- 1 kun | 2 hafta | 3 briefing
  ADD COLUMN period_start    timestamptz,
  ADD COLUMN src_fingerprint bytea,      -- md5 of "id:md5(text)" ro'yxati
  ADD COLUMN structured      jsonb,
  ADD COLUMN faithfulness    real,
  ADD COLUMN node_id         bigint REFERENCES retrieval_nodes(id) ON DELETE SET NULL;
```

`src_fingerprint` mavjud `msg_count` invalidatsiyasining o'rnini bosadi:
hozirgi mantiq (`digests.digest_for_day`, `cached.msg_count == len(msgs)`)
**tahrir va o'chirishni sezmaydi** — soni o'zgarmaydi. Fingerprint ikkalasini
ham qamrab oladi.

### 3.5 `node_entities` (3-bosqich)

```sql
CREATE TABLE node_entities (
  node_id bigint NOT NULL REFERENCES retrieval_nodes(id) ON DELETE CASCADE,
  entity  text   NOT NULL,
  kind    varchar(16) NOT NULL,   -- username | link | hashtag | domain
  PRIMARY KEY (node_id, entity, kind)
);
```

Regex bilan chiqariladi (`@username`, `t.me/...`, `#hashtag`, URL domeni) —
LLM chaqirilmaydi.

## 4. Ingest qatlami

`services/ingestion.py` o'zgarishsiz qoladi: faqat o'qish, `WAIT_BETWEEN_BATCHES`,
`limits_for` ramp-up, `FloodWait → SyncPaused → defer_by`. Qo'shiladigani —
DB'ga yozishdan oldingi **normalizatsiya**.

**Unicode sanitizatsiya** (`services/textnorm.py::sanitize`), ingest vaqtida
bir marta:

* NFC normalizatsiya;
* zero-width olib tashlash: `U+200B`–`U+200F`, `U+FEFF`;
* bidi boshqaruvi: `U+202A`–`U+202E`, `U+2066`–`U+2069`;
* Tags block: `U+E0000`–`U+E007F` (ko'rinmas prompt injection tashuvchisi);
* C0 boshqaruv belgilari, `\n` va `\t` dan tashqari.

`messages.text` ga **sanitizatsiya qilingan** matn yoziladi; xom matn
saqlanmaydi — media bilan bir xil prinsip ("kerak bo'lmagan narsa saqlanmaydi").
Bu `services/untrusted.py::neutralize` ni almashtirmaydi, balki uni
qo'llab-quvvatlaydi: `untrusted` — prompt chegarasidagi himoya, `textnorm` —
saqlash chegarasidagi.

**Til aniqlash**: `lingua-py`, ingest'da, `uz`/`ru`/`en`/`other` + `lang_conf`.
Qisqa matn (<12 belgi) → `other`. Bu retrieval'da ikkala tsvector'dan
qaysinisiga ko'proq ishonishni va evalda til kesimlarini beradi.

`lingua`'da **o'zbek modeli yo'q** — u faqat ru/en'ni tasdiqlaydi (low-accuracy
rejimda). Shu sabab `uz` alohida model bilan emas, script (Kirill/Lotin) +
marker stop-so'zlar heuristikasi bilan aniqlanadi (`ingestion.detect_lang`):
Kirill ustun bo'lsa va `lingua` rus deb tasdiqlasa → `ru`, aks holda noaniq;
Lotin ustun bo'lsa va `lingua` ingliz deb tasdiqlasa → `en`, aks holda `uz`
lotin deb qabul qilinadi. `lang_conf` shuning uchun **filtr emas, og'irlik** —
retrieval past ishonchli qatorni chiqarib tashlamaydi, faqat kamroq vazn beradi.

**Yangi maydonlar**: `fwd_from_peer_id` / `fwd_from_msg_id` / `fwd_date`
(dedup va "bu qayerdan kelgan?" savoli uchun), `top_msg_id` (forum topic),
`deleted_at` / `edited_at` (2-bosqich updates listener to'ldiradi).

**`chats.link_prefix`** sync vaqtida hisoblanadi — hozir har sitata uchun
`t.me` havolasi runtime'da yig'ilmoqda (`tools._line`), bu esa peer turini
qayta tekshirishni talab qiladi.

**Telethon `>=1.44`** (layer 227+) — `channels.SearchPosts(query=)` va
`CheckSearchPostsFlood` uchun shart.

## 5. Indeks qatlami

### 5.1 `uz_fold` — ikki tilda bir xil

Muammo: o'zbek tili uchun Postgres lug'ati yo'q, matn lotin va kirillda
aralash, apostrof uch xil kodlanadi (`'`, `ʻ`, `’`). Yechim — **fold funksiyasi**:

1. NFC → 2. lower → 3. apostrof sinfini bitta belgiga → 4. uz-kirill → lotin
transliteratsiya → 5. apostrofni butunlay o'chirish (`o'zbek` → `ozbek`).

Ikki joyda **aynan bir xil** amalga oshiriladi:
`services/textnorm.py::uz_fold` (Python) va `uz_fold(text)` — Postgres
`IMMUTABLE` funksiyasi (generated column uchun `IMMUTABLE` shart).
`tests/test_textnorm.py` ikkalasini ~500 ta namunada solishtiradi va CI'da
Postgres service bilan yuradi — drift jimgina indeksni buzmasin.

Stop-so'zlar `services/stopwords.py` ga ko'chiriladi: Zenodo o'zbek ro'yxatidan
tanlangan ~200 ta + rus + ingliz. Hozirgi `search._STOP_UZ_EN` / `_STOP_RU`
(~90 so'z, qo'lda) o'rnini bosadi.

### 5.2 Epizod segmentatsiyasi — LLM'siz

`tg_msg_id` tartibida yurish; yangi epizod boshlanadi, agar:

* oldingi xabardan **gap > 45 daqiqa** (kanal uchun 6 soat), yoki
* joriy epizodda **≥40 xabar**, yoki
* **≥6000 belgi**.

Qat'iy istisno: **reply-zanjir hech qachon bo'linmaydi** — `reply_to_msg_id`
bo'yicha union-find, bitta komponent bitta epizodda qoladi. Maqsad: median
8–15 xabar. Epizod matni `digests._line` formatida (`[HH:MM] #id who: text`) —
digest va retrieval bitta formatdan foydalanadi, ikkita render yo'lini
saqlashga hojat yo'q.

### 5.3 Kontekstli embedding

Xabar "ha, roziman" o'zicha embed qilinsa hech nimaga mos kelmaydi. Shuning
uchun har tugun **metadatadan bepul** prefiks oladi:

```
[chat: Ish guruhi (supergroup) | 2026-03-14 | speakers: Nurbek, Aziz |
 replying to Nurbek: "to'lovni 20-sanaga suramizmi"]
```

`embed = ctx_prefix + "\n" + text`. Indekslashda `task_type=RETRIEVAL_DOCUMENT`,
so'rovda `RETRIEVAL_QUERY` (hozir ikkalasi ham default bilan yuborilmoqda —
bepul aniqlik yo'qotish). Model `gemini-embedding-001` @768; `gemini-embedding-2`
ga o'tish — alohida re-embed ishi, `model` ustuni bo'yicha bosqichma-bosqich.

### 5.4 Digest → tugun

`chat_digests` yozuvi level 2 tugun sifatida `retrieval_nodes` ga kiritiladi va
embed qilinadi. Hafta tuguni **faqat kun tugunlaridan** quriladi, oy — faqat
haftadan: xom xabarlarni qayta o'qish yo'q, xarajat chiziqli o'sadi.

### 5.5 simhash

Har tugunda 64-bitli simhash (3-belgili shingle). Forward va sitata
takrorlarini **vektorsiz** topish uchun: Hamming ≤3 → dublikat nomzod.
Bu dedup'ning ikkinchi pog'onasi (birinchisi — sha1).

## 6. Retrieval (`services/search.py`)

`hybrid_search` ikki ro'yxat (fts + vector) o'rniga **to'rt + ikki** ro'yxatga
kengayadi, hammasi `asyncio.gather` bilan parallel:

| Yo'l | Nima | Izoh |
|---|---|---|
| `tsv_uz` | uz-lat va uz-cyr variantlari, `:*` prefiks | o'zbek qo'shimchalari |
| `tsv_ru` | ru variantlari, `russian` config | stemming ishlaydi |
| trigram | `strict_word_similarity(term, text_fold) >= 0.55` | imlo xatolari |
| vektor | **faqat asl so'rovning bitta** embeddingi | har variantga embedding = qimmat |
| tugunlar | `retrieval_nodes` (episode/day) tsv + vektor | ikkinchi RRF kirishi |

Trigram bo'yicha: hozirgi `func.similarity(Message.text, t)` butun xabar
uzunligiga nisbatan o'lchaydi — uzun xabarda har doim past ball. `strict_word_similarity`
so'z chegarasida o'lchaydi; agar Postgres versiyasi qo'llamasa,
`word_similarity >= 0.7` ga tushadi.

**Og'irlikli RRF.** `search.rrf` `w/(k+rank)` ga kengaytiriladi (k=60 saqlanadi).
Sukut og'irliklar: tsv 1.0, vektor 1.0, trigram 0.6, tugunlar 0.8 — eval bilan
sozlanadi.

**Ko'p tilli so'rov kengaytirish.** Bitta `Task.ROUTE` chaqiruvi, JSON:
`{uz_lat, uz_cyr, ru, en, synonyms, slang}`. Kesh: Redis, kalit
`md5(uz_fold(query))`, TTL 24 soat. Planner **untrusted matn ko'rmaydi** — faqat
foydalanuvchi savolini.

**`services/rerank.py`**: `rerank(query, candidates) -> list[Hit]`. Default —
LLM listwise (`Task.ROUTE`), dedup'dan keyingi top ≤60 nomzod, har biri uchun
faqat `ctx_prefix` + 300 belgi. `RERANK_PROVIDER=vertex|cohere` — keyinroq,
`config.py` orqali. **Dedup rerank'dan oldin majburiy**: aks holda model
o'nta bir xil forwardni tartiblaydi.

**`services/cluster.py`** (numpy/sklearn), to'rt pog'ona:

1. `sha1(normalized)` — aynan bir xil;
2. simhash Hamming ≤3;
3. cosine ≥0.93 — near-dup, `also_seen_in` ro'yxati bilan saqlanadi
   ("bu xabar 4 chatda forward bo'lgan" — javobda qimmatli ma'lumot);
4. agglomerative, average-linkage, cosine distance 0.20 — mavzu klasterlari.

Klaster yorliqlari: **bitta** Flash-Lite chaqiruvi, ≤8 so'z, imperativ naqshlar
strip qilinadi, kirish `<untrusted_data kind="labels">` konvertida.

`select_context` **yagona kontekst tanlash nuqtasi bo'lib qoladi** — barcha
yangi funksiyalar shu orqali o'tadi, "promptga N xabar tashlash"ning yangi
yo'llari yaratilmaydi.

## 7. Deep search loop

Alohida modul: **`services/deep_search.py`**. `services/agent.py` o'z holicha
qoladi — u interaktiv ReAct sikli (tool'lar, `MAX_CALLS_PER_TOOL`, `seen_calls`),
deep search esa qat'iy bosqichli quvur. Ikkalasini bitta funksiyaga
tiqishtirish ikkalasini ham buzadi.

```python
MAX_ROUNDS = 2
DEEP_RESULT_TOKENS = 20_000
SOFT_WALL_S, HARD_WALL_S = 60, 120

async def deep_search(session, *, account_id, question, on_event) -> DeepResult:
    plan = await plan_query(question)          # Task.ROUTE, JSON, untrusted YO'Q
    seen, clusters = set(), []
    for rnd in range(MAX_ROUNDS):
        lists = await asyncio.gather(*[
            local_four_way(session, sq, plan) for sq in plan.subqueries
        ], *(telegram_rungs(plan) if plan.needs_server else []))
        fused   = rrf(*lists, k=60)                       # og'irlikli
        deduped = cluster.dedup(fused)                    # sha1 → simhash → cosine
        clusters = cluster.topics(deduped)
        ranked  = await rerank(question, deduped[:60])    # Task.ROUTE
        if rnd + 1 == MAX_ROUNDS or elapsed() > SOFT_WALL_S:
            break
        refl = await reflect(plan, labels=[c.label for c in clusters],
                             coverage=coverage_stats())   # matn KO'RMAYDI
        if not refl.need_more or marginal_yield(clusters, seen) < 0.10:
            break
        plan = plan.extend(refl.subqueries)
    answer = await synthesize(question, ranked, clusters) # Task.DEEP, TOOLSIZ
    return DeepResult(answer, clusters, coverage_report())
```

**Byudjetlar kodda**: `MAX_ROUNDS=2`, `DEEP_RESULT_TOKENS=20_000`, 60 s soft /
120 s hard wall clock. Marginal-yield qoidasi: yangi raund ≥10% yangi klaster
hiti bermasa — to'xtash. Bu `MAX_CALLS_PER_TOOL` bilan bir mantiq: model
qoniqmasa cheksiz qayta so'raydi.

**Qamrov hisoboti — kodda**, LLM'da emas:

* `chats_searched / chats_total`, `messages_scanned`;
* qamrab olingan vaqt oralig'i;
* `synced_min_id != 1` bo'lgan chatlar → **"ma'lum bo'shliq"** deb alohida
  ro'yxatlanadi;
* Telegram server rung'lari ishlatildimi, qaysilari.

"Hamma xabarlar ko'rildi" iborasi hech qachon chiqmaydi.

**SSE `/api/deep/stream`** eventlari: `plan`, `search`, `cluster`, `rerank`,
`token`, `coverage`, `done`. Klaster yorliqlari javob matnidan **oldin**
ko'rsatiladi — foydalanuvchi 20 soniya bo'sh ekranga qaramaydi. Sitata id
fazosi yagona (`#msg_id` + chat). Narx `done` eventida; kunlik byudjet
(`services/limits.py`) so'rovdan **oldin** tekshiriladi.

**Xavfsizlik.** Planner va reflect untrusted matnni ko'rmaydi (dual-LLM
naqshi); sintezator ko'radi, lekin unda **tool yo'q** — u faqat matn qaytara
oladi. `taint` bayrog'i bosqich chegaralarida tekshiriladi: untrusted deb
belgilangan qiymat planner promptiga o'tolmaydi (test bilan qo'riqlanadi).

## 8. Telegram qamrovi

`allowlist.READ` ga **11 ta** metod qo'shiladi (hech biri destruktiv emas,
`DENIED` ga tegilmaydi — u faqat o'sadi):

| Burchak | TL metod | Allowlist | Bucket |
|---|---|---|---|
| Qaysi kunlarda mavzu bor | `messages.GetSearchResultsCalendar` (Telethon 1.45'da `q` parametri yo'q — faqat `filter`; `tools._calendar_supports_query` imzodan tekshiradi) | READ | search |
| Natija pozitsiyalari | `messages.GetSearchResultsPositions` | READ | search |
| Saved Messages dialoglari | `messages.GetSavedDialogs` | READ | history |
| Saved Messages tarixi | `messages.GetSavedHistory` | READ | history |
| O'qilmagan mention | `messages.GetUnreadMentions` | READ | history |
| O'qilmagan reaksiya | `messages.GetUnreadReactions` | READ | history |
| Forum topiclari | `messages.GetForumTopics` (`channels.*` allowlist'da alias sifatida qoladi) | READ | history |
| Topic ID bo'yicha | `messages.GetForumTopicsByID` (`channels.*` alias) | READ | history |
| O'xshash kanallar | `channels.GetChannelRecommendations` | READ | discover |
| Global post qidiruvi kvotasi | `channels.CheckSearchPostsFlood` | READ | discover |
| Post qayerga forward bo'lgan | `stats.GetMessagePublicForwards` | READ | search |
| *(alohida)* global post qidiruvi | `channels.SearchPosts` | READ ⚠️ | discover |

⚠️ **`channels.SearchPosts` — user ko'rib chiqishi kerak** (13-bo'lim). Tool
`allow_paid_stars` ni **hech qachon** yubormaydi; avval `CheckSearchPostsFlood`
chaqiriladi, `query=` faqat `remains > 0` yoki `query_is_free` bo'lganda
ishlatiladi, aks holda hashtag rejimiga yoki lokal qidiruvga qaytadi. Stars
sarflash — **moliyaviy amal**, `payments.SendStarsFormRequest` bilan bir sinf
(u `DENIED` da).

### 8.1 Token bucket'lar

Bitta `TG_SEARCH_RATE_PER_MINUTE` (hozir `tools._tg_rpc_guard` orqali
hamma serverga boradigan tool bitta kvotani bo'lishadi) **to'rtta sinfga**
bo'linadi — akkaunt bo'yicha, ±20% jitter bilan:

| Bucket | Tezlik | Burst | Kunlik | Kimlar |
|---|---|---|---|---|
| `history` | 40/min | 8 | — | GetHistory, GetSavedHistory, GetForumTopics, mentions |
| `search` | 12/min | 4 | — | Search, SearchGlobal, Calendar, Positions, PublicForwards |
| `resolve` | 4/min | 2 | 60 | ResolveUsername (**cache-first**, faqat miss'da) |
| `discover` | 3/min | 1 | 30 | ChannelRecommendations, SearchPosts, CheckSearchPostsFlood |

Bo'sh bucket → modelga matn: `rate_limited:<bucket>` + lokalga qaytish
ko'rsatmasi. `FloodWaitError` **retry qilinmaydi** — hozirgidek matn bo'lib
qaytadi (4.2-band: retry-storm yo'q).

### 8.2 Tool o'zgarishlari (`services/tools.py`)

`search_telegram` yangi parametrlar: `from_user`, `topic_id` (→ `top_msg_id`),
`filter` (`url|pinned|mentions|poll|photo|doc`), `since` / `until`.
`search_global`: `scope` (`channels|groups|users`), `archived` folder.

Yangi tool'lar: `search_calendar`, `list_forum_topics`, `discover_channels`,
`saved_messages_search`. Har biri `AGENT_SYSTEM_PROMPT` da ham tavsiflanadi —
`test_agent.py::test_prompts_are_stable_and_safe` registry'dagi har bir nom
promptda bo'lishini talab qiladi.

**Planner narvoni**: har savol **lokaldan** boshlanadi; server rung'lari faqat
lokal miss bo'lganda. Telefon raqami hech qayerda qaytarilmaydi — mavjud
qoida yangi tool'larga ham taalluqli.

### 8.3 Real-time updates listener (2-bosqich)

Faqat **worker** jarayonida (api'da emas — bitta sessiya, `AuthKeyDuplicated`
riski): `NewMessage` → upsert, `MessageEdited` → `edited_at` + qayta
fingerprint, `MessageDeleted` → `deleted_at`. Uzilishdan keyin `catch_up()`.
10 daqiqalik incremental cron zaxira sifatida qoladi.

## 9. Digest v2 (`services/digests.py`)

**L1 — kun.** Mavzu bloklariga segmentlash (reply-zanjir + gap, 5.2-band
bilan bir xil mantiq); dedup — takror forwardlar `(×N)` ko'rinishida, manba
`<untrusted_data kind="forward">` konvertida; map — Flash-Lite; chiqish —
`structured` JSON: `entities`, `decisions`, `action_items`, `open_questions`,
`links`, `notable[{msg_id, why}]`. **Chain-of-Density 2 raund** faqat
foydalanuvchi ko'radigan yakuniy digestda (oraliq map'da emas — qimmat).
Digest **manba tilida** yoziladi.

**L2 — hafta** (dushanba), faqat L1 tugunlaridan.

**L3 — shaxsiy briefing** (kunlik, kross-chat). Chat salience:
`Δmsg_count` joriy 28 kun medianasiga nisbatan, egasi mention qilinganmi,
ochiq savollar bormi. Top 3 chat to'liq, qolgani bir qatordan. Briefing
**UI tilida**, sitatalar **asl tilda**.

**Mexanik nazorat** (LLM-judge emas, kod):

* har bullet'da ≥1 `#msg_id` bo'lishi shart, regex bilan tekshiriladi;
* har id oynadagi id'lar to'plamida bo'lishi shart — aks holda bullet
  tashlanadi ("hallyutsinatsiya qilingan sitata" mumkin emas);
* qo'shtirnoq ichidagi matn manbada **substring** bo'lishi shart.

Ustiga 5–10% namunaviy **faithfulness auditor** (manbani ko'radi, Flash-Lite) →
`chat_digests.faithfulness`.

**Yetkazish** (3-bosqich): bot push, HTML `parse_mode`, ≤4096 belgi — TL;DR +
top 3, qolgani Mini App'ga chuqur havola. `accounts.tz`, `quiet_from` /
`quiet_to` hisobga olinadi.

**Batch API** embedding va digest uchun — ~50% arzon, kechiktirish muhim emas.

## 10. Baholash (`evals/`)

`evals/build_golden.py`:

1. 300–500 xabar **stratifikatsiya** bilan (chat, `lang`, uzunlik, yangilik);
2. Flash bilan savol generatsiyasi — **tillar ataylab aralashtiriladi**
   (ruscha savol o'zbekcha xabarga va aksincha);
3. dedup;
4. permissiv union top-50 filtri (nomzod javoblar);
5. **inson tekshiruvi** → 60–80 ta yakuniy savol;
6. `evals/golden.jsonl` — DB'da ham, git'da ham saqlanmaydi (`.gitignore`);
   real chat kontenti repoga tushmasligi shart.

Metrikalar: **Recall@20 (asosiy)**, nDCG@10, MRR@10, cluster-recall@k, sitata
aniqligi (judge sitatani ko'radi), qamrov da'vosi aniqligi (kodda tekshiriladi).
Kesimlar: `uz-lat`, `uz-cyr`, `ru-cyr`, `ru-translit`, `en`.

**≥3 yugurish, median + tarqalish** — `CLAUDE.md` dagi qoida
("bitta yugurish hech narsani isbotlamaydi", `scratchpad/bench.py` uslubi).

Real akkaunt ulangunicha: unit testlar + docker compose Postgres (CI'da service
sifatida) — `uz_fold` pariteti, RRF determinizmi, dedup, qamrov hisoboti.

### 10.1 Aniq buyruqlar

```bash
# 1) Golden to'plam qurish (real akkaunt + LLM kerak — DB'ga va providerga ulanadi)
python -m evals.build_golden --account-id 1 --n 400 --out evals/golden.jsonl --seed 1
# qo'lda yozilgan ko'p-bosqichli savollarni birlashtirish:
python -m evals.build_golden --account-id 1 --n 400 --out evals/golden.jsonl --hard evals/hard.jsonl

# 2) Baholash — bitta konfiguratsiya, kamida 3 yugurish (CLAUDE.md qoidasi)
python -m evals.run_eval --golden evals/golden.jsonl --config baseline --runs 3 \
    --out evals/runs/2026-09-15.json

# hammasini bitta chaqiruvda taqqoslash (baseline / expand / expand+rerank / deep)
python -m evals.run_eval --golden evals/golden.jsonl \
    --config baseline expand expand+rerank deep --runs 3

# yoki Makefile orqali:
make eval-golden account=1 args="--n 400 --seed 1"
make eval-run args="--golden evals/golden.jsonl --config baseline expand --runs 3"
```

`--config deep` `services/deep_search.py` mavjud bo'lmasa (2-bosqich hali
tugallanmagan bo'lsa) xatosiz o'tkazib yuboriladi — natijada shu
konfiguratsiya uchun `"skipped": true` + izoh. `evals/golden.jsonl` va
`evals/runs/` `.gitignore`'da — bu buyruqlarning natijasi hech qachon
commit qilinmaydi.

## 11. Narx

| Nima | Taxmin |
|---|---|
| Deep so'rov | ≈ **$0.05** (>85% — sintez, `Task.DEEP`) |
| Oddiy so'rov | ≈ **$0.005** |
| Indeks qurish (bir martalik) | ≈ **$25** |

Deep rejim UI'da **aniq tanlanadi** — tasodifan 10× qimmat so'rov ketmaydi.
Follow-up savollar uchun Gemini **context caching**. Flash-Lite **2026-10-16**
da retire bo'ladi — model nomi `config.py` orqali beriladi, kodda qattiq
yozilmaydi. Yangi model nomi `llm/pricing.py` ga **qaytgan** nom bilan
qo'shiladi, aks holda `estimate_cost` `None` qaytaradi va kunlik xarajat limiti
jimgina ishlamay qoladi.

## 12. Roadmap

| Bosqich | Fayllar | Testlar |
|---|---|---|
| **0 — joriy** | 5 ta HIGH xato: envelope escape, `input_peer` access_hash, duplicate row, confirm race, strict `zip` | `test_phase0.py`, `test_write_actions.py` |
| **1 — Retrieval poydevori** (akkauntsiz testlanadi) | migratsiya (ustunlar, generated columns, GIN, `lang`, fwd, `top_msg_id`, `deleted_at`/`edited_at`, `link_prefix`, `chat_digests` kengaytmasi); `services/textnorm.py` + SQL `uz_fold`; sanitizatsiya `ingestion.py` da; `services/stopwords.py`; trigram fix + `task_type` (`search.py`); so'rov kengaytirish + og'irlikli RRF; `services/rerank.py`; `services/cluster.py`; Telethon 1.44; `allowlist.py` +12; token bucket'lar (`tools.py`); `search_telegram` parametrlari + 4 yangi tool + `prompts.py` | `test_textnorm.py` (SQL/Python paritet), `test_search_quality.py`, `test_allowlist.py`, `test_tg_search.py`, CI'ga Postgres service + append-only trigger testi |
| **2 — Ierarxik indeks + deep search** ✅ *(qolgani pastda)* | ✅ `retrieval_nodes` / epizodlar / `halfvec` (migratsiya `0009`); ✅ kontekstli embedding; ✅ digest embedding + L2/L3 (`services/nodes.py`); ✅ `services/deep_search.py` (plan → fan-out → cluster → rerank → reflect → synth); ✅ SSE `/api/deep/stream` + UI (klaster yorliqlari, qamrov paneli); ✅ updates listener (`worker/updates.py`, `TG_UPDATES_LISTENER`); ✅ eval harness (`evals/`, `test_evals.py`) | ✅ `test_episodes.py`, `test_nodes.py`, `test_deep_search.py` (taint chegarasi), `test_updates.py`, `test_evals.py` |
| **3 — Qamrov + mahsulot** | ⬜ `discover_channels` / `SearchPosts` / forum tool'lar; ⬜ digest push + quiet hours; ⬜ alertlar; ⬜ `node_entities`; ⬜ golden set (real akkaunt bilan to'ldirish); ⬜ Batch API; ⬜ `gemini-embedding-2` re-embed | `evals/` harness, `test_digest_push.py` |

**2-bosqichdan qolgani** (3-bosqichga o'tdi, chunki server burchaklariga
bog'liq): deep search ichida **server pog'onalari** — `search_telegram` /
`search_global` ni fan-out'ga qo'shish hali yo'q (hozir faqat lokal DB
kanallari: `hybrid_search_multi` + `node_search`). Qamrov hisobotida
"sinxronlanmagan tarix" `gaps` bo'lib ko'rinadi, lekin avtomatik
to'ldirilmaydi. `test_coverage_report.py` alohida fayl sifatida yozilmadi —
qamrov matematikasi `test_deep_search.py` ichida qo'riqlanadi.

Har bosqich o'z testlari bilan tugaydi; CI invariant qadami
(`test_allowlist.py`, `test_peer_guard.py`, `test_write_actions.py`,
delete-grep) **yumshatilmaydi**.

## 13. Ochiq savollar / user qarori kerak

1. **`channels.SearchPosts` va Stars.** Metod allowlist'ga qo'shiladi, lekin
   pullik yo'li kodda butunlay yopiq (`allow_paid_stars` yuborilmaydi).
   Savol: umuman qo'shilsinmi? Stars sarflash — moliyaviy amal, `payments.*`
   `DENIED` bilan bir sinf. Agar user "yo'q" desa, `discover_channels` faqat
   `GetChannelRecommendations` ustida ishlaydi va hashtag rejimi qoladi.
   **Bu band tasdiqlanmaguncha 3-bosqichda bloklangan deb hisoblanadi.**
2. **Rerank provideri.** `RERANK_PROVIDER=vertex|cohere` — tashqi servisga
   nomzod matnlar (ya'ni **chat kontenti**) yuboriladi. Bu 3-invariantni
   buzmaydi (envelope o'z joyida), lekin yangi ma'lumot chiqish kanali ochadi.
   Default LLM listwise bilan qolamizmi?
3. **`gemini-embedding-2` migratsiyasi.** Re-embed = to'liq indeksni qayta
   hisoblash (≈$25 + vaqt). Qachon: `embedding-001` deprecate e'lon
   qilinganda avtomatmi yoki qo'lda? Ikki model yonma-yon turadimi
   (`model` ustuni buni qo'llaydi, lekin HNSW indeks ikkilanadi)?
4. **Quiet hours va `accounts.tz`.** Hozir `auto_reply_rules` da
   `quiet_from`/`quiet_to` **UTC soat** sifatida saqlanadi. Digest push uchun
   akkaunt vaqt zonasi kerak. `accounts.tz` qo'shilsa, mavjud auto-reply
   qoidalari ham unga o'tadimi (migratsiya kerak) yoki UTC'da qoladimi?

### Kodda sezilgan ziddiyatlar (qaror o'zgartirilmadi, faqat qayd)

* **Migratsiya raqami.** Qarorlarda "migratsiya 0007" deyilgan, lekin
  `migrations/versions/0007_agent_run_error.py` allaqachon mavjud
  (0-bosqich, `agent_runs.error`, `down_revision = "0006"`). 1-bosqich
  migratsiyasi **`0008`** bo'lishi kerak.
* **Digest invalidatsiyasi.** `CLAUDE.md` "Digest keshi `msg_count` bo'yicha
  invalidatsiya" deydi; v2 `src_fingerprint` ni joriy qiladi. Ikkalasi bir
  vaqtda ishlaydi (fingerprint `NULL` bo'lsa eski yo'l), lekin `CLAUDE.md`
  1-bosqichda yangilanishi kerak.
* **`edited_at` takrori.** `messages.edited_at` allaqachon bor
  (`db/models.py`). Qarordagi "yangi ustun `edited_at`" — amalda mavjud
  ustunni updates listener bilan **to'ldirish**; yuqorida `edited_at_seen`
  deb ajratildi, agar bu keraksiz bo'lsa 1-bosqichda olib tashlanadi.
* **Bitta rate limiter.** `tools._tg_rpc_guard` hozir barcha serverga
  boradigan tool'lar uchun **bitta** `RateLimiter` ishlatadi; to'rt bucket'ga
  bo'lish `tools.py` dagi ~8 ta chaqiruv nuqtasini o'zgartirishni talab
  qiladi — 1-bosqichda bir marta, keyin barqaror.
* **`search.rrf` imzosi.** Hozir `rrf(*ranked, k=60)` og'irliksiz; og'irlikli
  versiya imzoni o'zgartiradi (`rrf(*ranked, weights=..., k=60)`) —
  `hybrid_search` va `test_search_quality.py` mos ravishda yangilanadi.
