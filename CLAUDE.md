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

## Deep search (v2, 2-bosqich)

`services/deep_search.py` — `services/agent.py` dan **boshqa narsa**: model
tool tanlamaydi, bosqichlar kodda qattiq turadi:
PLAN → fan-out → dedup/cluster → rerank → (≤1 qo'shimcha raund) reflect →
synth → qamrov. Web: `POST /api/conversations/{id}/messages` `mode="deep"`
(oqimsiz,
`chat_service.send_message`) yoki `POST /api/deep/stream` (SSE,
`web/routers/deep.py`).

- **Dual-LLM taint guard.** `plan` va `reflect` prompt'lariga xabar matni
  **hech qachon** tushmaydi — `_Taint.assert_clean()` har chaqiruvdan oldin
  tekshiradi, buzilsa `DeepSearchError("taint")`. Yangi bosqich qo'shsangiz
  uni ham shu qo'riqchidan o'tkazing.
- **Bitta konvert.** Dalillar faqat bitta `<untrusted_data kind="evidence">`
  ichida, har satr `untrusted.neutralize` dan o'tgan. Ikkinchi konvert
  qo'shmang.
- **Qamrov (`Coverage`) kodda hisoblanadi**, modelga yozdirilmaydi. `gaps` =
  `synced_min_id != 1` chatlar; `server_rungs_used` / `server_calls` /
  `server_hits` / `server_skipped` / `external_hits` — server pog'onalarining
  **rost** hisobi. "Hamma xabarlarni ko'rdim" degan gap chiqmasin.
- **Yagona yozuvchi — `deep_search._persist`.** User turn + assistant turn +
  `agent_runs` bitta joyda yoziladi; boshqa joydan `conversation_messages`
  ga deep natijasini yozmang. Rejim `context.mode = "deep"` da.
- **Byudjetlar konstanta** (`MAX_ROUNDS=2`, `MAX_CANDIDATES=400`,
  `PER_SUBQUERY_LIMIT=40`, `DEEP_RESULT_TOKENS=20_000`, `SOFT_WALL_S=60`,
  `HARD_WALL_S=120`) — `MAX_CALLS_PER_TOOL` bilan bir mantiq: xavfsizlik
  qopqog'i. Wall clock'ni `.env` (`DEEP_SOFT_WALL_S`/`DEEP_HARD_WALL_S`)
  bekor qiladi, qolganini emas.
- `DEEP_RATE_PER_MINUTE` (default 3, 0 = deep butunlay o'chiq) —
  `limits.check_deep_rate`, foydalanuvchi bo'yicha. Kunlik byudjet
  tekshiruvi har doim **birinchi** LLM chaqirig'idan oldin.
- Fan-out'da har subquery **o'z sessiyasi** bilan ishlaydi (bitta
  `AsyncSession` parallelga xavfsiz emas). Reja oynasi (`since`/`until`)
  `hybrid_search_multi` ga ham, `node_search` ga ham uzatiladi.

### Server pog'onalari (3-bosqich)

Xom MTProto qidiruv mantiqi `services/tg_search.py` da — konvertsiz, tipli
(`ServerHits` / `ServerHit`). `tools.search_telegram` / `search_global` /
`search_calendar` / `saved_messages_search` / `discover_channels` shuning
ustidagi **yupqa o'ramchi**; ikkinchi nusxa yozmang. Shift bitta joyda:
`tools.bucket_guard` → `tools._buckets` (tool yo'li ham, deep search yo'li ham
bitta akkaunt kvotasini bo'lishadi).

- Chiqish sharti **kodda** (`deep_search.needs_server`), modelda emas:
  (a) bo'shliqli chat, (b) lokal hit `DEEP_SERVER_MIN_LOCAL_HITS` (8) dan kam,
  (c) `lookup`/`person` oynasi eng eski sinxronlangan xabardan oldin,
  (d) planner/reflect `server_scope` ni `global`/`discover` qilgan.
  `server_scope="none"` hammasini bekor qiladi.
- Pog'onalar tartibi: bo'shliqli chatlar (`messages.Search`, top 2 subquery) →
  global (`messages.SearchGlobal`) → `discover` (`channels.SearchPosts`) —
  oxirgisi **faqat** `server_scope=="discover"` da. Jami
  `DEEP_SERVER_MAX_CALLS` (6) chaqiruv.
- To'lgan bucket — xato emas: pog'ona **o'tkazib yuboriladi** va
  `coverage.server_skipped` ga yoziladi. Kutish yo'q, retry yo'q.
- Server hitlari alohida RRF ro'yxati (og'irlik 0.9). Lokal `chats` da bor
  chatdan kelgani `ingestion.upsert_messages` orqali saqlanadi (sitata lokal
  id fazasida qolsin); qolgani **saqlanmaydi**, manfiy sintetik id oladi va
  dalilda `[server external]` deb belgilanadi — u lokal indeks bilan
  solishtirilmagan.

### `retrieval_nodes` (`services/nodes.py`, `services/episodes.py`)

- Darajalar: **0** xom xabar · **1** epizod · **2** kun · **3** hafta.
  Sitata **har doim level 0** ga tushadi: `NodeHit` da `message_id` yo'q,
  shuning uchun tugun topilmalari `search.node_hits_to_messages` orqali
  xabarlarga qaytariladi va shundan keyingina `rrf` ga kiradi.
- **Hafta tuguni faqat kun tugunlaridan** quriladi (`build_week_nodes`) —
  xom xabardan emas. Bu yagona LLM ko'radigan joy va u begona matnni
  `<untrusted_data>` konvertida oladi.
- **Epizod segmentatsiyasi deterministik** — LLM'siz (gap/limit chegaralari +
  reply-zanjir union-find). Bir xil kirish → bir xil natija; bu yerga model
  qo'shmang.
- `TG_UPDATES_LISTENER` (`worker/updates.py`) — **faqat worker**, default
  `false`. `api` da yoqilmasin: bitta akkauntga ikkita listener = ortiqcha
  MTProto oqimi.

### `evals/` — o'lchov

```bash
make eval-golden account=1 args="--n 400 --seed 1"
make eval-run args="--golden evals/golden.jsonl --config baseline expand deep --runs 3"
```

Real DB/LLM'ga ulanadi, `pytest`/`ci` ularni yugurtirmaydi (`tests/test_evals.py`
faqat sof funksiyalarni tekshiradi). **Kamida 3 yugurish** — LLM tebranadi,
bitta yugurish hech narsani isbotlamaydi (narx qopqoqlari bilan bir qoida).
`evals/golden.jsonl` va `evals/runs/` commit qilinmaydi.

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

### Narx qopqoqlari — o'lchov bilan qo'yilgan

Uchtasi ham o'lchovdan keyin qo'shilgan, taxmindan emas. Olib tashlashdan oldin
`scratchpad/bench.py` uslubida qayta o'lchang (LLM tebranadi — **kamida 3 marta**
yugurtiring, bitta yugurish hech narsani isbotlamaydi).

- **Chat indeksi `runtime_note` ichida** (`tools.chat_index(compact=True)`).
  Usiz agent deyarli har so'rovda `list_chats` chaqirardi — bitta ortiqcha LLM
  aylanishi ≈ 4.3k token. Indeksning o'zi ≈ 350 token. System'ga **qo'ymang**:
  kesh prefiksi buziladi (`cached=4224` shu tufayli ishlaydi).
- **`MAX_CALLS_PER_TOOL = 3`** — model qoniqmasa `search_messages` ni boshqa
  so'zlar bilan 4 martagacha takrorlaydi (o'lchovda 25.8k token). Prompt'dagi
  "adjust once" buni to'xtatmadi.
- **`seen_calls`** — aynan bir xil `(tool, args)` ikkinchi marta bajarilmaydi;
  natija o'zgarmaydi, DB/MTProto ishi va token esa behuda ketadi.

To'silgan chaqiruv ham `calls_log` ga tushadi (nima so'ralganini ko'rish uchun) —
loglardagi tool soni **bajarilgan** soni bilan bir xil emas.
- Digest keshi (`chat_digests`) `src_fingerprint` bo'yicha invalidatsiya
  (`md5("<tg_msg_id>:<md5(text)>,…")`, SQL'da hisoblanadi) — qo'shilish,
  **tahrir va o'chirish** ham qamraladi; `msg_count` zaxira yo'l sifatida
  faqat fingerprint `NULL` bo'lgan eski qatorlar uchun qoladi.
- Yangi tool qo'shsangiz uni `AGENT_SYSTEM_PROMPT` da ham tavsiflang —
  `test_agent.py::test_prompts_are_stable_and_safe` buni talab qiladi
  (registry'dagi har bir nom promptda bo'lishi shart).

### Qidiruvning ikki yo'li

- `search_messages` — **lokal DB** (FTS + pgvector, RRF). Reyting bor, ma'noni
  tushunadi, bepul. Standart yo'l.
- `search_telegram` / `search_global` — **Telegram serveri**
  (`messages.Search` / `messages.SearchGlobal`, allowlist'da READ). Sinxronlanmagan
  tarixni ham ko'radi va `search_global` chat bilmasdan qidiradi, lekin faqat
  kalit so'z bo'yicha va har chaqiruv real MTProto so'rovi.
- `TG_SEARCH_RATE_PER_MINUTE` (akkaunt bo'yicha, 0 = tool'lar o'chiq) —
  agent bir savolda bir necha marta chaqirishi mumkin, shuning uchun shift
  kerak. `FloodWaitError` **retry qilinmaydi** — modelga matn bo'lib qaytadi
  va u lokal qidiruvga o'tadi (4.2-band: retry-storm yo'q).
- Telethon entity'lari `m.sender` / `m.chat` keshidan olinadi; `get_sender()` /
  `get_chat()` chaqirmang — ular allowlist'da yo'q metodlarga borishi mumkin.

### A'zolar ro'yxati

- `list_members` — `channels.GetParticipants` (allowlist'da READ). Katta
  kanalda admin huquqi kerak, `private` chatda umuman yo'q, bir so'rovda
  maksimum 200 ta.
- **Telefon raqami hech qachon qaytarilmaydi** — natija LLM promptiga tushadi,
  raqam u yerda kerak emas. `test_list_members_never_leaks_phone` qo'riqlaydi.
  Yangi maydon qo'shsangiz shu mezon bilan o'lchang: "buni modelga berish
  shartmi?"
- Shift `_tg_rpc_guard` orqali — `search_*` bilan **bitta** `TG_SEARCH_RATE_PER_MINUTE`
  kvotasini bo'lishadi, chunki hammasi bir akkauntning MTProto kvotasini yeydi.

### Profil tool'lari va peer turi

`get_chat_info` / `get_user_info` / `get_reactions` / `get_post_comments` —
serverga boradi, shift bir xil.

- **Peer turiga qarab boshqa TL metod kerak.** DB'dagi `chats.type` ustuniga
  ishonmang — haqiqiy `InputPeer` turini tekshiring:
  `InputPeerChannel` → `channels.GetFullChannel` (kanal **va** supergruh),
  `InputPeerChat` → `messages.GetFullChat` (oddiy guruh),
  `InputPeerUser` → `users.GetFullUser`. Noto'g'rilari `TypeError` beradi
  (`Cannot cast InputPeerChat to InputChannel` — jonli sinovda tutilgan).
- **Telegram cheklovlari xato emas, holat** — modelga tushunarli matn qaytarish
  kerak, aks holda u qayta-qayta urinadi:
  kanalda `GetMessageReactionsList` → `BroadcastForbiddenError` (kim bosgani
  oshkor qilinmaydi, faqat umumiy son); bog'langan guruhsiz kanalda
  `GetReplies` → `MsgIdInvalidError`; `stats.*` → admin + minimal obunachi.
- **Telefon raqami hech qayerda qaytarilmaydi** (`list_members`, `get_user_info`) —
  testlar qo'riqlaydi.
- Ishlatilmayotgan READ metodlar hali bor (`contacts.GetContacts`,
  `messages.GetMessageReadParticipants`, `stats.*`). Ular ataylab qo'shilmagan:
  maxfiylik qiymati past, token narxi yuqori. Qo'shsangiz shu mezon bilan
  o'lchang.

## Briefing (3-bosqich/B)

- Bitta jadval, migratsiya `0010`: `briefings` (`uq_briefing_account_day` —
  bitta akkaunt + bitta **lokal** kun uchun bitta qator, cron shu kalitga
  tayanib idempotent bo'ladi) va `accounts.tz` / `quiet_from` / `quiet_to` /
  `briefing_enabled` / `briefing_hour` / `briefing_chats` ustunlari. Bu
  ustunlar **Alertlar** bo'limidagi jim soatlar bilan **bir xil** — ikkinchi
  nusxa qo'shmang.
- `services/briefing.py` — yagona nuqta. Uch qaror:
  1. **Salience kodda, sintez modelda.** Chat muhimligi — 28 kunlik mediana
     z-og'ishi + egasi tilga olingani (+2) + ochiq savol (+1 har biri).
     Model tartib tanlamaydi, faqat tanlangan `BRIEFING_TOP_CHATS` haqida
     yozadi.
  2. **Sitatalar mexanik tekshiriladi.** Har `#<id>` manba tugunlari
     matnidagi id to'plamida bo'lishi shart; bo'lmasa havola o'chiriladi,
     havolasiz qolgan qator **tashlanadi** — hallyutsinatsiya qilingan
     sitata chiqishga yetmaydi.
  3. **"Kim ko'p yozdi" kodda sanaladi**, modeldan so'ralmaydi.
- Kun tugunlari matni — begona odamlar yozgani (invariant 3): bitta
  `<untrusted_data kind="digests">` konvertida, `untrusted.neutralize`dan
  o'tgan holda. Chain-of-Density raundlari ham shu konvertdan foydalanadi
  (arzon `Task.ROUTE`); asosiy sintez bitta `Task.SEARCH` chaqiruvi.
- Yetkazish **faqat** akkaunt egasiga (`accounts.user_id → users.tg_user_id`),
  `bot/push.py::send_html` orqali — MTProto bu yo'lda umuman ishlatilmaydi
  (hech narsa Telegram akkaunt nomidan yuborilmaydi). `deliver_account`
  `Briefing.delivered_at` ni tekshiradi — qayta yuborish yo'q.
- **Yetkazish bandlash bilan** (migratsiya `0014`, `briefings.attempts`):
  push'dan **oldin** shartli `UPDATE ... WHERE delivered_at IS NULL`
  bajariladi va `RETURNING attempts` bo'sh kelsa qator boshqa tik tomonidan
  olingan — hech narsa yuborilmaydi. Xato bo'lsa bandlash bo'shatiladi;
  `BRIEFING_MAX_DELIVERY_ATTEMPTS` (3) dan keyin voz kechiladi
  (`delivered_at` qoladi, `delivery_error = "gave_up_after_N: ..."`).
  Tekshir-keyin-yoz ko'rinishiga qaytarmang — bu aynan o'sha poyga edi.
- `build_briefing` boshida kunlik byudjet (`limits.check_daily_budget`) —
  briefing foydalanuvchi so'ramagan fon xarajati va `record_usage` orqali
  o'sha limitga kiradi. To'lgan bo'lsa nashr o'tkazib yuboriladi (xato emas).
- Jim soatlar — **yetkazishni** to'xtatadi (`delivery_check` →
  `in_quiet_hours`), **qurishni** emas: kun tugunlari mavjud bo'lganda
  briefing baribir tayyorlanadi, faqat push kutadi (keyingi yetkazish
  cronida `delivered_at is None` bo'lgani qayta tekshiriladi).
- Cron: `build_briefings` (har soat, `minute={45}`) — lokal vaqti 02:00
  bo'lgan akkauntlar uchun **kechagi** briefingni quradi; `deliver_briefings`
  (`minute={6,21,36,51}`, har 15 daqiqa) tayyor-lekin-yetkazilmagan
  briefinglarni push qiladi. `worker/tasks.PHASE3B_CRONS`,
  `settings.briefing_enabled` — global o'chiruv kaliti.

## Alertlar (3-bosqich/C)

- Ikki jadval, migratsiya `0011`: `alert_rules` (kalit so'z yoki semantik
  qoida, `chat_id` NULL = barcha chatlar) va `alert_events`
  (`uq_alert_event(rule_id, message_id)` — **dedup bazada**, kodda emas).
- `services/alerts.py` — yagona nuqta. `compile_keyword_query` naqshni
  `search.fold_terms` orqali **indeks bilan bir xil** fold fazosiga tushiradi
  (`tsv_uz`), va alohida **xom** variantini `tsv_ru` uchun quradi. Ikki kanal
  kerak: `simple` konfiguratsiyada ruscha stemming yo'q.
- Uchta qopqoq olib tashlanmaydi: kursor (`last_checked_msg_id`, har skaner
  faqat yangi xabarlarni ko'radi), `cooldown_minutes` (`(rule, chat)` bo'yicha
  — gapiradigan chat aks holda soatiga o'nlab push beradi) va
  `MAX_EVENTS_PER_PUSH = 5` ("va yana N ta").
- **Kursor ikki tomondan klamplangan** (`_scope_max_id`, migratsiya `0014`):
  faqat `messages.created_at` ≥5 s oldin bo'lgan qatorlargacha (READ
  COMMITTED da kichikroq id keyinroq ko'rinadi) va semantik qoida uchun
  faqat **vektori bor** eng katta id gacha. Ikkinchisisiz hali embed
  qilinmagan yangi xabarlar kursor ortida qolib abadiy o'tkazib yuborilardi
  (`_semantic_stmt` `message_embeddings` bilan INNER JOIN qiladi).
  Shu sababli semantik qoida `EMBED_MIN_CHARS` (20) dan qisqa xabarlarni
  **umuman** ko'rmaydi — UI'da ham shunday yozilgan.
- **O'zimizniki emas** filtri `messages.out` ga tayanadi (`sender_id` emas):
  shaxsiy chatda `from_id` bo'sh keladi, ya'ni o'z xabaringiz o'z alertingizni
  uyg'otib, cooldown'ni yeb qo'yardi.
- Push matni **element bo'yicha** yig'iladi (`render_push`): sig'magan
  elementlar butunligicha tashlanadi va "+N" ga qo'shiladi. Tayyor HTML'ni
  belgilar soni bo'yicha kesish yo'q — kesish `<a href="…` o'rtasiga tushsa
  Telegram xabarni rad etadi va hodisa abadiy yetkazilmay qolardi.
  `bot/push._hard_split` ham teg va `&entity;` ichida kesmaydi.
- Yetkazish oynasi `created_at` bo'yicha tartiblanadi (`rule_id` emas — aks
  holda gapiradigan qoida boshqalarni ochlikda qoldiradi); xabari
  o'chirilgan hodisa `delivery_error="deleted"` bilan yopiladi;
  `ALERT_MAX_DELIVERY_ATTEMPTS` (3, `alert_events.attempts`) dan keyin voz
  kechiladi.
- Har begona satr (chat sarlavhasi, qoida nomi, snippet) bitta yordamchidan
  o'tadi: `_safe_html` = `neutralize` → kesish → `html.escape`. Kesish
  escape'dan **keyin** bo'lsa `&amp;` yarim qolib, Telegram xabarni rad etadi.
- Semantik qoidaning vektori `alert_rules.embedding` da **keshlanadi**;
  `pattern`/`kind` o'zgarsa router uni `NULL` ga tushiradi. Skaner har 5
  daqiqada yuguradi — keshsiz bu kunlik ortiqcha embedding xarajati.
- Push **faqat** egasining boshqaruv boti chatiga (`accounts.user_id →
  users.tg_user_id`), `bot.push.send_html` orqali. Xabar matni
  `untrusted.neutralize` **va** `html.escape` dan o'tadi (`alerts.snippet`):
  biri konvert tegini, ikkinchisi Telegram HTML tegini o'ldiradi.
- Jim soatlar — B'ning `accounts.tz/quiet_from/quiet_to` ustunlari,
  `alerts.is_quiet()` sof funksiyasi (yarim tun orqali o'raladi). Jim
  soatlarda **yetkazish** to'xtaydi, **skaner** emas: kursor oldinga siljib
  turmasa, tong otganda bir yugurishda butun tun qayta skanerlanardi.
- Cron: `scan_alerts` (har 5 daqiqa, 1/6/11…) va `deliver_alerts_task`
  (4/9/14…), `worker/tasks.PHASE3C_CRONS`. `sync_account` oxirida yangi
  xabar bo'lsa `scan_alerts` darhol navbatga tushadi.

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
- **QR login — asosiy yo'l** (`auth_flow.start_qr`, `/api/auth/qr` + `/qr/poll`).
  Telefon ham, kod ham kerak emas. Token ~1 daqiqada eskiradi, shuning uchun
  har oqim uchun fon vazifasi (`_watch_qr`) `qr.wait()` / `qr.recreate()` siklini
  yuritadi; frontend 2 soniyada bir holatni so'raydi. Oqim tugaganda
  `_cancel_watcher` **majburiy** — aks holda task yopilgan klientda qoladi.
- QR rasmi **server tomonda** chiziladi (`qrcode` → SVG → `data:` URI) va
  `<img src>` bo'lib beriladi. Brauzerga QR kutubxonasi qo'shilmaydi (CDN yo'q)
  va `innerHTML` ishlatilmaydi.
- Kod yo'li fallback: Telegram faol sessiya ko'rsa kodni **SMS emas, ilovaga**
  yuboradi (`code_type=app`). Shuning uchun `auth.ResendCode` (`/api/auth/resend`)
  bor — `next_type` bo'sh bo'lsa Telegram qayta yuborishga ruxsat bermaydi va
  UI tugmani ko'rsatmaydi. Resend `attempts` ni **nolga tushirmaydi**.

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
| `ROUTE` | `claude-haiku-4-5` | `gemini-2.5-flash-lite` | `deepseek-v4-flash` |
| `SEARCH` | `claude-opus-5` | `gemini-2.5-flash` | `deepseek-v4-flash` |
| `TOOLS` | `claude-opus-5` | `gemini-2.5-flash` | `deepseek-v4-flash` |
| `DEEP` | `claude-opus-5` | `gemini-2.5-pro` | `deepseek-v4-pro` |
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
- DeepSeek V4: `deepseek-v4-flash` va `deepseek-v4-pro` — API `/models`
  faqat shu ikkitasini beradi. **Ikkalasida ham function calling bor**,
  `temperature` ham qabul qilinadi. `_TOOL_CAPABLE` ga qo'shilmagan yangi
  model nomi = `Task.TOOLS` jimgina Gemini'ga tushadi, xatosiz.
- Eski `deepseek-chat` / `deepseek-reasoner` nomlari ishlaydi, lekin API
  **ikkalasini ham v4-flash** ga yo'naltiradi — `reasoner` deb so'rash
  kuchliroq model bermaydi. `.env` da konkret v4 nomini yozing.
- DeepSeek kontekst **1M** (chiqish 384K gacha). Shunga qaramay
  `search.select_context` byudjetidan o'ting — narx uchun, sig'im uchun emas.
- Javobdagi `model` maydoni so'ralganidan farq qilishi mumkin (alias) —
  `pricing.py` ga **qaytgan** nomni qo'shing, aks holda `estimate_cost`
  `None` qaytaradi va kunlik xarajat limiti jimgina ishlamay qoladi.

**Model override (`LLM_MODEL_*`).** `LLM_TASK_*` provider bilan modelni birga
beradi, `LLM_MODEL_ROUTE|SEARCH|TOOLS|DEEP|EMBED` esa **faqat model nomini**
almashtiradi — provider tanlash o'zgarmaydi. Model tanlangan providerning
`capabilities()` iga mos kelmasa router **aniq xato** tashlaydi (tool'siz model
`TOOLS` ga, chat modeli `EMBED` ga tushmaydi) — jimgina eski modelga qaytish
yo'q. `gemini-2.5-flash-lite` 2026-10-16 da eskiradi (`router.RETIRING`): 45 kun
qolganda startupda `llm.model_retiring` ogohlantirishi chiqadi, lekin default
**o'zgartirilmaydi** — vorisga `LLM_MODEL_ROUTE` bilan, o'lchovdan keyin o'ting.
Embedding tomonida `EMBED_MODEL` / `EMBED_DIMS`; model almashsa
`python -m app.main reembed --model X` (hech narsa o'chirmaydi, qatorlar
yonma-yon, qidiruv joriy modelni filtrlaydi) — batafsil `docs/llm.md`.

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
