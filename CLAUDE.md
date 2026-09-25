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

- **Intent avval, qidiruv keyin.** Rejim (`fast|deep|exhaustive`) —
  **qopqoq**, buyruq emas: u qidiruv qanchalik chuqur bora olishini aytadi,
  qidirish kerakligini emas. Tartib: (a) kod darvozasi
  `deep_search.direct_intent` (deterministik, uz/ru/en: salom/`meta`/qisqa
  gap-so'z) → (b) planner intenti (`meta|chitchat|about_prior` ham bor) →
  (c) **faqat** qidiruv kerak bo'lsa baho → tasdiq → skan. Qidiruvsiz turn'da
  `estimate` hodisasi ham, `scan_runs` qatori ham yo'q va deep shift sloti
  `limits.refund_deep_rate` bilan qaytariladi (baho yo'li bilan bir xil).
- **Qidiruvsiz javob — `deep_search.answer_direct`**: bitta arzon chaqiruv
  (`Task.SEARCH`), `meta` da imkoniyatlar **statik matndan**
  (`prompts_deep.capabilities_text`, uch tilda) — model funksiya o'ylab
  topmasin. Yozuv o'sha yagona yozuvchida (`_persist`), `context.direct=True`,
  `context.intent=<intent>`, qamrov da'vosi yo'q. `about_prior` esa qidirmaydi,
  lekin **oldingi dalillarni** `_refs_to_hits` bilan qayta o'qiydi; havola
  bo'lmasa quvur baribir qidiradi.
- **`stages` hodisasi birinchi keladi** — shu turn'da yuguradigan bosqichlar
  ro'yxati; UI reyki (`static/pages/ask/deep.js`) shundan quriladi, qattiq
  yozilgan 9 talikdan emas. Baho kartasi faqat `estimate` hodisasi kelganda,
  `deep`/`fast` da esa faqat `est_total_usd > $0.05` bo'lsa (`deep.js::estimateCardVisible`,
  `WINDOW_CONFIRM_MIN_USD` bilan bir); `exhaustive` da doim.
- **Dual-LLM taint guard.** `plan` va `reflect` prompt'lariga xabar matni
  **hech qachon** tushmaydi — `_Taint.assert_clean()` har chaqiruvdan oldin
  tekshiradi, buzilsa `DeepSearchError("taint")`. Yangi bosqich qo'shsangiz
  uni ham shu qo'riqchidan o'tkazing.
- **Bitta konvert.** Dalillar faqat bitta `<untrusted_data kind="evidence">`
  ichida, har satr `untrusted.neutralize` dan o'tgan. Ikkinchi konvert
  qo'shmang. Shakl ixcham (`build_evidence_envelope`): `## topic` →
  `### <chat> (@username | id N)` → `#### YYYY-MM-DD` → satr
  `[#c<chat>m<msg>] HH:MM kim: matn` — chat nomi va sana har satrda
  takrorlanmaydi (Gemini'da raqam = token; 300 satrlik fixture'da −22%,
  `tests/test_evidence_format.py`). Byudjet kesimi reyting tartibida, chat
  ichidagi chiqish vaqt tartibida. Sitata tokeni o'zgarmagan — uni
  qisqartirmang: `validate_citations` va havola shunga tayanadi.
  Map bo'lagi (`build_map_envelope`) ham shu shaklda (`### chat (id N)` →
  `#### kun` → `[#id] HH:MM kim: matn`), filtr partiyasi — `### chat` →
  `#i YYYY-MM-DD kim: matn` (indeks o'zgarmagan). Satr shaklini o'zgartirsangiz
  **prompt'dagi tavsifni ham** (`DEEP_SYNTH_PROMPT` "## Evidence", `SCALE_MAP_PROMPT`,
  `SCALE_FILTER_PROMPT`) va `DEEP_PROMPT_VERSION` ni yangilang — e7aacff dan
  keyin sintez prompt'i hali ham eski `[chat | date | sender]` qavsini
  tasvirlab turgan edi.
- **Vaqt oynasi savoldan — kodda** (`services/timewin.py`, uz/ru/en, akkaunt
  `tz` sida): bugun/kecha/shu hafta/oxirgi N daqiqa|soat|kun|hafta|oy|yil,
  "…dan beri", aniq sana va oraliq, "butun davr" (`since=None`).
  `parse_plan` uni planner ustidan qo'yadi (bo'sh yoki kengroq → kodniki,
  torroq → modelniki) va `plan.window_from_question` ni yoqadi — UI "vaqt
  oralig'ini ko'rsating" ogohlantirishi **faqat** shu bayroq bo'lmaganda
  chiqadi. Oyna bucket gridiga ham o'tadi: ≤31 kunlik oyna → **kunlik**
  bucket (oylik emas) va har bucket oynaga qisiladi.
- **`plan.retrieval_kind`** (`topic|window|person`) — quvurning shakli.
  `window` (`services/window_search.py`): "kimlar yozgan", "qaysi chatlar
  faol edi" — bucket rejasi, tirsak va mavzu filtri **yo'q**, oynadagi
  xabarlar `count(*)` bilan sanaladi. **Mavzusiz** oynada (`plan.entities`
  bo'sh) `WINDOW_MAP_MAX_MSGS` (400) dan ko'p xabar bo'lsa — **chat bo'yicha
  namuna** (`stratified_quotas`: hajmga mutanosib, har chat ≥5 ≤60, chat
  ichida eng yangisi; `pull_window_sample`), bu reja, kesish emas
  (`truncated` yo'q), qamrov "N xabar oynada · 400 tasi o'qildi (chat bo'yicha
  namuna) · hisoblar aniq" deydi, sanoq blokida chat bo'yicha **aniq**
  sonlar. Mavzuli oynada ≤`SOFT_MAX_EVIDENCE` bo'lsa hammasi, ko'pi
  `(chat, kun)` kvotasi + filtr ("M dan N ko'rildi"). Map kirishi har doim
  ≤400 (`window_chunks`: chatlar ketma-ket, 80 lik bo'laklar —
  (chat, yuboruvchi) bo'yicha bo'lish 16 bo'lak qopqog'i bilan katta qismini
  map'dan tashqarida qoldirardi); sintezga ketadigan 150 satr chatlar bo'ylab
  navbat bilan (`stratify_order`, ilgari eng eski 150 tasi edi). Jonli
  o'lchov (conv 285, 12,461 xabar): 805 o'qilgan, 16 map bo'lagi, 17,949
  chiqish tokeni, $0.127. Filtr/map'dan oldin `dedup_window` (≥24 belgili
  matnlar, `scale_search.dedup_fast`, O(n)) kross-post takrorlarni vakilga
  yig'adi — `dropped_duplicates` qamrovda, takrorlar "ko'rildi" ga kiradi;
  qisqa "rahmat"lar tegilmaydi. `inbox` = `out=false` **va** (shaxsiy chat yoki
  mening xabarimga javob) — `@eslatma` sxemada yo'q va buni qamrov aytadi.
  Darvoza kodda (`WINDOW_LIST_RE` + oyna, `TOPIC_MARK_RE` bo'lsa `topic`):
  "bugun IT ta'lim **haqida** nima deyishdi" — mavzu skani, oyna bir kun
  bo'lsa ham.
  **Yo'nalish** `plan.direction` (`detect_direction`, uz/ru/en): `in` (=
  `inbox`, moslik taxallusi), `out` ("kimga yozdim" → `messages.out`), `both`
  ("kim bilan gaplashdim" → men shu oynada yozgan chatlar). Reyting/sanoq
  savoli (`AGGREGATE_RE`, mavzusiz) → `window_stats`: `plan → stats → write`,
  javob **DB agregatlaridan** (`stats_window`, chat/yuboruvchi, ulush %) +
  top 10 chatdan ≤3 namuna; pull/filtr/map yo'q, tasdiq so'ralmaydi.
  Mavzuli savolda `out` kaskad pull'iga filtr bo'lib o'tadi (`PullSpec.out_only`).
- **Qamrov (`Coverage`) kodda hisoblanadi**, modelga yozdirilmaydi. `gaps` =
  `synced_min_id != 1` chatlar; `server_rungs_used` / `server_calls` /
  `server_hits` / `server_skipped` / `external_hits` — server pog'onalarining
  **rost** hisobi. "Hamma xabarlarni ko'rdim" degan gap chiqmasin.
- **Tashqi manba da'vosi kod bilan qo'riqlanadi.** Sintez turn'iga
  `deep_search.external_fact()` bitta fakt satri qo'yadi
  (`external_evidence: none` / `N hit(s) from @a, @b`) va prompt "tashqi
  kanaldan" deyishni faqat shunda ruxsat etadi. Sitata validatsiyasi bilan
  bir joyda `external_claim_note()` post-tekshiruvi: javob da'vo qilib,
  tashqi dalil bo'lmasa javob **qayta yozilmaydi**, oxiriga halol callout
  qo'shiladi va `deep.external_claim_unbacked` log'ga tushadi.
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
- **Fan-out kechikishi — trigram kanali** (2026-09-25, 2.7M xabar, "oxirgi
  3 oyda ijara narxi qanday o'zgargan?", 6 subquery, butun akkaunt, 116
  kunlik oyna). Oldin: `fan_out` 108/109/112 s (jonli: 60 s soft-wall bilan
  kesilgan `search_local` 80–87 s), shundan `_trigram_multi` bitta so'rovda
  27–104 s; qolgan kanallar ≤2.7 s (tsv_uz), vektor ≤0.6 s, tugun ≤1.7 s.
  `EXPLAIN (ANALYZE, BUFFERS)` ikki sababni ko'rsatdi va ikkalasi kodda:
  (1) `work_mem` 4MB da trigram∩sana bitmap'i **lossy** (36k sahifa,
  306k qator recheck, 101 s) → `search.TRGM_WORK_MEM="32MB"` (`SET LOCAL`,
  `tune_trgm`) → lossy 0, 33k recheck, izolyatsiyada 9.5–13 s;
  (2) planner `<<%` selektivligini ba'zi termlarda ~200 qator deb baholab
  sana indeksi bilan `BitmapAnd` qilmaydi → butun tarix recheck, 179 s →
  `TRGM_MULTI_TIMEOUT_MS=25_000` SAVEPOINT ichida (`search.trigram_timeout`,
  kanal bo'sh, tranzaksiya sog'). So'rov embedding'i leksik kanallar bilan
  parallel (`hybrid_search_multi` → `_hybrid_multi_channels`).
  Keyin: `fan_out` 23.6/22.5/24.3 s, `fan_out`+dedup+rerank 27.5/26.3/28.0 s
  (oldin 112/113/116 s), nomzodlar 156 → 156, dalil 40 → 40 (bir xil).
  `bench.py --sections fanout --account 2 --keep-stats`: trigram 4MB p50
  75.5 s → 32MB 9.5 s. Oyna filtri har kanal SQL'ida allaqachon bor edi;
  `PER_SUBQUERY_LIMIT`/`MAX_CANDIDATES`/`TRGM_CANDIDATE_CAP` o'zgarmagan.
  Trigram hali ham eng sekin kanal (6 parallel ostida ~18–20 s) — keyingi
  qadam `(published_at, text_fold)` GIN yoki planner'ga ishonmaydigan
  so'rov shakli, **avval o'lchov**.

### Orkestrator (`services/orchestrator.py`, `services/skills.py`)

Deep/exhaustive kirishi (`/api/deep*`, `chat_service` deep yo'li) endi
**orkestrator**: planner katalogdan "eng arzon yetarli yo'l" qadamlarini
tanlaydi, executor ularni tartibda yuritadi. `fast`/agent yo'li o'zgarmagan.

- **Katalog** (`skills.SKILLS`): `answer_direct`, `clarify`, `reuse_prior`,
  `chat_digest`, `stats`, `compute`, `window`, `search_local`,
  `search_server`, `discover`, `scan_corpus`, `write_propose`. Har biri —
  mavjud funksiya ustidagi **yupqa o'ram** (fan_out/run_window/run_scale/
  escalate/digests); narx, vaqt, `min_mode`, `needs_confirm`, `quarantined`
  e'lon qilingan. Yangi qidiruv mantiqini bu yerga yozmang — eski modulga
  yozing va o'rang. Yangi ko'nikma karantinda bo'lsa `QUARANTINED_SKILLS`
  testi ongli qaror talab qiladi.
- **Planner** — bitta chaqiruv, `ORCHESTRATOR_PLAN_TASK` (`search` sukut |
  `route`; `orchestrator.plan_task()`) (`ORCH_PLAN_PROMPT` =
  `DEEP_PLAN_PROMPT` + katalog), eski reja maydonlari `parse_plan` dan
  o'tadi (oyna, `retrieval_kind`, `force_all_chats`, `DISCOVER_RE`,
  follow-up — hammasi joyida). Qoidalar: arzon birinchi; `chat_digest`/
  `stats`/`compute` qidiruvdan oldin; lokal → server → skan; `scan_corpus`
  faqat ko'p chat x uzun davr + mavzu; `clarify` parametr yo'q bo'lsa;
  "bu chat" = tanlangan chat; ≤4 qadam; rejim qopqog'idan yuqori yo'q.
- **Planner vazifasi A/B** (2026-09-25, `docs/cost-research.md`): `route`
  (flash-lite) 17 deep turn'dan 15 tasida `search` bilan aynan bir xil
  qadamlar, yaroqsiz JSON 0, planner narxi ≈ −68%. Lekin planner jami deep
  narxining ≈19% i — tejash ≈13%, qoida esa ≥30% talab qiladi → sukut
  **`search`**. Bitta yugurishdagi 1–3 senariy flip — tebranish (bir xil
  konfiguratsiyada 16/18 va 17/18). Yaroqsiz JSON → `orch.plan_invalid`,
  `planner_ok=False`; `orch.plan` log'ida `plan_task`.
- **Qo'riqchilar** (`apply_guards`) planner ustida, faqat aniq holatlarda:
  direct/reuse, terminal yolg'iz, sanoq darvozasi → `stats`, mavzusiz oyna →
  `window`, **1–3 chatga `scan_corpus` yo'q**, `compute` oldidan raqam
  manbasi, discover, bo'sh reja → `default_steps`. Qopqoqdan yuqori
  ko'nikma pastga tushiriladi. Server qo'riqchisi (`needs_server`) qidiruvdan
  keyin; replan ≤1 (dalil < 3, ≤2 qadam, `Task.ROUTE`, faqat sanoq).
- **Tanlangan chatlar** — `DeepIn.chat_ids` (UI "Kontekst chatlari",
  `answer.js`), chat API'da `context.peer_id` → pinned chat. "hamma/barcha
  chatlar" bo'lmasa `plan.chats` = tanlangan.
- **Raqamlar kodda**: `stats`/`chat_digest` faktlari `count(*)` dan,
  `compute` — `ast` oq ro'yxatli evaluator (`eval` **yo'q**). Faktlar evidence
  konvertiga, `unbacked_numbers` javobdagi faktsiz raqamni log/kontekstga
  yozadi (javob qayta yozilmaydi).
- **Taint**: karantin ko'nikma chiqishi plan/replan'ga faqat sanoq bo'lib
  o'tadi; dayjest matni `_Taint.add_texts` bilan qo'riqchida;
  `assert_clean("replan")`.
- **Qaytish**: `ORCHESTRATOR=false` → eski `_run_pipeline`
  (`run_deep_search(orchestrate=...)`, kutubxona sukuti `False`).

### Suhbatga bog'liqlik (follow-up)

Planner endi oxirgi `DEEP_HISTORY_TURNS` (6) turn'ni ham ko'radi
(`load_history` → `RunContext.history`), aks holda "chuqurroq qidir" da aynan
shu so'zlarni qidirardi.

- **Taint qoidasi ikki xil.** User matni — o'zining matni, ochiq ketadi.
  Oldingi **javob** begona xabarlardan sintez qilingan → faqat
  `<untrusted_data kind="prior_answer">` konvertida (`render_history`), ichi
  `neutralize` dan o'tgan. Tekshiruv **ikki xil og'irlikda**: xabar matni
  planner promptida topilsa `assert_clean` → `DeepSearchError` (qattiq xato);
  oldingi javob konvertdan chiqib ketsa esa `prior_leak` → `True`, yugurish
  **tarixsiz davom etadi** (`deep.taint_prior_skipped`). Sabab: 24 belgilik
  bo'lak savolni takrorlagan javobda har turn'da yolg'on pozitiv berardi va
  suhbat doimiy 500 bilan o'lardi. Endi chegara 48 belgi, `plan:` metasi va
  user matni qoldiqdan olib tashlanadi.
  Xabar matni esa baribir **hech qayerda** bo'lmaydi (konvert ichida ham).
  Oldingi rejaning `subqueries`/`entities`/`chats` i — bizning planner
  yasagan tuzilma, oddiy JSON bo'lib beriladi.
- Reja `standalone_question` + `intent_follow_up`
  (`none|deeper|repeat|correction|context_switch|reformat|explain`) +
  `needs_retrieval` qaytaradi; kengaytirish **kodda** (`_apply_follow_up`):
  `deeper` → oldingi subquery'lar + yangilari, `levels=[0,1,2]`,
  `server_scope ≥ gaps`, 2-raund majburiy; `repeat` → expansion+HyDE;
  `correction`/`context_switch` → eski reja **meros qilinmaydi**.
- `needs_retrieval=False` (faqat `reformat`/`explain`) → fan-out o'tkazib
  yuboriladi, oldingi turn dalillari `context.evidence_refs`
  (`[{chat_id, tg_msg_id}]`, ≤60) dan `_refs_to_hits` bilan qayta o'qiladi.
  Model yolg'on aytsa ham havola bo'lmasa — quvur baribir qidiradi.

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
- Pog'onalar tartibi: `discover` (`channels.SearchPosts`) — **faqat**
  `server_scope=="discover"` da, lekin shunda **birinchi va zaxiralangan**
  1 chaqiruv (bepul slot tugasa +2 tekin zaxira yo'l, `contacts.Search`) →
  bo'shliqli chatlar (`messages.Search`, top 2 subquery) → global
  (`messages.SearchGlobal`). Ilgari `discover` oxirida turardi va shu bilan
  qolgan ikki pog'onani **butunlay o'chirardi** — bitta noto'g'ri `discover`
  butun server qamrovini yo'q qilardi. Jami `DEEP_SERVER_MAX_CALLS` (6)
  chaqiruv.
- **`channels.SearchPosts` Telegram Premium talab qiladi.** Premium'siz
  akkauntda u `PremiumAccountRequiredError` beradi → `note_code="premium_required"`
  (xato emas, holat; `tg_search.PREMIUM_MISSING` da akkaunt bo'yicha bir marta
  eslab qolinadi) va **tekin zaxira yo'l** ishga tushadi:
  `tg_search.server_discover_by_name` = `contacts.Search` (nom bo'yicha ochiq
  kanallar, lokal `chats` dagilari tashlanadi) + top 3 kanalda `messages.Search`.
  **So'rov strategiyasi ikki xil**: `contacts.Search` ga mavzu iborasi emas,
  `tg_search.discover_name_queries` chiqargan ≤2 ta qisqa "nomga o'xshagan"
  variant boradi (reja `entities`i, keyin subquery boshi — stop-so'z va joy
  kelishigi qo'shimchasi olib tashlanadi; jonli misol: `"IT ta'lim
  O'zbekistonda"` `contacts.Search`da 0 ta kanal beradi, `"IT ta'lim"` esa
  beradi), `messages.Search` esa **asl mavzu matnini** oladi (`topic_query`) —
  bu to'liq matn qidiruvi va iborani tushunadi. **Mavzu nomzodlari birinchi,
  joy nomi/umumiy so'zlar ("O'zbekiston", "Toshkent", "yangiliklar", ...)
  oxirida** va qopqoqqa sig'masa butunlay tashlanadi; byudjet ruxsat bersa
  ikkala `contacts.Search` ham **yuboriladi** (birinchisining to'la natijasi
  ikkinchisini bekor qilmaydi — aks holda joy nomi mavzu so'rovini yer edi),
  kanal tanlovi esa so'rovlar bo'yicha **round-robin** (gigant umumiy kanal
  a'zolar soni bilan mavzu kanallarini siqib chiqarmaydi). Kanalda mavzu
  matni 0 bersa — **o'sha kanalda** qisqa mavzu nomi bilan bir marta qayta
  qidiriladi (byudjet qolgan bo'lsa).
  Peer javob bilan kelgan access hash'dan yasaladi (`get_entity()` yo'q), natija
  tashqi (`[server external]`, `t.me/<username>/<id>`). Bitta implementatsiya —
  `tools.discover_channels` ham, `deep_search.escalate` ham shuni chaqiradi.
  Byudjet: 1 flood/premium tekshiruvi + ≤2 `contacts.Search` + ≤3
  `messages.Search`, `SERVER_OTHER_RESERVE=1` esa gap-chat/global
  pog'onalariga qoladi (`top_k` shunga qarab `allow - len(name_queries)`
  bilan qisqartiriladi).
  `contacts.Search` **o'z bucket'ida** (`BUCKET_CONTACTS`, 3/min burst 2) —
  `BUCKET_DISCOVER` (burst 1) da probe yagona tokenni yeb, zaxira yo'l hech
  qachon ishga tushmasdi (jonli: `server_skipped=["discover"]`,
  `external_hits=0`). `PREMIUM_MISSING` — jarayon xotirasida (tez yo'l) **va**
  Redis'da 7 kun TTL bilan (`tg:premium_missing:<account_id>`) — sovuq `api`
  jarayoni ham qayta probe qilmaydi; Premium sotib olinsa bayroq ≤7 kunda
  o'zi eskiradi.
- To'lgan bucket — xato emas: pog'ona **o'tkazib yuboriladi** va
  `coverage.server_skipped` ga yoziladi. Kutish yo'q, retry yo'q.
- Server hitlari alohida RRF ro'yxati (og'irlik 0.9). Lokal `chats` da bor
  chatdan kelgani `ingestion.upsert_messages` orqali saqlanadi (sitata lokal
  id fazasida qolsin); qolgani **saqlanmaydi**, manfiy sintetik id oladi va
  dalilda `[server external]` deb belgilanadi — u lokal indeks bilan
  solishtirilmagan.

### Korpus miqyosidagi kaskad (`services/scale_search.py`)

Rejimlar: **`fast`** — yuqoridagi fan-out quvuri, o'zgarmagan; **`deep`** /
**`exhaustive`** — `deep_search._run_pipeline` dan `scale_search.run_scale`
ga o'tadi va fan-out→dedup→rerank **o'rniga** turadi. Reja, taint, sintez,
qamrov va sitata validatsiyasi — o'sha (`deep_search.py`).

    plan_buckets (chat x ISO-hafta, tugun vektorlari bo'yicha;
                  deep <=200, exhaustive <=400 — o'lchovdan, quyiga qarang)
      -> pull_candidates (bucket ichida 4 kanalli RRF, k=30, parallel
         sessiyalar, elbow kesish [500, 3000], <=75% tashlanadi)
      -> dedup_fast (sha1 + **banded** simhash, O(n))
      -> filter_batched (ROUTE/Flash-Lite, 50 satrlik partiya, 0..3 ball)
      -> select_evidence (ball + chat/hafta kvotasi, <=1000)
      -> map (bir jinsli, bitta chat, 80 xabarlik bo'lak; <=15 dalilda
         umuman o'tkazib yuboriladi) -> reduce -> da'volar dayjesti

- **Dedup hovuzda O(n) bo'lishi shart.** `cluster.dedup_simhash` har imzoni
  saqlanganlarning hammasi bilan solishtiradi; 20k nomzodda bu o'ladi.
  `dedup_banded` imzoni 4 ta 16-bitli bandga bo'ladi (`BAND_BUCKET_MAX=32`
  chelak qopqog'i bilan) — kosinus/klaster esa **faqat** filtrdan keyingi
  ≤1000 ta ustida (o'sha yerda O(n²) halol).
- **Karantin.** `filter`/`map`/`reduce` prompt'lari begona matn ko'radi va
  `deep_search.QUARANTINED_STAGES` da turadi; `plan`/`reflect` esa **hech
  qachon** — `_Taint.assert_clean` ularda qizaradi. Yangi bosqich qo'shsangiz
  uni ro'yxatga qo'shish **ongli qaror** bo'lsin.
- **Narx qopqog'i kodda**: `COST_CEILING = {fast .01, deep .10, exhaustive
  .50}` filtr va map orasida tekshiriladi; oshsa bosqich to'xtaydi va
  `coverage.stopped_by="ceiling"`, `truncated=True` chiqadi. To'xtash sabablari:
  `yield | ceiling | saturation | wall | complete | estimate_only`.
- **To'xtash sababi bosqich bo'yicha yoziladi** (`pull_stopped_by` /
  `filter_stopped_by`, migratsiya 0021), umumiysi `combine_stops` bilan
  hisoblanadi: wall/ceiling — yield/complete dan ustun. Ilgari `filter`
  hovuzni tugatib "complete" deb yozar va pull qo'ygan "wall" ni **yuvardi**.
- **`estimate()` bitta skan tokenidan oldin** hisoblanadi. Nomzod sanog'i
  kalibrlangan: `min(FILTER_MAX, max(ELBOW_MIN, bucket x k x EST_PULL_YIELD))`,
  keep-rate `EST_KEEP_RATE=0.08` — xom `bucket x k` har doim qopqoqqa to'yinar
  va baho **har doim bir xil** chiqardi. `est_cost_usd` = skan (qopqoq shuni
  o'lchaydi), `est_total_usd` = skan + sintez. `exhaustive` API'da
  `confirm: true` talab qiladi — aks holda `estimate` + `done
  {needs_confirm:true}` qaytadi, yugurish to'xtaydi, deep shift sloti
  **qaytariladi** (`limits.refund_deep_rate`) va `scan_runs` ga
  `stopped_by="estimate_only"` qatori yoziladi.
- **Oynasiz `exhaustive` butun sinxronlangan oraliqni rejalashtiradi**: tugun
  bucket'lari + **hamma** nishon chat uchun `(chat, oy)` gridi. Tugunlar faqat
  oxirgi haftalarda bo'lsa reja 5-7 bucket / 1 hafta bilan chegaralanardi.
- **`plan_buckets`: grid avval, qopqoq keyin.** Donadorlik oynadan
  (`grid_grain`): ≤31 kun — kun, ≤`WEEK_GRID_MAX_DAYS` (400) — ISO-hafta,
  undan uzun/chegarasiz — oy. Teng ballda ko'p xabarli bucket oldinda,
  `BucketSummary` qopqoqdan oldingi jami (`buckets_total`), chatlar va
  `weeks`/`days` ni bahoga beradi — UI "400 / 12 062 bo'lak · 38 hafta"
  deydi, hech qachon "0 hafta" emas.
- **Kaskadda ikkinchi raund yo'q** (`reflect` chaqirilmaydi): bucket fazosi
  allaqachon aylanib chiqilgan, "boshqa so'z bilan qidir" u yerda qamrov
  emas, faqat narx qo'shadi.
- Kesh `relevance_cache` (migratsiya 0020), kalit
  `md5(uz_fold(standalone_question))`; audit `scan_runs` — har yugurish uchun
  bitta qator (`buckets_planned/scanned`, `filtered`, `kept`, `stopped_by`).
  `scan_runs.candidates` — **tirsakdan oldingi** hovuz (`stats.pulled`), ya'ni
  "nechta nomzod ko'rildi" degan savolning halol javobi.
- **`BUCKET_MAX_DEEP=200` o'lchovdan.** 2.66M xabarli jonli korpusda
  (faqat o'qish, 3 savol) `pull_candidates`: cap=100 p50 40.1 s / p95 41.3 s ·
  cap=200 p50 34.9 s / p95 40.5 s · cap=400 p50 47.0 s / **p95 75.2 s**.
  `deep` da pull byudjeti o'lchov paytida ≈53 s edi (`PULL_RESERVE_S=45`),
  hozir ≈70 s (`PULL_RESERVE_S=30` — pull'dan keyingi ish o'lchovda 30-40 s),
  ya'ni 400 p95 da javobni jimgina kesardi.
  `plan_buckets` p50 ≈1.1 s (cap'dan **deyarli mustaqil**). Narx bucket
  soniga chiziqli emas: katta chatlarning bir nechta oyi vaqtni yeydi.
- Pull byudjeti tugasa qolgan bucket'lar qaralmaydi va bu
  `stopped_by="wall"` bo'lib qamrovga tushadi — jim o'tkazilmaydi.
- Tugun embedding'lari bo'lmasa (`retrieval_node_embeddings` bo'sh) reja
  butunlay `(chat, oy)` zaxira yo'liga tushadi — quvur ishlaydi, lekin
  granularlik oylik. Prod'da bu hozir shunday.
- **`truncated` faqat rejalashtirilgan ish bajarilmaganda** rost bo'ladi:
  `stopped_by` `wall`/`ceiling` yoki `pool_capped` (`is_truncating`).
  `yield`/`saturation`/`complete` — truncation **emas**, oddiy to'yinish.
  Tirsak kesishi (reytingdan past nomzodlarni tashlash) ham o'zi bilan
  `truncated=True` opkelmaydi — u neytral holda `Coverage.ranked` /
  `dropped_by_ranking` orqali `coverage_disclosure` ga chiqadi.
- Konstantalar o'lchovdan: `evals/perf/bench.py --sections scale` va
  `evals/calibrate_filter.py` (filtr chegarasi uchun recall jadvali).
  O'zgartirishdan oldin **kamida 3 marta** qayta o'lchang.
- TODO: `exhaustive` hozircha sinxron (`EXHAUSTIVE_WALL_S=600`). Batch
  API / worker vazifasiga ko'chirish seam'i — `filter_batched`.

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
- **Takroriy qidiruv qopqog'i.** Yugurish ichida `(cached)` natija
  ko'ringandan keyin server qidiruv tool'lari (`search_telegram`/
  `search_global`/`search_calendar`/`discover_channels`) shu turn uchun
  bloklanadi — foydalanuvchi ochiq-oydin chuqurroq/qayta qidirishni
  so'ramasa (`agent.wants_deeper_search`). Bloklangan chaqiruv ham
  `calls_log` ga `blocked="repeat_guard"` bilan tushadi, `context_used.
  repeat_guard` ham belgilanadi. Suhbatning so'nggi user turn'idan biriga
  aynan/deyarli aynan mos savol (`agent.is_repeat_turn` — fold qilingan aynan
  moslik yoki Jaccard ≥0.9, kamida 3 token, "chuqurroq" so'ralmagan bo'lsa)
  server tool'larini **birinchi chaqiruvdanoq** to'sadi, bajariladigan
  chaqiruvlar qopqog'ini 2 taga tushiradi va `runtime_note`ga qator qo'shadi.
  Raundda qo'riqchi chaqiruvni to'sib, **hech narsa bajarilmagan** bo'lsa
  sikl darhol to'xtaydi (`stop_reason="repeat_guard"`, byudjet bilan bir xil
  "javob ber" turn'i) — ilgari model to'silgan chaqiruvni qayta yozib
  iteratsiya shiftigacha aylanardi.
- **Suhbat keshi (`services/tool_cache.py`)** — `seen_calls` bitta yugurish
  ichida ishlaydi, bu esa **turn'lar orasida**: kalit `(conversation_id, tool,
  buklangan args)`, TTL 10 daqiqa, Redis (yo'q bo'lsa kesh ham yo'q). Faqat
  o'qish tool'lari va faqat `ok` natija. Keshdan kelgan matn `(cached)` bilan
  belgilanadi va byudjetga **baribir** sanaladi (aks holda cheksiz sikl).
  Kalit suhbatga bog'langan: tool natijasi Telegram kontenti, u suhbatdan
  chiqmasin.
- **`AGENT_MAX_EXECUTED_CALLS` (6) va `AGENT_MAX_COST_USD` ($0.15)** — yugurish
  bo'yicha qopqoqlar (`conv-20260917-144833`: turnda 6-11 bajarilgan chaqiruv,
  $0.11-$0.40). Yetilganda tool'lar o'chadi va "bor narsa bilan javob ber"
  turn'i qo'shiladi; `stop_reason` (`answered|call_budget|cost_budget|repeat_guard|no_progress|iterations`)
  `context_used` ga yoziladi. Iteratsiya shifti yolg'iz o'zi yetmagan.
- **`no_progress`** — ikki ketma-ket raundda bajarilgan chaqiruvlar hech narsa
  bermagan (xato / bo'sh / faqat keshdan, `agent.unproductive`) va model yangi
  tool nomini sinamayapti → tool'lar o'chadi, bitta "javob ber" turn'i.
  Qopqoqlar (`call_cap`, `cost_cap`, iteratsiya) o'zgarmagan — bu ulardan
  **oldin** to'xtaydi.
- **Chat maslahatlari (`agent.chat_hints`)** — savoldagi nom (uz_fold, o'zbek
  kelishik qo'shimchalari olib tashlanadi, `difflib` ≥0.8) akkauntning
  **hamma** sinxronlangan chati sarlavhasi/`@username` bilan kodda
  solishtiriladi; ≤3 ta `chat_hint: "<title>" [id:N] <tur> msgs= last=`
  qatori `runtime_note` ga. Nega: indeks faqat 25 ta faol chat — conv 284
  ("Otabek bilan oxirgi marta nima gaplashganmiz?") da "Otabek" shaxsiy
  chatlari unda yo'q edi va agent `list_chats → search_global → get_user_info
  → list_members ×2` bilan 7 iteratsiya ($0.105) aylanib noto'g'ri javob
  berdi. Prompt: nomlangan odam — o'qiladigan **chat** (`get_recent_messages`
  / `search_messages` shu `chat` bilan, 1-3 chaqiruv), `search_global` /
  `get_user_info` / `list_members` emas. `_resolve_chat` indeksdagi
  `[id:N]` shaklini ham qabul qiladi (conv 284 da `"[id:3836]"` rad etilgan).
  Umumiy so'zlar (`chat`, `kanal`, `hafta`, `bilan`…) `_HINT_STOPWORDS` da.
- **Bir xil so'zlarni serverda takrorlash to'siladi**: lokal `search_messages`
  ≥1 topilma bergan so'rov `search_telegram`/`search_global` bilan qayta
  yugurtirilmaydi (lokal 0 bergan bo'lsa — ochiq, bu sinxronlanmagan tarix).

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
- **Chat profil rasmlari — "media saqlanmaydi" invariantiga chegaralangan
  istisno** (user aniq so'radi: "chatlarning imagelarini ham ko'rsat").
  Faqat **profil rasmi** yuklanadi va `DATA_DIR/avatars` da saqlanadi (≤512 KB,
  JPEG/PNG, `chats.avatar_*` meta, migratsiya 0017); xabar mediasi (photo,
  video, document, voice) **hech qachon** yuklab olinmaydi. `upload.GetFile`
  allowlist'da READ, lekin uni chaqiradigan **yagona modul**
  `services/avatars.py` — boshqa joyda `download_media` / `download_file` /
  `GetFileRequest` yo'q va buni `tests/test_avatars.py` grep bilan qo'riqlaydi.
  Yuklash `history` bucket'idan o'tadi, FloodWait retry qilinmaydi
  (`chats.avatar_error` ga matn bo'lib tushadi), `avatar_photo_id` o'zgarmasa
  fayl qayta olinmaydi. `AVATARS_ENABLED=false` — butunlay o'chiradi.

## Kuzatuv va limitlar (8-bosqich)

- Metrikalar `app/observability.py` — nomlar barqaror (`tgai_*`), yangi
  metrikani shu yerga qo'shing; heartbeat cron oxirida (`heartbeat(name)`).
- Kunlik byudjet (`services/limits.py`) chat so'rovidan **oldin** tekshiriladi;
  agent/digest/judge chaqiruvlari ham `conversation_messages` tokenlariga kiradi.
- **Limit foydalanuvchi bo'yicha sozlanadi** (migratsiya 0023, `user_budgets`):
  `limits.effective_budget` = override (`token_budget`/`cost_budget_usd`, NULL
  = global `LLM_DAILY_*`) + faol "bugungi qo'shimcha" (`boost_*`,
  `boost_until > now`, standart muddat — akkaunt `tz` sidagi yarim tun,
  faol boost ustiga qo'shiladi), keyin ixtiyoriy qattiq shift
  `LLM_DAILY_TOKEN_BUDGET_MAX` / `LLM_DAILY_COST_BUDGET_USD_MAX` (0 = yo'q;
  faqat foydalanuvchi o'zgartirgan qiymatga qo'llanadi). **Har** tekshiruv
  (`daily_usage` → `check_daily_budget` / `assert_within_budget` /
  `blocked_user_ids` / `account_budget_blocked`) shu yerdan o'tadi — global
  sozlamani boshqa joyda to'g'ridan-to'g'ri solishtirmang. Xom qator Redis'da
  60 s keshlanadi (`limits:budget:<uid>`, yozuvda o'chiriladi); Redis nosoz →
  DB, DB ham nosoz → global (fail-open). API: `GET/PUT /api/settings/budget`,
  `POST/DELETE /api/settings/budget/boost` (CSRF + cookie), UI — Sozlamalar →
  Byudjet (bitta limit foydalanuvchining barcha akkauntlari uchun umumiy).
- CI invariant qadamini (`tests/test_allowlist.py`, `test_peer_guard.py`,
  `test_write_actions.py`, delete-grep) yumshatmang.
- **Worker metrikalari alohida portdan.** `api` dagi `/metrics` faqat api
  jarayonini ko'radi; ingest/cron/alert hisoblagichlari worker'da o'sadi.
  `WORKER_METRICS_PORT` (default 0 = o'chiq) berilsa `worker/settings.py::startup`
  `observability.start_metrics_server()` ni chaqiradi. Portni tashqariga
  ochmang — token yo'q.
- Judge va rasm generatsiya xarajati ham kunlik byudjetga kiradi: ikkalasi
  yashirin suhbatga (`"Judge"` / `"Images"`) `assistant` qatori bo'lib yoziladi
  (briefing bilan bir uslub), chunki `limits.daily_usage` faqat
  `conversation_messages` ni o'qiydi. Rasm tokenlari **taxmin**
  (`images.IMAGE_OUTPUT_TOKENS`, rasm = 1290 output token).

### Hardening (web perimetri)

- **Logout server tomonda bekor qiladi.** Cookie o'chirish yetarli emas edi:
  nusxasi 14 kun ishlardi. `web/security.py::revoke_sessions` Redis'ga
  `web:sess:revoked:<uid>` = hozirgi vaqtni yozadi, `identity_of` esa `iat <=`
  chegarali cookie'ni rad etadi. Redis yo'q → **fail-open** (ogohlantirish
  log'i): imzo va TTL baribir kuchda. Cookie'da `v` (`SESSION_VERSION`)
  da'vosi bor — global rotatsiya migratsiyasiz.
- **`X-Forwarded-For` ga ishonilmaydi**, agar TCP peer `TRUSTED_PROXIES`
  (CIDR, default bo'sh) ichida bo'lmasa. Zanjir o'ngdan chapga yuriladi —
  birinchi ishonchsiz hop. Proxy ortida ishlatsangiz env'ni to'ldiring.
- **Log maskalash rekursiv** (`app/logging.py::scrub`): ichma-ich dict/list,
  kalit nomi bo'yicha (`token|secret|password|session|api_key|authorization|otp|code`,
  `_KEY_ALLOW` istisnolari bilan) va matn ichidagi sir shakllari bo'yicha
  (Telethon session string, `sk-`/`AIza`, bot tokeni). Sentry `before_send`
  ham shu funksiyani chaqiradi — implementatsiya bitta. Token **sonlari**
  (`tokens_in|tokens_out|tokens_thinking|tokens_cached` — faqat shu aniq
  nomlar) `_KEY_ALLOW` da: aks holda `token` bo'lagi ularni yashirib, xarajatni
  log'dan tiklab bo'lmasdi.
- Konteyner `app` (UID 1000) bo'lib yuguradi, `.dockerignore` `.env*`, `data/`,
  `tests/`, `.git` ni kontekstdan chiqaradi, `HEALTHCHECK` → `GET /healthz`
  (auth yo'q, `{"ok": true}`). Bot/worker/migrate'da healthcheck compose'da
  o'chirilgan.
- Snapshot cron bitta yugurishda `SNAPSHOT_MAX_MESSAGES_PER_RUN` (5000) post
  oladi, **eng eski `captured_at`** dan boshlab — katta kanal qolganlarini
  ochlikda qoldirmaydi.

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
- **Mini App `initData`** (`POST /api/auth/telegram`, `auth.verify_init_data`):
  kalit = `HMAC_SHA256("WebAppData", BOT_TOKEN)`, imzo shu kalit bilan
  `data_check_string` (kalitlar alifbo tartibida, `k=v`, `\n`) ustidan;
  taqqoslash `hmac.compare_digest` — oddiy `!=` imzoni belgima-belgi tiklash
  uchun vaqt kanalini ochadi. `auth_date` ≤ 24 soat (`MINIAPP_INIT_DATA_MAX_AGE`),
  kelajakdagi sana ham rad etiladi. **Bot token faqat HMAC kaliti** — log'ga,
  xato matniga, javobga hech qachon tushmaydi.
  `initData` — bu **kirish**, ulanish emas: `tg_user_id` mavjud `users` qatoriga
  mos kelmasa `403 auth.err.not_linked` va akkaunt **yaratilmaydi** (MTProto
  sessiyasi faqat QR login'da tug'iladi). IP bo'yicha shift `/api/auth/phone`
  bilan bir xil limiter'da.
- `DELETE /api/accounts/{id}` (uzish) `auth.LogOut` chaqirmaydi — u `DENIED`
  (invariant 1). Faqat bizdagi session envelope o'chadi (`revoke_session`),
  status `revoked`, joblar bekor qilinadi; ma'lumot qoladi.

## Ban riskini kamaytirish

- `TG_DEVICE_MODEL` / `TG_SYSTEM_VERSION` / `TG_APP_VERSION` — o'zgarmas.
- `FloodWaitError` da har doim kutiladi, retry-storm yo'q.
- Yangi akkaunt: birinchi sutkada cheklangan sync.
- `max_jobs = 4` — parallellikni oshirmang.

## Ma'lumot modeli — nozik nuqtalar

- **Havola yagona joydan**: `analytics.msg_link` → `chats.link_prefix`
  (`t.me/c/<id>/` yopiq, `t.me/<username>/` ochiq), yo'q bo'lsa `None`.
  Modelga faqat tool natijasidagi havola beriladi — u o'zi `t.me` URL yasamasin.
- **`senders` migratsiya 0019 dan keyingi tarixni ko'radi**; eskisi uchun
  `backfill_senders` (cron soatiga :25, `python -m app.main backfill-senders
  --account N`) har chatdan oxirgi 200 xabarni o'qib `m.sender` dan nom yig'adi.
  Telefon raqami hech qachon olinmaydi.
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
| `ROUTE` | `claude-haiku-4-5` | `gemini-3.5-flash-lite` | `deepseek-v4-flash` |
| `SEARCH` | `claude-opus-5` | `gemini-3.5-flash` | `deepseek-v4-flash` |
| `TOOLS` | `claude-opus-5` | `gemini-3.5-flash` | `deepseek-v4-flash` |
| `DEEP` | `claude-opus-5` | `gemini-3.1-pro-preview` | `deepseek-v4-pro` |
| `EMBED` | ❌ yo'q | `gemini-embedding-001` | ❌ yo'q |
| `IMAGE` | ❌ yo'q | `gemini-2.5-flash-image` | ❌ yo'q |

2026-09: Gemini 2.5 avlod suhbat modellari (`flash-lite`/`flash`/`pro`) yangi
API kalitlariga **yopiq** (404) — Gemini ustuni shuning uchun 3.x. Embedding
va rasm o'zgarmagan (alohida o'tish yo'li — `EMBED_MODEL`/`reembed`,
`LLM_MODEL_IMAGE` — sifat/moslik o'lchovi talab qiladi).

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
- **Gemini'da fikrlash tokenlari `max_tokens` byudjetidan yeyiladi.** Jonli
  yugurishda `gemini-3.1-pro-preview` sintezda `tokens_out=2677` bilan
  `MAX_TOKENS` qaytardi (byudjet 4000) va javob jumla o'rtasida kesildi.
  Adapter endi byudjetni aniq beradi (`gemini.thinking_config`,
  `GEMINI_THINKING_BUDGET`, default 1024): `gemini-3*` → `thinking_level`,
  `gemini-2.5*` → `thinking_budget`, ikkalasida `include_thoughts=False`.
  0 = sozlamani yubormaslik. Daraja **vazifa bo'yicha**
  (`GEMINI_THINKING_LEVEL_ROUTE|SEARCH|TOOLS|DEEP`, default
  `minimal/low/low/low`, `router.thinking_level`): ROUTE flash-lite'ning o'z
  sukuti `minimal` — ilgari hammaga `low` berilib arzon vazifada fikrlash
  oshirilardi. `minimal` ni bilmaydigan model (3.7/3.8-flash, 3.1-pro) `low`
  oladi. Chaqiruv bo'yicha override `LLM().chat(..., thinking=...)` — faqat
  dayjest sintezi (`synth_budget(...).kind == "digest"`) ishlatadi,
  `DEEP_DIGEST_THINKING` (default `minimal`, `""` = o'chiq). Deep sintez qattiq qopqog'i `SYNTH_MAX_TOKENS=8000`, lekin
  `max_tokens` **niyat bo'yicha** (`deep_search.synth_budget`, kodda):
  oyna/dayjest/sanoq → 3000, mavzu → 6000, `explain`/`deeper`/"batafsil"
  (`_DETAIL_RE`, uz/ru/en) → 8000; user turn'ida `answer_length: ~N tokens`
  satri (N = byudjetning 60%), `context.answer_budget` ga yoziladi.
  Provayderga ketadigan `max_tokens` = byudjet + `SYNTH_THINKING_RESERVE`
  (2500, ≤8000): Gemini fikrlashi `max_tokens` dan yeydi — jonli conv 288 da
  3000 lik dayjestda `low` fikrlash 2,488 token olib, javob 508 tokenda
  `MAX_TOKENS` bilan kesildi. Uzunlikni `answer_length:` boshqaradi.
  `finish_reason` length bo'lsa **bitta** davom ettirish chaqirig'i faqat
  batafsil so'ralganda qilinadi; qisqa byudjet kesilsa — davom yo'q,
  `context.truncated_answer=True` (jonli conv 285: oyna dayjesti 17,949
  chiqish tokeni edi). `SYNTH_MAX_TOKENS` ni pasaytirmang — shu yerdan boshqaring.
  `finish_reason` ikkala yo'lda ham (`agent` va `deep`) kontekstga yoziladi.
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
- **Thinking va kesh tokenlari hisobda.** Gemini `candidates_token_count`
  fikrlashni o'z ichiga olmaydi, Google esa uni output narxida oladi —
  `ChatResult.tokens_thinking` / `tokens_cached` (`usage.thinking_tokens` /
  `cached_tokens`) → `conversation_messages.tokens_thinking/tokens_cached`
  (migratsiya 0022, nullable), `context_used`, `agent.done`, `llm.chat`.
  Narx: `pricing.result_cost(res)` / `estimate_cost(..., tokens_thinking=,
  tokens_cached=, tier=)` — thinking output narxida, cached `CACHED_PRICES`
  da (yo'q bo'lsa to'liq narx). `limits.daily_usage` thinking'ni ham sanaydi.
  Yangi yig'uvchi yozsangiz `estimate_cost(in, out)` ni to'g'ridan-to'g'ri
  chaqirmang — thinking yo'qoladi va xarajat pastki chegara bo'lib qoladi.
- **Flex tier faqat fon ishlarida** (`LLM().chat(..., tier=background_tier())`,
  `LLM_FLEX_BACKGROUND`, default true): worker digest'lari, hafta tugunlari,
  briefing, judge. Yarim narx, 1-15 daq kechikish; 429/503 dan keyin adapter
  bir marta standart tier'da uradi (`llm.flex_fallback`),
  `ChatResult.service_tier` rostini aytadi. Interaktiv yo'lga (agent, deep,
  chat) Flex **qo'ymang** — `test_llm_thinking_tier.py` grep bilan qo'riqlaydi.

**Model override (`LLM_MODEL_*`).** `LLM_TASK_*` provider bilan modelni birga
beradi, `LLM_MODEL_ROUTE|SEARCH|TOOLS|DEEP|EMBED|IMAGE` esa **faqat model
nomini** almashtiradi — provider tanlash o'zgarmaydi. Model tanlangan
providerning `capabilities()` iga mos kelmasa router **aniq xato** tashlaydi
(tool'siz model `TOOLS` ga, chat modeli `EMBED`/`IMAGE` ga tushmaydi) — jimgina
eski modelga qaytish yo'q. `gemini-2.5-flash-lite`/`-flash`/`-pro` 2026-09'da
yangi mijozlarga yopilgan (`router.RETIRING`): istalgan `Task` shu modelga
tushsa (default yoki override orqali) startupda `llm.model_retiring`
ogohlantirishi chiqadi (avval faqat ROUTE edi — endi hammasi). Default endi
3.x (yuqoridagi jadval) — keyingi almashtirish ham avval o'lchov, keyin
`LLM_MODEL_*` bilan. Embedding tomonida `EMBED_MODEL` / `EMBED_DIMS`; model
almashsa `python -m app.main reembed --model X` (hech narsa o'chirmaydi,
qatorlar yonma-yon, qidiruv joriy modelni filtrlaydi) — batafsil `docs/llm.md`.

**"Yonma-yon" sxemada qo'riqlanadi** (migratsiya `0016`):
`message_embeddings` PK = `(message_id, model)`,
`retrieval_node_embeddings` PK = `(node_id, model)`. Kalitni bir ustunga
qaytarmang — u holda "joriy model bo'yicha yetishmayapti" tanlovi bilan
INSERT to'qnashadi (`UniqueViolation` sikli) va model almashtirish
qidiruvni **bo'sh** qoldiradi. Har bir yozuvchi
(`worker.embed_messages`, `worker.reembed_all`, `nodes.embed_nodes`,
`embed_batches._write_vectors`) `ON CONFLICT (id, model) DO NOTHING` bilan
yozadi; joriy model nomi **yagona manba** — `llm.embed_model_name()`
(`LLM_TASK_EMBED` → `LLM_MODEL_EMBED` → `EMBED_MODEL`).
`EMBED_DIMS` ≠ `embed_dim` bo'lsa har INSERT yiqiladi —
`python -m app.main check` va api/worker startupi buni aytadi.

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
