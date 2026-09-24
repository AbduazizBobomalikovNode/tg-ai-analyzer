# HTTP API

Bazaviy URL — `WEBAPP_BASE_URL` (lokal `http://localhost:8080`). Hamma `/api/*`
JSON. Auth — imzolangan cookie (`tgai_session`), login oqimidan keyin beriladi.
**O'zgartiruvchi so'rovlar** (`POST/PATCH/DELETE`) `X-Requested-With: fetch`
header'ini talab qiladi (CSRF), aks holda `403 {"detail":{"code":"csrf"}}`.
Xato formati: `{"detail": {"code": "<i18n kalit>", ...}}` — kalitlar
`app/i18n/locales/*.json` da (`auth.err.*`, `pool.err.*`, `chat.err.*`,
`action.err.*`, `sync.err.*`).

## Sahifalar
`GET /` → `/login` (sessiya yo'q) · `/onboarding` (tugatilmagan) · `/today` ·
`GET /login` · `GET /onboarding` · `GET /today` · `GET /chats[/{id}]` ·
`GET /ask[/{id}]` (`GET /chat` → 302 `/ask`) · `GET /alerts` · `GET /actions` ·
`GET /settings` · `GET /dashboard` · `GET /briefings[/{day}]` ·
`?lang=uz|ru|en` (cookie'ga yoziladi) · `GET /health`.

## Auth (telefon → kod → 2FA)
| Metod | Yo'l | Tana | Javob |
|---|---|---|---|
| POST | `/api/auth/phone` | `{phone}` | `{flow_id, status:"code_sent", code_type, code_length, timeout}` (IP rate limit) |
| POST | `/api/auth/code` | `{flow_id, code}` | `{status:"needs_2fa"}` yoki `{status:"done", account_id}` + cookie |
| POST | `/api/auth/password` | `{flow_id, password}` | `{status:"done", account_id}` + cookie |
| POST | `/api/auth/cancel` | `{flow_id}` | |
| POST | `/api/auth/logout` | | cookie o'chadi (session saqlanadi) |
| POST | `/api/auth/telegram` | `{init_data}` | Mini App: `initData` HMAC tekshiriladi → `{status:"done", user_id}` + cookie. `403 auth.err.not_linked` — bu Telegram akkaunt ulanmagan (**akkaunt yaratmaydi**, QR login kerak); `400 auth.err.init_data` / `init_data_expired`; IP rate limit |
| GET | `/api/me` | | `{user: {id, tg_user_id, username, display_name, avatar_url, prefs}, accounts[{…, chats_total, chats_synced}], max_accounts}` |

`done` bo'lganda `sync_account` navbatga tushadi (Redis bo'lsa).

## Telegram (o'qish)
| Metod | Yo'l | Izoh |
|---|---|---|
| GET | `/api/accounts/{id}/dialogs?limit&refresh` | jonli MTProto dialoglar |
| GET | `/api/accounts/{id}/dialogs/{peer_id}/messages?limit` | jonli oxirgi xabarlar (≤200) |
| GET | `/api/accounts/{id}/chats` | DB registry + sync progress (`progress`, `sync_state`, `write_mode`, …) |
| POST | `/api/accounts/{id}/sync` | to'liq ingestion (navbat) |
| POST | `/api/accounts/{id}/chats/{chat_id}/sync` | bitta chat |
| PATCH | `/api/accounts/{id}/chats/{chat_id}/write_mode` | `{mode: read_only|write_with_confirm|autonomous}` |

## AI chat
| Metod | Yo'l | Tana / javob |
|---|---|---|
| GET | `/api/conversations?limit=1..100&before=<ISO updated_at>` | `{items:[{id,title,account_id,updated_at}], next_before}` — kursorli sahifalash (`next_before` null bo'lsa oxiri) |
| POST | `/api/conversations` | `{account_id?, title?}` |
| PATCH | `/api/conversations/{id}` | `{title}` (≤120, zararsizlantiriladi) → `{id, title, account_id}` |
| GET | `/api/conversations/{id}/messages?limit=1..500&before=<msg id>` | `{conversation, items:[…, actions?], next_before}` — **eng yangisi oxirida**, `before` dan eski qatorlar |
| DELETE | `/api/conversations/{id}` | |
| POST | `/api/conversations/{id}/messages` | `{text, mode?: auto|agent|direct|deep, account_id?, deep?, context?: {account_id, peer_id, limit?, strategy?: auto|recent|search|window}}` → `{conversation_id, user_message_id, assistant_message_id, text, model, provider, tokens_in, tokens_out, latency_ms, cost_usd, context, actions[]}` |
| POST | `/api/conversations/{id}/messages/stream` | o'sha tana → **SSE** (quyida) |
| POST | `/api/conversations/{id}/messages/{mid}/regenerate` | `{context?, mode?, deep?, account_id?, stream?}` — oxirgi user turn'ini qayta yugurtiradi (eski javob o'chadi, matn saqlangan savoldan olinadi). Javob `/messages` bilan bir xil; `stream: true` → SSE. `409 chat.err.not_last` — bu oxirgi turn emas |
| POST | `/api/conversations/{id}/messages/{mid}/rate` | `{rating: 1|-1|0, comment?}` |

`context` javobda: `strategy`, `source` (`db`/`live`/`tools`), `est_tokens`,
`truncated`, `hits`, agent rejimida `tools`, `tool_calls`, `iterations`, `run_id`,
`action_ids`.

### SSE: `POST /api/conversations/{id}/messages/stream`

Ramka `POST /api/deep/stream` bilan **bir xil** (`event: <nom>` + `data: <json>`
+ bo'sh satr) — `EventSource` emas, `fetch` + `ReadableStream`; CSRF odatdagidek
`X-Requested-With: fetch`. Auth (401) va CSRF (403) oqim boshlanishidan oldin,
odatdagi HTTP status bilan; **byudjet/shift xatolari esa `error` hodisasi
bo'lib** keladi (status allaqachon 200).

| Hodisa | `data` |
|---|---|
| `status` | `{stage: "thinking"｜"tool"｜"writing", tool?: "<tool nomi>"}` |
| `token` | `{text: "<~400 belgili bo'lak>"}` — hammasi ketma-ket qo'shiladi |
| `done` | `{conversation_id, user_message_id, assistant_message_id, text, model, provider, tokens_in, tokens_out, cost_usd, latency_ms, context, actions: [...]}` |
| `error` | `{code: "chat.err.<x>", detail: "<qisqa>"}` |

Tartib: `thinking` → (`tool`)* → `writing` → `token`* → `done`. LLM fasadida
hozircha rost oqim yo'q, `token` bo'laklari tayyor javobdan kesiladi —
shartnoma oqim qo'shilganda ham o'zgarmaydi. Saqlash oqimsiz yo'l bilan
**aynan bir xil** (yagona yozuvchi `chat_service.send_message`).

### SSE: `POST /api/deep/stream`

Tana: `{conversation_id, question, mode?: fast|deep|exhaustive, deep?, confirm?}`.
Rejim — **qopqoq**, buyruq emas: u qidiruv qanchalik chuqur bora olishini
aytadi, qidirish kerakligini emas.

| Hodisa | `data` |
|---|---|
| `stages` | `{stages: ["plan", …], mode, direct?}` — **birinchi** hodisa: shu turn'da yuguradigan bosqichlar. Reja tanlovdan oldin kelgani uchun ro'yxat bir marta kengayishi mumkin (`["plan"]` → to'liq ro'yxat) |
| `plan` | `{standalone_question, intent, intent_follow_up, needs_retrieval, retrieval_kind, inbox, window_from_question, window_kind, window_tz, since, until, chats, subqueries, …}` |
| `estimate` | `{mode, buckets, weeks, range, est_messages, est_cost_usd, est_total_usd, ceiling_usd, retrieval_kind, window_from_question}` — **faqat** kaskad/oyna rejimida va **faqat** qidiruv kerak bo'lganda |
| `buckets`/`pull`/`filter`/`select`/`map`/`reduce` | kaskad progressi (`scale_search`) |
| `search`/`cluster`/`rerank` | fan-out (`fast`) progressi |
| `token` | `{text}` — javob bo'laklari |
| `coverage` | qamrov (`Coverage.to_dict()`) — qidiruvsiz turn'da **yuborilmaydi** |
| `done` | `{citations, invalid_citations, cost_usd, tokens_in, tokens_out, model, provider, latency_ms, assistant_message_id, conversation_id}`; `{needs_confirm: true, estimate}` — tasdiq kutilyapti; `{direct: true, intent}` — qidiruvsiz javob |
| `error` | `{code: "chat.err.<x>", detail}` |

Uch ssenariy:

* **qidiruvsiz** (`salom`, "sen nimalar qila olasan", "2-bandni tushuntir"):
  `stages ["plan","write"]` → `plan` → `token`* → `done {direct:true,intent}`.
  `estimate` yo'q, `scan_runs` qatori yo'q, deep shift sloti **qaytariladi**;
* **tasdiq** (`exhaustive`, `confirm:false`): `stages` → `plan` → `estimate` →
  `done {needs_confirm:true}` (slot qaytariladi, `scan_runs.stopped_by =
  "estimate_only"`);
* **to'liq yugurish**: `stages` → `plan` → (`estimate` → kaskad | fan-out) →
  `token`* → `coverage` → `done`.

`plan.retrieval_kind` quvurning **shaklini** aytadi:

* `topic` (sukut) — mavzu skani: kaskad (`deep`/`exhaustive`) yoki fan-out
  (`fast`), bosqichlar o'zgarmagan;
* `window` — vaqt bilan chegaralangan ro'yxat ("bugun menga kimlar yozgan",
  "kecha qaysi chatlar faol edi"). Bosqichlar: `stages ["plan","estimate",
  "pull","select","map","write"]` — bucket rejasi, tirsak va mavzu filtri
  **yo'q**. `coverage` da `retrieval_kind`, `inbox` va `window
  {from,to,tz,total,taken}` bo'ladi ("oynada M xabardan N ko'rildi").
  `exhaustive` bu yo'lda tasdiqni **faqat** skan bahosi `$0.05` dan oshsa
  yoki oyna qopqoqqa sig'masa so'raydi;
* `person` — nomi aytilgan odam/chat; hozircha `topic` quvuri bilan yuguradi.

Oyna (`since`/`until`) savolning o'zidan **kodda** olinadi
(`services/timewin.py`, akkaunt mintaqasida): planner bo'sh qoldirsa yoki
kengroq bersa kod qo'yadi, torroq bergani qoladi. `window_from_question`
shuni bildiradi — UI "vaqt oralig'ini ko'rsating" ogohlantirishini aynan shu
bayroqqa qarab yashiradi. `since: null` + `window_kind: "all_time"` —
"butun davr", ya'ni quyi chegara yo'q.

## Sozlamalar va akkaunt (ui/R1b)
| Metod | Yo'l | Tana / javob |
|---|---|---|
| GET | `/api/settings/prefs` | `{prefs: {theme: system｜light｜dark, locale: ""｜uz｜ru｜en, default_mode, default_strategy, deep_default: bool, density: comfortable｜compact, onboarding_done: bool, pinned_chats: [id], recent_chats: [id]}}` |
| PUT | `/api/settings/prefs` | qisman JSON → merge qilingan `{prefs}`. **Faqat tanilgan kalitlar**, noma'lum/buzuq qiymat jimgina tashlanadi (422 emas) |
| DELETE | `/api/accounts/{id}` | akkauntni uzadi → `{status:"revoked", id}`. Saqlangan MTProto session o'chadi, status `revoked`, joblar bekor qilinadi; **ma'lumot qoladi**. `auth.LogOut` chaqirilmaydi (DENIED) |
| GET | `/api/accounts/{id}/sync/status` | `{running: bool, chats: [{chat_id, state, progress, synced_total, total_estimate, error}], counts: {<state>: n}, poll_seconds}` — yengil (dialoglar RPC'si yo'q), 5 s poll uchun. `chats` = hozir `running`/`failed` bo'lganlar (≤50) |
| GET | `/api/accounts/{id}/sync/stream` | SSE: har 2 s `status` (yuqoridagi tana), sync tugagach `done {running:false}`. `GET` — CSRF kerak emas, `EventSource` ham ishlaydi |

`prefs.locale` to'ldirilgan bo'lsa sahifa tili **shundan** olinadi (`?lang=` va
`lang` cookie mantiqi o'z joyida qoladi; aniq `?lang=` bir safarga ustun).

## Static va kesh
`asset("app.css")` (Jinja global) → `/static/app.css?v=<sha1[:10]>`. `v=` bo'lgan
so'rov `Cache-Control: public, max-age=31536000, immutable`, versiyasiz —
`no-cache`. HTML sahifalar: `Cache-Control: no-cache` + `ETag` (mos kelsa 304).
`/static/vendor/telegram-web-app.js` — lokal nusxa, CDN yo'q (`vendor/VENDOR.md`).

## Yozish amallari
| Metod | Yo'l | Izoh |
|---|---|---|
| GET | `/api/actions?status=proposed&run_id&limit` | foydalanuvchining amallari |
| POST | `/api/actions/{id}/confirm` | bajaradi → `{status: executed|failed, result_msg_id, error}`; xatolar `409 wrong_status`, `410 expired`, `429 rate_limited`, `403 blocked/read_only` |
| POST | `/api/actions/{id}/reject` | |

## Dashboard
`GET /api/stats/overview?days=7|30|90` — `totals` (so'rovlar, tokenlar, xarajat,
latency, qoniqish, auto-baho), `daily[]`, `models[]`, `strategies[]`,
`top_chats[]`, `review[]`, `actions{}`. `GET /api/stats/ingestion` — akkaunt/chat/
xabar/embedding/snapshot holati.

## Kontent (7-bosqich)
| Metod | Yo'l | Izoh |
|---|---|---|
| GET | `/api/images/{id}` | egasiga rasm fayli (private cache) |
| GET/PUT | `/api/accounts/{id}/chats/{chat_id}/autoreply` | qoida `{enabled, trigger, keywords, instructions, max_per_hour, quiet_from, quiet_to}` |
| GET | `/api/accounts/{id}/chats/{chat_id}/scheduled` | Telegram-side rejalashtirilgan postlar |
