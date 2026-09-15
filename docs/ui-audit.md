# Frontend va API auditi — redizayn oldidan (2026-09-16, `v2-deep-search`)

## Templates (`src/app/web/templates/`)

5 ta fayl, 4 sahifa: `base.html` (shell, `window.TGAI={strings,lang}`, `md.js` har
doim), `login.html` (telefon/kod/2FA + QR), `chat.html` (sidebar + topbar + xabarlar +
composer: ctx‑limit slider, mode, strategy, deep checkbox), `dashboard.html` (KPI, 3 chart,
jadvallar + alert CRUD formasi), `briefings.html` (`page-dash` qayta ishlatilgan).
Muammolar: lang‑switch va brand bloklari 4 joyda nusxa, macro/partial yo'q, `/briefings`
ga hech qayerdan havola yo'q, sozlamalar sahifasi yo'q (alert — dashboard'da, briefing —
briefings'da, write‑mode/autoreply — chat topbar'da). i18n: `pages.py::_strings()` prefiks
filtri (`web. auth.err. pool.err. chat.err. action.err. sync.err. alert.err.
briefing.err. error.`), butun lug'at har sahifaga.

## Static (`src/app/web/static/`)

`app.css` 38KB (310 token, light/dark faqat `prefers-color-scheme`, `[data-theme]`
yo'q, breakpoint 1100/860/480, safe‑area faqat sidebar'da), `chat.js` 38KB bitta IIFE
(663 qator), `md.js` 10KB (escape‑first, `safeUrl` http/https/mailto/tg, rasm faqat
`/api/images/<id>`, mermaid lazy `securityLevel: strict`, tema bir marta), `dashboard.js`
11KB (qo'lda SVG), `alerts.js` 8KB (o'z `esc` nusxasi), `login.js` 7.6KB, `briefings.js`
3.8KB, `vendor/mermaid.min.js` 2.57MB. ES modul yo'q; global `window.TGAI` (+
`TGAI_BRIEF_DAY`). `innerHTML` inventar: hammasi `esc()` yoki `md()` orqali — invariant
saqlanadi; bitta yumshoq nuqta: `chat.js:124` `p.image_url` `<img src>`ga sxema
tekshiruvisiz (`safeHref` bor, ishlatilmagan). Fetch wrapper 5 xil nusxa (401 redirect
ba'zilarida yo'q). SSE parser (`parseFrame`) to'g'ri (CRLF, multi‑line data, 2MB cap).
**Telegram Mini App integratsiyasi yo'q** (script, initData, theme, BackButton — hech biri).
Drawer breakpoint JS'da (`860`) va CSS'da alohida.

## API (UI ishlatadigani)

- Pages: `/`, `/login`, `/chat`, `/dashboard`, `/briefings[/{day}]`.
- Auth (`/api/auth/*`): phone, qr, qr/poll, resend, code, password, cancel, logout;
  `GET /api/me` → `{user, accounts[], max_accounts}`.
- TG (`/api/accounts/{id}/…`): `dialogs` (jonli MTProto), `dialogs/{peer}/messages`
  (UI ishlatmaydi), `chats` (DB: id, peer_id, title, type, username, is_admin,
  is_writable, write_mode, participants_count, sync_state, synced_total, total_estimate,
  progress, sync_error, last_sync_at, last_message_at), `sync`, `chats/{id}/sync`.
- Conversations: list/create/delete, `GET /{id}/messages`, `POST /{id}/messages`
  (mode auto|agent|direct|deep, **bloklovchi**, streaming yo'q), rate.
- Deep: `POST /api/deep/stream` (SSE plan|search|cluster|rerank|token|coverage|done|
  error), `POST /api/deep`.
- Actions: list, confirm, reject, write_mode PATCH. Content: images, autoreply GET/PUT,
  scheduled. Stats: overview, system, ingestion (faqat user darajasi). Alerts CRUD +
  events. Briefings list/day + settings GET/PUT.

## Yo'q narsalar (UI uchun kerak)

Chat avatarlari (ustun/fayl/endpoint/TL yo'q), bio/about (ustun yo'q; `GetFull*`
allowlist'da, faqat agent tool'i ishlatadi), a'zolar soni (bor, ko'rsatilmaydi), per‑chat
analitika (`analytics.chat_stats` faqat agent tool'i orqali — HTTP yo'q), sync progress
push (15s poll), UI uchun xabar qidiruvi (`/api/search` yo'q), chat qidiruvi server
tomonda yo'q (client `.includes()`), suhbat nomini o'zgartirish, pagination, sozlamalar
endpointlari (tema/til/default rejim), akkauntni uzish, `/api/me` da ism/avatar, jonli
xabar oqimi.

## DB ustunlari (`chats`)

`id, account_id, tg_peer_id, access_hash, type, title, username, is_writable, is_admin,
participants_count, write_mode, sync_state, synced_min_id, synced_max_id, synced_total,
last_sync_at, total_estimate, sync_error, last_message_at, link_prefix`. Yo'q: avatar,
about, is_verified/scam, unread_count, pinned/folder, last_message_text.

## Avatar uchun allowlist

`channels.GetFullChannel`, `users.GetFullUser`, `messages.GetFullChat` — READ'da bor.
`upload.GetFileRequest` — **yo'q** (deny‑by‑default), `photos.GetUserPhotos` — yo'q (kerak
emas, faqat tarix uchun). Telethon `download_profile_photo`: entity'dagi `photo` →
`InputPeerPhotoFileLocation` → `upload.GetFile` (chunk'lar); boshqa DC bo'lsa
`auth.Export/ImportAuthorization` (INTERNAL'da bor). **Minimal qo'shimcha:
`upload.GetFileRequest` (faqat o'qish).** Saqlash: `services/images.py` naqshi
(`DATA_DIR/…`, magic‑byte sniff, egalik tekshiruvi, `Cache-Control: private`).

## Backend prerekvizitlar (checklist)

- Allowlist `upload.GetFileRequest`; migratsiya `chats.about, avatar_*, is_verified,
  is_scam, is_fake, unread_count, last_message_text`; `services/avatars.py`; worker
  task (history bucket, FloodWait); `GET /api/accounts/{id}/chats/{chat_id}/avatar`.
- `_chat_out` kengaytmasi (about, avatar_url, unread, verified…).
- `GET …/chats/search?q=` (uz‑fold), `GET …/messages/search?q=&chat_id=`
  (`search.hybrid_search`, LLM'siz), `GET …/chats/{id}/stats?days=`
  (`analytics.chat_stats`), `GET …/chats/{id}/digests`.
- `POST /api/conversations/{id}/messages/stream` (SSE, `deep.py` framing), `PATCH
  /api/conversations/{id}` (title), pagination, per‑chat sync status.
- Sozlamalar guruhi (tema, til, default rejim/strategiya), `DELETE /api/accounts/{id}`,
  `/api/me` kengaytmasi.
- Mini App: `telegram-web-app.js` lokal, `initData` HMAC (keyingi bosqich), theme →
  token'lar, `BackButton`/`expand()`.
- Frontend: bitta `api()` helper, Jinja macro/partial, `[data-theme]` + toggle + mermaid
  re‑init, `chat.js` modullarga, `p.image_url` sxema tekshiruvi, safe‑area composer/
  topbar, `vh → dvh`, breakpoint bitta manba.
