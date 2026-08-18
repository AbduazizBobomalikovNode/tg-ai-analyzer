# HTTP API

Bazaviy URL — `WEBAPP_BASE_URL` (lokal `http://localhost:8080`). Hamma `/api/*`
JSON. Auth — imzolangan cookie (`tgai_session`), login oqimidan keyin beriladi.
**O'zgartiruvchi so'rovlar** (`POST/PATCH/DELETE`) `X-Requested-With: fetch`
header'ini talab qiladi (CSRF), aks holda `403 {"detail":{"code":"csrf"}}`.
Xato formati: `{"detail": {"code": "<i18n kalit>", ...}}` — kalitlar
`app/i18n/locales/*.json` da (`auth.err.*`, `pool.err.*`, `chat.err.*`,
`action.err.*`, `sync.err.*`).

## Sahifalar
`GET /` → `/chat` yoki `/login` · `GET /login` · `GET /chat` · `GET /dashboard` ·
`?lang=uz|ru|en` (cookie'ga yoziladi) · `GET /health`.

## Auth (telefon → kod → 2FA)
| Metod | Yo'l | Tana | Javob |
|---|---|---|---|
| POST | `/api/auth/phone` | `{phone}` | `{flow_id, status:"code_sent", code_type, code_length, timeout}` (IP rate limit) |
| POST | `/api/auth/code` | `{flow_id, code}` | `{status:"needs_2fa"}` yoki `{status:"done", account_id}` + cookie |
| POST | `/api/auth/password` | `{flow_id, password}` | `{status:"done", account_id}` + cookie |
| POST | `/api/auth/cancel` | `{flow_id}` | |
| POST | `/api/auth/logout` | | cookie o'chadi (session saqlanadi) |
| GET | `/api/me` | | `{user, accounts[], max_accounts}` |

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
| GET | `/api/conversations` | `{items:[{id,title,account_id,updated_at}]}` |
| POST | `/api/conversations` | `{account_id?, title?}` |
| GET | `/api/conversations/{id}/messages` | `{conversation, items:[…, actions?]}` |
| DELETE | `/api/conversations/{id}` | |
| POST | `/api/conversations/{id}/messages` | `{text, mode?: auto|agent|direct, account_id?, deep?, context?: {account_id, peer_id, limit?, strategy?: auto|recent|search|window}}` → `{text, model, provider, tokens_in, tokens_out, latency_ms, cost_usd, context, actions[]}` |
| POST | `/api/conversations/{id}/messages/{mid}/rate` | `{rating: 1|-1|0, comment?}` |

`context` javobda: `strategy`, `source` (`db`/`live`/`tools`), `est_tokens`,
`truncated`, `hits`, agent rejimida `tools`, `tool_calls`, `iterations`, `run_id`,
`action_ids`.

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
