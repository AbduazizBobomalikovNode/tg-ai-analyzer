# Agent, tool'lar, yozish amallari

## Rejimlar (`chat_service.choose_mode`)

| Rejim | Qachon | Nima bo'ladi |
|---|---|---|
| `direct` | salomlashish / qisqa matn / aniq strategiya (`search/window/recent`) / sinxron chat yo'q | kontekst oldindan (`select_context`) → bitta LLM chaqiruv |
| `agent` | `auto` + sinxron chat bor + savol ma'lumot haqida; yoki UI'da "Agent" | tool sikli |
| `auto` | default | yuqoridagini o'zi tanlaydi |

## Read-only tool'lar (`services/tools.py`)

| Tool | Argumentlar | Natija |
|---|---|---|
| `list_chats` | — | sinxron chatlar (id, tur, msgs, admin, pinned) |
| `search_messages` | `chat?, query, limit≤30, days?` | hybrid qidiruv, ranked, `#id date author (views/re/fwd): text t.me/link` |
| `get_recent_messages` | `chat?, limit≤100` | oxirgilar |
| `get_message_context` | `chat?, message_id, radius≤15` | ±N xabar |
| `get_window_digest` | `chat?, days≤62` | kunlik digest'lar (kesh) |
| `get_chat_stats` | `chat?, days≤366, top_n≤10` | `analytics.chat_stats` JSON: post, ko'rish sum/avg/median, reaksiya, forward, o'sish (snapshot), kun/soat, media, top postlar, oldingi davr |

`chat` — DB id / @username / sarlavha bo'lagi; bo'lmasa UI'da pinned chat.
Natija ≤9k belgi, `<untrusted_data source="tool:…">` konvertida; xato — matn.

## Sikl (`services/agent.py`)

`Task.TOOLS`; ≤`AGENT_MAX_ITERATIONS` (5); iteratsiyada ≤4 tool (qolganiga
"skipped" natija); natijalar umumiy byudjeti `AGENT_TOOL_RESULT_TOKENS` (12k) —
oshsa kesiladi va modelga "javob ber"; oxirgi raundda tool berilmaydi; sikl
tool bilan tugasa majburiy tool'siz chaqiruv. Har chaqiruv `agent_actions`
(executed/failed). Natija: `tokens_in/out`, `iterations`, `tool_calls`,
`action_ids` → `conversation_messages.context`.

## Yozish tool'lari (`services/write_tools.py`)

`send_message(chat?, text, reply_to?, schedule_at?)`, `edit_message(chat?,
message_id, text)`, `forward_message(chat, from_chat, message_id, drop_author?)`,
`pin_message(chat?, message_id, silent?)`. Spec'lar agentga faqat akkauntda
`write_mode != read_only` chat bo'lsa beriladi. Chaqiruv → `build_proposal`
(validatsiya + `assert_writable`) → `agent_actions.proposed`; modelga "tasdiq
kutmoqda, yuborildi dema". `autonomous` chat → darhol `execute_action`.

## Tasdiqlash (`services/actions.py`, UI)

Javob ostida karta: tool, matn, chat, muddat → ✅ / ❌. `confirm_action`:
egalik → TTL (`WRITE_PROPOSAL_TTL_HOURS`) → `execute_action`: `assert_writable`
→ rate limit (`WRITE_RATE_PER_HOUR`) → `read_only` tekshiruvi → MTProto
(`send_message/edit_message/pin_message/forward_messages`) → `executed`
(`result_msg_id`) yoki `failed` (`error`). Per-chat rejim topbar'da (✍️);
`autonomous` — akkauntga bitta chat.

## Baholash

👍/👎 (`rating`), fon auto-judge (`Task.ROUTE`, savol+javob): `auto_relevance`,
`auto_usefulness`, `auto_grounded`, `auto_note`. Dashboard "ko'rib chiqish
kerak": 👎 yoki auto ≤2 yoki asossiz.
