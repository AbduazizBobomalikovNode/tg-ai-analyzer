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
`calls_executed`, `stop_reason`, `action_ids` → `conversation_messages.context`.

Yugurish bo'yicha ikkita qo'shimcha qopqoq (jonli o'lchovdan,
`evals/runs/conv-20260917-144833.md`: 24 turn, $2.71, turnda 6-11 bajarilgan
chaqiruv):

* `AGENT_MAX_EXECUTED_CALLS` (6) — **bajarilgan** (to'silmagan) chaqiruvlar;
* `AGENT_MAX_COST_USD` (0.15) — `llm/pricing.estimate_cost` bo'yicha yig'ma taxmin.

Ikkalasida ham: qolgan chaqiruvlar bajarilmaydi, "bor narsa bilan javob ber"
user turn'i qo'shiladi va keyingi LLM chaqiruvi **tool'siz** ketadi.
`stop_reason` = `answered | call_budget | cost_budget | iterations`.

Uchinchi qopqoq — takroriy qidiruv: lokal `search_messages` shu so'zlar bilan
≥1 topilma bergan bo'lsa, aynan o'sha so'rov `search_telegram` / `search_global`
bilan serverda **takrorlanmaydi** (lokal 0 bergan bo'lsa — ochiq, o'sha yerda
haqiqatan yangi ma'lumot bor).

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

## Kontent tool'lari (7-bosqich)

| Tool | Nima |
|---|---|
| `generate_image(prompt, style?)` | Gemini image → `/api/images/<id>`; modelga markdown `![..](url)`; keyin `send_message(image_id)` |
| `list_scheduled_messages(chat?)` | Telegram-side rejalashtirilganlar (READ) |
| `send_message(..., image_id?, schedule_at?)` | rasm bilan (caption ≤1024) va/yoki rejalashtirilgan — baribir taklif |

Auto-reply (`services/autoreply.py`): qoida → worker (5 daq) → `draft_reply`
(`AUTOREPLY_PROMPT`, `SKIP` mumkin) → `write_tools.propose_or_execute(send_message,
reply_to)`. Soatiga `max_per_hour`, jim soatlar, faqat begona xabarlar, boshlang'ich
nuqta — qoida yaratilgan paytdagi eng yangi xabar.
