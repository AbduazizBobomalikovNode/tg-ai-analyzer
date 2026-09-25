# Conversational eval suite — schema

Har bir `*.yaml` fayl bitta kategoriyaning senariylarini saqlaydi. Runner:
`evals/conv_runner.py` (qarang: shu papkadagi `../conv_runner.py` docstring).

Senariy = **bitta suhbat** (`POST /api/conversations`) + ketma-ket turn'lar.
Suhbat holati (tarix) turn'lar orasida saqlanadi — multi-turn moslashuv shu
orqali o'lchanadi.

```yaml
version: 1
scenarios:
  - id: prec-01                 # majburiy, global unikal, `<cat>-<nn>`
    title: "..."                # inson uchun
    category: precise           # absurd|precise|account|global|time|people|
                                # engagement|digest|multiturn|safety|mode|economy|
                                # daily
    locale: uz                  # uz|ru|en — so'rov tili (Accept-Language ham shu)
    priority: 1                 # 1 = live yugurishda majburiy, 2 = kengaytirilgan
    chat_hint: "Honadoshlar"    # ixtiyoriy: chat sarlavhasining bo'lagi.
                                # Runner uni /api/accounts/{acc}/chats/search
                                # orqali `{id, peer_id, title}` ga aylantiradi va
                                # `context` sifatida yuboradi (pinned chat).
    turns:
      - user: "..."             # foydalanuvchi matni
        mode: auto              # auto (agent/direct) | deep | exhaustive (ikkalasi
                                # ham deep_search quvuri, /api/deep, `mode` maydoni
                                # bilan; `exhaustive` tasdiqsiz faqat baho beradi)
        expect:
          tools_any: [search_messages, search_calendar]   # kamida bittasi chaqirilsin
          tools_all: [get_chat_stats]                     # hammasi chaqirilsin (ixtiyoriy)
          tools_none: [discover_channels, send_message]   # hech biri chaqirilmasin
          max_tool_calls: 4                               # bajarilgan chaqiruvlar shifti
          scope: chat           # chat | account | global | none
          scope_chat_hint: "aiogram"  # `scope: chat` uchun senariy `chat_hint`
                                # o'rniga shu bilan solishtiriladi — kontekst
                                # almashtiruvchi turn'lar uchun (multi-04 t2)
          must_reference_prior: false   # javob oldingi turn mavzusiga suyanishi shart
          must_change_context: false    # bu turn yangi tool/qidiruv qilishi shart
          keywords_any: ["..."]         # javob matnida (case-insensitive) biror biri
          keywords_none: ["..."]        # javobda bo'lmasligi kerak
          forbid_regex: ["\\+?\\d[\\d\\s-]{7,}"]  # javobda hech qaysi naqsh mos kelmasin
          must_propose_action: false    # `agent_actions(proposed)` paydo bo'lsin
          max_cost_usd: 0.05
          judge: "one-line rubric"      # LLM-judge (Task.ROUTE) uchun mezon

          # Faqat `mode: deep`/`exhaustive` uchun — `DeepResult.to_dict()`
          # (`deep_search.py`) dan `_normalize_deep` orqali keladi, hammasi
          # ixtiyoriy:
          direct: false                # `DeepResult.direct` (qidiruvsiz javob)
          intent: "greeting"           # `DeepResult.intent` — greeting|meta|chitchat
          intent_follow_up: "deeper"   # `DeepPlan.intent_follow_up` (`plan` hodisasi)
          needs_retrieval: true        # `DeepPlan.needs_retrieval`
          needs_confirm: true          # `DeepResult.needs_confirm` (`exhaustive`, tasdiqsiz)
          retrieval_kind: "window"     # `DeepPlan`/`Coverage.retrieval_kind` — topic|window|person
          direction: "out"             # `Coverage.direction` — ""|in|out|both
          window_stats: true           # `Coverage.window_stats` — sanoq yo'li (xabar tortilmaydi)
          server_rungs_used: true      # `Coverage.server_rungs_used`
          external_hits_min: 1         # `Coverage.external_hits` >= shu son
```

## `scope` semantikasi

Runner tool chaqiruvlaridagi `chat` argumentlaridan va `calls_log` dagi
`chat` meta'sidan foydalanib qamrovni aniqlaydi:

| scope     | talab                                                              |
|-----------|--------------------------------------------------------------------|
| `chat`    | barcha tool chaqiruvlari **bitta** chat bilan chegaralangan (odatda `chat_hint`) |
| `account` | chat argumentisiz (yoki ≥ 2 xil chat) lokal/akkaunt bo'ylab qidiruv |
| `global`  | `discover_channels` yoki `search_global` — akkaunt tashqarisi/bo'ylab |
| `none`    | hech qanday tool chaqirilmasligi kerak                              |

## Kutilmalar semantikasi

* Hamma `expect.*` maydonlari **ixtiyoriy**; berilmagani tekshirilmaydi.
* `tools_*` faqat **bajarilgan** chaqiruvlarga tegishli emas: `calls_log` da
  to'silgan (`seen_calls` / `MAX_CALLS_PER_TOOL`) chaqiruvlar ham bor
  (CLAUDE.md, "To'silgan chaqiruv"). Runner ikkalasini ham hisoblaydi va
  `blocked` sonini alohida qaytaradi; `max_tool_calls` faqat **bajarilgan**
  chaqiruvlarni sanaydi.
* `deep` turn'lar `calls_log` bermaydi (quvurda tool yo'q) — ular uchun
  `tools_*` e'tiborsiz, `scope` esa `context.clusters` / `context.citations`
  dagi chat id'lardan hisoblanadi.
* `judge` — bitta `Task.ROUTE` chaqiruvi; unga savol tarixi + javob beriladi,
  `{"pass": bool, "reason": "..."}` qaytadi. Judge **tavsiya**, xatoning yakuniy
  hukmi emas (CLAUDE.md, "Auto-baho").

## Xarajat

Live yugurish qimmat (real akkaunt, real model). `priority: 1` to'plami
≤ 60 normal + ≤ 10 deep turn'ga sig'adi; `priority: 2` — qo'lda
(`--priority 2`) yugurtiriladi.

Turn soni — byudjetning **yomon** o'lchovi. 2026-09-17 yugurishida tool'lar
tiklangandan keyin bitta turn $0.40 gacha chiqdi (agent iteratsiya shiftigacha
tool zanjirini davom ettirdi), to'plam esa kunlik `LLM_DAILY_TOKEN_BUDGET` ni
yeb qo'ydi: qolgan 38 turn `chat.err.budget` / `chat.err.rate_limited` bilan
qaytdi va o'lchov yarim qoldi. Shuning uchun `--max-cost-usd` (default `1.0`)
bor — jami xarajat shiftga yetsa qolgan senariylar yugurtirilmaydi va hisobot
sarlavhasida "to'liq yugurmadi" ogohlantirishi chiqadi. Yugurishdan oldin
kunlik byudjetda joy borligini tekshiring.
