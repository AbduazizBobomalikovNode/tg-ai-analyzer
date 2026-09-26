# Orkestrator kengaytmasi — `services/skills_ext.py`

`skills.register_extra()` import paytida `skills_ext.register(SKILLS)` ni
chaqiradi; ko'nikmalar planner katalogiga (`skills.catalog_text`) o'zi tushadi.
Yadro nomlari almashtirilmaydi. `skills_ext` `app.services.skills` ni modul
darajasida **import qilmaydi** (`Skill`/`SkillResult` chaqiruv paytida olinadi) —
aks holda `skills_ext` birinchi import qilinganda `register_extra` yarim
yuklangan modulni ko'rib, hech narsani ro'yxatga olmasdi.

| Ko'nikma | Argumentlar | Rejim | Karantin | Tasdiq | Terminal |
|---|---|---|---|---|---|
| `format_output` | `format` (table/chart/list), `chart` (bar/line/pie), `series` (ranking/per_day/facts/weekday/hour), `title` | fast | yo'q | yo'q | yo'q |
| `create_alert` | `kind` (keyword/semantic), `pattern`, `chat_id`, `cooldown_minutes` | fast | yo'q | har rejimda | ha |
| `set_briefing` | `hour`, `chat_ids` | fast | yo'q | har rejimda | ha |
| `remind` | `when` (ibora/ISO, zaxira), `text`, `job_ids` | fast | yo'q | har rejimda | ha |
| `cancel_reminder` | `job_ids` | fast | yo'q | har rejimda | ha |
| `send_result` | `chat_id` / `chat`, `text_from` (answer/facts) | fast | **ha** | har rejimda | yo'q (o'zi to'xtaydi) |

"Tasdiq" — `Skill.confirm_modes` (katalogda "asks for confirmation"). Bu
`agent_actions` tasdig'i: ko'nikma `needs_confirm` natijasini **qo'ymaydi**,
ya'ni skan baho kartasi chiqmaydi.

## `format_output` (`services/output_formats.py`)

LLM yo'q. Hafta kuni / soat taqsimoti (`stats(by=…)` → `wd_<kun>_n`, `hr_<HH>_n`,
`distribution_rows`) chat reytingidan **ustun** — aks holda "haftaning qaysi kuni …
jadval" da blok reyting bo'lib chiqardi. Raqam faqat `ctx.facts` dan (`stats`: `top{i}_n`, `total`, …) yoki
`count(*)` dan (`count_series` — oynadagi xabarlar lokal kun/hafta/oy
bo'yicha, `accounts.tz`). Chiqish:

- GFM jadval (raqamli ustun `---:`), yoki
- ` ```chart ` fence — `md.js::chartSvg` sxemasi:
  `{"type":"bar"|"line","title":…,"labels":[…≤60],"series":[{"name":…,"values":[…]}≤6]}`
  (`pie` so'ralsa `bar` yoziladi — md.js pie chizmaydi), yoki
- raqamli ro'yxat.

Blok `ctx.claims` ga `pre_rendered_block … end_pre_rendered_block` bilan
o'ralib qo'shiladi (sintez konvertiga tushadi) va `SkillResult.notes` da
`FORMAT_NOTE` (sintezga "o'zgartirmasdan ko'chir" satri) qaytadi.
**Orkestrator ulagichi:** `_finish_answer` da sintezdan keyin
`output_formats.ensure_blocks(answer, ctx.claims)` — model blokni tashlab
ketsa ham u javob oxiriga qo'shiladi. Chat sarlavhalari katakka
`untrusted.neutralize` + `|` qochirish bilan tushadi; xabar matni blokka
hech qachon tushmaydi.

## Avtomatlashtirish (`services/automation.py`, `services/scheduled.py`)

Oqim bitta, yozish amallari bilan bir xil:

```
skill → automation.build_* (kodda tahlil + tekshiruv, faqat o'qish)
      → write_tools.propose_automation → agent_actions(proposed)
      → Actions sahifasi: foydalanuvchi tasdig'i (actions.confirm_action)
      → actions.execute_action → _execute_automation → automation.execute
      → alerts.create_rule | briefing.apply_settings | scheduled.create_reminder
        | scheduled.cancel_reminders
```

- `write_tools.AUTOMATION_TOOL_NAMES` — `WRITE_TOOL_NAMES` ga **qo'shilmagan**
  (u Telegram yozish tool'lari; testlar qo'riqlaydi). Delete yo'q: bekor
  qilish `status='cancelled'`, briefingni o'chirish `briefing_enabled=false`.
- `autonomous` bu amallarga tegishli emas: `propose_automation` hech qachon
  bajarmaydi, `_execute_automation` esa `confirmed=False` ni
  `needs_confirm` bilan rad etadi.
- Taklif `args` i (nima yaratiladi + `summary` matni) append-only
  `agent_actions` da muzlaydi; bajarishda egalik qayta tekshiriladi:
  `args.account_id == run.account_id`, `account.user_id == run.user_id`,
  chat → akkaunt.
- Niyat **foydalanuvchi matnida** tekshiriladi (`ALERT_INTENT_RE`,
  `BRIEFING_INTENT_RE`, `REMIND_INTENT_RE`, `SEND_INTENT_RE`) — planner xato
  qilsa ham taklif yaratilmaydi.
- Kartada (`actions._preview`) `text = args.summary` — `/actions` va `/ask`
  kartalari o'zgarishsiz ko'rsatadi; `pages/actions.js` faqat ikonka qo'shdi.

### Eslatmalar

- Vaqt kodda: `scheduled.parse_due` (uz/ru/en, "ertaga 10 da", "2 soatdan
  keyin", "через час", "tomorrow at 8pm", "30 sentyabr 14:00",
  "30.09.2026"), **akkaunt tz** sida; o'tmish yoki >366 kun — rad. Planner
  `when` faqat zaxira.
- Matn: ajratuvchi `:` dan keyingi qism, bo'lmasa buyruq/vaqt so'zlarisiz qoldiq.
- Migratsiya `0025`: `scheduled_jobs.attempts/delivered_at/error` + qisman
  indeks `ix_scheduled_jobs_due (type, run_at) WHERE status='pending'`.
- Cron `worker.tasks.deliver_reminders` — har daqiqa. `due_ids` → har biri
  o'z tranzaksiyasida `deliver_one`: push'dan **oldin** shartli UPDATE
  (`pending → sending`, `delivered_at`, `attempts+1`, `RETURNING`) — ikkinchi
  tik yubormaydi; xato → `pending`, `REMINDER_MAX_DELIVERY_ATTEMPTS` (3) dan
  keyin `failed` (`gave_up_after_N`). Push **faqat** `bot.push.send_html`
  orqali egasining `users.tg_user_id` siga, `neutralize` + `html.escape`.
  Jim soatlar qo'llanmaydi (vaqtni foydalanuvchi o'zi aytgan).

## `send_result`

"topib Y ga yubor" / "shu xulosani guruhga tashla". Rejaning **oxirgi**
qadami; `stop=True` qaytaradi, shuning uchun `_finish_terminal` javobni
yozadi va `done.actions` da karta chiqadi (ikkinchi sintez yo'q).

Matn manbasi:
- `text_from="facts"` — kodda hisoblangan faktlar + format bloklari (chart'siz);
- joriy turn'da dalil/da'vo bo'lsa — **bitta** `DS.synthesize` (yagona
  dalil konverti, `taint.add`), shu matn ham javob, ham yuboriladigan matn;
- aks holda — suhbatning oxirgi assistant javobi (DB'dan to'liq).

Joriy turn qidirib hech narsa topmasa (`ranked` bo'sh va `chat_total`/`total` = 0 —
`turn_found_nothing`) taklif **yaratilmaydi**, javob oxiriga `send_empty` izohi
qo'shiladi: "topilmadi" matni uchinchi odamga yuborilmasin.

Manzil (`skills_ext.send_target`): (1) suhbat holatidagi yechim
(`conv_state.resolve_name` — "Otabekka yubor", "unga yubor"), planner boshqa
shu nomli chat id sini bergan bo'lsa ham; (2) planner chat id si; (3) nom ≥2
shaxsiy chatga **teng** mos kelsa — `send_which` bilan so'raladi, taklif yo'q;
(4) aks holda nom `write_tools` ga.

`clean_for_send`: sitata tokenlari, qamrov/qadamlar izohi va ```chart
olib tashlanadi, ≤4000 belgi. Taklif `write_tools.propose_or_execute
("send_message")` — `assert_writable`, peer guard, `read_only`,
`autonomous` qoidalari o'zgarmagan. Manzil aniq aytilmasa — so'raladi.

## Orkestrator bilan ma'lum chegaralar (ulagich egasi uchun)

1. `apply_guards` G1/G2 (`DIRECT_FOLLOW_UPS` → `answer_direct`,
   `needs_retrieval=False` → `reuse_prior`) terminal tekshiruvidan **oldin**
   ishlaydi: planner "shu xulosani guruhga tashla" ni `reformat`/`explain`
   deb belgilasa, `send_result` / avtomatlashtirish qadami tashlanadi.
   Ularni G1/G2 dan ozod qilish kerak.
2. `parse_steps` `WRITE_INTENT_RE` ni faqat `write_propose` ga qo'llaydi —
   bizniki o'z darvozasiga ega (ko'nikma ichida), bu yetarli.
3. `ensure_blocks` ulagichi (`_finish_answer`) — yuqoriga qarang.
4. `mode_note` hozir faqat qat'iy ro'yxatdan yig'iladi; `res.notes` dagi
   `FORMAT_NOTE` sintezga tushishi uchun ularni ham qo'shish kerak.
