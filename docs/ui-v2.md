# UI v2 — dizayn spetsifikatsiyasi

Manba: `docs/ui-research.md` (tadqiqot) + `docs/ui-audit.md` (joriy holat).
Bu hujjat — **shartnoma**: token nomlari, sinf nomlari, modul API'lari va
komponent holatlari shu yerda qayd etilgan. Kod bilan ajralib ketsa —
hujjat emas, kod noto'g'ri (yoki hujjatni yangilang).

Stack o'zgarmaydi: FastAPI + Jinja + vanilla ES modullar, bundler yo'q, CDN
yo'q, `static/md.js` yagona Markdown renderer.

Bosqich: **P0 poydevor + P1 karkas** bajarildi. P2 (AI javob yuzasi), P3
(chat ro'yxati), P4 (analitika/onboarding) — keyingi raund.

---

## 1. Axborot arxitekturasi va marshrutlar

```
/            → akkaunt yo'q → /login; onboarding tugamagan → /onboarding; aks holda /today
/login       ✅ QR-first onboarding
/ask         ✅ AI suhbat (3-panel, oqim, deep, sitatalar, amal kartalari)
/ask/{id}    ✅ suhbatga chuqur havola
/chat        ↪️ 302 → /ask (eski havolalar uchun; autentifikatsiyadan keyin)
/dashboard   ✅ token/narx/latency/sifat/sync (alertlar → /alerts)
/briefings   ✅ kunlik briefing
/today       ✅ uy: briefing + alert lentasi + "so'rang" + sync/plitalar
/chats       ✅ chat ro'yxati (mobil to'liq sahifa, desktop 2-panel)
/chats/{id}  ✅ chat tafsiloti
/alerts      ✅ alert qoidalari (karta + sheet forma) va hodisalar
/actions     ✅ tasdiq kutayotgan yozish amallari (rail badge)
/settings    ✅ akkaunt, write-mode, auto-reply, jim soatlar, til, maxfiylik
/onboarding  ✅ 4 qadam: QR → ishonch kartasi → sync progress → 3 savol
```

**R2 dan keyin kutilayotgan marshrut qolmadi** — IA dagi hamma sahifa jonli.
Marshrutlar ikki joyda e'lon qilinadi (mexanizm saqlanadi: kelajakdagi yangi
sahifa yoqilgunicha `none`/`null` bo'lib turadi va o'lik havola bermaydi):

| Joy | Nima uchun |
|---|---|
| `templates/_shell.html` → `ROUTES` | navigatsiya: `none` bo'lsa element ko'rinadi, lekin `<a>` emas (`is-disabled`, `aria-disabled`) |
| `templates/base.html` → `window.TGAI.routes` | ⌘K palitrasi va JS navigatsiya |

Yangi sahifani yoqish uchun **shu ikki xaritani** o'zgartiriladi, boshqa hech
nimani. `tests/test_web.py::test_routes_map_has_no_dead_links` har bir `null`
bo'lmagan qiymat haqiqatan ochilishini **va** IA dagi to'qqizta sahifa
(`today, chats, ask, alerts, actions, settings, dashboard, briefings,
onboarding`) jonli ekanini tekshiradi. Shablonli qiymat (`chatDetail:
"/chats/{id}"`) — `lib/palette-cmd.js` `{id}` ni almashtiradi.

---

## 2. Karkas (layout)

Yagona breakpoint: **`--bp-mobile: 860px`** (`css/tokens.css`).
CSS media query'da 860px yozilgan; JS uni `lib/store.js::mobileQuery()`
orqali **o'sha o'zgaruvchidan o'qib** `matchMedia` quradi — ikkalasi bir
manbadan. Ikkinchi chegara `--bp-wide: 1200px` (3-panel).

```
.shell (column, 100svh)
  .shell-row (flex: 1)
    .rail            ≥861px, 64px
    .shell-main (column)
      [topbar]
      .shell-body    data-pane="list" | "content"
        .pane-list   336px (861–1199: 300px)
        .pane-content
  .tabbar            ≤860px, 56px + safe-area
```

Tab bar **`position: fixed` emas** — oqimdagi flex element. Shu sababli
hech qayerda "kontent tab bar ostida qolmasin" deb `padding-bottom`
hisoblanmaydi va klaviatura ochilganda sakramaydi.

| Kenglik | Panel soni | Navigatsiya |
|---|---|---|
| ≤860 | 1 (`data-pane` tanlaydi) | pastki tab bar: Today · Chats · Ask · Alerts · More |
| 861–1199 | 2 (ro'yxat + kontent) | 64px rail |
| ≥1200 | 3 (rail + ro'yxat + kontent, `max-inline-size: 1100px`) | 64px rail |

### Xavfsiz zona va klaviatura

- `--sa-top/-bottom/-left/-right` — `env(safe-area-inset-*)`; Telegram
  ichida `--tg-content-safe-area-inset-*` ustun.
- `--kb` — `visualViewport` dan hisoblanadi (`lib/theme.js::watchKeyboard`).
  WebKit 217754: klaviatura ochilganda `safe-area-inset-bottom` **kichraymaydi**,
  ya'ni composer klaviatura ostida qoladi.
- `--pad-bottom: max(var(--sa-bottom), var(--kb))` — composer, tab bar va
  toast shuni ishlatadi.
- Top bar: `padding-block-start: calc(<pad> + var(--sa-top))`.

---

## 3. Ekranlar (ASCII wireframe)

### 3.1 Login / onboarding

```
MOBIL (≤860)                        DESKTOP (≥861)
┌──────────────────────────┐        ┌───────────────────────────────────────────┐
│ safe-area                │        │                                           │
│ ┌──────────────────────┐ │        │        ┌───────────────────────────┐      │
│ │ ✦ tg-ai   UZ RU EN ☾ │ │        │        │ ✦ tg-ai      UZ RU EN  ☾  │      │
│ │ Telegram akkauntini  │ │        │        │ Telegram akkauntini ulang │      │
│ │ ulang                │ │        │        │                           │      │
│ │ ┌──────────────────┐ │ │        │        │ ┌───────────────────────┐ │      │
│ │ │ +998 90 123 4567 │ │ │        │        │ │ +998 90 123 45 67     │ │      │
│ │ └──────────────────┘ │ │        │        │ └───────────────────────┘ │      │
│ │ [  Kod yuborish    ] │ │        │        │ [    Kod yuborish       ] │      │
│ │ ──── yoki ────       │ │        │        │ ──────── yoki ─────────   │      │
│ │ [  QR bilan kirish ] │ │        │        │ [   QR bilan kirish     ] │      │
│ │ 🔒 kod faqat shu     │ │        │        │ (lock) kod faqat shu      │      │
│ │    sahifada          │ │        │        │        sahifada           │      │
│ └──────────────────────┘ │        │        └───────────────────────────┘      │
└──────────────────────────┘        └───────────────────────────────────────────┘
```

QR bosilganda kartaning o'zi QR ekraniga almashadi (server chizgan
`data:` SVG). Login sahifasida rail/tab bar **yo'q** — foydalanuvchi hali
kirmagan.

### 3.2 Chats (ro'yxat)

```
MOBIL                               DESKTOP (3-panel)
┌──────────────────────────┐        ┌────┬──────────────────┬───────────────────┐
│ Chatlar         ⌕   ⋯    │        │ ✦  │ Chatlar      ⌕   │ Ish chati         │
├──────────────────────────┤        │────│ [All][Ch][Gr][Pr]│ 12k xabar · 2 soat│
│ [All][Chan][Grp][Priv]   │        │ ⌂  ├──────────────────┤───────────────────│
├──────────────────────────┤        │ ▣  │ ◐ Ish chati      │ ┌───────────────┐ │
│ ◐ Ish chati       ▁▃▅▂   │        │ ✦  │   12k · 2 soat ▁▃│ │ tahlil, grafik│ │
│   12k · 2 soat oldin  ●  │        │ ⌁  │ ◑ Oila           │ │               │ │
│ ◑ Oila            ▁▁▃▁   │        │ ✓  │   340 · kecha    │ └───────────────┘ │
│   340 · kecha         ●  │        │ ▤  │ ◒ Kanal          │                   │
│ ◒ Kanal #tech     ▅▇▆▃   │        │────│   sync yo'q      │                   │
│   sync yo'q       [sync] │        │ ⌘  │                  │                   │
├──────────────────────────┤        │ ☾  │                  │                   │
│ ⌂    ▣    ✦    ⌁    ☰   │        │ ⋯  │                  │                   │
└──────────────────────────┘        └────┴──────────────────┴───────────────────┘
```

Qator 64px: avatar 44px (rasm yoki inisiallar peer rangida) → sarlavha +
subtitle → o'ng ustun (tur, 14 kunlik sparkline, sync nuqtasi).
`content-visibility: auto` + `contain-intrinsic-size: 0 64px`.

### 3.3 Chat tafsiloti

```
MOBIL                               DESKTOP
┌──────────────────────────┐        ┌────┬──────────────────┬───────────────────┐
│ ‹  Ish chati        ⋯    │        │ ✦  │ Chatlar      ⌕   │ ‹ Ish chati    ⋯  │
├──────────────────────────┤        │ ⌂  │ ◐ Ish chati ◀    ├───────────────────┤
│   ╭────╮                 │        │ ▣  │ ◑ Oila           │  ╭────╮ Ish chati │
│   │ IC │ Ish chati       │        │ ✦  │ ◒ Kanal          │  │ IC │ @ishchat  │
│   ╰────╯ @ishchat        │        │ ⌁  │                  │  ╰────╯ guruh·48  │
│  guruh · 48 a'zo         │        │ ✓  │                  │  bio 2 qator…     │
│  bio 2 qator + ko'proq   │        │ ▤  │                  │  [synced·12k·2h]  │
│  [ synced · 12k · 2h ]   │        │────│                  │  Ask Sync Alerts  │
│  [Ask][Sync][Alerts]     │        │ ⌘  │                  │  ┌──────┬──────┐  │
│  ┌──────────┬─────────┐  │        │ ☾  │                  │  │faollik│ top  │  │
│  │ faollik  │  top    │  │        │ ⋯  │                  │  └──────┴──────┘  │
├──────────────────────────┤        │    │                  │                   │
│ ⌂    ▣    ✦    ⌁    ☰   │        └────┴──────────────────┴───────────────────┘
└──────────────────────────┘        Mobilda: push (BackButton = history.back)
```

### 3.4 Ask (AI suhbat)

```
MOBIL                               DESKTOP
┌──────────────────────────┐        ┌────┬───────────────────────────────────────┐
│ Suhbat         ⌕  ✎  ⋯   │        │ ✦  │ Suhbat          ctx: Ish chati  ⌕ ⋯   │
├──────────────────────────┤        │ ⌂  ├───────────────────────────────────────┤
│  ┌────────────────────┐  │        │ ▣  │      ┌─────────────────────────────┐  │
│  │ savol              │  │        │ ✦◀ │      │ savol                       │  │
│  └────────────────────┘  │        │ ⌁  │      └─────────────────────────────┘  │
│ ┌ manbalar ──────────┐   │        │ ✓  │  ┌ manbalar ▸ ▸ ▸ ──────────────────┐ │
│ │ ◐Ish ◑Oila ◒Kanal  │   │        │ ▤  │  │ ◐Ish·4  ◑Oila·2  ◒Kanal·1        │ │
│ └────────────────────┘   │        │────│  └──────────────────────────────────┘ │
│  javob matni¹ … ²        │        │ ⌘  │  javob matni¹ …  (68ch o'lchov)  ²    │
│  ⊙ plan ⊙ search ◉ write │        │ ☾  │  ⊙plan ⊙search ◉write  (bosqichlar)   │
│  ▸ Qamrov · 3 chat · 8k  │        │ ⋯  │  ▸ Qamrov · 3 chat · 8k xabar         │
│  ⧉ ⤴ ↻ 👍 👎              │        │    │  ⧉ ⤴ ↻ ± (javob amallari)             │
├──────────────────────────┤        │    │  ┌───────────────────────────────┐    │
│ [auto][ctx 50][deep]     │        │    │  │ Savol yozing…          [Ask]  │    │
│ ┌────────────────┐ [→]   │        │    │  └───────────────────────────────┘    │
│ └────────────────┘       │        └────┴───────────────────────────────────────┘
│ ⌂    ▣    ✦    ⌁    ☰   │        Composer: padding-bottom = max(safe, --kb)
└──────────────────────────┘
```

### 3.5 Today (uy)

```
MOBIL                               DESKTOP
┌──────────────────────────┐        ┌────┬──────────────────────────────────────┐
│ Bugun          ⌕    ☾    │        │ ✦  │ Bugun                        ⌕   ☾   │
├──────────────────────────┤        │ ⌂◀ ├──────────────────────────────────────┤
│ ┌ Briefing ───────────┐  │        │ ▣  │ ┌ Briefing ────────┐ ┌ Alertlar ───┐ │
│ │ TL;DR 3 qator       │  │        │ ✦  │ │ TL;DR 3 qator    │ │ ⚑ to'lov ×2 │ │
│ │ ▸ to'liq            │  │        │ ⌁  │ │ ▸ to'liq         │ │ ⚑ shartnoma │ │
│ └─────────────────────┘  │        │ ✓  │ └──────────────────┘ └─────────────┘ │
│ ┌ Alertlar ───────────┐  │        │ ▤  │ ┌ So'rang ───────────────────────┐   │
│ │ ⚑ to'lov ×2         │  │        │────│ │ [Bugun nima muhim?]           │   │
│ │ ⚑ shartnoma         │  │        │ ⌘  │ │ chip chip chip                │   │
│ └─────────────────────┘  │        │ ☾  │ └───────────────────────────────┘   │
│ ┌ So'rang ────────────┐  │        │ ⋯  │ ┌ Kutayotgan amallar (2) ───────┐   │
│ │ [ savol yozing… ]   │  │        │    │ └───────────────────────────────┘   │
│ └─────────────────────┘  │        └────┴──────────────────────────────────────┘
│ ⌂    ▣    ✦    ⌁    ☰   │
└──────────────────────────┘
```

### 3.6 Settings

```
MOBIL                               DESKTOP (2-panel)
┌──────────────────────────┐        ┌────┬──────────────────┬───────────────────┐
│ ‹ Sozlamalar             │        │ ✦  │ Sozlamalar       │ Akkaunt           │
├──────────────────────────┤        │ ⌂  │ ▸ Akkaunt        │ ╭──╮ Ali          │
│ ▸ Akkaunt                │        │ ▣  │ ▸ Yozish rejimi  │ ╰──╯ +998••••567  │
│ ▸ Yozish rejimi          │        │ ✦  │ ▸ Jim soatlar    │ [ Uzish ]         │
│ ▸ Jim soatlar            │        │ ⌁  │ ▸ Model/byudjet  │                   │
│ ▸ Model va byudjet       │        │ ✓  │ ▸ Til va mavzu   │ Mavzu             │
│ ▸ Til va mavzu           │        │ ▤  │ ▸ Maxfiylik      │ (Tizim)(Yorug')   │
│ ▸ Maxfiylik              │        │────│                  │ (Qorong'u)        │
│ [ Chiqish ]              │        │ ⌘  │                  │                   │
├──────────────────────────┤        │ ☾  │                  │ Til  UZ  RU  EN   │
│ ⌂    ▣    ✦    ⌁    ☰   │        │ ⋯  │                  │                   │
└──────────────────────────┘        └────┴──────────────────┴───────────────────┘
```

---

## 4. Token'lar v2 (`css/tokens.css`)

Uch qatlam saqlanadi: xom palitra (`app.css`) → semantik token → Telegram
mapping. Yangi token'lar **faqat** `css/tokens.css` da.

### 4.1 Masofa va tipografika

| Token | Qiymat |
|---|---|
| `--s-1 … --s-7` | 4 / 8 / 12 / 16 / 24 / 32 / 48 px |
| `--fs-xs … --fs-2xl` | `clamp()` suyuq: 11–12 / 12.5–14 / 14–15.5 / 15.5–17 / 18–22 / 22–30 |
| `--lh-tight / --lh-body / --lh-answer` | 1.25 / 1.55 / 1.65 |
| `--measure` / `--measure-narrow` | `68ch` / `52ch` |

### 4.2 Karkas

`--bp-mobile: 860px`, `--bp-wide: 1200px`, `--rail-w: 64px`,
`--list-w: 336px`, `--content-max: 1100px`, `--topbar-h: 52px`,
`--tabbar-h: 56px`, `--tap: 44px`.

### 4.3 Yangi semantik ranglar

| Token | Yorug' | Qorong'u | Vazifa |
|---|---|---|---|
| `--cite` | `#0b6ea8` | `#7cc4fa` | sitata chipi matni |
| `--cite-bg` | `#eaf5fd` | `#16304a` | sitata chipi foni |
| `--cite-border` | `#b9dffb` | `#1d4162` | chip chegarasi |
| `--stage-idle` | `#cbd5e0` | `#33455a` | bosqich: kutmoqda |
| `--stage-active` | `#0d78b8` | `#5eb5f7` | bosqich: bajarilyapti |
| `--stage-done` | `#0f9960` | `#3fc08a` | bosqich: tugadi |
| `--stage-failed` | `#e5484d` | `#ff6b6b` | bosqich: xato |
| `--skeleton` / `--skeleton-hi` | `#e6ebf1` / `#f2f6fa` | `#1e2b38` / `#27364a` | yuklanish |
| `--scrim` | `rgba(23,33,43,.42)` | `rgba(0,0,0,.6)` | modal orqa fon |

### 4.4 Peer palitrasi (7 rang) — **kontrast o'lchangan**

`peerColor(peer_id) = peer_id % 7 + 1` (satr ustida mod — katta id'lar
`Number` ga sig'maydi). Har bir juftning WCAG kontrast nisbati:

| # | Yorug' fon | ink | ratio | Qorong'u fon | ink | ratio |
|---|---|---|---|---|---|---|
| 1 qizil | `#c0392b` | `#ffffff` | **5.44** | `#ef8d84` | `#3a0f0a` | **7.04** |
| 2 to'q sariq | `#c05621` | `#ffffff` | **4.57** | `#efa06a` | `#3a1c07` | **7.34** |
| 3 binafsha | `#7a3fa8` | `#ffffff` | **6.77** | `#c79bea` | `#2a1038` | **7.57** |
| 4 yashil | `#2f7a45` | `#ffffff` | **5.26** | `#79ce93` | `#0c2c17` | **7.97** |
| 5 moviy | `#0e7490` | `#ffffff` | **5.36** | `#6fc7dd` | `#042b36` | **7.76** |
| 6 ko'k | `#2159b8` | `#ffffff` | **6.59** | `#8db4f2` | `#0b2049` | **7.55** |
| 7 pushti | `#b03a7a` | `#ffffff` | **5.63** | `#eb9bc6` | `#3a0e28` | **7.88** |

Hammasi ≥ 4.5:1 (AA, normal matn). Yorug' fon ranglari panel (`#fff`)
ustida ham ≥ 4.57:1 — ya'ni klaster chipi **matni** sifatida ham yaroqli.
Yangi rang qo'shsangiz nisbatni qayta hisoblang va shu jadvalga yozing.

### 4.5 Tema tanlash

Uch qatlam, spetsifiklik bo'yicha ustun-ma'ustun:

1. `:root` — yorug' (default = "system").
2. `@media (prefers-color-scheme: dark) :root:not([data-theme="light"])` (0,2,0).
3. `:root[data-theme="dark"]` / `:root[data-theme="light"]` (0,2,0, keyinroq).

`[data-theme]` har doim ustun turadi, shu jumladan `app.css` dagi eski
`prefers-color-scheme` blokidan (0,1,0) ham. Rejim `localStorage`
(`tgai.theme`) da; `base.html` dagi **inline bootstrap** uni CSS'dan
oldin `<html data-theme>` ga qo'yadi — usiz qorong'u temada har sahifada
oq chaqnash bo'ladi. `<meta name="theme-color">` ikkita (media bilan),
runtime'da `lib/theme.js` yangilaydi.

### 4.6 Telegram mapping (`:root.in-telegram`)

`--bg ← --tg-theme-secondary-bg-color`, `--panel ← bg-color`,
`--panel-2 ← section-bg-color`, `--text`, `--muted ← hint-color`,
`--accent ← button-color`, `--accent-ink ← button-text-color`,
`--accent-text ← link-color`, `--danger ← destructive-text-color`.
**Har biri fallback bilan** (`var(--tg-theme-x, bizning-qiymat)`): mijoz
kalitni bermasa UI o'z palitrasida qoladi. `.shell` balandligi
`--tg-viewport-stable-height`.

---

## 5. Komponentlar (holatlar bilan)

`css/components.css`. ⚑ = hozir bor, ⏳ = P2/P3.

| Komponent | Sinf | Holatlar |
|---|---|---|
| ⚑ Icon | `.icon`, `.icon-btn-v2` | default · hover · `aria-pressed` · `.spin` |
| ⚑ Avatar | `.avatar` (`.sm`/`.lg`) | photo · initials(`.peer-N`) · `.is-loading` · badge |
| ⚑ Skeleton | `.skeleton`, `.skeleton-row` | shimmer (reduced-motion'da o'chadi) |
| ⚑ EmptyState | `.empty-v2` | icon + title + text + actions |
| ⚑ Toast | `.toast-root`, `.toast` | `.ok` · `.err` · info · undo tugmasi |
| ⚑ Segment | `.segment` | `aria-selected` |
| ⚑ SyncPill | `.sync-pill` | idle · `.running` · `.done` · `.failed` |
| ⚑ CommandPalette | `.cmdk` | yopiq · bo'sh so'rov · natijalar · `.cmdk-empty` |
| ⚑ Sheet | `.sheet` | yopiq · ochiq (260ms) |
| ⚑ Rail / TabBar | `.rail`, `.tabbar` | default · `aria-current` · `.is-disabled` · badge |
| ⚑ CitationChip | `.cite-chip` | idle · hover · coarse-pointer (26px) |
| ⚑ StageTimeline | `.stage`, `.stage-item[data-state]` | pending · active · done · failed |
| ⏳ ChatRow | — | default·hover·selected·multi·unread·syncing·not-synced·skeleton |
| ⏳ ClusterChip, CoveragePanel, ActionCard, Composer, AnswerBlock | — | P2/P3 |

---

## 6. Harakat

Faqat `transform` / `opacity` / `clip-path` animatsiya qilinadi.
Hammasi `@media (prefers-reduced-motion: reduce)` ostida `.001ms` ga
tushadi (`css/base.css`).

| Harakat | Davomiylik | Ease | Token |
|---|---|---|---|
| hover / rang | 110ms | `--ease` | `--t-hover` |
| chip pop | 180ms | `--ease-out-back` | `--t-chip` |
| hovercard / kartalar | 140ms (+200ms kechikish) | `--ease` | `--t-card` |
| sheet | 260ms | `cubic-bezier(.32,.72,0,1)` | `--t-sheet`, `--ease-sheet` |
| panel / list→detail | 220ms | `--ease` | `--t-pane` |
| skeleton shimmer | 1.3s cheksiz | linear | — |
| spinner | 1s cheksiz | linear | `.spin` |

---

## 7. A11y qoidalari

- Har ikonkali tugmada `aria-label` (+ `title`); ikonka `aria-hidden`.
- `aria-current="page"` — faol nav elementi; `.is-disabled` +
  `aria-disabled="true"` — hali yo'q marshrut.
- Skip-link (`.skip-link`) → `#main-content`.
- Fokus halqasi `:focus-visible` (2px `--accent`, 2px offset); sichqoncha
  bosishida ko'rinmaydi.
- Fokus tuzog'i: `lib/dom.js::trapFocus` — palitra, sheet, modal.
- ⌘K: `role="combobox"` + `aria-controls` + `aria-activedescendant`,
  ro'yxat `role="listbox"`, element `role="option"` + `aria-selected`.
- Teginish nishoni ≥ **44px** (`--tap`; `@media (pointer: coarse)` da
  tugma va ro'yxat qatorlariga ham).
- Rang yolg'iz ma'no tashimaydi: sync holati rang **va** matn, bosqich
  rang **va** yorliq.
- `aria-live="polite"` — toast ildizi va KPI plitalari; javob hududida
  faqat **bosqich** o'zgarishida (token bo'yicha emas — spam bo'ladi).
- Emoji ikonka yo'q: ekran o'quvchi emojini ism deb o'qiydi va
  platformalarda har xil ko'rinadi. Test: `test_ui_strings_carry_no_emoji_icons`,
  `test_pages_reference_the_icon_sprite_not_emoji`.
- `text-transform: uppercase` ismlarga qo'llanmaydi (faqat `.block-title`
  kabi yorliqlarga).

---

## 8. i18n qoidalari

- Uch til: `uz` (default) / `ru` / `en`; yangi matn **uchchalasiga**
  (`src/app/i18n/locales/*.json`). Test:
  `test_new_ui_strings_exist_in_all_three_locales`.
- Server → brauzer: `pages.py::_strings()` prefiks filtri (`web.`,
  `*.err.`, `error.`) → `window.TGAI.strings` → `lib/i18n.js::t()`.
- Kalit topilmasa **kalitning o'zi** qaytadi — yo'q tarjima UI'da ko'rinib
  qoladi.
- Ko'plik: `lib/fmt.js::plural(key, n)` → `Intl.PluralRules` →
  `key.one|few|many|other`. Rus tilida uchala shakl majburiy
  (1 сообщение / 2 сообщения / 5 сообщений).
- Sana/son/nisbiy vaqt — `Intl.*` orqali, locale `window.TGAI.lang` dan
  (`uz-UZ` / `ru-RU` / `en-US`).
- Yorliqda emoji yo'q (§7).

---

## 9. Unumdorlik byudjeti

| Nima | Byudjet | Hozir |
|---|---|---|
| CSS (kritik yadro: tokens+base) | ≤ 12 KB | ~7 KB |
| CSS (jami, marshrut bo'yicha) | ≤ 50 KB | ~57 KB (app.css 40 KB — P3'da bo'linadi) |
| JS (marshrut, gzʻsiz) | ≤ 150 KB | shell ~26 KB + sahifa (ask modullari ~97 KB) |
| Ikonka sprite | ≤ 25 KB | 20 KB (67 ikonka) |
| FCP (mobil) | < 1s | brauzer QA'da o'lchanadi |

Qoidalar:

- `vendor/mermaid.min.js` **2.57 MB** — faqat `md.js` ichidagi lazy
  yuklash orqali va faqat ```mermaid bloki bo'lsa. Hech qachon `<script>`
  bo'lib sahifaga qo'shilmaydi.
- `vendor/telegram-web-app.js` — faqat Telegram konteksti aniqlanganda
  (`location.hash` da `tgWebApp` yoki `window.TelegramWebviewProxy`).
  Oddiy brauzerda umuman yuklanmaydi.
- `@import` **ishlatilmaydi** (ketma-ket yuklash) — bir nechta `<link>`,
  kritik yadro birinchi.
- `modulepreload` — `lib/shell.js`, `lib/theme.js`, `lib/palette-cmd.js`.
- Static URL'lar `asset()` bilan fingerprint qilinadi (R1b) → uzoq
  `immutable` kesh; HTML `no-cache` + ETag.
- Uzun ro'yxatlar: `content-visibility: auto` + `contain-intrinsic-size`.

---

## 10. Fayl va modul xaritasi

```
src/app/web/
  templates/
    base.html        qobiq: tema bootstrap, CSS zanjiri, window.TGAI, md.js
    _macros.html     icon · brand · lang_switch · avatar · skeleton_rows
                     · empty_state · toast_root       (`|safe` yo'q)
    _shell.html      rail · tabbar · topbar · pane'lar · More sheet · ⌘K
    login.html       base.html (karkassiz — kirmagan foydalanuvchi)
    chat.html        _shell.html → pane_content = eski `.app`
    dashboard.html   _shell.html → pane_content (scrolls)
    briefings.html   _shell.html → pane_content (scrolls)

  static/
    icons.svg              Lucide sprite, 67 `<symbol>` (ISC)
    vendor/LICENSE-lucide.txt
    app.css                eski sahifa uslublari + §11 v2 tuzatishlari
    css/tokens.css         token'lar v2, tema qatlamlari, Telegram mapping
    css/base.css           karkas asosi, a11y utilitalari, reduced-motion
    css/shell.css          rail · tabbar · topbar · pane'lar
    css/components.css     avatar · skeleton · toast · cmdk · sheet · chip
    md.js                  KLASSIK skript (node `eval` bilan yuklaydi) —
                           ESM qilinmaydi, `window.TGAI.md/hydrate/esc`
    pages/ask.js + pages/ask/*.js   (ES modullar, /ask kirish nuqtasi)
    lib/*.js               ES modullar (quyida)
```

### Modul API'lari (`static/lib/`)

```js
// md.js — `static/md.js` ustidan ko'prik (yagona renderer saqlanadi)
esc(s) -> string;  md(src) -> html;  hydrate(root) -> Promise<void>

// dom.js
$(sel, root?) -> Element|null;  $$(sel, root?) -> Element[]
el(spec, props?, children?) -> HTMLElement      // innerHTML ISHLATMAYDI
append(node, children);  clear(node);  replace(node, children)
trapFocus(container) -> () => void
debounce(fn, ms=80) -> fn & {cancel()}
ready(fn);  esc   // md.js dan re-export

// i18n.js
t(key, vars?) -> string;  has(key) -> bool;  tAny(keys, vars?);  lang() -> "uz"|"ru"|"en"

// api.js
class ApiError { code, status, detail }
api(method, path, body?, {signal?, raw?}) -> Promise<any>
get(path, opts?);  post(path, body?, opts?);  put;  patch;  del
errorOf(res) -> Promise<ApiError>

// sse.js
SSE_MAX_BUFFER = 2 * 1024 * 1024
parseFrame(frame) -> {kind, data} | null
stream(path, body, onEvent, {signal?}) -> Promise<void>

// fmt.js
num(n, opts?);  compact(n);  pct(x, digits?);  usd(x);  ms(v);  bytes(b)
dateTime(v);  date(v);  time(v);  relTime(v, now?);  listDate(v, now?)
weekdayDate(v) -> string               // "Dushanba, 15 sentyabr" (bugungi sarlavha)
plural(key, n, vars?) -> string        // Intl.PluralRules
// oy/kun/nisbiy-vaqt matnlari `lang()==="uz"` da `web.fmt.*` lug'atidan
// (Chromium'ning ba'zi ICU to'plamlarida `uz-UZ` uchun bu nomlar yo'q —
// `Intl.DateTimeFormat` "M09"/"Tue" qaytaradi); ru/en hamon `Intl` orqali.

// icons.js
icon(name, {label?, cls?, size?}) -> SVGElement
iconHTML(name, {cls?}) -> string       // nom [a-z0-9-] ga cheklanadi
iconButton(name, label, {cls?, onclick?, attrs?, type?}) -> HTMLButtonElement

// palette.js   (parite: services/textnorm.py, migratsiya 0008)
APOSTROPHES;  uzNorm(text);  uzFold(text)
fuzzy(needle, hay) -> {score, positions} | null
rank(q, items, keyFn, limit=20) -> [{item, score, positions}]
peerColor(id) -> 1..7;  peerClass(id) -> "peer-N";  initials(title) -> string

// store.js
createStore(initial, {persist?}) -> {get, set, update, subscribe}
ui              // {theme, density, recents, lastAccountId}, localStorage
pushRecent({id, title, kind});  mobileQuery() -> MediaQueryList;  isMobile()

// theme.js
MODES = ["system","light","dark"];  mode();  effective() -> "light"|"dark"
apply(m?) -> effective;  set(m);  cycle();  init()
// hodisa: window "tgai:theme" {mode, effective} — mermaid shu yerda qayta init

// tg.js  (Telegram tashqarisida hammasi no-op)
webApp();  inTelegram();  init() -> bool        // ready() + expand() + theme
backButton(show, onBack?);  openLink(url)
haptic.impact(style?);  haptic.notify(type?);  haptic.select()

// keys.js
bind(combo, fn) -> () => void;  inEditable(target);  label(combo) -> "⌘K"|"Ctrl K"

// toast.js
toast(message, {kind?, ms?, action?}) -> dismiss;  toastError(err)

// palette-cmd.js  (⌘K)
init();  open(prefill?);  close();  toggle()
setChats(rows);  addAction({id, label, icon, run})

// shell.js  — sahifa kirish nuqtasi
initShell();   // theme + tg + cmdk + More sheet + tema tugmalari
export { cmdk, theme, tg }
```

**Nega `md.js` modul emas:** `tests/test_md_renderer.py` uni node'da
`eval(fs.readFileSync(...))` bilan yuklaydi (`require` konteksti). ESM
qilinsa bu test yiqiladi. Shuning uchun u klassik skript bo'lib qoladi va
`lib/md.js` uning globallarini import qilinadigan qilib beradi —
renderer hamon **yagona**.

---

## 11. Testlar

| Test | Nimani qo'riqlaydi |
|---|---|
| `test_web.py::test_pages_render_with_the_navigation_shell` | rail · tabbar · cmdk · toast · sheet · skip-link har sahifada |
| `…::test_pages_reference_the_icon_sprite_not_emoji` | sprite ishlatiladi, HTML'da emoji yo'q |
| `…::test_theme_bootstrap_runs_before_paint` | `data-theme` skripti CSS'dan **oldin**, CSS zanjiri to'liq |
| `…::test_routes_map_has_no_dead_links` | `TGAI.routes` dagi har bir URL ochiladi |
| `…::test_new_ui_strings_exist_in_all_three_locales` | uz/ru/en + ru ko'plik shakllari |
| `…::test_ui_strings_carry_no_emoji_icons` | yorliqlarda emoji yo'q |
| `…::test_macros_and_shell_templates_exist_and_are_used` | sahifalar `_shell.html` dan meros, eski nusxalar yo'q |
| `…::test_static_assets_are_versioned_in_html` | `asset()` fingerprint'i ishlayapti |
| `test_js_lib.py::test_uz_fold_matches_python` | JS fold ↔ Python fold **pariteti** (20 ta korpus) |
| `…::test_fuzzy_subsequence_and_ranking` | subsequence, reyting tartibi, mos emaslik |
| `…::test_peer_color_is_deterministic_and_in_range` | 1..7, deterministik, katta id'lar |
| `…::test_initials_are_grapheme_safe` | emoji/kirill butun qoladi |
| `…::test_every_js_file_parses` | `node --check` har bir faylga |
| `…::test_sprite_has_every_icon_the_templates_use` | so'ralgan `#name` sprite'da bor |
| `test_md_renderer.py` (o'zgarmagan) | escape-first renderer |

---

## 12. 2-bosqich uchun ish (sahifa bo'yicha)

| Sahifa | Qilinadigan ish |
|---|---|
| **`/today`** (yangi) | `pages.py` marshruti; briefing kartasi (`/api/briefings`), alert lentasi (`/api/alerts/*/events`), "so'rang" maydoni + taklif chiplari; `ROUTES.today` va `TGAI.routes.today` ni yoqish |
| **`/chats`** (yangi) | `pane_list` + ChatRow komponenti; avatar endpoint (R1a), sparkline (`/chats/{id}/stats`), filtr segmenti URL holati bilan, `/` bilan ro'yxat qidiruvi (`lib/palette.js::rank`), multi-select |
| **`/chats/{id}`** (yangi) | sarlavha (avatar 64px, bio clamp, sync pill), tez amallar, analitika plitalari; mobil push + `tg.backButton(true)` |
| **`/ask`** (tayyor) | `chat.js` (39 KB) modullarga boʻlindi: `pages/ask.js` + `pages/ask/{state,conversations,composer,messages,answer,deep,actions,panels}.js` (~97 KB, izohlar bilan); `lib/api.js` va `lib/sse.js` ga oʻtdi; javob yuzasi: manbalar tasmasi, `.cite-chip` + hovercard, bosqich timeline’i, klaster chiplari, qamrov paneli, oqim (`/messages/stream` va `/api/deep/stream`), amal kartalari |
| **`/alerts`** (yangi) | alert CRUD dashboard'dan ko'chadi; hodisa lentasi |
| **`/actions`** (yangi) | tasdiq kutayotgan amallar; rail/tabbar badge (`#rail-actions-badge`) to'ldiriladi |
| **`/dashboard`** | plitalar → `lib/fmt.js`; chart'lar lazy `import()`; qolgan emoji (👍 👎 ★) sprite'ga |
| **`/briefings`** | `/today` ichiga singadi yoki arxiv bo'lib qoladi; sozlamalar `/settings` ga |
| **`/settings`** (yangi) | tema/til/default rejim (prefs API), write-mode, jim soatlar, akkauntni uzish |
| **`/login`** | onboarding 4 ekran (QR → ishonch kartasi → sync progress → 3 taklif) |

Umumiy: view transitions (`document.startViewTransition`, feature-detect),
`lib/tg.js::haptic` ni amallarga ulash, uPlot (vendor, lazy) analitika
uchun, `app.css` ni sahifa bo'yicha bo'lish.

---

## 13. Ochiq qolgan / kelishilmagan

- `app.css` (40 KB) hali yagona fayl — eski sinf nomlari `chat.js` va
  `dashboard.js` bilan shartnoma, shuning uchun bo'lish P3'ga qoldi.
- Modul ichidagi `import`lar fingerprint'siz (`/static/lib/*.js`);
  kirish nuqtasi `asset()` bilan versiyalangan. To'liq versiyalash uchun
  import map kerak — R1b bilan kelishiladi.
- `chat.js`/`dashboard.js` da bir nechta emoji qoldi (👍/👎 baho tugmalari,
  ★ reyting, 🔧/🤖 badge'lar) — ular klient tomonda chiziladi, server
  HTML'ida yo'q, shuning uchun test ushlamaydi. 2-bosqichda sprite'ga.
- Avatar rasmi yo'q (backend endpoint R1a'da) — hozir faqat inisiallar.
