# UI/UX redizayn — tadqiqot xulosasi (2026-09-16)

Manba: mustaqil tadqiqot agenti (WebSearch), joriy stack auditidan keyin. Bu hujjat
`docs/ui-v2.md` (dizayn spetsifikatsiyasi) uchun asos. Stack: FastAPI + Jinja + vanilla
JS (ES modullar), bitta CSS, bundler yo'q, CDN yo'q, `md.js` yagona renderer.

## 1. Etalonlar va "ushlab turuvchi" mexanikalar

- **Perplexity** — deep rejim uchun eng yaqin etalon: (1) bosqichli progress yuklanish
  holati sifatida (plan → searching → clusters → ranking → writing bosqichlari deep
  rejimning *qahramoni* bo'lishi kerak, status qatori emas); (2) da'vo oxirida raqamli
  sitata chip'i, hover‑karta manba parchasi bilan; (3) "editorial calm" — gradient/
  glassmorphism yo'q, javob markazda.
- **Claude / ChatGPT** — 65–72ch o'lchov, javob amallari hover'da (copy/regenerate/
  feedback), follow‑up chip'lar.
- **Linear / Raycast** — ⌘K palitra: subsequence (fuzzy) moslik, yozishdan oldin ham
  foydali (recents, tez amallar), bo'limlar (Chats / Actions / Go to), destruktiv amal
  navigatsiyaga o'xshamasin.
- **Telegram Mini App** (qattiq cheklovlar): `--tg-theme-*` (14 kalit), `--tg-safe-area-
  inset-*` va `--tg-content-safe-area-inset-*`, `--tg-viewport-stable-height`,
  `WebApp.expand()` init'da (aks holda app "buzilgan" ko'rinadi), `BackButton`,
  `MainButton`, `HapticFeedback`, `theme_changed`.
- Retention: skeleton'lar spinner'dan ~20–30% tezroq his qilinadi; optimistik UI faqat
  arzon, qaytariladigan amallarda (👍/👎, pin, filtr) — Telegram'ga yozishda hech qachon
  (4‑invariant); o'rgatuvchi bo'sh holatlar (kontekstli taklif chip'lar); mikro‑
  animatsiya faqat sabab‑oqibatni ko'rsatganda, hammasi `prefers-reduced-motion` ichida.

## 2. Ma'lumot arxitekturasi

```
/            → akkaunt yo'q → /login, aks holda /today
/login       → QR‑first onboarding (mavjud)
/onboarding  → ulash → sync progress → birinchi savol takliflari
/today       → uy: kunlik briefing + alert lentasi + "so'rang" maydoni
/chats       → chat ro'yxati (mobil: to'liq sahifa; desktop: 2‑panel)
/chats/{id}  → chat tafsiloti: sarlavha (avatar, bio, a'zolar, sync), analitika
/ask[/{conv}]→ AI suhbat (auto/agent/direct/deep)
/alerts      → alertlar va qoidalar
/actions     → tasdiq kutayotgan yozish amallari (badge)
/dashboard   → token/narx/latency/sifat/sync
/settings    → akkaunt, chat bo'yicha yozish rejimi, jim soatlar, model/byudjet, til
```

Navigatsiya: **mobil (<860px)** pastki tab bar 5 ta (`Today · Chats · Ask · Alerts ·
More`), 56px + `env(safe-area-inset-bottom)`, tafsilot sahifalari *push* (Telegram
BackButton = history back). **Planshet (860–1200)** 2 panel + 64px ikonka rail.
**Desktop (≥1200)** 3 panel: rail 64px / ro'yxat 320–360px / kontent (max 1100px).
⌘K/Ctrl+K palitra hamma joyda; `/` ro'yxat qidiruvi, `Esc` yopadi, `↑↓ Enter`,
`⌘Enter` "shu chat haqida so'ra". Onboarding 4 ekran: QR → "nimani o'qiymiz" ishonch
kartasi (faqat o'qish, media yuklanmaydi, telefon saqlanmaydi) → jonli sync progress
("hozir boshlash mumkin" chiqish yo'li bilan) → 3 ta generatsiya qilingan birinchi savol.

## 3. Dizayn tizimi (framework'siz)

Mavjud ikki qatlamli (raw → semantic) token'lar saqlanadi; uchinchi qatlam — Telegram
mapping faqat `:root.in-telegram` da (`--bg ← --tg-theme-secondary-bg-color`, `--panel ←
bg_color`, `--panel-2 ← section_bg_color`, `--text`, `--muted ← hint_color`, `--accent ←
button_color`, `--accent-ink ← button_text_color`, `--accent-text ← link_color`,
`--danger ← destructive_text_color`), `theme_changed` da qayta; kontrast kafolatlangan
juftlar (`--accent-ink` on `--accent`) runtime'da tekshiriladi.

Token'lar (light / dark): `--bg #eef1f5/#0b1219`, `--panel #fff/#151f2a`, `--panel-2
#f4f7fa/#1b2733`, `--panel-3 #e9eef4/#22303e`, `--text #17212b/#e8eef5`, `--muted
#55636f/#93a4b7`, `--faint #667689/#78899d`, `--border #e3e8ee/#24323f`, `--accent
#0d78b8/#5eb5f7`, `--accent-text #0b6ea8/#7cc4fa`, `--accent-ink #fff/#06202f`; yangi:
`--cite #0b6ea8/#7cc4fa`, `--cite-bg #eaf5fd/#16304a`, `--stage-done #0f9960/#3fc08a`,
`--stage-active`, `--skeleton #e6ebf1/#1e2b38`, `--peer-1..7` (Telegram 7 rangi:
red/orange/violet/green/cyan/blue/pink, `hash(peer_id) % 7`, har biriga `-ink` juft
≥4.5:1). Spacing 4pt (`--s-1..7`: 4/8/12/16/24/32/48). Tip: `clamp()` bilan fluid
(`--fs-xs … --fs-2xl`), body 15–16px, javob o'lchovi `max-width: 68ch`.

Zamonaviy CSS (2026 Baseline): container queries komponentlar uchun (media query faqat
shell/panel soni uchun); `100svh` shell, `dvh` faqat kerakda; klaviatura ochilganda
`safe-area-inset-bottom` kichraymaydi (WebKit 217754) → `visualViewport` bilan `--kb`
var, `padding-bottom: max(env(safe-area-inset-bottom), var(--kb))`; same‑document
View Transitions (`document.startViewTransition`, feature‑detect), `@starting-style`,
`content-visibility: auto` + `contain-intrinsic-size: 0 64px` ro'yxat qatorlarida +
IntersectionObserver paging (virtualizatsiya faqat >2000 chat bo'lsa); `focus-visible`;
touch target **44px** loyiha minimumi.

Shrift: default **system stack**; ixtiyoriy Inter variable (OFL, kirill bor) faqat
subset (`unicode-range` lotin ~45KB + kirill ~35KB, `font-display: swap`); Geist kirillsiz
(rad), Manrope zich ro'yxatda yomon (rad). Ikonka: **Lucide (ISC)** bitta inline SVG
sprite (`/static/icons.svg`, `<use href="#name">`), 40–50 ikonka ≈ 8–12KB; emoji
ikonkalar o'chiriladi.

## 4. AI javob prezentatsiyasi

Tipografiya 16px/1.65, 68ch; `md.js` yagona renderer bo'lib qoladi (kengaytiriladi).
Sitata `#c<chat>m<msg>` → inline superscript chip (touch'da 44px hit), hover‑karta
(200ms in / 120ms out): avatar + chat, sender, sana, 3 qatorli parcha, "Telegram'da
ochish ↗"; touch'da bottom sheet. Javob tepasida gorizontal scroll‑snap **manbalar
tasmasi** (avatar, chat, N sitata). Deep rejim: **bosqich timeline** (desktop chap rail,
mobil gorizontal), holatlar pending/active/done/failed, bajarilganlari avtomatik
yig'iladi (kengaytiriladi). **Klaster chip'lar** peer palitra rangida, bosilganda
sitatalarni filtrlaydi — eng "ushlab turuvchi" interaksiya, darhol/optimistik.
**Qamrov paneli**: bitta "ishonch %" raqami YO'Q; nimaga qaralgani/qaralmaganini
ko'rsatish (`searched / synced / total` per chat, "qaralmadi: 3 chat (sync yo'q)" +
"sync qilish" tugmasi), `<details>` ichida "Qamrov · N chat · M xabar". Streaming:
yakuniy layout ichiga oqim (tugaganda sakramaydi), 1em blok kursor. Javob footer:
copy · share · regenerate · 👍/👎 (optimistik) + 2–3 follow‑up chip; >1500 so'zda TOC.
Mobil jadvallar `overflow-x:auto` + chap ustun sticky; kod bloklari copy tugmasi bilan.

## 5. Chat ro'yxati va tanlash

Qator 64px (container‑query): avatar 44px → sarlavha + subtitle (oxirgi faollik /
a'zolar) → o'ng ustun: tur belgisi, 14 kunlik sparkline (48×16 SVG), sync nuqta.
Avatar: haqiqiy rasm bo'lsa; fallback — inisiallar deterministik Telegram peer rangida
(grapheme‑safe, kirill/emoji). Filtrlar sticky segment: `All · Channels · Groups ·
Private` + `Not synced`, URL holati (`/chats?type=channel`). Qidiruv lokal fuzzy
(subsequence) sarlavha + username ustida, **uz‑fold** normalizatsiya (oʻ/o'/ө→o, gʻ→g,
apostrof yo'q, kirill↔lotin juftlar), moslik belgilanadi, 80ms debounce. **Multi‑select**
(long‑press / hover checkbox) → "N chat · So'rash" — kross‑chat savol uchun kontekst
chip'lar. Bo'limlar: Pinned → Recent (5) → All. Chat sarlavhasi: avatar 64px, sarlavha,
@username, tur, a'zolar soni, **bio 2 qator clamp + "ko'proq"**, sync pill (synced ·
12k · 2h ago / syncing 43% / not synced + CTA), tez amallar: Ask · Sync · Alerts · Write.

## 6. Analitika vizuallari

Sparkline/bar/heatmap — qo'lda SVG/CSS (`color-mix(in oklab, …)` 7×24 grid); vaqt
qatori uchun **uPlot** (MIT, ~50KB, vendor, faqat analitika marshrutida lazy import).
Pie yo'q. Rang bilan birga belgi/yorliq (rang yolg'iz ma'no tashimaydi).

## 7. A11y va i18n

`lang` atributi javob bloklarida ham (ru/uz kontent, en UI); ismlarda `text-transform:
uppercase` yo'q; kontrast har yangi token uchun tekshiriladi; klaviatura: nav → ro'yxat →
kontent, ⌘K, `j/k`, `Esc`, roving tabindex; `aria-live="polite"` javob hududida faqat
bosqich o'zgarishida (token bo'yicha emas), `role="status"` timeline'da, har ikonka
tugmada `aria-label`, `aria-expanded`; logical properties; ru plural (`Intl.PluralRules`).

## 8. Unumdorlik byudjeti

JS < 150KB gz / marshrut, CSS < 50KB, FCP < 1s mobil. **`vendor/mermaid.min.js` 2.57MB**
— faqat lazy va aniq "diagrammani chizish" tugmasi ortida. Native ES modullar
(`<script type="module">`, marshrut kirish nuqtasi + `lib/`: `api.js dom.js sse.js
fmt.js md.js icons.js palette.js store.js theme.js`), `modulepreload`, dinamik
`import()` deep/actions/uPlot/mermaid uchun; `chat.js` (38KB) bo'linadi. Static
fayllarga fingerprint + `Cache-Control: immutable`, HTML `no-cache` + ETag. CSS: ~12KB
kritik yadro + marshrut bo'yicha.

## Komponentlar (holatlar bilan)

ChatRow (default·hover·selected·multi·unread·syncing·not‑synced·skeleton), Avatar
(photo·initials·loading·badge), CommandPalette, Composer (idle·typing·context‑chips·
streaming·error·over‑budget), AnswerBlock, CitationChip (idle·hover·active·broken),
StageTimeline, ClusterChip, CoveragePanel, ActionCard (proposed·confirming·executing·
sent·failed·expired), SyncPill, Toast (undo bilan), Sheet/Drawer, EmptyState, Skeleton.

## Harakat

Hover 110ms `--ease`; chip pop 180ms `--ease-out-back`; hovercard 140ms + 200ms delay;
sheet 260ms `cubic-bezier(.32,.72,0,1)`; panel/list→detail 220ms view transition;
kursor 1s `steps(2)`. Faqat `transform/opacity/clip-path` animatsiya qilinadi.

## Ustuvorlik

P0 poydevor: fingerprint + cache; mermaid gate; ES modullar + `chat.js` bo'linishi;
token v2 + Telegram mapping + `expand()`; Lucide sprite, 44px. P1 shell: 3‑panel /
tab bar; `/today`, `/chats`, `/chats/{id}`; ⌘K; skeleton/bo'sh holatlar. P2 AI yuzasi:
javob redizayni, manba tasmasi, sitata chip/hovercard, timeline, klaster chip'lar,
qamrov paneli, streaming. P3 chat ro'yxati: avatarlar, sparkline, filtrlar URL, fuzzy
uz‑fold, multi‑select. P4: analitika, onboarding, view transitions, haptics, a11y.
