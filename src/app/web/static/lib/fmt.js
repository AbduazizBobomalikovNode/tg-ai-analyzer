/* lib/fmt.js — sana, son, ko'plik. Hammasi `Intl` ustida (locale server'dan).
 *
 * Rus tilida ko'plik uch shaklda (1 сообщение / 2 сообщения / 5 сообщений),
 * o'zbek va inglizda ikkita. Shuning uchun `plural()` `Intl.PluralRules` ga
 * tayanadi va kalitni `key.one` / `key.few` / `key.many` / `key.other`
 * ko'rinishida qidiradi.
 */

import { lang, t, has } from "./i18n.js";

const LOCALE = { uz: "uz-UZ", ru: "ru-RU", en: "en-US" };
const locale = () => LOCALE[lang()] || "uz-UZ";

const memo = new Map();
const fmt = (kind, opts) => {
  const key = `${kind}|${locale()}|${JSON.stringify(opts)}`;
  let f = memo.get(key);
  if (!f) {
    f = kind === "n" ? new Intl.NumberFormat(locale(), opts)
      : kind === "rt" ? new Intl.RelativeTimeFormat(locale(), opts)
        : kind === "pr" ? new Intl.PluralRules(locale(), opts)
          : new Intl.DateTimeFormat(locale(), opts);
    memo.set(key, f);
  }
  return f;
};

const toDate = (v) => (v instanceof Date ? v : new Date(typeof v === "number" ? v : String(v)));
const ok = (d) => d instanceof Date && !Number.isNaN(d.getTime());

/*
 * Chromium'ning ba'zi qurilmalarida (kichik ICU to'plami) `uz-UZ` uchun
 * oy/kun nomlari yo'q: `Intl.DateTimeFormat("uz-UZ", {month:"long"})`
 * "sentyabr" o'rniga "M09" qaytaradi, `{weekday:"long"}` — "Tue" (ingliz
 * standarti). Raqamli formatlash (soat/daqiqa) va rus/ingliz ishlaydi —
 * muammo faqat matnli oy/kun nomlarida. Shuning uchun `uz` uchun bu
 * nomlar `Intl` ga tayanmaydi, `web.fmt.*` lug'atidan olinadi.
 */
const monthShortText = (d) =>
  lang() === "uz" ? t(`web.fmt.month_short.${d.getMonth() + 1}`) : fmt("dt", { month: "short" }).format(d);
const monthFullText = (d) =>
  lang() === "uz" ? t(`web.fmt.month_full.${d.getMonth() + 1}`) : fmt("dt", { month: "long" }).format(d);
const weekdayFullText = (d) =>
  lang() === "uz" ? t(`web.fmt.weekday_full.${d.getDay()}`) : fmt("dt", { weekday: "long" }).format(d);

/** Son: `12 345`. `opts` — `Intl.NumberFormat` opsiyalari. */
export const num = (n, opts) => (n == null ? "—" : fmt("n", opts).format(Number(n)));

/** Ixcham son: `12k`, `3,4M` (ro'yxat va plitalar uchun). */
export const compact = (n) => (n == null ? "—" : fmt("n", { notation: "compact", maximumFractionDigits: 1 }).format(Number(n)));

/** Foiz: `0.734` → `73%`. */
export const pct = (x, digits = 0) => (x == null ? "—" : fmt("n", { style: "percent", maximumFractionDigits: digits }).format(Number(x)));

/** Pul: `$0.0042` / `$1.23` / `$0.00` (L8: nol **2** kasr, faqat `0 < x < 0.01` da 4). */
export const usd = (x) => {
  if (x == null) return "—";
  const n = Number(x);
  return `$${n > 0 && n < 0.01 ? n.toFixed(4) : n.toFixed(2)}`;
};

/** Davomiylik: `1.4s`, `320ms`, `2m 05s`. */
export function ms(v) {
  if (v == null) return "—";
  const n = Number(v);
  if (n < 1000) return `${Math.round(n)}ms`;
  if (n < 60000) return `${(n / 1000).toFixed(1)}s`;
  const s = Math.round(n / 1000);
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

/** Kun + oy (yilsiz): `15 sen`. `date()`/`dateTime()`/`listDate()` shu yerdan. */
const dayMonth = (d) => `${d.getDate()} ${monthShortText(d)}`;

/**
 * Ilova bo'yicha **yagona** to'liq sana-vaqt formati: `15 sen, 14:32`
 * (`docs/ui-v2.md` § 4.1 — bir xil ko'rinish hamma joyda, locale bo'yicha
 * boshqa-boshqa tartib yo'q).
 */
export const dateTime = (v) => { const d = toDate(v); return ok(d) ? `${dayMonth(d)}, ${time(d)}` : "—"; };

/** Faqat sana: `15 sen 2026`. */
export const date = (v) => { const d = toDate(v); return ok(d) ? `${dayMonth(d)} ${d.getFullYear()}` : "—"; };

/** Faqat vaqt: `08:31` — raqamli, `Intl` hamma joyda ishlaydi. */
export const time = (v) => { const d = toDate(v); return ok(d) ? fmt("dt", { hour: "2-digit", minute: "2-digit" }).format(d) : "—"; };

/** Kun nomi + kun + to'liq oy: `Dushanba, 15 sentyabr` (bugungi sarlavha uchun). */
export const weekdayDate = (v) => { const d = toDate(v); return ok(d) ? `${weekdayFullText(d)}, ${d.getDate()} ${monthFullText(d)}` : ""; };

const UNITS = [
  ["year", 31536000], ["month", 2592000], ["week", 604800],
  ["day", 86400], ["hour", 3600], ["minute", 60], ["second", 1],
];

/**
 * `uz` uchun qo'lda: `Intl.RelativeTimeFormat("uz-UZ", {numeric:"auto"})`
 * ICU'siz muhitda lokalizatsiya qilinmagan qisqartma qaytaradi (masalan
 * `-4 w`, manfiy belgi bilan). Bu yerda hamma narsa `t("web.fmt.rel_*")`
 * dan — **hech qachon manfiy** (kelajak/soat farqi bo'lsa ham `Math.abs`).
 */
function relTimeUz(diffSeconds) {
  const abs = Math.round(Math.abs(diffSeconds));
  if (abs < 45) return t("web.fmt.now");
  if (abs < 3600) return t("web.fmt.rel_minutes", { n: Math.max(1, Math.round(abs / 60)) });
  if (abs < 86400) return t("web.fmt.rel_hours", { n: Math.max(1, Math.round(abs / 3600)) });
  const days = Math.round(abs / 86400);
  if (days === 1) return t("web.fmt.yesterday");
  if (days < 7) return t("web.fmt.rel_days", { n: days });
  if (days < 30) return t("web.fmt.rel_weeks", { n: Math.max(1, Math.round(days / 7)) });
  if (days < 365) return t("web.fmt.rel_months", { n: Math.max(1, Math.round(days / 30)) });
  return t("web.fmt.rel_years", { n: Math.max(1, Math.round(days / 365)) });
}

/** Nisbiy vaqt: `2 soat oldin`, `5 kun oldin`. 45 soniyagacha — "hozir". */
export function relTime(v, now = Date.now()) {
  const d = toDate(v);
  if (!ok(d)) return "—";
  const diff = (d.getTime() - now) / 1000;
  if (lang() === "uz") return relTimeUz(diff);
  const abs = Math.abs(diff);
  if (abs < 45) return t("web.fmt.now");
  for (const [unit, secs] of UNITS) {
    if (abs >= secs || unit === "second") {
      return fmt("rt", { numeric: "auto" }).format(Math.round(diff / secs), unit);
    }
  }
  return "—";
}

/** Bugun bo'lsa vaqt, shu yil bo'lsa kun+oy, aks holda to'liq sana. */
export function listDate(v, now = Date.now()) {
  const d = toDate(v);
  if (!ok(d)) return "";
  const n = new Date(now);
  if (d.toDateString() === n.toDateString()) return time(d);
  if (d.getFullYear() === n.getFullYear()) return dayMonth(d);
  return date(d);
}

/**
 * Ko'plik: `plural("web.n.messages", 5)` → `web.n.messages.many` kalitini
 * (ru: one/few/many/other; uz/en: one/other) `{n}` bilan to'ldiradi.
 * Kerakli shakl lug'atda bo'lmasa `.other` ga tushadi.
 */
export function plural(key, n, vars) {
  const cat = fmt("pr").select(Number(n));
  const full = `${key}.${cat}`;
  const chosen = has(full) ? full : `${key}.other`;
  return t(chosen, { n: num(n), count: n, ...vars });
}

/** Bayt: `1,2 MB`. */
export function bytes(b) {
  if (b == null) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0, v = Number(b);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${fmt("n", { maximumFractionDigits: i ? 1 : 0 }).format(v)} ${u[i]}`;
}
