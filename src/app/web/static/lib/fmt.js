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

/** Son: `12 345`. `opts` — `Intl.NumberFormat` opsiyalari. */
export const num = (n, opts) => (n == null ? "—" : fmt("n", opts).format(Number(n)));

/** Ixcham son: `12k`, `3,4M` (ro'yxat va plitalar uchun). */
export const compact = (n) => (n == null ? "—" : fmt("n", { notation: "compact", maximumFractionDigits: 1 }).format(Number(n)));

/** Foiz: `0.734` → `73%`. */
export const pct = (x, digits = 0) => (x == null ? "—" : fmt("n", { style: "percent", maximumFractionDigits: digits }).format(Number(x)));

/** Pul: `$0.0042` / `$1.23`. */
export const usd = (x) => (x == null ? "—" : `$${Number(x) < 0.01 ? Number(x).toFixed(4) : Number(x).toFixed(2)}`);

/** Davomiylik: `1.4s`, `320ms`, `2m 05s`. */
export function ms(v) {
  if (v == null) return "—";
  const n = Number(v);
  if (n < 1000) return `${Math.round(n)}ms`;
  if (n < 60000) return `${(n / 1000).toFixed(1)}s`;
  const s = Math.round(n / 1000);
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

/** To'liq sana-vaqt: `14.09.2026, 08:31`. */
export const dateTime = (v) => { const d = toDate(v); return ok(d) ? fmt("dt", { dateStyle: "short", timeStyle: "short" }).format(d) : "—"; };

/** Faqat sana: `14 sen 2026`. */
export const date = (v) => { const d = toDate(v); return ok(d) ? fmt("dt", { day: "numeric", month: "short", year: "numeric" }).format(d) : "—"; };

/** Faqat vaqt: `08:31`. */
export const time = (v) => { const d = toDate(v); return ok(d) ? fmt("dt", { hour: "2-digit", minute: "2-digit" }).format(d) : "—"; };

const UNITS = [
  ["year", 31536000], ["month", 2592000], ["week", 604800],
  ["day", 86400], ["hour", 3600], ["minute", 60], ["second", 1],
];

/** Nisbiy vaqt: `2 soat oldin`, `5 kun oldin`. 45 soniyagacha — "hozir". */
export function relTime(v, now = Date.now()) {
  const d = toDate(v);
  if (!ok(d)) return "—";
  const diff = (d.getTime() - now) / 1000;
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
  if (d.getFullYear() === n.getFullYear()) return fmt("dt", { day: "numeric", month: "short" }).format(d);
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
