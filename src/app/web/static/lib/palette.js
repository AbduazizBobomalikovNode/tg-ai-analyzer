/* lib/palette.js — peer rangi, inisiallar, o'zbekcha fold va fuzzy moslik.
 *
 * `uzFold()` — `src/app/services/textnorm.py::uz_fold` ning **aynan
 * nusxasi** (u ham Postgres `uz_fold()` bilan paritetda). Uchalasi bir
 * fazoda bo'lmasa qidiruv natijalari brauzerda va serverda ajralib
 * ketadi: `oʻzbek`, `o'zbek`, `ozbek`, `ўзбек` — to'rttasi ham `ozbek`.
 * Parite testi: `tests/test_js_lib.py::test_uz_fold_matches_python`.
 *
 * Tartib (Python bilan bir xil): NFC → lower → apostrof sinfini o'chirish
 * → kirill→lotin (bitta o'tish).
 */

/* ── o'zbekcha fold ──────────────────────────────────────────────────── */

/** Apostrof sinfi — `textnorm.APOSTROPHES`. Fold'da butunlay o'chiriladi. */
export const APOSTROPHES = "'‘’‛ʻʼʹ´`ʽ";

const APO_RE = new RegExp(`[${APOSTROPHES.replace(/[\\\]^-]/g, "\\$&")}]`, "g");

/** Kirill → lotin. `textnorm._CYR_TO_LAT` bilan bir xil. */
const CYR = {
  "ё": "yo", "ю": "yu", "я": "ya", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
  "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "j",
  "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
  "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
  "х": "x", "ы": "i", "э": "e",
  "ғ": "g", "қ": "q", "ҳ": "h", "ў": "o",
  "ъ": "", "ь": "",
};
const CYR_RE = new RegExp(`[${Object.keys(CYR).join("")}]`, "g");

/** NFC → lower → apostrofsiz. Transliteratsiya **yo'q** (`uz_norm`). */
export function uzNorm(text) {
  if (!text) return "";
  return String(text).normalize("NFC").toLowerCase().replace(APO_RE, "");
}

/** `uzNorm` + kirill→lotin (`uz_fold`). */
export function uzFold(text) {
  if (!text) return "";
  return uzNorm(text).replace(CYR_RE, (c) => CYR[c]);
}

/* ── fuzzy (subsequence) moslik ──────────────────────────────────────── */

/**
 * `needle` `hay` ichida **ketma-ketlik** bo'lib uchraydimi.
 *
 * Linear/Raycast uslubi: harflar tartibi saqlansa yetarli (`chsh` →
 * "Cheklangan **sh**artnoma"). Ikkala tomon ham `uzFold` dan o'tadi.
 *
 * @param {string} needle  foydalanuvchi yozgani
 * @param {string} hay     nomzod (chat sarlavhasi, @username, marshrut)
 * @returns {{score: number, positions: number[]}|null}  mos kelmasa null.
 *   `score` kattaroq = yaxshiroq; `positions` — **fold qilingan** `hay`
 *   ichidagi indekslar (belgilashda o'sha satr ustida ishlatiladi).
 */
export function fuzzy(needle, hay) {
  const n = uzFold(needle), h = uzFold(hay);
  if (!n) return { score: 0, positions: [] };
  if (!h) return null;

  const positions = [];
  let hi = 0, score = 0, streak = 0;
  for (let ni = 0; ni < n.length; ni++) {
    const ch = n[ni];
    const found = h.indexOf(ch, hi);
    if (found < 0) return null;
    // qo'shni belgi — ketma-ketlik bonusi; so'z boshi — alohida bonus
    if (found === hi && ni > 0) { streak++; score += 4 + streak; }
    else { streak = 0; score += 1; }
    if (found === 0 || /[\s._\-/@]/.test(h[found - 1])) score += 6;
    positions.push(found);
    hi = found + 1;
  }
  // qisqaroq nomzod — aniqroq moslik
  score += Math.max(0, 24 - h.length) / 4;
  if (h.startsWith(n)) score += 12;
  return { score, positions };
}

/**
 * Nomzodlarni fuzzy bo'yicha saralaydi.
 * @param {string} q
 * @param {T[]} items
 * @param {(it: T) => string} key  taqqoslanadigan matn
 * @param {number} [limit]
 * @returns {{item: T, score: number, positions: number[]}[]}
 * @template T
 */
export function rank(q, items, key, limit = 20) {
  const out = [];
  for (const item of items) {
    const m = fuzzy(q, key(item));
    if (m) out.push({ item, score: m.score, positions: m.positions });
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, limit);
}

/* ── peer rangi va inisiallar ────────────────────────────────────────── */

/**
 * Peer id → 1..7 (Telegram'ning 7 rangi). Katta id'lar `Number` ga
 * sig'masligi mumkin, shuning uchun mod **satr ustida** hisoblanadi —
 * aniq va deterministik.
 */
export function peerColor(id) {
  const s = String(id ?? "").replace(/\D/g, "");
  if (!s) return 1;
  let r = 0;
  for (const d of s) r = (r * 10 + (d.charCodeAt(0) - 48)) % 7;
  return r + 1;
}

/** `peerColor` → CSS sinfi (`components.css` dagi `.peer-N`). */
export const peerClass = (id) => `peer-${peerColor(id)}`;

const seg = (() => {
  try { return new Intl.Segmenter(undefined, { granularity: "grapheme" }); } catch { return null; }
})();

/** Matnni grapheme'larga bo'ladi (emoji, kirill birikmalari butun qoladi). */
function graphemes(s) {
  if (seg) return Array.from(seg.segment(s), (g) => g.segment);
  return Array.from(s); // fallback: kod nuqtasi (surrogat juftlar butun)
}

/** Raqam, tinish belgisi yoki matn/valyuta/o'zgartiruvchi belgisidan (`|`,
 * `@`, `#`, `$`) iborat grapheme — inisialga nomzod emas. Piktografik
 * belgilar (emoji, `\p{So}`) bundan mustasno — ular butun holda saqlanadi. */
const SKIP_RE = /^[\p{Nd}\p{P}\p{Sm}\p{Sc}\p{Sk}]+$/u;

/** So'z ichidagi birinchi "foydali" (raqam/tinish belgisi bo'lmagan) grapheme. */
function firstUsableGrapheme(word) {
  for (const g of graphemes(word)) {
    if (!SKIP_RE.test(g)) return g;
  }
  return null;
}

/**
 * Inisiallar: harflar (yoki emoji) bor birinchi ikki so'zning birinchi
 * grapheme'si. Faqat raqam/tinish belgisidan iborat so'zlar (`"1-"`, `"|"`)
 * o'tkazib yuboriladi — lekin so'z ichida ular harfdan oldin kelsa ham
 * birinchi harf topiladi (`"1-A"` → `"A"`). Bo'sh sarlavha → `?` (so'z
 * umuman yo'q); faqat raqam/tinish belgili so'zlar bo'lsa → `•`.
 */
export function initials(title) {
  const words = String(title || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "?";
  const usable = words.filter((w) => firstUsableGrapheme(w) != null);
  if (!usable.length) return "•";
  const first = firstUsableGrapheme(usable[0]);
  if (usable.length === 1) return first.toLocaleUpperCase();
  const second = firstUsableGrapheme(usable[1]);
  return (first + second).toLocaleUpperCase();
}
