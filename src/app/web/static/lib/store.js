/* lib/store.js — kichik kuzatiladigan (observable) holat + localStorage.
 *
 * Framework yo'q, shuning uchun eng kichik shartnoma: `get/set/subscribe`.
 * Saqlanadigani faqat **qaytariladigan ko'rinish sozlamalari**: tema,
 * zichlik, oxirgi ochilgan chatlar. Sir, token yoki chat matni bu yerga
 * **hech qachon** yozilmaydi (localStorage — XSS uchun o'lja).
 */

const PREFIX = "tgai.";

/** localStorage — xususiy oynada yoki bloklanganda tashlaydi. */
function readLS(key, fallback) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw == null ? fallback : JSON.parse(raw);
  } catch { return fallback; }
}
function writeLS(key, value) {
  try {
    if (value == null) localStorage.removeItem(PREFIX + key);
    else localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch { /* kvota/xususiy oyna — sozlama shu sessiyada qoladi */ }
}

/**
 * @param {Record<string, unknown>} initial
 * @param {{persist?: string[]}} [opts]  qaysi kalitlar saqlanadi
 */
export function createStore(initial, opts = {}) {
  const persist = new Set(opts.persist || []);
  const state = { ...initial };
  for (const k of persist) { const v = readLS(k, undefined); if (v !== undefined) state[k] = v; }

  const subs = new Map();   // kalit ("*" = hammasi) → Set<fn>

  const notify = (key, value) => {
    for (const fn of subs.get(key) || []) fn(value, key);
    for (const fn of subs.get("*") || []) fn(value, key);
  };

  return {
    get: (key) => (key === undefined ? { ...state } : state[key]),
    set(key, value) {
      if (Object.is(state[key], value)) return value;
      state[key] = value;
      if (persist.has(key)) writeLS(key, value);
      notify(key, value);
      return value;
    },
    update(key, fn) { return this.set(key, fn(state[key])); },
    /** @returns {() => void} obunani bekor qiluvchi */
    subscribe(key, fn) {
      if (!subs.has(key)) subs.set(key, new Set());
      subs.get(key).add(fn);
      return () => subs.get(key)?.delete(fn);
    },
  };
}

/** Ilova bo'yicha yagona do'kon (ko'rinish sozlamalari). */
export const ui = createStore(
  { theme: "system", density: "cozy", recents: [], lastAccountId: null },
  { persist: ["theme", "density", "recents", "lastAccountId"] },
);

/** Oxirgi ochilganlar ro'yxati (MRU, maksimum 8). */
export function pushRecent(entry) {
  if (!entry || entry.id == null) return;
  const list = (ui.get("recents") || []).filter((r) => String(r.id) !== String(entry.id));
  list.unshift({ id: entry.id, title: String(entry.title || "").slice(0, 120), kind: entry.kind || "chat" });
  ui.set("recents", list.slice(0, 8));
}

/**
 * `--bp-mobile` dan `matchMedia` quradi — breakpoint **bitta manba**
 * (tokens.css). O'zgaruvchi o'qilmasa 860px ga tushadi.
 */
export function mobileQuery() {
  let bp = "860px";
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue("--bp-mobile").trim();
    if (/^\d+(\.\d+)?px$/.test(v)) bp = v;
  } catch { /* SSR/test */ }
  return window.matchMedia(`(max-width: ${bp})`);
}

/** Hozir mobil kenglikdami. */
export const isMobile = () => mobileQuery().matches;
