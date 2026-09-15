/* lib/i18n.js — `window.TGAI.strings` ustidan `t()`.
 *
 * Lug'atni server beradi (`pages.py::_strings()` prefiks filtri). Kalit
 * topilmasa **kalitning o'zi** qaytadi — bu ataylab: yo'q tarjima UI'da
 * ko'rinib qoladi va test uni ushlaydi.
 */

const dict = () => (typeof window !== "undefined" && window.TGAI && window.TGAI.strings) || {};

/** Joriy til kodi (`uz` | `ru` | `en`). */
export const lang = () => (typeof window !== "undefined" && window.TGAI && window.TGAI.lang) || "uz";

/**
 * Tarjima. `{name}` shablonlari `vars` dan to'ldiriladi.
 * @param {string} key
 * @param {Record<string, unknown>} [vars]
 * @returns {string}
 */
export function t(key, vars) {
  let v = dict()[key];
  if (v == null) v = key;
  if (!vars) return String(v);
  return String(v).replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
}

/** Kalit lug'atda bormi (fallback tanlash uchun). */
export const has = (key) => key in dict();

/** Bir nechta kalitdan birinchi mavjudini tarjima qiladi. */
export function tAny(keys, vars) {
  for (const k of keys) if (has(k)) return t(k, vars);
  return t(keys[keys.length - 1], vars);
}
