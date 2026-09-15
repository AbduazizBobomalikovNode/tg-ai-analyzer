/* lib/md.js — `static/md.js` (klassik skript) ustidan ESM ko'prigi.
 *
 * Nega ko'prik: `md.js` node'da `eval()` bilan yuklanadi
 * (`tests/test_md_renderer.py`), ya'ni **ESM bo'lolmaydi**. Shu sababli u
 * o'zgarmaydi va `window.TGAI.md/hydrate/esc` global bo'lib qoladi; modul
 * kodi esa import qilib ishlatadi. Renderer hamon **yagona** (invariant).
 *
 * Fallback `esc` faqat `md.js` yuklanmagan holat uchun (masalan modul
 * testi) — mantiq `md.js` dagi bilan bir xil.
 */

const G = () => (typeof window !== "undefined" && window.TGAI) || {};

const _ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const _fallbackEsc = (s) => String(s).replace(/[&<>"']/g, (c) => _ESC[c]);

/** HTML-escape. `md.js` dagi `esc` ning aynan o'zi. */
export function esc(s) {
  const fn = G().esc;
  return fn ? fn(s) : _fallbackEsc(s);
}

/** Markdown → xavfsiz HTML (escape-first). `md.js` yuklanmagan bo'lsa — escape. */
export function md(src) {
  const fn = G().md;
  return fn ? fn(src) : _fallbackEsc(src);
}

/** Mermaid bloklarini chizadi (lazy). Har doim Promise qaytaradi. */
export function hydrate(root) {
  const fn = G().hydrate;
  return fn ? Promise.resolve(fn(root)) : Promise.resolve();
}
