/* lib/keys.js — global klaviatura yorliqlari.
 *
 * Qoida: matn kiritilayotgan joyda yorliq **ishlamaydi** (`/` slash chat
 * xabarida yozilishi kerak). Istisno — `Escape` va `⌘K`/`Ctrl+K`, ular
 * har doim ishlaydi.
 */

const EDITABLE = new Set(["INPUT", "TEXTAREA", "SELECT"]);

/** Fokus matn maydonidami. */
export function inEditable(target) {
  const n = target;
  if (!n || !n.tagName) return false;
  return EDITABLE.has(n.tagName) || n.isContentEditable === true;
}

const handlers = new Map();   // "mod+k" → Set<fn>

/** Hodisadan normal kalit: `mod+k`, `shift+/`, `escape`. */
function comboOf(e) {
  const parts = [];
  if (e.metaKey || e.ctrlKey) parts.push("mod");
  if (e.altKey) parts.push("alt");
  if (e.shiftKey && e.key.length > 1) parts.push("shift");
  parts.push(String(e.key).toLowerCase());
  return parts.join("+");
}

const ALWAYS = new Set(["escape", "mod+k"]);

function onKeyDown(e) {
  const combo = comboOf(e);
  const set = handlers.get(combo);
  if (!set || !set.size) return;
  if (inEditable(e.target) && !ALWAYS.has(combo)) return;
  for (const fn of set) {
    if (fn(e) !== false) e.preventDefault();
  }
}

let bound = false;

/**
 * Yorliq bog'laydi.
 * @param {string} combo  `mod+k`, `/`, `escape`, `shift+?`
 * @param {(e: KeyboardEvent) => boolean|void} fn  `false` qaytarsa
 *   `preventDefault()` chaqirilmaydi.
 * @returns {() => void} uzish
 */
export function bind(combo, fn) {
  const key = combo.toLowerCase();
  if (!handlers.has(key)) handlers.set(key, new Set());
  handlers.get(key).add(fn);
  if (!bound) { document.addEventListener("keydown", onKeyDown); bound = true; }
  return () => handlers.get(key)?.delete(fn);
}

/** Platformaga mos ko'rinish: `⌘K` yoki `Ctrl K`. */
export function label(combo) {
  const mac = typeof navigator !== "undefined" && /mac|iphone|ipad/i.test(navigator.platform || navigator.userAgent || "");
  return combo
    .split("+")
    .map((p) => (p === "mod" ? (mac ? "⌘" : "Ctrl") : p === "shift" ? "⇧" : p.length === 1 ? p.toUpperCase() : p[0].toUpperCase() + p.slice(1)))
    .join(mac ? "" : " ");
}
