/* lib/dom.js — DOM yordamchilari.
 *
 * Qoida: bu yerda `innerHTML` **umuman yo'q**. `el()` faqat
 * `document.createElement` + `textContent` ishlatadi, shuning uchun
 * begona matn (chat sarlavhasi, sender ismi) hech qachon HTML bo'lib
 * talqin qilinmaydi. Markdown chizish yagona istisno —
 * `lib/md.js::md()` chiqishi (`md.js` escape-first).
 */

export { esc } from "./md.js";

/** Birinchi mos element (`root` — ixtiyoriy kontekst). */
export const $ = (sel, root = document) =>
  sel[0] === "#" && !/[ .>[\],:]/.test(sel) ? root.getElementById?.(sel.slice(1)) ?? root.querySelector(sel) : root.querySelector(sel);

/** Barcha mos elementlar — haqiqiy massiv (map/filter ishlaydi). */
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const BOOL_ATTRS = new Set(["hidden", "disabled", "checked", "selected", "readonly", "required", "open", "autofocus", "multiple"]);

/**
 * Element quruvchi.
 *
 *   el("button.btn.primary", { onclick, "aria-label": "x" }, "Yubor")
 *   el("li", { class: "row" }, [avatarEl, el("span", null, title)])
 *
 * @param {string} spec  `tag`, `tag.cls`, `tag#id.cls` (tag majburiy emas → div)
 * @param {object|null} props  atributlar; `on*` — hodisa, `dataset` — obyekt,
 *   `style` — obyekt, `class`/`className` — sinf, qolgani `setAttribute`.
 * @param {*} children  matn, Node, yoki ular massivi (null/undefined tashlanadi)
 * @returns {HTMLElement}
 */
export function el(spec, props, children) {
  const m = /^([a-zA-Z][\w-]*)?(#[\w-]+)?((?:\.[\w-]+)*)$/.exec(spec || "div");
  if (!m) throw new Error(`el: noto'g'ri spec: ${spec}`);
  const node = document.createElement(m[1] || "div");
  if (m[2]) node.id = m[2].slice(1);
  if (m[3]) node.className = m[3].slice(1).split(".").join(" ");

  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class" || k === "className") node.className = [node.className, v].filter(Boolean).join(" ");
    else if (k === "dataset") for (const [dk, dv] of Object.entries(v)) { if (dv != null) node.dataset[dk] = String(dv); }
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k === "text") node.textContent = String(v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (BOOL_ATTRS.has(k)) { if (v) node.setAttribute(k, ""); }
    else node.setAttribute(k, String(v));
  }
  append(node, children);
  return node;
}

/** Bola(lar)ni qo'shadi: Node → o'zi, qolgani → matn tugun. */
export function append(node, children) {
  if (children == null) return node;
  for (const c of Array.isArray(children) ? children : [children]) {
    if (c == null || c === false) continue;
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

/** Elementni bo'shatadi (`innerHTML = ""` o'rniga — tez va xavfsiz). */
export function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/** `clear` + `append`. */
export const replace = (node, children) => append(clear(node), children);

/** Fokus tuzog'i: `Tab` konteynerdan chiqmaydi. `stop()` qaytaradi. */
export function trapFocus(container) {
  const SEL = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
  const onKey = (e) => {
    if (e.key !== "Tab") return;
    const items = $$(SEL, container).filter((n) => n.offsetParent !== null || n === document.activeElement);
    if (!items.length) return;
    const first = items[0], last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };
  container.addEventListener("keydown", onKey);
  return () => container.removeEventListener("keydown", onKey);
}

/** `fn` ni `ms` tinchlikdan keyin chaqiradi (ro'yxat qidiruvi uchun 80ms). */
export function debounce(fn, ms = 80) {
  let id = 0;
  const wrapped = (...args) => { clearTimeout(id); id = setTimeout(() => fn(...args), ms); };
  wrapped.cancel = () => clearTimeout(id);
  return wrapped;
}

/** DOM tayyor bo'lganda (yoki darhol, agar allaqachon tayyor bo'lsa). */
export function ready(fn) {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", fn, { once: true });
  else fn();
}
