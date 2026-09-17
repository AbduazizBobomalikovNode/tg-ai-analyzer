/* lib/toast.js — qisqa bildirishnoma (ixtiyoriy "Bekor qilish" bilan).
 *
 * `alert()` o'rniga: sahifani bloklamaydi va `aria-live` orqali ekran
 * o'quvchiga yetadi. Matn har doim `textContent` — HTML emas.
 */

import { el, $ } from "./dom.js";
import { icon } from "./icons.js";
import { t } from "./i18n.js";

const ICONS = { ok: "check", err: "alert-triangle", info: "info" };

/**
 * @param {string} message
 * @param {{kind?: "ok"|"err"|"info", ms?: number, action?: {label: string, run: () => void}}} [opts]
 * @returns {() => void} erta yopish
 */
export function toast(message, opts = {}) {
  const root = $("#toast-root");
  if (!root) return () => {};
  const kind = opts.kind || "info";
  const node = el(`div.toast.${kind}`, { role: kind === "err" ? "alert" : "status" }, [
    icon(ICONS[kind] || "info"),
    el("span", { style: { flex: "1 1 auto", minInlineSize: "0" } }, message),
  ]);
  if (opts.action) {
    node.appendChild(el("button", {
      type: "button",
      onclick: () => { opts.action.run(); dismiss(); },
    }, opts.action.label));
  }
  root.appendChild(node);
  const timer = setTimeout(dismiss, opts.ms ?? (kind === "err" ? 6000 : 3500));
  function dismiss() { clearTimeout(timer); node.remove(); }
  return dismiss;
}

/** `ApiError` uchun qulaylik: `code` allaqachon tarjima qilingan. */
export const toastError = (err) => toast(err?.message || t("error.generic"), { kind: "err" });
