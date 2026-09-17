/* pages/ask/panels.js — varaq (sheet), lightbox va inline banner.
 *
 * Bitta varaq konteyneri (`#ask-sheet`) barcha modal mazmun uchun: chat
 * tanlash, expert sozlamalari, sitata kartasi, rasm, mundarija, amallar.
 * Nega bitta: fokus tuzog'i, `Esc`, backdrop va scroll-lock mantiqini bir
 * marta yozamiz — beshta nusxada uchtasi doim nosoz bo'ladi.
 *
 * Bu yerda `innerHTML` **yo'q**: mazmun `el()` bilan quriladi.
 */

import { $, el, clear, replace, trapFocus } from "../../lib/dom.js";
import { get, put, patch } from "../../lib/api.js";
import { icon, iconButton } from "../../lib/icons.js";
import { t } from "../../lib/i18n.js";
import { dateTime } from "../../lib/fmt.js";
import { toast, toastError } from "../../lib/toast.js";
import { haptic } from "../../lib/tg.js";
import { safeHref } from "./state.js";

let untrap = null;
let lastFocus = null;
let onClosed = null;

const sheet = () => $("#ask-sheet");
const body = () => $("#ask-sheet-body");

/**
 * Varaqni ochadi.
 *
 * @param {string} title  sarlavha (matn — `textContent`)
 * @param {Node|Node[]} content
 * @param {{onClose?: () => void, cls?: string}} [opts]
 */
export function openSheet(title, content, opts = {}) {
  const box = sheet();
  if (!box) return;
  closeSheet();                       // ochiq varaq ustiga ochilmasin
  lastFocus = document.activeElement;
  onClosed = opts.onClose || null;
  $("#ask-sheet-title").textContent = title;
  const bd = body();
  bd.className = ["sheet-body", opts.cls].filter(Boolean).join(" ");
  replace(bd, content);
  box.hidden = false;
  untrap = trapFocus(box);
  haptic.impact("light");
  // Birinchi fokuslanadigan element — sarlavha emas, mazmundagi boshqaruv.
  const first = bd.querySelector("input, button, select, textarea, a[href]");
  (first || $("#ask-sheet-close")).focus();
}

/** Varaqni yopadi (ochiq bo'lmasa — no-op). */
export function closeSheet() {
  const box = sheet();
  if (!box || box.hidden) return;
  box.hidden = true;
  clear(body());
  untrap?.(); untrap = null;
  const cb = onClosed; onClosed = null;
  lastFocus?.focus?.();
  cb?.();
}

/** Sahifa yuklanganda bir marta: yopish tugmasi, backdrop, `Esc`. */
export function initSheet() {
  const box = sheet();
  if (!box) return;
  $("#ask-sheet-close")?.addEventListener("click", closeSheet);
  box.addEventListener("mousedown", (e) => { if (e.target === box) closeSheet(); });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!box.hidden) { e.preventDefault(); closeSheet(); }
  });
}

/* ── tasdiq ───────────────────────────────────────────────────────── */

/**
 * `confirm()` o'rnini bosuvchi varaq. `window.confirm` Telegram Mini App
 * ichida bloklanadi va sahifa oqimini to'xtatadi.
 *
 * @returns {Promise<boolean>}
 */
export function confirmSheet(title, text, okLabel, { danger = false } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (value) => { if (!done) { done = true; resolve(value); } };
    openSheet(title, [
      el("p.sheet-text", null, text),
      el("div.sheet-actions", null, [
        el("button.btn.ghost", { type: "button", onclick: () => { finish(false); closeSheet(); } }, t("web.close")),
        el(`button.btn.${danger ? "danger" : "primary"}`, {
          type: "button",
          onclick: () => { finish(true); closeSheet(); },
        }, okLabel),
      ]),
    ], { onClose: () => finish(false) });
  });
}

/* ── lightbox ─────────────────────────────────────────────────────── */

/** Rasmni to'liq ko'rsatadi. Manba **faqat** `/api/images/<id>` bo'lishi mumkin. */
export function openLightbox(src, alt = "") {
  if (!/^\/api\/images\/[A-Za-z0-9-]{8,40}$/.test(String(src))) return;
  openSheet(alt || t("web.ask.image"), el("div.lightbox", null,
    el("img", { src, alt, loading: "lazy", decoding: "async" })), { cls: "sheet-body wide" });
}

/* ── inline banner ────────────────────────────────────────────────── */

/**
 * Composer ustidagi banner (byudjet, shift, sessiya xatosi).
 *
 * Toast emas: bu xatolar **harakat talab qiladi** (kuting, rejimni
 * o'zgartiring) va 3 soniyada yo'qolib ketmasligi kerak.
 *
 * @param {"err"|"warn"|"info"} kind
 * @param {string} message
 * @param {{action?: {label: string, run: () => void}}} [opts]
 */
export function banner(kind, message, opts = {}) {
  const root = $("#banners");
  if (!root) return () => {};
  const node = el(`div.ask-banner.${kind}`, { role: kind === "err" ? "alert" : "status" }, [
    icon(kind === "err" ? "alert-triangle" : "info"),
    el("span.ask-banner-text", null, message),
  ]);
  if (opts.action) {
    node.appendChild(el("button.btn.ghost.small", {
      type: "button",
      onclick: () => { opts.action.run(); node.remove(); },
    }, opts.action.label));
  }
  node.appendChild(iconButton("x", t("web.close"), { onclick: () => node.remove() }));
  root.appendChild(node);
  return () => node.remove();
}

/** Barcha bannerlarni tozalaydi (yangi savol yuborilganda). */
export const clearBanners = () => clear($("#banners"));

/* ── tashqi havola ────────────────────────────────────────────────── */

/**
 * "Telegram'da ochish" havolasi. `safeHref` o'tmasa — `null` (tugma
 * umuman chizilmaydi, `href="#"` qoldirilmaydi).
 */
export function externalLink(url, label) {
  const href = safeHref(url);
  if (!href) return null;
  return el("a.btn.ghost.small", {
    href, target: "_blank", rel: "noopener noreferrer",
  }, [el("span", null, label), icon("external-link")]);
}

/* ── chat sozlamalari: yozish rejimi · auto-reply · rejalashtirilganlar ──
 *
 * Bular kontekstga tanlangan **bitta** chatga tegishli. Uchalasi ham
 * serverdagi mavjud endpoint'larga boradi; UI hech qanday yozish amalini
 * o'zi bajarmaydi (6-bosqich invarianti).
 */

const WRITE_MODES = ["read_only", "write_with_confirm", "autonomous"];

/** Yozish rejimi. `autonomous` — alohida tasdiq (akkauntga bitta chat). */
export async function openWriteModeSheet(chat, accountId) {
  const seg = el("div.segment", { role: "group", "aria-label": t("web.chat.write_mode") });
  const current = chat.write_mode || "read_only";
  for (const mode of WRITE_MODES) {
    const btn = el("button", {
      type: "button", "aria-selected": String(mode === current),
    }, t(`web.chat.wm_${mode}`));
    btn.addEventListener("click", async () => {
      if (mode === "autonomous") {
        const ok = await confirmSheet(
          t("web.chat.wm_autonomous"), t("web.chat.wm_autonomous_warn"),
          t("web.chat.wm_autonomous"), { danger: true },
        );
        if (!ok) return;
      }
      try {
        const r = await patch(`/api/accounts/${accountId}/chats/${chat.id}/write_mode`, { mode });
        chat.write_mode = r.write_mode;
        toast(t("web.chat.wm_saved"), { kind: "ok" });
        closeSheet();
      } catch (err) { toastError(err); }
    });
    seg.appendChild(btn);
  }
  openSheet(`${t("web.chat.write_mode")} · ${chat.title || ""}`, [
    seg, el("p.form-note", null, t("web.ask.wm_note")),
  ]);
}

const TRIGGERS = ["questions", "mentions", "keywords", "all"];

/** Auto-reply qoidasi. Javob **hech qachon** o'zi ketmaydi — taklif bo'lib keladi. */
export async function openAutoReplySheet(chat, accountId) {
  let rule = null;
  try {
    rule = (await get(`/api/accounts/${accountId}/chats/${chat.id}/autoreply`)).rule;
  } catch { /* qoida hali yo'q */ }
  const r = rule || {
    enabled: false, trigger: "questions", keywords: "", instructions: "",
    max_per_hour: 5, quiet_from: null, quiet_to: null,
  };
  const field = (node, label) =>
    el("label.form-field", null, [el("span", null, label), node]);

  const enabled = el("input", { type: "checkbox", name: "enabled", checked: !!r.enabled });
  const trigger = el("select", { name: "trigger" }, TRIGGERS.map((v) =>
    el("option", { value: v, selected: r.trigger === v }, t(`web.ar.trig_${v}`))));
  const keywords = el("input", { type: "text", name: "keywords", value: r.keywords || "" });
  const instructions = el("textarea", {
    name: "instructions", rows: "3", placeholder: t("web.ar.instructions_ph"),
  }, r.instructions || "");
  const maxPerHour = el("input", {
    type: "number", name: "max_per_hour", min: "1", max: "60", value: String(r.max_per_hour || 5),
  });
  const quietFrom = el("input", { type: "number", min: "0", max: "23", placeholder: "22", value: r.quiet_from ?? "" });
  const quietTo = el("input", { type: "number", min: "0", max: "23", placeholder: "7", value: r.quiet_to ?? "" });

  const form = el("form.ask-form", null, [
    (chat.write_mode || "read_only") === "read_only"
      ? el("div.ask-banner.warn", null, [icon("alert-triangle"), el("span", null, t("web.ar.needs_write"))])
      : null,
    field(enabled, t("web.ar.enabled")),
    field(trigger, t("web.ar.trigger")),
    field(keywords, t("web.ar.keywords")),
    field(instructions, t("web.ar.instructions")),
    field(maxPerHour, t("web.ar.max_per_hour")),
    field(el("span.row", null, [quietFrom, quietTo]), t("web.ar.quiet")),
    el("p.form-note", null, t("web.ar.note")),
    el("div.sheet-actions", null,
      el("button.btn.primary", { type: "submit" }, t("web.ar.save"))),
  ]);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const num = (node) => (node.value === "" ? null : Number(node.value));
    try {
      await put(`/api/accounts/${accountId}/chats/${chat.id}/autoreply`, {
        enabled: enabled.checked,
        trigger: trigger.value,
        keywords: keywords.value,
        instructions: instructions.value,
        max_per_hour: Number(maxPerHour.value) || 5,
        quiet_from: num(quietFrom),
        quiet_to: num(quietTo),
      });
      toast(t("web.ar.saved"), { kind: "ok" });
      closeSheet();
    } catch (err) { toastError(err); }
  });
  openSheet(`${t("web.ar.title")} · ${chat.title || ""}`, form);
}

/** Telegram server tomonda rejalashtirilgan postlar (faqat o'qish). */
export async function openScheduledSheet(chat, accountId) {
  openSheet(`${t("web.sched.title")} · ${chat.title || ""}`, el("p.muted", null, "…"));
  const box = $("#ask-sheet-body");
  try {
    const data = await get(`/api/accounts/${accountId}/chats/${chat.id}/scheduled`);
    const items = data.items || [];
    replace(box, items.length
      ? items.map((m) => el("div.sched-item", null, [
        el("div.muted", null, [
          icon("clock"),
          el("span", null, `${dateTime(m.date)} · #${m.msg_id}`),
          m.has_media ? icon("image") : null,
        ]),
        el("div", null, m.text || ""),
      ]))
      : el("p.muted", null, t("web.sched.empty")));
  } catch (err) {
    replace(box, el("div.ask-banner.err", null, [icon("alert-triangle"), el("span", null, err.message)]));
  }
}
