/* pages/ask/actions.js — yozish amali kartalari (tasdiq kutayotgan).
 *
 * Shartnoma o'zgarmaydi: karta hech narsa yubormaydi, u faqat
 * `POST /api/actions/{id}/confirm|reject` chaqiradi — guard, rate limit,
 * TTL va `read_only` tekshiruvi serverda (6-bosqich invarianti).
 *
 * Holatlar: proposed → confirming → executing → executed | rejected |
 * failed | blocked | expired. `expired` **klient tomondagi** ko'rinish:
 * TTL o'tgan kartada tugmalar o'chadi va server baribir 410 qaytaradi.
 */

import { $, el, replace } from "../../lib/dom.js";
import { post, get } from "../../lib/api.js";
import { icon } from "../../lib/icons.js";
import { t } from "../../lib/i18n.js";
import { dateTime } from "../../lib/fmt.js";
import { toastError } from "../../lib/toast.js";
import { S, emit } from "./state.js";
import { hooks, chatAvatar } from "./messages.js";
import { openSheet } from "./panels.js";

const IMAGE_RE = /^\/api\/images\/[A-Za-z0-9-]{8,40}$/;

const KIND_ICON = {
  send_message: "send",
  edit_message: "edit-3",
  forward_message: "share-2",
  pin_message: "pin",
  send_image: "image",
};

const label = (tool) => {
  const key = `web.act.tool_${tool}`;
  const got = t(key);
  return got === key ? tool : got;
};

/* ── TTL hisoblagichi ─────────────────────────────────────────────── */

const SECOND = 1000;
const timers = new Set();

/** Qolgan vaqt: `12:34` yoki `1s 05d`. Muddati o'tgan bo'lsa `null`. */
function remaining(expiresAt) {
  const end = Date.parse(expiresAt || "");
  if (Number.isNaN(end)) return "";
  const left = Math.floor((end - Date.now()) / SECOND);
  if (left <= 0) return null;
  const h = Math.floor(left / 3600), m = Math.floor((left % 3600) / 60), s = left % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

/** Bitta kartaning hisoblagichini yuritadi; karta DOM'dan ketsa to'xtaydi. */
function startCountdown(node, expiresAt, onExpire) {
  const out = node.querySelector(".act-ttl");
  if (!out) return;
  const tick = () => {
    if (!node.isConnected) { clearInterval(id); timers.delete(id); return; }
    const left = remaining(expiresAt);
    if (left === null) {
      clearInterval(id); timers.delete(id);
      onExpire();
      return;
    }
    out.textContent = t("web.ask.act_ttl", { t: left });
  };
  const id = setInterval(tick, SECOND);
  timers.add(id);
  tick();
}

/** Sahifadan chiqishda (yoki tred qayta chizilganda) barcha hisoblagichlar. */
export function stopCountdowns() {
  for (const id of timers) clearInterval(id);
  timers.clear();
}

/* ── karta ────────────────────────────────────────────────────────── */

/**
 * Bitta amal kartasi.
 * @param {object} a  `ActionView.to_dict()`
 * @returns {HTMLElement}
 */
export function actionCard(a) {
  const p = a.preview || {};
  const chat = p.chat_id != null ? S.byId.get(Number(p.chat_id)) : null;
  const node = el(`div.act-card.is-${a.status}`, { dataset: { aid: a.id } });

  const head = el("div.act-head", null, [
    icon(KIND_ICON[a.tool] || "edit-3"),
    el("b.act-title", null, label(a.tool)),
    el("span.act-status", null, t(`web.act.status_${a.status}`)),
  ]);
  node.appendChild(head);

  const target = el("div.act-target", null, [
    chatAvatar(chat || { title: p.chat || "?", id: p.chat_id || 0 }),
    el("span", null, p.chat || t("web.ask.act_no_chat")),
  ]);
  node.appendChild(target);

  if (IMAGE_RE.test(String(p.image_url || ""))) {
    node.appendChild(el("img.gen-img", { src: p.image_url, alt: "", loading: "lazy" }));
  }
  if (p.text) node.appendChild(el("div.act-text", null, String(p.text)));

  const meta = [];
  if (p.message_id) meta.push(`#${p.message_id}`);
  if (p.reply_to) meta.push(t("web.ask.act_reply", { id: p.reply_to }));
  if (p.schedule_at) meta.push(dateTime(p.schedule_at));
  if (p.drop_author) meta.push(t("web.ask.act_copy"));
  if (p.image_id) meta.push(t("web.act.image"));
  if (a.result_msg_id) meta.push(t("web.act.result", { id: a.result_msg_id }));
  if (a.error) meta.push(String(a.error));
  if (a.block_reason) meta.push(String(a.block_reason));
  if (meta.length) node.appendChild(el("div.act-meta", null, meta.join(" · ")));

  if (a.status !== "proposed") return node;

  const ttl = el("span.act-ttl.muted");
  const confirm = el("button.btn.primary.small", { type: "button" }, t("web.act.confirm"));
  const reject = el("button.btn.ghost.small", { type: "button" }, t("web.act.reject"));
  const btns = el("div.act-btns", null, [ttl, el("span.spacer"), reject, confirm]);
  node.appendChild(btns);

  const expire = () => {
    node.className = "act-card is-expired";
    replace(btns, el("span.act-status", null, t("web.ask.act_expired")));
  };
  const run = async (kind, btn) => {
    for (const b of [confirm, reject]) b.disabled = true;
    btn.textContent = kind === "confirm" ? t("web.act.confirming") : t("web.ask.act_rejecting");
    node.classList.add(kind === "confirm" ? "is-confirming" : "is-rejecting");
    try {
      const updated = await post(`/api/actions/${a.id}/${kind}`, {});
      node.replaceWith(actionCard(updated));
    } catch (err) {
      toastError(err);
      node.classList.remove("is-confirming", "is-rejecting");
      for (const b of [confirm, reject]) b.disabled = false;
      confirm.textContent = t("web.act.confirm");
      reject.textContent = t("web.act.reject");
    }
    refreshPending();
  };
  confirm.addEventListener("click", () => run("confirm", confirm));
  reject.addEventListener("click", () => run("reject", reject));
  if (a.expires_at) startCountdown(node, a.expires_at, expire);
  return node;
}

/** Bir nechta karta (javob ostidagi guruh). Bo'sh bo'lsa `null`. */
export function actionCards(list) {
  if (!list || !list.length) return null;
  return el("div.act-group", null, list.map(actionCard));
}

/* ── kutayotganlar: badge va varaq ────────────────────────────────── */

/** `GET /api/actions?status=proposed` — badge va varaq uchun. */
export async function refreshPending() {
  try {
    const data = await get("/api/actions?status=proposed&limit=50");
    S.pending = data.items || [];
  } catch {
    S.pending = [];
  }
  paintBadge();
  emit("pending");
}

function paintBadge() {
  const n = S.pending.length;
  for (const id of ["#actions-badge", "#rail-actions-badge"]) {
    const badge = $(id);
    if (!badge) continue;
    badge.textContent = String(n);
    badge.hidden = n === 0;
  }
  const btn = $("#btn-actions-open");
  if (btn) btn.hidden = n === 0;
}

/** Kutayotgan amallar varag'i. */
export function openPendingSheet() {
  const body = S.pending.length
    ? el("div.act-group", null, S.pending.map(actionCard))
    : el("p.muted", null, t("web.pending.empty"));
  openSheet(t("web.pending.title"), body);
}

/** Sahifa yuklanganda bir marta.
 *
 * Varaq ochiq turganda kartani tasdiqlash `refreshPending()` ni chaqiradi,
 * lekin ro'yxat DOM'da allaqachon almashtirilgan (`node.replaceWith`) —
 * shuning uchun varaq qayta chizilmaydi, faqat badge yangilanadi.
 */
export function initActions() {
  hooks.actionCards = actionCards;
  $("#btn-actions-open")?.addEventListener("click", openPendingSheet);
  refreshPending();
}
