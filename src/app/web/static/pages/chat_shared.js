/* pages/chat_shared.js — `/chats` va `/chats/{id}` uchun umumiy bo'laklar.
 *
 * Nega alohida modul: ikkala sahifa ham bir xil akkaunt tanlovi, bir xil
 * prefs (pin/recent) va bir xil chat qatori elementlariga tayanadi. Nusxa
 * ikkita bo'lsa ular jimgina ajralib ketadi (audit § Templates dagi
 * brand/lang-switch hikoyasi).
 *
 * XAVFSIZLIK: bu yerda `innerHTML` **yo'q**. Chat sarlavhasi, bio va
 * oxirgi xabar matni — Telegram kontenti, ya'ni ishonchsiz ma'lumot;
 * hammasi `el()` / `textContent` orqali chiziladi.
 */

import { el } from "../lib/dom.js";
import { icon } from "../lib/icons.js";
import { t } from "../lib/i18n.js";
import { get, put } from "../lib/api.js";
import { peerClass, initials, uzFold } from "../lib/palette.js";
import { ui } from "../lib/store.js";

/* ── /api/me (bir marta) ───────────────────────────────────────────── */

let mePromise = null;

/** `{user, accounts, max_accounts}` — sahifa umri davomida bir marta. */
export function me() {
  if (!mePromise) mePromise = get("/api/me");
  return mePromise;
}

/** Faol (revoke qilinmagan) akkauntlar. */
export const activeAccounts = (data) =>
  (data.accounts || []).filter((a) => a.status !== "revoked");

/**
 * Qaysi akkaunt bilan ishlaymiz: oxirgi tanlangani (localStorage), agar u
 * ro'yxatda bo'lmasa — birinchi faol, u ham bo'lmasa birinchi mavjud.
 */
export function pickAccount(accounts) {
  const last = ui.get("lastAccountId");
  const byId = accounts.find((a) => String(a.id) === String(last));
  return byId || accounts.find((a) => a.status !== "revoked") || accounts[0] || null;
}

export const rememberAccount = (id) => ui.set("lastAccountId", id == null ? null : Number(id));

/* ── prefs (pinned / recent) ───────────────────────────────────────── */

let prefsCache = null;

export async function prefs() {
  if (!prefsCache) prefsCache = (await get("/api/settings/prefs")).prefs || {};
  return prefsCache;
}

/** Lokal keshni yangilab, serverga PUT qiladi. Xato — chaqiruvchida. */
export async function savePrefs(patch) {
  const res = await put("/api/settings/prefs", patch);
  prefsCache = res.prefs || { ...(prefsCache || {}), ...patch };
  return prefsCache;
}

let recentTimer = 0;

/**
 * Ochilgan chatni `recent_chats` boshiga qo'yadi (debounce 600ms).
 *
 * Xato **jim yutiladi**: bu qulaylik sozlamasi, u tufayli sahifada
 * ogohlantirish chiqmasligi kerak.
 */
export function pushRecentChat(chatId) {
  const id = Number(chatId);
  if (!id) return;
  clearTimeout(recentTimer);
  recentTimer = setTimeout(async () => {
    try {
      const p = await prefs();
      const list = [id, ...(p.recent_chats || []).filter((x) => Number(x) !== id)].slice(0, 10);
      if (JSON.stringify(list) === JSON.stringify(p.recent_chats || [])) return;
      await savePrefs({ recent_chats: list });
    } catch { /* qulaylik sozlamasi — jim */ }
  }, 600);
}

/* ── chat qatorining bo'laklari ────────────────────────────────────── */

/** Chat turi → i18n yorlig'i (`supergroup` ham "guruh"). */
export function typeLabel(type) {
  const kind = type === "supergroup" ? "group" : type || "chat";
  return t(`web.chat.kind_${kind}`);
}

/** Avatar: rasm bo'lsa rasm (lazy), aks holda peer rangida inisiallar. */
export function avatarEl(chat, cls = "") {
  const node = el(`span.avatar${cls ? `.${cls}` : ""}`, {
    class: peerClass(chat.peer_id ?? chat.id),
    "aria-hidden": "true",
  });
  if (chat.avatar_url) {
    const img = el("img", {
      src: chat.avatar_url, alt: "", loading: "lazy", decoding: "async",
      // Rasm kelmasa (fayl o'chgan, 404) inisiallarga tushamiz.
      onerror: () => { img.remove(); node.appendChild(document.createTextNode(initials(chat.title))); },
    });
    node.appendChild(img);
  } else {
    node.appendChild(document.createTextNode(initials(chat.title)));
  }
  return node;
}

/** Sync holati nuqtasi (rang **va** `title` — rang yolg'iz ma'no tashimaydi). */
export function syncDot(chat) {
  const state = chat.sync_state || "idle";
  return el("span.sync-dot", { dataset: { state }, title: syncLabel(chat), "aria-hidden": "true" });
}

export function syncLabel(chat) {
  const state = chat.sync_state || "idle";
  if (state === "running") return t("web.chats.sync_running", { pct: chat.progress ?? 0 });
  if (state === "failed") return t("web.chats.sync_failed");
  if (state === "done" || (chat.synced_total || 0) > 0) return t("web.chats.sync_synced");
  return t("web.chats.sync_none");
}

/** Tasdiqlangan/scam/fake belgilari (matn bilan — rang yolg'iz emas). */
export function flagNodes(chat, { verifiedIcon = true } = {}) {
  const out = [];
  if (chat.is_verified && verifiedIcon) {
    out.push(icon("shield-check", { cls: "verified", label: t("web.chats.verified") }));
  }
  if (chat.is_scam) out.push(el("span.flag-badge", null, t("web.chats.scam")));
  if (chat.is_fake) out.push(el("span.flag-badge", null, t("web.chats.fake")));
  return out;
}

/**
 * Fuzzy mosligini belgilaydi. `positions` — **fold qilingan** satr
 * indekslari; fold uzunligi asl uzunlikdan farq qilsa (кирилл → 2 harf)
 * belgilash tashlanadi va sof matn qaytadi (`palette-cmd.js` bilan bir xil
 * qoida).
 */
export function highlight(text, positions) {
  const span = el("span");
  const src = String(text || "");
  if (!positions || !positions.length || uzFold(src).length !== src.length) {
    span.textContent = src;
    return span;
  }
  const set = new Set(positions);
  let run = "", marked = false;
  const flush = () => {
    if (!run) return;
    span.appendChild(marked ? el("mark", null, run) : document.createTextNode(run));
    run = "";
  };
  for (let i = 0; i < src.length; i++) {
    const hit = set.has(i);
    if (hit !== marked) { flush(); marked = hit; }
    run += src[i];
  }
  flush();
  return span;
}

/* ── marshrutlar ───────────────────────────────────────────────────── */

const R = () => (typeof window !== "undefined" && window.TGAI && window.TGAI.routes) || {};

export const chatHref = (id) => `/chats/${encodeURIComponent(id)}`;

/** `/ask?chats=1,2,3` — R2-B shu query'ni o'qiydi. */
export function askHref(ids) {
  const base = R().ask || "/ask";
  const list = (Array.isArray(ids) ? ids : [ids]).filter(Boolean).join(",");
  return list ? `${base}?chats=${encodeURIComponent(list)}` : base;
}

/** Alert sahifasi hali yoqilmagan bo'lsa `null` (o'lik havola yo'q). */
export const alertsHref = (chatId) =>
  R().alerts ? `${R().alerts}?chat=${encodeURIComponent(chatId)}` : null;

/* ── yozish rejimi ─────────────────────────────────────────────────── */

export const WRITE_MODES = ["read_only", "write_with_confirm", "autonomous"];

export const writeModeLabel = (mode) =>
  t(`web.chats.wm_${mode === "write_with_confirm" ? "confirm" : mode === "autonomous" ? "autonomous" : "read_only"}`);
