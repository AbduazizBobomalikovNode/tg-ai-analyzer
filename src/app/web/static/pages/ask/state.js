/* pages/ask/state.js — `/ask` sahifasining yagona holati va umumiy yordamchilar.
 *
 * Nega markazda: `chat.js` da holat besh joyda tarqalgan edi va "qaysi chat
 * kontekstda" degan savolga javob DOM'dan o'qilardi. Bu yerda bitta obyekt
 * (`S`) va kichik hodisa shinasi (`on`/`emit`) — modullar bir-birini import
 * qilmasdan xabarlashadi, ya'ni sikl bog'liqlik yo'q.
 *
 * Modul yuklanganda **hech qanday DOM'ga tegilmaydi** (node testi shu
 * fayldan `safeHref` ni import qiladi).
 */

import { put } from "../../lib/api.js";

/** Sahifa konfiguratsiyasi (shablondan `initAsk()` orqali keladi). */
export const cfg = {
  conversationId: 0,
  contextDefault: 200,
  contextMax: 2000,
};

/** Yagona holat. Modullar o'qiydi; yozgandan keyin `emit()` bilan xabar beradi. */
export const S = {
  // akkaunt / chatlar
  accounts: [],
  accountId: null,
  chats: [],                 // `chat_out` qatorlari
  byId: new Map(),           // chat_id → chat
  byPeer: new Map(),         // tg peer_id → chat
  contextChats: [],          // tanlangan chat **id**lari (chip'lar)

  // suhbat
  conversations: [],
  convId: 0,
  convTitle: "",
  nextBefore: null,          // "eskiroq" sahifa kursori (xabar id)

  // composer sozlamalari (prefs'dan boshlanadi)
  mode: "auto",
  strategy: "auto",
  ctxLimit: 200,
  deep: false,

  // oqim
  busy: false,
  abort: null,               // AbortController

  // amallar
  pending: [],
};

/* ── hodisa shinasi ────────────────────────────────────────────────── */

const subs = new Map();

/** `name` hodisasiga obuna. `() => void` qaytaradi. */
export function on(name, fn) {
  if (!subs.has(name)) subs.set(name, new Set());
  subs.get(name).add(fn);
  return () => subs.get(name)?.delete(fn);
}

/** Hodisani yuboradi. Tinglovchi xatosi qolganlarini to'xtatmaydi. */
export function emit(name, data) {
  for (const fn of subs.get(name) || []) {
    try { fn(data); } catch (err) { console.warn(`ask:${name}`, err); }
  }
}

/* ── chat yordamchilari ────────────────────────────────────────────── */

/** Chatlar ro'yxatini holatga yozadi va indekslarni qayta quradi. */
export function setChats(items) {
  S.chats = items || [];
  S.byId = new Map(S.chats.map((c) => [Number(c.id), c]));
  S.byPeer = new Map(S.chats.map((c) => [Number(c.peer_id), c]));
  // ⌘K palitrasi shu ro'yxatdan oziqlanadi — qo'shimcha so'rov yo'q.
  if (typeof window !== "undefined") {
    window.TGAI = window.TGAI || {};
    window.TGAI.chats = S.chats;
    window.dispatchEvent(new CustomEvent("tgai:chats", { detail: S.chats }));
  }
  emit("chats");
}

/** Kontekstda tanlangan chat obyektlari (yo'qolganlari tashlanadi). */
export const contextChatObjects = () =>
  S.contextChats.map((id) => S.byId.get(Number(id))).filter(Boolean);

/** Chat tanlangan/olib tashlangan. Takror tanlash — almashtirgich. */
export function toggleContextChat(id) {
  const n = Number(id);
  const i = S.contextChats.indexOf(n);
  if (i >= 0) S.contextChats.splice(i, 1);
  else S.contextChats.push(n);
  emit("context");
}

export function clearContextChats() {
  S.contextChats = [];
  emit("context");
}

/* ── prefs ─────────────────────────────────────────────────────────── */

let prefsTimer = 0;
let prefsQueue = {};

/**
 * Sozlamani serverda saqlaydi (`PUT /api/settings/prefs`), **birlashtirib**:
 * slayderni sudrash yuzlab so'rov yubormasin. Xato jimgina yutiladi —
 * saqlanmagan ko'rinish sozlamasi uchun toast ko'rsatish shovqin.
 */
export function savePref(key, value) {
  prefsQueue[key] = value;
  clearTimeout(prefsTimer);
  prefsTimer = setTimeout(async () => {
    const body = prefsQueue;
    prefsQueue = {};
    try { await put("/api/settings/prefs", body); } catch { /* sozlama — kritik emas */ }
  }, 600);
}

/** `/api/me` dagi prefs'ni composer holatiga ko'chiradi. */
export function applyPrefs(prefs) {
  if (!prefs) return;
  if (typeof prefs.default_mode === "string" && prefs.default_mode !== "deep") {
    S.mode = prefs.default_mode;
  }
  if (typeof prefs.default_strategy === "string") S.strategy = prefs.default_strategy;
  if (typeof prefs.deep_default === "boolean") S.deep = prefs.deep_default;
  emit("prefs");
}

/* ── xavfsizlik ────────────────────────────────────────────────────── */

/**
 * Tashqi havola uchun yagona filtr: **faqat** `http(s)://` va `tg://`.
 * `javascript:`, `data:`, protokolsiz va bo'sh qiymat → `null`, ya'ni
 * chaqiruvchi `<a>` o'rniga oddiy matn chizadi.
 *
 * Boshida bo'shliq/boshqaruv belgilari bo'lishi mumkin (`\njavascript:`
 * hiylasi) — shuning uchun avval trim.
 */
export function safeHref(u) {
  if (typeof u !== "string") return null;
  const v = u.trim();
  return /^(https?:\/\/|tg:\/\/)/i.test(v) ? v : null;
}

/**
 * Matn kirillmi (blok `lang` atributi uchun evristika).
 *
 * Nega kerak: `lang` bo'lmasa ekran o'quvchi ruscha matnni o'zbekcha
 * ovoz bilan o'qiydi va tirnoq/tire kerning'i ham noto'g'ri bo'ladi.
 * Harflarning **yarmidan ko'pi** kirill bo'lsa `ru` deb qaraymiz.
 */
export function guessLang(text) {
  const s = String(text || "");
  let cyr = 0, lat = 0;
  for (const ch of s) {
    if (/[Ѐ-ӿ]/.test(ch)) cyr++;
    else if (/[a-z]/i.test(ch)) lat++;
  }
  if (cyr + lat < 12) return "";       // juda qisqa — taxmin qilmaymiz
  return cyr > lat ? "ru" : "uz";
}
