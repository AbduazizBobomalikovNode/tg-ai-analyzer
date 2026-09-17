/* lib/shell.js — karkas kirish nuqtasi (har sahifada bir marta).
 *
 * Vazifasi: tema, Telegram Mini App, ⌘K palitrasi, mobil "More" varag'i,
 * tema tugmasi va `Esc`. Sahifaga xos mantiq bu yerda **yo'q**.
 */

import { $, $$, trapFocus } from "./dom.js";
import * as themeMod from "./theme.js";
import * as tg from "./tg.js";
import * as cmdk from "./palette-cmd.js";
import { bind } from "./keys.js";
import { t } from "./i18n.js";
import { get, post } from "./api.js";

/*
 * ⌘K "Chatlar" bo'limi — H3: `cmdk.setChats()` sahifa moduli chaqirmasa
 * (masalan `/today`, `/settings`, `/onboarding` — bularda chat ro'yxati
 * ekranga chiqmaydi) bo'lim doim bo'sh qolardi. Shu sababli karkas o'zi
 * bir marta yuklaydi: `/api/me` → birinchi faol (yoki birinchi) akkaunt →
 * uning chatlari. `sessionStorage`da 5 daqiqa keshlanadi — har sahifada
 * qayta so'ramaslik uchun (akkaunt/chat ro'yxati shu oraliqda kamdan-kam
 * o'zgaradi).
 */
const CHATS_CACHE_KEY = "tgai.cmdk.chats";
const CHATS_CACHE_MS = 5 * 60 * 1000;

function readChatsCache() {
  try {
    const raw = sessionStorage.getItem(CHATS_CACHE_KEY);
    if (!raw) return null;
    const { at, rows } = JSON.parse(raw);
    if (!Array.isArray(rows) || !(Date.now() - at < CHATS_CACHE_MS)) return null;
    return rows;
  } catch { return null; }
}

function writeChatsCache(rows) {
  try { sessionStorage.setItem(CHATS_CACHE_KEY, JSON.stringify({ at: Date.now(), rows })); }
  catch { /* kvota/xususiy oyna — shu sessiyada qayta so'raladi */ }
}

/**
 * `/api/me` + birinchi akkauntning chatlari. Sof funksiya (DOM'siz) —
 * testda `getFn` mock bilan almashtiriladi.
 * @param {typeof get} [getFn]
 * @returns {Promise<any[]>}
 */
export async function fetchPaletteChats(getFn = get) {
  const cached = readChatsCache();
  if (cached) return cached;
  const me = await getFn("/api/me");
  const acc = (me.accounts || []).find((a) => a.status === "active") || (me.accounts || [])[0];
  if (!acc) return [];
  const data = await getFn(`/api/accounts/${acc.id}/chats?limit=1000`);
  const rows = data.items || [];
  writeChatsCache(rows);
  return rows;
}

async function loadPaletteChats() {
  try {
    const rows = await fetchPaletteChats();
    if (rows.length) cmdk.setChats(rows);
  } catch { /* palitra chatlarsiz ham ishlaydi */ }
}

function initMoreSheet() {
  const sheet = $("#more-sheet");
  if (!sheet) return;
  let untrap = null, last = null;
  const open = () => {
    last = document.activeElement;
    sheet.hidden = false;
    untrap = trapFocus(sheet);
    $(".sheet-item", sheet)?.focus();
    tg.haptic.impact("light");
  };
  const close = () => {
    if (sheet.hidden) return;
    sheet.hidden = true;
    untrap?.(); untrap = null;
    last?.focus?.();
  };
  for (const b of $$("[data-sheet-open]")) b.addEventListener("click", open);
  for (const b of $$("[data-sheet-close]", sheet)) b.addEventListener("click", close);
  sheet.addEventListener("mousedown", (e) => { if (e.target === sheet) close(); });
  bind("escape", () => { if (sheet.hidden) return false; close(); return true; });
}

/**
 * More sheet'dagi "Chiqish" (L5). Ilgari faqat `pages/ask.js` bog'lardi —
 * ya'ni boshqa sahifalarda (`/today`, `/settings`, `/alerts`, ...) shu
 * tugma bosilganda hech narsa bo'lmasdi. Karkas endi o'zi bog'laydi.
 */
function initLogout() {
  const btn = $("#btn-logout-sheet");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    try { await post("/api/auth/logout"); } catch { /* cookie baribir tozalanadi */ }
    window.location.href = "/login";
  });
}

function initThemeButtons() {
  const paint = () => {
    const m = themeMod.mode();
    for (const b of $$("[data-theme-toggle]")) {
      const lbl = `${t("web.theme.title")}: ${t(`web.theme.${m}`)}`;
      b.setAttribute("aria-label", lbl);
      b.title = lbl;
      const txt = b.querySelector("[data-theme-label]");
      if (txt) txt.textContent = t(`web.theme.${m}`);
      const use = b.querySelector("use");
      if (use) {
        const name = m === "dark" ? "moon" : m === "light" ? "sun" : "monitor";
        const href = `${(window.TGAI && window.TGAI.sprite) || "/static/icons.svg"}#${name}`;
        use.setAttribute("href", href);
      }
    }
  };
  for (const b of $$("[data-theme-toggle]")) {
    b.addEventListener("click", () => { themeMod.cycle(); tg.haptic.select(); });
  }
  window.addEventListener("tgai:theme", paint);
  paint();
}

/** Karkasni ishga tushiradi. Sahifa skripti buni birinchi chaqiradi. */
export function initShell() {
  themeMod.init();
  tg.init();
  // Tafsilot sahifasida Telegram orqaga tugmasi = tarix.
  tg.backButton(document.body.dataset.detail === "1");
  cmdk.init();
  initMoreSheet();
  initThemeButtons();
  initLogout();
  loadPaletteChats();
  // Sahifa o'zi yangiroq ro'yxat yuklagan bo'lsa (masalan `/alerts`,
  // `/settings`) — palitra ham, kesh ham shundan yangilanadi.
  window.addEventListener("tgai:chats", () => {
    const rows = (window.TGAI && window.TGAI.chats) || [];
    if (rows.length) { writeChatsCache(rows); cmdk.setChats(rows); }
  });
}

export { cmdk, themeMod as theme, tg };
