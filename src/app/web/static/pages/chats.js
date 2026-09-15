/* pages/chats.js — `/chats` ro'yxat paneli (R2-A).
 *
 * Vazifalari: akkaunt tanlovi · qidiruv (lokal fuzzy → server zaxirasi) ·
 * filtr segmenti URL holati bilan · Pinned/Recent/All bo'limlari ·
 * `IntersectionObserver` bilan sahifalash · ko'p tanlash → `/ask?chats=` ·
 * qator kontekst menyusi (Ask · Sync · Pin · Alerts · Write mode).
 *
 * Qoidalar:
 *   * `innerHTML` **yo'q** — har bir tugun `lib/dom.js::el()` bilan.
 *   * URL holati (`?type=`, `?unsynced=1`) `pushState` bilan — brauzer
 *     orqaga tugmasi filtrni tiklaydi.
 *   * Ro'yxat bir marta olinadi (`/chats?limit=1000`) va brauzerda
 *     filtrlanadi; ro'yxat qopqoqqa tegsa (`truncated`) qidiruv serverga
 *     o'tadi — aks holda foydalanuvchi ko'rmagan chatni izlay olmaydi.
 */

import { $, el, clear, replace, debounce, trapFocus } from "../lib/dom.js";
import { icon, iconButton } from "../lib/icons.js";
import { t } from "../lib/i18n.js";
import { get, post, patch as apiPatch } from "../lib/api.js";
import { toast, toastError } from "../lib/toast.js";
import { rank } from "../lib/palette.js";
import { compact, listDate, plural } from "../lib/fmt.js";
import { inEditable } from "../lib/keys.js";
import { haptic } from "../lib/tg.js";
import {
  me, pickAccount, rememberAccount, prefs, savePrefs,
  avatarEl, syncDot, syncLabel, typeLabel, flagNodes, highlight,
  chatHref, askHref, alertsHref, WRITE_MODES, writeModeLabel,
} from "./chat_shared.js";

/** Bir "sahifa" — shuncha qator chiziladi, keyin sentinel kutadi. */
const PAGE = 100;
/** `/chats` so'rovi qopqog'i (server `le=1000`). */
const FETCH_LIMIT = 1000;
/** Uzoq bosish (mobil ko'p tanlash) — ms. */
const LONG_PRESS_MS = 500;
/** Ko'rinadigan "Recent" qatorlari. */
const RECENT_SHOWN = 5;

const state = {
  accountId: 0,
  selectedId: 0,
  rows: [],          // barcha yuklangan chatlar
  truncated: false,  // ro'yxat qopqoqqa tegdimi (server qidiruviga o'tamiz)
  serverRows: null,  // server qidiruvi natijasi (null = lokal)
  q: "",
  type: "",
  unsynced: false,
  pinned: [],
  recent: [],
  entries: [],       // chizishga tayyor: {kind:"sec"|"row", …}
  drawn: 0,
  selected: new Set(),
};

let refs = {};
let observer = null;

/* ── URL holati ────────────────────────────────────────────────────── */

function readUrl() {
  const p = new URLSearchParams(location.search);
  state.type = ["channel", "group", "private"].includes(p.get("type") || "") ? p.get("type") : "";
  state.unsynced = p.get("unsynced") === "1";
  state.q = p.get("q") || "";
}

function writeUrl(push) {
  const p = new URLSearchParams();
  if (state.type) p.set("type", state.type);
  if (state.unsynced) p.set("unsynced", "1");
  if (state.q) p.set("q", state.q);
  const url = `${location.pathname}${p.toString() ? `?${p}` : ""}`;
  if (url === `${location.pathname}${location.search}`) return;
  if (push) history.pushState({ chats: true }, "", url);
  else history.replaceState({ chats: true }, "", url);
}

/* ── filtr / saralash ──────────────────────────────────────────────── */

const isGroup = (c) => c.type === "group" || c.type === "supergroup";

function matchesFilters(c) {
  if (state.type === "channel" && c.type !== "channel") return false;
  if (state.type === "group" && !isGroup(c)) return false;
  if (state.type === "private" && c.type !== "private") return false;
  if (state.unsynced && (c.synced_total || 0) > 0) return false;
  return true;
}

const byRecency = (a, b) =>
  String(b.last_message_at || "").localeCompare(String(a.last_message_at || "")) ||
  (b.id - a.id);

/**
 * Ko'rsatiladigan yozuvlar ro'yxatini quradi.
 *
 * So'rov bo'lsa — bo'limlarsiz yagona reytingli ro'yxat (bo'limlar
 * natijani parchalab, "eng yaxshi moslik qayerda" degan savol tug'diradi).
 */
function buildEntries() {
  const source = state.serverRows || state.rows;
  const pool = source.filter(matchesFilters);
  const out = [];

  if (state.q.trim()) {
    const ranked = state.serverRows
      ? pool.map((item) => ({ item, positions: [] }))
      : rank(state.q, pool, (c) => `${c.title || ""} ${c.username ? `@${c.username}` : ""}`, 500);
    for (const { item, positions } of ranked) out.push({ kind: "row", chat: item, positions });
    state.entries = out;
    return;
  }

  const byId = new Map(pool.map((c) => [Number(c.id), c]));
  const used = new Set();
  const section = (label, ids) => {
    const items = ids.map((id) => byId.get(Number(id))).filter((c) => c && !used.has(c.id));
    if (!items.length) return;
    out.push({ kind: "sec", label });
    for (const c of items) { used.add(c.id); out.push({ kind: "row", chat: c, positions: [] }); }
  };

  section(t("web.chats.sec_pinned"), state.pinned);
  section(t("web.chats.sec_recent"), state.recent.slice(0, RECENT_SHOWN));

  const rest = pool.filter((c) => !used.has(c.id)).sort(byRecency);
  if (rest.length) {
    if (out.length) out.push({ kind: "sec", label: t("web.chats.sec_all") });
    for (const c of rest) out.push({ kind: "row", chat: c, positions: [] });
  }
  state.entries = out;
}

/* ── qator ─────────────────────────────────────────────────────────── */

function subtitleOf(c) {
  if (c.last_message_text) return String(c.last_message_text).replace(/\s+/g, " ").slice(0, 160);
  if (c.participants_count) return plural("web.n.members", c.participants_count);
  return syncLabel(c);
}

function rowNode(chat, positions) {
  const id = Number(chat.id);
  const li = el("li.chat-row", {
    dataset: { id: String(id) },
    "aria-current": id === state.selectedId ? "true" : null,
  });
  if (state.selected.has(id)) li.classList.add("is-selected");

  const check = el("input.chat-check", {
    type: "checkbox",
    checked: state.selected.has(id),
    "aria-label": t("web.chats.select"),
    onclick: (e) => e.stopPropagation(),
    onchange: (e) => toggleSelect(id, e.target.checked),
  });

  const link = el("a.chat-row-main", { href: chatHref(id) }, [
    el("span.chat-title", null, [highlight(chat.title || `#${id}`, positions), ...flagNodes(chat)]),
    el("span.chat-sub", null, subtitleOf(chat)),
  ]);

  const side = el("span.chat-side", null, [
    el("span.chat-side-row", null, [
      chat.last_message_at ? el("span", null, listDate(chat.last_message_at)) : null,
      syncDot(chat),
    ]),
    el("span.chat-side-row", null, [
      el("span.type-badge", null, typeLabel(chat.type)),
      chat.unread_count ? el("span.unread-pill", null, compact(chat.unread_count)) : null,
    ]),
  ]);

  const menuBtn = iconButton("more-horizontal", t("web.chats.menu"), {
    cls: "chat-menu-btn",
    onclick: (e) => { e.preventDefault(); e.stopPropagation(); openMenu(chat, e.currentTarget); },
  });

  li.append(check, avatarEl(chat), link, side, menuBtn);
  wireLongPress(li, id);
  return li;
}

/** Mobil ko'p tanlash: uzoq bosish. Sichqoncha bilan — hoverdagi checkbox. */
function wireLongPress(li, id) {
  let timer = 0;
  const start = (e) => {
    if (e.pointerType === "mouse") return;
    timer = setTimeout(() => {
      timer = 0;
      haptic.impact("medium");
      toggleSelect(id, !state.selected.has(id));
    }, LONG_PRESS_MS);
  };
  const cancel = () => { clearTimeout(timer); timer = 0; };
  li.addEventListener("pointerdown", start);
  for (const ev of ["pointerup", "pointercancel", "pointerleave", "pointermove"]) {
    li.addEventListener(ev, cancel);
  }
  // Tanlash rejimida qatorga bosish navigatsiya emas, tanlovni o'zgartiradi.
  li.addEventListener("click", (e) => {
    if (!state.selected.size) return;
    if (e.target.closest("button")) return;
    e.preventDefault();
    toggleSelect(id, !state.selected.has(id));
  });
}

/* ── chizish ───────────────────────────────────────────────────────── */

function drawMore() {
  const list = refs.rows;
  let drawn = 0;
  while (state.drawn < state.entries.length && drawn < PAGE) {
    const entry = state.entries[state.drawn++];
    if (entry.kind === "sec") list.appendChild(el("li.chat-sec", { role: "presentation" }, entry.label));
    else { list.appendChild(rowNode(entry.chat, entry.positions)); drawn++; }
  }
  refs.sentinel.hidden = state.drawn >= state.entries.length;
}

function render() {
  buildEntries();
  clear(refs.rows);
  state.drawn = 0;
  drawMore();
  if (!state.entries.length) {
    refs.rows.appendChild(el("li.chat-sec", { role: "presentation" },
      state.q ? t("web.chats.no_results") : t("web.state.empty")));
  }
  renderSelbar();
  paintFilters();
}

function paintFilters() {
  for (const b of refs.typeSeg.querySelectorAll("button")) {
    b.setAttribute("aria-checked", String((b.dataset.type || "") === state.type));
    b.tabIndex = (b.dataset.type || "") === state.type ? 0 : -1;
  }
  refs.unsynced.setAttribute("aria-pressed", String(state.unsynced));
}

/* ── ko'p tanlash ──────────────────────────────────────────────────── */

function toggleSelect(id, on) {
  if (on) state.selected.add(id); else state.selected.delete(id);
  const li = refs.rows.querySelector(`.chat-row[data-id="${id}"]`);
  if (li) {
    li.classList.toggle("is-selected", on);
    const box = li.querySelector(".chat-check");
    if (box) box.checked = on;
  }
  refs.pane.classList.toggle("is-selecting", state.selected.size > 0);
  renderSelbar();
}

function clearSelection() {
  const ids = [...state.selected];
  state.selected.clear();
  for (const id of ids) {
    const li = refs.rows.querySelector(`.chat-row[data-id="${id}"]`);
    if (li) { li.classList.remove("is-selected"); const b = li.querySelector(".chat-check"); if (b) b.checked = false; }
  }
  refs.pane.classList.remove("is-selecting");
  renderSelbar();
}

function renderSelbar() {
  const n = state.selected.size;
  if (!n) { clear(refs.selbar); return; }
  replace(refs.selbar, el("div.selbar", { role: "region", "aria-label": t("web.chats.selected", { n }) }, [
    el("span.selbar-count", null, t("web.chats.selected", { n })),
    el("button.btn.ghost.small", { type: "button", onclick: clearSelection }, t("web.chats.clear_selection")),
    el("a.btn.primary.small", { href: askHref([...state.selected]) }, [icon("sparkles"), " ", t("web.chats.ask")]),
  ]));
}

/* ── kontekst menyusi ──────────────────────────────────────────────── */

let closeMenu = null;

function menuItem(iconName, label, onclick, extra) {
  return el("button", { type: "button", onclick, ...(extra || {}) }, [icon(iconName), label]);
}

function openMenu(chat, anchor) {
  closeMenu?.();
  const id = Number(chat.id);
  const pinned = state.pinned.map(Number).includes(id);
  const alerts = alertsHref(id);

  const menu = el("div.ctx-menu", { role: "menu", "aria-label": t("web.chats.menu") });
  const body = () => {
    clear(menu);
    menu.append(
      el("a", { href: askHref([id]), role: "menuitem" }, [icon("sparkles"), t("web.chats.ask")]),
      menuItem("refresh-cw", t("web.chats.sync"), () => { syncChat(id); closeMenu(); }, { role: "menuitem" }),
      menuItem(pinned ? "pin-off" : "pin", pinned ? t("web.chats.unpin") : t("web.chats.pin"),
        () => { togglePin(id, !pinned); closeMenu(); }, { role: "menuitem" }),
      alerts ? el("a", { href: alerts, role: "menuitem" }, [icon("bell"), t("web.chats.alerts")]) : null,
      el("div.ctx-sec", { role: "presentation" }, t("web.chats.write_mode")),
      ...WRITE_MODES.map((mode) => menuItem(
        chat.write_mode === mode ? "check" : "chevron-right",
        writeModeLabel(mode),
        () => { setWriteMode(chat, mode); closeMenu(); },
        { role: "menuitemradio", "aria-checked": String(chat.write_mode === mode) },
      )),
    );
  };
  body();

  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  const w = menu.offsetWidth || 220, h = menu.offsetHeight || 260;
  menu.style.insetInlineStart = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.left - w + r.width))}px`;
  menu.style.insetBlockStart = `${Math.max(8, Math.min(window.innerHeight - h - 8, r.bottom + 6))}px`;

  const untrap = trapFocus(menu);
  const onDown = (e) => { if (!menu.contains(e.target)) closeMenu(); };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); closeMenu(); } };
  document.addEventListener("mousedown", onDown, true);
  document.addEventListener("keydown", onKey, true);
  menu.querySelector("a,button")?.focus();

  closeMenu = () => {
    document.removeEventListener("mousedown", onDown, true);
    document.removeEventListener("keydown", onKey, true);
    untrap();
    menu.remove();
    closeMenu = null;
    anchor.focus?.();
  };
}

/* ── amallar ───────────────────────────────────────────────────────── */

async function syncChat(id) {
  try {
    await post(`/api/accounts/${state.accountId}/chats/${id}/sync`);
    toast(t("web.chats.sync_started"), { kind: "ok" });
  } catch (err) { toastError(err); }
}

async function syncAll() {
  try {
    await post(`/api/accounts/${state.accountId}/sync`);
    toast(t("web.chats.sync_started"), { kind: "ok" });
  } catch (err) { toastError(err); }
}

/** Optimistik: darhol chiziladi, server rad etsa qaytariladi + toast. */
async function togglePin(id, on) {
  const before = state.pinned.slice();
  state.pinned = on
    ? [Number(id), ...before.filter((x) => Number(x) !== Number(id))]
    : before.filter((x) => Number(x) !== Number(id));
  render();
  try {
    await savePrefs({ pinned_chats: state.pinned });
  } catch (err) {
    state.pinned = before;
    render();
    toast(t("web.chats.pin_failed"), { kind: "err" });
    void err;
  }
}

async function setWriteMode(chat, mode) {
  const before = chat.write_mode;
  chat.write_mode = mode;
  try {
    await apiPatch(`/api/accounts/${state.accountId}/chats/${chat.id}/write_mode`, { mode });
    toast(t("web.chats.wm_saved", { mode: writeModeLabel(mode) }), { kind: "ok" });
  } catch (err) {
    chat.write_mode = before;
    toastError(err);
  }
}

/* ── qidiruv ───────────────────────────────────────────────────────── */

const runSearch = debounce(async () => {
  writeUrl(false);
  if (!state.q.trim() || !state.truncated) { state.serverRows = null; render(); return; }
  try {
    const url = `/api/accounts/${state.accountId}/chats/search?q=${encodeURIComponent(state.q)}`
      + (state.type ? `&type=${encodeURIComponent(state.type === "group" ? "supergroup" : state.type)}` : "")
      + "&limit=100";
    state.serverRows = (await get(url)).items || [];
  } catch {
    state.serverRows = null;   // server qidiruvi yiqildi — lokal ro'yxat qoladi
  }
  render();
}, 160);

/* ── yuklash ───────────────────────────────────────────────────────── */

function banner(kind, text, action) {
  replace(refs.banner, el(`div.chats-banner${kind ? `.${kind}` : ""}`, null, [
    el("div", null, text),
    action ? el("div.row", null, action) : null,
  ]));
}

async function loadChats() {
  const data = await get(`/api/accounts/${state.accountId}/chats?limit=${FETCH_LIMIT}`);
  state.rows = data.items || [];
  state.truncated = state.rows.length >= FETCH_LIMIT;
  // ⌘K palitrasi shu ro'yxatdan oziqlanadi (qo'shimcha so'rov yo'q).
  window.TGAI.chats = state.rows;
  window.dispatchEvent(new Event("tgai:chats"));
  if (!state.rows.length) {
    banner("", t("web.chats.empty_none_text"),
      el("button.btn.primary.small", { type: "button", onclick: syncAll }, [icon("refresh-cw"), " ", t("web.chats.sync")]));
  } else {
    clear(refs.banner);
  }
}

function renderAccounts(accounts) {
  if (accounts.length < 2) { refs.account.hidden = true; return; }
  refs.account.hidden = false;
  const sel = refs.accountSelect;
  clear(sel);
  for (const a of accounts) {
    sel.appendChild(el("option", { value: String(a.id), selected: a.id === state.accountId },
      a.label || `#${a.id}`));
  }
  sel.onchange = async () => {
    state.accountId = Number(sel.value);
    rememberAccount(state.accountId);
    refs.skeletons.hidden = false;
    await loadChats();
    refs.skeletons.hidden = true;
    render();
  };
}

function renderSuggestions() {
  const host = $("#chats-suggestions");
  if (!host) return;
  const base = (window.TGAI.routes && window.TGAI.routes.ask) || "/ask";
  clear(host);
  for (const key of ["web.chats.suggest_1", "web.chats.suggest_2", "web.chats.suggest_3"]) {
    const text = t(key);
    host.appendChild(el("a.suggest-chip", { href: `${base}?q=${encodeURIComponent(text)}` }, text));
  }
}

async function boot() {
  let data;
  try { data = await me(); } catch (err) { refs.skeletons.hidden = true; toastError(err); return; }

  const accounts = data.accounts || [];
  if (!accounts.length) {
    refs.skeletons.hidden = true;
    banner("", t("web.chats.no_accounts_text"),
      el("a.btn.primary.small", { href: "/login" }, t("web.chats.connect")));
    return;
  }

  const acc = pickAccount(accounts);
  state.accountId = Number(acc.id);
  rememberAccount(state.accountId);
  renderAccounts(accounts);
  if (acc.status === "revoked") {
    banner("warn", t("web.chats.revoked"), el("a.btn.small", { href: "/login" }, t("web.chats.relogin")));
  }

  try {
    const [, p] = await Promise.all([loadChats(), prefs().catch(() => ({}))]);
    state.pinned = (p.pinned_chats || []).map(Number);
    state.recent = (p.recent_chats || []).map(Number);
  } catch (err) {
    toastError(err);
  }
  refs.skeletons.hidden = true;
  render();
}

/* ── kirish nuqtasi ────────────────────────────────────────────────── */

/** @param {{selectedId?: number}} [opts] */
export function initChats(opts = {}) {
  refs = {
    pane: $("#chats-pane"),
    rows: $("#chat-rows"),
    sentinel: $("#chats-sentinel"),
    search: $("#chats-search"),
    typeSeg: $("#chats-type"),
    unsynced: $("#chats-unsynced"),
    banner: $("#chats-banner"),
    selbar: $("#chats-selbar"),
    account: $("#chats-account"),
    accountSelect: $("#chats-account-select"),
    skeletons: $(".skeletons", $("#chats-pane")),
    syncAll: $("#chats-sync-all"),
  };
  if (!refs.pane || !refs.rows) return;
  state.selectedId = Number(opts.selectedId || 0);

  readUrl();
  refs.search.value = state.q;

  refs.search.addEventListener("input", () => { state.q = refs.search.value; runSearch(); });
  refs.typeSeg.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-type]");
    if (!b) return;
    state.type = b.dataset.type || "";
    writeUrl(true);
    render();
  });
  refs.unsynced.addEventListener("click", () => {
    state.unsynced = !state.unsynced;
    writeUrl(true);
    render();
  });
  refs.syncAll?.addEventListener("click", syncAll);

  // `/` — ro'yxat qidiruvi. Capture fazasida: `lib/keys.js` hujjatga
  // ko'pikda (bubble) osilgan, shuning uchun ⌘K palitrasi ochilmaydi.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey || inEditable(e.target)) return;
    e.preventDefault();
    e.stopPropagation();
    refs.search.focus();
    refs.search.select();
  }, true);

  window.addEventListener("popstate", () => { readUrl(); refs.search.value = state.q; render(); });

  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver((entries) => {
      if (entries.some((x) => x.isIntersecting)) drawMore();
    }, { rootMargin: "400px" });
    observer.observe(refs.sentinel);
  } else {
    refs.pane.addEventListener("scroll", () => {
      if (refs.pane.scrollTop + refs.pane.clientHeight > refs.pane.scrollHeight - 400) drawMore();
    });
  }

  renderSuggestions();
  paintFilters();
  boot();
}

export { state as chatsState };
