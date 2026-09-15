/* lib/palette-cmd.js — ⌘K buyruq palitrasi.
 *
 * Bo'limlar: Chats (fuzzy, uz-fold) · Go to (marshrutlar) · Actions.
 * Yozishdan **oldin** ham foydali: oxirgi ochilganlar va tez amallar
 * ko'rinadi (Raycast/Linear naqshi).
 *
 * Marshrutlar `window.TGAI.routes` dan olinadi — 2-bosqichda yangi
 * sahifalar yoqilganda faqat o'sha xarita o'zgaradi, bu yerda emas.
 * Yoqilmagan marshrut palitrada **ko'rsatilmaydi** (o'lik havola yo'q).
 */

import { $, $$, el, clear, trapFocus, debounce } from "./dom.js";
import { t } from "./i18n.js";
import { icon } from "./icons.js";
import { rank, uzFold, peerClass, initials } from "./palette.js";
import { ui } from "./store.js";
import * as themeMod from "./theme.js";
import { haptic } from "./tg.js";
import { bind, label as keyLabel } from "./keys.js";

let box = null, input = null, list = null, root = null;
let items = [], cursor = 0, untrap = null, lastFocus = null;
let chats = [];
const extraActions = [];

/** Sahifa o'z chat ro'yxatini beradi (chat.js allaqachon yuklagan). */
export function setChats(rows) {
  chats = (rows || []).map((c) => ({
    id: c.id, peer_id: c.tg_peer_id ?? c.peer_id ?? c.id,
    title: c.title || "", username: c.username || "", type: c.type || "chat",
  }));
}

/** Sahifaga xos amal qo'shish: `{id, label, icon, run}`. */
export function addAction(action) { extraActions.push(action); }

const routes = () => (window.TGAI && window.TGAI.routes) || {};

function goEntries() {
  const R = routes();
  const defs = [
    ["today", "web.nav.today", "home"],
    ["chats", "web.nav.chats", "messages-square"],
    ["ask", "web.nav.ask", "sparkles"],
    ["alerts", "web.nav.alerts", "bell"],
    ["actions", "web.nav.actions", "check-check"],
    ["briefings", "web.nav.briefings", "inbox"],
    ["dashboard", "web.nav.dashboard", "bar-chart-3"],
    ["settings", "web.nav.settings", "settings"],
  ];
  return defs
    .filter(([k]) => typeof R[k] === "string" && R[k])
    .map(([k, key, ico]) => ({
      kind: "go", id: `go:${k}`, label: t(key), icon: ico,
      run: () => { window.location.href = R[k]; },
    }));
}

function actionEntries() {
  const out = [...extraActions];
  out.push({
    kind: "action", id: "act:theme", icon: "sun",
    label: t("web.cmd.toggle_theme"),
    meta: t(`web.theme.${themeMod.mode()}`),
    run: () => { themeMod.cycle(); render(input.value); return "keep"; },
  });
  for (const l of ["uz", "ru", "en"]) {
    out.push({
      kind: "action", id: `act:lang:${l}`, icon: "external-link",
      label: t("web.cmd.language", { lang: l.toUpperCase() }),
      run: () => {
        const u = new URL(window.location.href);
        u.searchParams.set("lang", l);
        window.location.href = u.toString();
      },
    });
  }
  return out;
}

function chatEntries() {
  return chats.map((c) => ({
    kind: "chat", id: `chat:${c.id}`, chat: c,
    label: c.title || String(c.peer_id),
    meta: c.username ? `@${c.username}` : t(`web.chat.kind_${c.type}`),
    run: () => {
      const R = routes();
      window.location.href = R.chatDetail ? R.chatDetail.replace("{id}", String(c.id)) : `${R.ask || "/chat"}?chat=${encodeURIComponent(c.id)}`;
    },
  }));
}

/** Belgilangan matn — `positions` fold indekslari, **hech qachon** HTML. */
function labelNode(text, positions) {
  const span = el("span.cmdk-label");
  if (!positions || !positions.length) { span.textContent = text; return span; }
  // fold uzunligi asl uzunlikdan farq qilishi mumkin (кирилл → 2 harf):
  // farq bo'lsa belgilashni tashlab, sof matn ko'rsatamiz.
  if (uzFold(text).length !== text.length) { span.textContent = text; return span; }
  const set = new Set(positions);
  let run = "", markRun = false;
  const flush = () => {
    if (!run) return;
    span.appendChild(markRun ? el("mark", null, run) : document.createTextNode(run));
    run = "";
  };
  for (let i = 0; i < text.length; i++) {
    const hit = set.has(i);
    if (hit !== markRun) { flush(); markRun = hit; }
    run += text[i];
  }
  flush();
  return span;
}

function group(title) {
  return el("li.cmdk-group", { role: "presentation" }, title);
}

function row(entry, idx, positions) {
  const li = el("li.cmdk-item", {
    role: "option", id: `cmdk-opt-${idx}`,
    "aria-selected": String(idx === cursor),
    dataset: { idx: String(idx) },
    onclick: () => runAt(idx),
    onmousemove: () => { if (cursor !== idx) { cursor = idx; paintCursor(); } },
  });
  if (entry.kind === "chat") {
    li.appendChild(el(`span.avatar.sm.${peerClass(entry.chat.peer_id)}`, null, initials(entry.label)));
  } else {
    li.appendChild(icon(entry.icon || "chevron-right"));
  }
  li.appendChild(labelNode(entry.label, positions));
  if (entry.meta) li.appendChild(el("span.cmdk-meta", null, entry.meta));
  return li;
}

function paintCursor() {
  for (const n of $$(".cmdk-item", list)) {
    const on = Number(n.dataset.idx) === cursor;
    n.setAttribute("aria-selected", String(on));
    if (on) { input.setAttribute("aria-activedescendant", n.id); n.scrollIntoView({ block: "nearest" }); }
  }
}

function render(q) {
  const query = (q || "").trim();
  clear(list);
  items = [];

  const sections = [
    [t("web.cmd.sec_chats"), chatEntries()],
    [t("web.cmd.sec_goto"), goEntries()],
    [t("web.cmd.sec_actions"), actionEntries()],
  ];

  for (const [title, pool] of sections) {
    const matched = query
      ? rank(query, pool, (e) => `${e.label} ${e.meta || ""}`, title === t("web.cmd.sec_chats") ? 8 : 6)
      : pool.slice(0, title === t("web.cmd.sec_chats") ? 5 : 6).map((item) => ({ item, positions: [] }));
    if (!matched.length) continue;
    list.appendChild(group(title));
    for (const { item, positions } of matched) {
      const idx = items.length;
      items.push(item);
      list.appendChild(row(item, idx, positions));
    }
  }

  if (!items.length) list.appendChild(el("li.cmdk-empty", { role: "presentation" }, t("web.cmd.empty")));
  cursor = 0;
  paintCursor();
}

function runAt(idx) {
  const entry = items[idx];
  if (!entry) return;
  haptic.select();
  const keep = entry.run();
  if (keep !== "keep") close();
}

function onKey(e) {
  if (e.key === "Escape") { e.preventDefault(); close(); return; }
  if (e.key === "ArrowDown" || (e.key === "n" && e.ctrlKey)) { e.preventDefault(); cursor = Math.min(items.length - 1, cursor + 1); paintCursor(); }
  else if (e.key === "ArrowUp" || (e.key === "p" && e.ctrlKey)) { e.preventDefault(); cursor = Math.max(0, cursor - 1); paintCursor(); }
  else if (e.key === "Home") { e.preventDefault(); cursor = 0; paintCursor(); }
  else if (e.key === "End") { e.preventDefault(); cursor = items.length - 1; paintCursor(); }
  else if (e.key === "Enter") { e.preventDefault(); runAt(cursor); }
}

const onInput = debounce(() => render(input.value), 80);

/** Palitrani ochadi (markup `_shell.html` da, boshida `hidden`). */
export function open(prefill = "") {
  if (!root || !root.hidden) return;
  lastFocus = document.activeElement;
  root.hidden = false;
  input.value = prefill;
  render(prefill);
  input.focus();
  untrap = trapFocus(box);
}

export function close() {
  if (!root || root.hidden) return;
  root.hidden = true;
  untrap?.(); untrap = null;
  input.setAttribute("aria-activedescendant", "");
  if (lastFocus && lastFocus.focus) lastFocus.focus();
}

export const toggle = () => (root && root.hidden ? open() : close());

/** Sahifa kirish nuqtasidan bir marta. Markup topilmasa jim chiqadi. */
export function init() {
  root = $("#cmdk");
  if (!root) return;
  box = $(".cmdk-box", root);
  input = $("#cmdk-input", root);
  list = $("#cmdk-list", root);

  input.addEventListener("input", onInput);
  box.addEventListener("keydown", onKey);
  root.addEventListener("mousedown", (e) => { if (e.target === root) close(); });

  const kbd = $(".cmdk-kbd", root);
  if (kbd) kbd.textContent = keyLabel("mod+k");

  bind("mod+k", () => toggle());
  bind("/", (e) => { if (!root.hidden) return false; open(); return true; });
  bind("escape", () => { if (root.hidden) return false; close(); return true; });

  for (const b of $$('[data-cmdk-open]')) b.addEventListener("click", () => open());
  ui.subscribe("theme", () => { if (!root.hidden) render(input.value); });
}
