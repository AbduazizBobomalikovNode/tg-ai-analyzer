/* pages/alerts.js — `/alerts`: qoidalar (CRUD) va hodisa lentasi.
 *
 * Eski `static/alerts.js` (dashboard ichidagi jadval) o'rniga. Farqlar:
 *   * `innerHTML` **umuman yo'q** — hamma narsa `el()` bilan quriladi, ya'ni
 *     chat sarlavhasi va hodisa matni (ikkalasi ham Telegram kontenti)
 *     hech qachon HTML bo'lib talqin qilinmaydi;
 *   * chat tanlash — raqam emas, fuzzy tanlagich (`lib/palette.js::rank`),
 *     ya'ni `ozbek` / `o'zbek` / `ўзбек` bir xil topiladi;
 *   * yoqish/o'chirish **optimistik**: switch darhol o'zgaradi, xato bo'lsa
 *     orqaga qaytadi va toast chiqadi.
 *
 * `?chat=<id>` — sahifa ochilishida forma o'sha chatga sozlanadi (chat
 * tafsiloti sahifasidan "shu chatga alert" tugmasi shu yo'l bilan keladi).
 */

import { $, $$, el, replace, clear, ready, trapFocus, debounce } from "/static/lib/dom.js";
import { t } from "/static/lib/i18n.js";
import { get, post, put, del } from "/static/lib/api.js";
import { icon } from "/static/lib/icons.js";
import { relTime } from "/static/lib/fmt.js";
import { rank, peerClass, initials } from "/static/lib/palette.js";
import { toast, toastError } from "/static/lib/toast.js";

/** Semantik qoida uchun minimal naqsh uzunligi (worker `EMBED_MIN_CHARS`). */
const SEMANTIC_MIN_CHARS = 20;
const PATTERN_MAX_CHARS = 300;

const state = {
  rules: [],
  chats: new Map(),      // id → {id, title, type, avatar_url, peer_id}
  accountId: null,
  editing: null,         // tahrirlanayotgan qoida yoki null
  chatId: null,          // formadagi tanlangan chat
  pickerRows: [],
  pickerCursor: 0,
};

/* ── Yordamchilar ────────────────────────────────────────────────────── */

const chatTitle = (id) => (id == null ? t("web.alerts.chat_all") : (state.chats.get(id)?.title || `#${id}`));

function avatarFor(id, title) {
  const row = id != null ? state.chats.get(id) : null;
  const node = el(`span.avatar.sm.${peerClass(row?.peer_id ?? id ?? 0)}`, { "aria-hidden": "true" });
  if (row && row.avatar_url) node.appendChild(el("img", { src: row.avatar_url, alt: "", loading: "lazy", decoding: "async" }));
  else node.textContent = initials(title || "?");
  return node;
}

function setErr(msg) {
  const node = $("#al-err");
  if (node) node.textContent = msg || "";
}

/* ── Qoida kartasi ───────────────────────────────────────────────────── */

function ruleCard(rule) {
  const sw = el("button.switch", {
    type: "button",
    role: "switch",
    "aria-checked": String(Boolean(rule.enabled)),
    "aria-label": t(rule.enabled ? "web.alerts.off" : "web.alerts.on"),
    onclick: () => toggleRule(rule, sw),
  });

  const facts = el("div.rule-facts", null, [
    el("span.pill", null, t(`web.alerts.kind_${rule.kind}`)),
    el("span.pill", null, [avatarFor(rule.chat_id, chatTitle(rule.chat_id)), chatTitle(rule.chat_id)]),
    rule.kind === "semantic" ? el("span.pill", null, `${t("web.alerts.threshold")} ${Number(rule.threshold).toFixed(2)}`) : null,
    el("span.pill", null, t("web.alerts.cooldown_n", { n: rule.cooldown_minutes })),
  ]);

  return el(`div.card-v2.rule-card${rule.enabled ? "" : ".is-off"}`, { dataset: { id: rule.id } }, [
    el("div.rule-top", null, [
      el("span.rule-ico", null, icon(rule.kind === "semantic" ? "brain" : "hash")),
      el("div.row-body", null, [
        el("div.rule-name", null, rule.label || rule.pattern),
        el("div.rule-pattern", null, rule.pattern),
      ]),
      sw,
    ]),
    facts,
    el("div.rule-btns", null, [
      el("button.btn.ghost.small", { type: "button", onclick: () => openForm(rule) }, [icon("edit-3"), t("web.alerts.edit")]),
      el("button.btn.ghost.small", { type: "button", onclick: () => showEvents(rule.id) }, [icon("list"), t("web.alerts.events")]),
      el("button.btn.ghost.small", { type: "button", onclick: () => removeRule(rule) }, [icon("trash-2"), t("web.alerts.delete")]),
    ]),
  ]);
}

function rulesEmpty() {
  return el("div.empty-v2", null, [
    el("span.empty-ico", null, icon("bell")),
    el("h2", null, t("web.alerts.empty_title")),
    el("p", null, t("web.alerts.empty_text")),
    el("div.empty-actions", null, [
      el("button.btn.primary", { type: "button", onclick: () => openForm(null) }, t("web.alerts.new")),
    ]),
  ]);
}

function paintRules() {
  const box = $("#al-rules");
  if (!box) return;
  if (!state.rules.length) { replace(box, rulesEmpty()); return; }
  replace(box, state.rules.map(ruleCard));
}

/* ── CRUD ────────────────────────────────────────────────────────────── */

async function toggleRule(rule, sw) {
  const next = !rule.enabled;
  sw.setAttribute("aria-checked", String(next));   // optimistik
  sw.disabled = true;
  try {
    const res = await put(`/api/alerts/${rule.id}`, { enabled: next });
    Object.assign(rule, res.rule || {});
    paintRules();
  } catch (err) {
    sw.setAttribute("aria-checked", String(rule.enabled));   // orqaga
    toastError(err);
  } finally {
    sw.disabled = false;
  }
}

async function removeRule(rule) {
  if (!window.confirm(t("web.alerts.confirm_delete"))) return;
  try {
    await del(`/api/alerts/${rule.id}`);
    state.rules = state.rules.filter((r) => r.id !== rule.id);
    paintRules();
    toast(t("web.alerts.deleted"), { kind: "ok" });
  } catch (err) { toastError(err); }
}

/* ── Hodisalar ───────────────────────────────────────────────────────── */

function eventRow(ev) {
  const body = el("div.row-body", null, [
    el("div.row-title", null, ev.chat_title || `#${ev.chat_id}`),
    el("div.event-text", null, ev.text || ""),
    el("div.row-sub", null, relTime(ev.created_at)),
  ]);
  const avatar = avatarFor(ev.chat_id, ev.chat_title);
  return ev.link
    ? el("a.row-item.event-item", { href: ev.link, target: "_blank", rel: "noopener" }, [avatar, body, icon("external-link")])
    : el("div.row-item.event-item", null, [avatar, body]);
}

async function loadEvents() {
  const box = $("#al-events");
  if (!box) return;
  const rules = state.rules;
  if (!rules.length) { replace(box, rulesEmpty()); return; }
  replace(box, el("p.muted", null, t("web.state.loading")));

  const settled = await Promise.allSettled(rules.map((r) => get(`/api/alerts/${r.id}/events?limit=20`)));
  const groups = [];
  settled.forEach((res, i) => {
    if (res.status !== "fulfilled") return;
    const items = res.value.items || [];
    if (!items.length) return;
    groups.push(el("section.event-group", null, [
      el("h3", null, rules[i].label || rules[i].pattern),
      ...items.map(eventRow),
    ]));
  });
  if (!groups.length) {
    replace(box, el("div.empty-v2", null, [
      el("span.empty-ico", null, icon("inbox")),
      el("h2", null, t("web.alerts.no_events")),
    ]));
    return;
  }
  replace(box, groups);
}

function showEvents(ruleId) {
  selectTab("events");
  loadEvents().then(() => {
    const first = $(`#al-events .event-group`);
    if (first) first.scrollIntoView({ block: "nearest" });
    void ruleId;
  });
}

/* ── Tab'lar ─────────────────────────────────────────────────────────── */

function selectTab(name) {
  for (const key of ["rules", "events"]) {
    const btn = $(`#tab-${key}`);
    const panel = $(`#panel-${key}`);
    if (btn) btn.setAttribute("aria-selected", String(key === name));
    if (panel) panel.hidden = key !== name;
  }
  if (name === "events") loadEvents();
}

/* ── Chat tanlagich (fuzzy) ──────────────────────────────────────────── */

function closePicker() {
  const list = $("#al-chat-list");
  const input = $("#al-chat-q");
  if (list) { list.hidden = true; clear(list); }
  if (input) input.setAttribute("aria-expanded", "false");
  state.pickerRows = [];
}

function pickChat(id) {
  state.chatId = id;
  const note = $("#al-chat-current");
  if (note) note.textContent = `${t("web.alerts.chat")}: ${chatTitle(id)}`;
  const input = $("#al-chat-q");
  if (input) input.value = id == null ? "" : chatTitle(id);
  closePicker();
}

function paintPicker() {
  const list = $("#al-chat-list");
  if (!list) return;
  clear(list);
  state.pickerRows.forEach((row, i) => {
    list.appendChild(el("li", {
      role: "option",
      "aria-selected": String(i === state.pickerCursor),
      onclick: () => pickChat(row.id),
      onmousemove: () => { state.pickerCursor = i; paintPicker(); },
    }, [
      avatarFor(row.id, row.title),
      el("span.picker-label", null, row.title),
    ]));
  });
  list.hidden = state.pickerRows.length === 0;
  const input = $("#al-chat-q");
  if (input) input.setAttribute("aria-expanded", String(state.pickerRows.length > 0));
}

function searchChats(q) {
  const all = [{ id: null, title: t("web.alerts.chat_all") }, ...state.chats.values()];
  state.pickerRows = q.trim()
    ? rank(q, all, (c) => `${c.title} ${c.username || ""}`, 10).map((r) => r.item)
    : all.slice(0, 10);
  state.pickerCursor = 0;
  paintPicker();
}

function initPicker() {
  const input = $("#al-chat-q");
  if (!input) return;
  const run = debounce(() => searchChats(input.value), 80);
  input.addEventListener("input", run);
  input.addEventListener("focus", () => searchChats(input.value));
  input.addEventListener("keydown", (e) => {
    if (!state.pickerRows.length) return;
    if (e.key === "ArrowDown") { e.preventDefault(); state.pickerCursor = Math.min(state.pickerRows.length - 1, state.pickerCursor + 1); paintPicker(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); state.pickerCursor = Math.max(0, state.pickerCursor - 1); paintPicker(); }
    else if (e.key === "Enter") { e.preventDefault(); pickChat(state.pickerRows[state.pickerCursor].id); }
    else if (e.key === "Escape") { closePicker(); }
  });
  document.addEventListener("mousedown", (e) => {
    const picker = $("#al-chat-picker");
    if (picker && !picker.contains(e.target)) closePicker();
  });
}

/* ── Forma (sheet) ───────────────────────────────────────────────────── */

let untrap = null;
let lastFocus = null;

function paintKind() {
  const kind = $("#al-kind")?.value || "keyword";
  for (const node of $$(".semantic-only")) node.hidden = kind !== "semantic";
}

function openForm(rule) {
  const sheet = $("#al-sheet");
  if (!sheet) return;
  state.editing = rule;
  setErr("");

  $("#al-sheet-title").textContent = t(rule ? "web.alerts.edit" : "web.alerts.new");
  $("#al-kind").value = rule ? rule.kind : "keyword";
  $("#al-pattern").value = rule ? rule.pattern : "";
  $("#al-label").value = rule ? rule.label : "";
  $("#al-cooldown").value = rule ? rule.cooldown_minutes : 30;
  $("#al-threshold").value = rule ? Number(rule.threshold).toFixed(2) : "0.80";
  if (rule && rule.account_id) $("#al-account").value = String(rule.account_id);
  pickChat(rule ? rule.chat_id : state.chatId);
  paintKind();

  lastFocus = document.activeElement;
  sheet.hidden = false;
  untrap = trapFocus(sheet);
  $("#al-pattern").focus();
}

function closeForm() {
  const sheet = $("#al-sheet");
  if (!sheet || sheet.hidden) return;
  sheet.hidden = true;
  untrap?.(); untrap = null;
  state.editing = null;
  lastFocus?.focus?.();
}

function readForm() {
  const kind = $("#al-kind").value;
  const pattern = $("#al-pattern").value.trim();
  const accountId = Number($("#al-account").value || 0);
  if (!accountId) return { error: t("web.alerts.err_account") };
  if (!pattern) return { error: t("web.alerts.err_pattern") };
  if (pattern.length > PATTERN_MAX_CHARS) return { error: t("web.alerts.err_pattern_long", { n: PATTERN_MAX_CHARS }) };
  if (kind === "semantic" && pattern.length < SEMANTIC_MIN_CHARS) {
    return { error: t("web.alerts.err_semantic_short", { n: SEMANTIC_MIN_CHARS }) };
  }
  return {
    body: {
      account_id: accountId,
      chat_id: state.chatId,
      kind,
      label: $("#al-label").value.trim(),
      pattern,
      threshold: Number($("#al-threshold").value || 0.8),
      cooldown_minutes: Number($("#al-cooldown").value || 30),
      enabled: state.editing ? state.editing.enabled : true,
    },
  };
}

async function submitForm(e) {
  e.preventDefault();
  setErr("");
  const { body, error } = readForm();
  if (error) { setErr(error); return; }
  const btn = $("#al-save");
  btn.disabled = true;
  try {
    if (state.editing) {
      const res = await put(`/api/alerts/${state.editing.id}`, body);
      const i = state.rules.findIndex((r) => r.id === state.editing.id);
      if (i >= 0) state.rules[i] = res.rule;
    } else {
      const res = await post("/api/alerts", body);
      state.rules.unshift(res.rule);
    }
    paintRules();
    closeForm();
    toast(t("web.alerts.saved"), { kind: "ok" });
  } catch (err) {
    setErr(err.message || t("error.generic"));
  } finally {
    btn.disabled = false;
  }
}

/* ── Yuklash ─────────────────────────────────────────────────────────── */

async function loadChats(accountId) {
  state.chats.clear();
  if (!accountId) return;
  try {
    const data = await get(`/api/accounts/${accountId}/chats?limit=500`);
    for (const c of data.items || []) {
      state.chats.set(c.id, {
        id: c.id, title: c.title || `#${c.id}`, type: c.type,
        username: c.username || "", avatar_url: c.avatar_url || "", peer_id: c.peer_id ?? c.id,
      });
    }
    if (window.TGAI) { window.TGAI.chats = data.items || []; window.dispatchEvent(new CustomEvent("tgai:chats")); }
  } catch (err) { toastError(err); }
}

async function loadRules() {
  try {
    state.rules = (await get("/api/alerts")).items || [];
  } catch (err) { state.rules = []; toastError(err); }
  paintRules();
}

async function main() {
  const sel = $("#al-account");
  let me = { accounts: [] };
  try { me = await get("/api/me"); } catch (err) { toastError(err); }
  clear(sel);
  for (const a of me.accounts || []) {
    sel.appendChild(el("option", { value: String(a.id) },
      `${a.label || a.tg_account_id || `#${a.id}`}${a.status !== "active" ? ` (${a.status})` : ""}`));
  }
  const active = (me.accounts || []).find((a) => a.status === "active") || (me.accounts || [])[0];
  state.accountId = active ? active.id : null;
  if (active) sel.value = String(active.id);

  sel.addEventListener("change", async () => {
    state.accountId = Number(sel.value || 0) || null;
    await loadChats(state.accountId);
    pickChat(null);
  });

  await loadChats(state.accountId);

  // `?chat=<id>` — formani shu chatga sozlaydi va darhol ochadi.
  const qChat = new URLSearchParams(window.location.search).get("chat");
  state.chatId = qChat && /^\d+$/.test(qChat) ? Number(qChat) : null;

  await loadRules();
  if (state.chatId != null) openForm(null);
}

ready(() => {
  $("#tab-rules")?.addEventListener("click", () => selectTab("rules"));
  $("#tab-events")?.addEventListener("click", () => selectTab("events"));
  $("#al-new")?.addEventListener("click", () => { state.chatId = null; openForm(null); });
  $("#al-kind")?.addEventListener("change", paintKind);
  $("#al-form")?.addEventListener("submit", submitForm);
  for (const b of $$("[data-sheet-cancel]")) b.addEventListener("click", closeForm);
  $("#al-sheet")?.addEventListener("mousedown", (e) => { if (e.target.id === "al-sheet") closeForm(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#al-sheet")?.hidden) closeForm();
  });
  initPicker();
  main();
});
