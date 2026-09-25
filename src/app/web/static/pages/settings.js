/* pages/settings.js — `/settings`.
 *
 * Sakkiz bo'lim, bitta sahifa. Har bo'lim **mustaqil** yuklanadi va o'z
 * endpoint'iga yozadi: ko'rinish sozlamalari `PUT /api/settings/prefs`,
 * yozish rejimi `PATCH .../write_mode`, auto-reply va briefing o'z
 * formalarida. Bittasi yiqilsa qolgani ishlayveradi.
 *
 * `prefs` yozish **debounce** bilan: segment tugmasini ketma-ket bosish
 * har bosishda so'rov yubormasin.
 */

import { $, $$, el, replace, clear, ready, trapFocus, debounce } from "/static/lib/dom.js";
import { t, has } from "/static/lib/i18n.js";
import { get, put, patch, post, del } from "/static/lib/api.js";
import { icon } from "/static/lib/icons.js";
import { compact, usd, dateTime } from "/static/lib/fmt.js";
import { peerClass, initials } from "/static/lib/palette.js";
import { toast, toastError } from "/static/lib/toast.js";
import * as themeMod from "/static/lib/theme.js";
import { ui } from "/static/lib/store.js";
import { openSheet } from "/static/lib/sync-center.js";

const WRITE_MODES = ["read_only", "write_with_confirm", "autonomous"];
const state = { me: null, account: null, chats: [], prefs: {}, briefChats: null, unlinkId: null };

/**
 * `display_name` hozircha Telegram profilida alohida ism bo'lmasa
 * `username`/`id<tg_user_id>` ga tushadi — u holda ism yo'q, faqat
 * `@username` ko'rsatiladi (L1, `pages/today.js::realName` bilan bir xil
 * mezon; ism hech qachon "to'qib chiqarilmaydi").
 */
function displayValue(user) {
  if (!user) return "";
  const dn = String(user.display_name || "").trim();
  const uname = String(user.username || "").trim();
  const isReal = dn && dn !== uname && !(user.tg_user_id != null && dn === `id${user.tg_user_id}`);
  if (isReal) return dn;
  return uname ? `@${uname}` : dn;
}

/* ── prefs ───────────────────────────────────────────────────────────── */

const savePrefs = debounce(async (patchBody) => {
  try {
    const res = await put("/api/settings/prefs", patchBody);
    state.prefs = res.prefs || state.prefs;
  } catch (err) { toastError(err); }
}, 250);

function segment(id, attr, value, onPick) {
  const box = $(id);
  if (!box) return;
  const paint = (v) => {
    for (const b of $$("button", box)) b.setAttribute("aria-selected", String(b.dataset[attr] === v));
  };
  paint(value);
  for (const b of $$("button", box)) {
    b.addEventListener("click", () => { paint(b.dataset[attr]); onPick(b.dataset[attr]); });
  }
}

function switchBtn(id, checked, onPick) {
  const btn = $(id);
  if (!btn) return;
  btn.setAttribute("aria-checked", String(Boolean(checked)));
  btn.addEventListener("click", () => {
    const next = btn.getAttribute("aria-checked") !== "true";
    btn.setAttribute("aria-checked", String(next));
    onPick(next);
  });
}

/* ── 1. Profil va akkauntlar ─────────────────────────────────────────── */

function accountCard(a) {
  const statusCls = a.status === "active" ? "ok" : a.status === "revoked" ? "err" : "warn";
  const kids = [
    el("div.acct-top", null, [
      el(`span.avatar.sm.${peerClass(a.tg_account_id || a.id)}`, { "aria-hidden": "true" },
        initials(a.label || `#${a.id}`)),
      el("div.row-body", null, [
        el("div.row-title", null, a.label || `#${a.id}`),
        el("div.row-sub", null, t("web.settings.chats_synced", {
          synced: a.chats_synced || 0, total: a.chats_total || 0,
        })),
      ]),
      el(`span.pill.${statusCls}`, null, has(`web.settings.status_${a.status}`) ? t(`web.settings.status_${a.status}`) : a.status),
    ]),
  ];
  const btns = [];
  if (a.status === "active") {
    // Sync markazi — `/today` bandi va `/chats` ikonkasi bilan **bitta**
    // komponent (`lib/sync-center.js`).
    btns.push(el("button.btn.small", {
      type: "button",
      onclick: () => openSheet(a.id),
    }, [icon("refresh-cw"), t("web.sync.details")]));
  } else {
    btns.push(el("a.btn.small", { href: "/onboarding" }, [icon("qr-code"), t("web.settings.relogin")]));
  }
  btns.push(el("button.btn.ghost.small", {
    type: "button",
    onclick: () => openUnlink(a),
  }, [icon("log-out"), t("web.settings.unlink")]));
  kids.push(el("div.form-actions", null, btns));
  return el("div.card-v2.acct-card", null, kids);
}

function paintAccounts() {
  const box = $("#set-accounts");
  if (!box) return;
  const accounts = state.me.accounts || [];
  replace(box, accounts.length
    ? accounts.map(accountCard)
    : el("p.set-note", null, t("web.chat.no_accounts")));
  const add = $("#set-add-account");
  if (add) add.hidden = accounts.length >= Number(state.me.max_accounts || 1);
}

/* ── Uzish tasdig'i ──────────────────────────────────────────────────── */

let untrap = null;

function openUnlink(a) {
  const sheet = $("#unlink-sheet");
  if (!sheet) return;
  state.unlinkId = a.id;
  $("#unlink-text").textContent = t("web.settings.unlink_text", { name: a.label || `#${a.id}` });
  sheet.hidden = false;
  untrap = trapFocus(sheet);
  $("#unlink-yes").focus();
}

function closeUnlink() {
  const sheet = $("#unlink-sheet");
  if (!sheet || sheet.hidden) return;
  sheet.hidden = true;
  untrap?.(); untrap = null;
  state.unlinkId = null;
}

async function doUnlink() {
  const id = state.unlinkId;
  if (!id) return;
  try {
    await del(`/api/accounts/${id}`);
    toast(t("web.settings.unlinked"), { kind: "ok" });
    closeUnlink();
    state.me = await get("/api/me");
    paintAccounts();
  } catch (err) { toastError(err); }
}

/* ── 4. Chat bo'yicha yozish rejimi ──────────────────────────────────── */

function chatAvatar(c) {
  const node = el(`span.avatar.sm.${peerClass(c.peer_id ?? c.id)}`, { "aria-hidden": "true" });
  if (c.avatar_url) node.appendChild(el("img", { src: c.avatar_url, alt: "", loading: "lazy", decoding: "async" }));
  else node.textContent = initials(c.title || "?");
  return node;
}

function writeModeRow(c) {
  const sel = el("select", { "aria-label": t("web.chat.write_mode") });
  for (const mode of WRITE_MODES) sel.appendChild(el("option", { value: mode }, t(`web.chat.wm_${mode}`)));
  sel.value = c.write_mode || "write_with_confirm";
  sel.disabled = !c.is_writable;
  sel.addEventListener("change", async () => {
    const prev = c.write_mode;
    try {
      const res = await patch(`/api/accounts/${state.account.id}/chats/${c.id}/write_mode`, { mode: sel.value });
      c.write_mode = res.write_mode;
      toast(t("web.chat.wm_saved"), { kind: "ok" });
    } catch (err) {
      sel.value = prev || "write_with_confirm";
      toastError(err);
    }
  });
  return el("div.row-item", null, [
    chatAvatar(c),
    el("div.row-body", null, [
      el("div.row-title", null, c.title || `#${c.id}`),
      el("div.row-sub", null, c.is_writable ? t(`web.chat.kind_${c.type}`) : t("web.chat.wm_not_writable")),
    ]),
    sel,
  ]);
}

function paintWriteModes() {
  const box = $("#set-write-modes");
  if (!box) return;
  const rows = state.chats.filter((c) => c.is_writable).slice(0, 200);
  replace(box, rows.length ? rows.map(writeModeRow) : el("p.set-note", null, t("web.state.empty")));
}

/* ── 5. Auto-reply ───────────────────────────────────────────────────── */

function arNote(msg) { const n = $("#ar-note"); if (n) n.textContent = msg || ""; }

function fillArSelect() {
  const sel = $("#ar-chat");
  if (!sel) return;
  clear(sel);
  sel.appendChild(el("option", { value: "" }, t("web.settings.pick_chat")));
  for (const c of state.chats.filter((x) => x.is_writable)) {
    sel.appendChild(el("option", { value: String(c.id) }, c.title || `#${c.id}`));
  }
  sel.addEventListener("change", loadAutoReply);
}

function setAr(rule) {
  $("#ar-enabled").setAttribute("aria-checked", String(Boolean(rule && rule.enabled)));
  $("#ar-trigger").value = (rule && rule.trigger) || "questions";
  $("#ar-keywords").value = (rule && rule.keywords) || "";
  $("#ar-instructions").value = (rule && rule.instructions) || "";
  $("#ar-max").value = (rule && rule.max_per_hour) || 5;
  $("#ar-qfrom").value = rule && rule.quiet_from != null ? rule.quiet_from : "";
  $("#ar-qto").value = rule && rule.quiet_to != null ? rule.quiet_to : "";
}

async function loadAutoReply() {
  const chatId = Number($("#ar-chat").value || 0);
  arNote("");
  if (!chatId || !state.account) { setAr(null); return; }
  try {
    const data = await get(`/api/accounts/${state.account.id}/chats/${chatId}/autoreply`);
    setAr(data.rule);
  } catch (err) { arNote(err.message || t("error.generic")); }
}

async function saveAutoReply(e) {
  e.preventDefault();
  const chatId = Number($("#ar-chat").value || 0);
  if (!chatId || !state.account) { arNote(t("web.settings.pick_chat")); return; }
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  try {
    await put(`/api/accounts/${state.account.id}/chats/${chatId}/autoreply`, {
      enabled: $("#ar-enabled").getAttribute("aria-checked") === "true",
      trigger: $("#ar-trigger").value,
      keywords: $("#ar-keywords").value.trim(),
      instructions: $("#ar-instructions").value.trim(),
      max_per_hour: Number($("#ar-max").value || 5),
      quiet_from: num("#ar-qfrom"),
      quiet_to: num("#ar-qto"),
    });
    arNote(t("web.ar.saved"));
  } catch (err) { arNote(err.message || t("error.generic")); }
}

/* ── 6. Briefing va jim soatlar ──────────────────────────────────────── */

function fillTimezones(current) {
  const sel = $("#b-tz");
  if (!sel) return;
  let zones = [];
  try { zones = Intl.supportedValuesOf("timeZone"); } catch { zones = []; }
  if (!zones.length) zones = ["UTC", "Asia/Tashkent", "Europe/Moscow"];
  if (current && !zones.includes(current)) zones.unshift(current);
  clear(sel);
  for (const z of zones) sel.appendChild(el("option", { value: z }, z));
  sel.value = current || "Asia/Tashkent";
}

function paintBriefChats(selected) {
  const box = $("#b-chats");
  if (!box) return;
  state.briefChats = selected;   // null = barcha chatlar
  const rows = state.chats.slice(0, 200).map((c) => {
    const cb = el("input", { type: "checkbox" });
    cb.checked = selected == null ? false : selected.includes(c.id);
    cb.addEventListener("change", () => {
      const set = new Set(state.briefChats || []);
      if (cb.checked) set.add(c.id); else set.delete(c.id);
      state.briefChats = set.size ? [...set] : null;
    });
    return el("label.row-item", null, [cb, chatAvatar(c), el("div.row-body", null, el("div.row-title", null, c.title || `#${c.id}`))]);
  });
  replace(box, rows.length ? rows : el("p.set-note", null, t("web.state.empty")));
}

async function loadBriefing() {
  if (!state.account) return;
  let data;
  try { data = await get(`/api/settings/briefing?account_id=${state.account.id}`); } catch { return; }
  const s = data.settings || {};
  $("#b-enabled").setAttribute("aria-checked", String(Boolean(s.enabled)));
  $("#b-hour").value = s.hour ?? 8;
  fillTimezones(s.tz);
  if (s.quiet_from) $("#b-qfrom").value = s.quiet_from;
  if (s.quiet_to) $("#b-qto").value = s.quiet_to;
  paintBriefChats(s.chats);
}

async function saveBriefing(e) {
  e.preventDefault();
  const note = $("#brief-note");
  if (!state.account) return;
  try {
    await put("/api/settings/briefing", {
      account_id: state.account.id,
      tz: $("#b-tz").value,
      hour: Number($("#b-hour").value || 8),
      enabled: $("#b-enabled").getAttribute("aria-checked") === "true",
      quiet_from: $("#b-qfrom").value || null,
      quiet_to: $("#b-qto").value || null,
      chats: state.briefChats,
    });
    if (note) note.textContent = t("web.brief.saved");
  } catch (err) {
    if (note) note.textContent = err.message || t("error.generic");
  }
}

/* ── 3. Model / byudjet (faqat o'qish) ───────────────────────────────── */

async function loadSystem() {
  const box = $("#set-system");
  if (!box) return;
  let sys;
  try { sys = await get("/api/stats/system"); } catch { return; }
  const b = sys.budget || {};
  const rows = [
    [t("web.dash.budget"), `${compact(b.tokens || 0)}${b.token_limit ? ` / ${compact(b.token_limit)}` : ""} · ${usd(b.cost_usd || 0)}${b.cost_limit ? ` / ${usd(b.cost_limit)}` : ""}`],
    ["v", String(sys.version || "")],
  ];
  replace(box, rows.map(([k, v]) => el("div", null, [el("span", null, k), el("b", null, v)])));
}

/* ── 3b. Kunlik byudjet ──────────────────────────────────────────────── */
/*
 * Limit **foydalanuvchi** bo'yicha (barcha akkauntlar uchun umumiy) — karta
 * har akkaunt sozlamalarida turadi, `account_id` faqat vaqt zonasi uchun
 * (boost standart muddati — shu zonadagi keyingi yarim tun).
 */

const bud = { data: null };

const budQuery = () => (state.account ? `?account_id=${state.account.id}` : "");
const budNote = (msg) => { const n = $("#bud-note"); if (n) n.textContent = msg || ""; };

function budMeter(label, used, limit, fmtVal, key) {
  const pctVal = limit ? Math.min(100, (100 * used) / limit) : 0;
  const bar = el("i", { style: { inlineSize: `${pctVal.toFixed(1)}%` } });
  if (limit && pctVal >= 100) bar.classList.add("is-hot");
  else if (limit && pctVal >= 80) bar.classList.add("is-warn");
  const text = t(key, { used: fmtVal(used), limit: limit ? fmtVal(limit) : t("web.budget.unlimited") });
  return el("div.bud-meter", null, [
    el("div.bud-meter-top", null, [el("span", null, label), el("b", null, text)]),
    el("div.progress", {
      role: "progressbar",
      "aria-label": text,
      "aria-valuemin": "0",
      "aria-valuemax": "100",
      "aria-valuenow": String(Math.round(pctVal)),
    }, bar),
  ]);
}

function paintBudget(d) {
  bud.data = d;
  const box = $("#bud-usage");
  if (!box) return;
  const u = d.usage_24h || {};
  const lim = d.limits || {};
  const kids = [
    budMeter(`${t("web.budget.usage")} · ${t("web.budget.requests", { n: u.requests || 0 })}`,
      Number(u.tokens || 0), lim.tokens, compact, "web.budget.tokens_of"),
    budMeter("", Number(u.usd || 0), lim.usd, usd, "web.budget.usd_of"),
  ];
  const status = [el(`span.pill.${lim.source === "custom" ? "accent" : "ok"}`, null,
    t(lim.source === "custom" ? "web.budget.source_custom" : "web.budget.source_global"))];
  if (d.boost) status.push(el("span.pill.ok", null, t("web.budget.boost_pill")));
  if (d.last_hit && d.last_hit.at) {
    status.push(el("span.field-note", null, t("web.budget.last_hit", { time: dateTime(d.last_hit.at) })));
  }
  kids.push(el("div.bud-status", null, status));
  if (d.blocked) {
    const msg = [t(`web.budget.blocked_${d.blocked === "cost" ? "cost" : "tokens"}`)];
    const at = d.resets_hint && d.resets_hint.at;
    if (at) msg.push(t("web.budget.resets_at", { time: dateTime(at) }));
    kids.push(el("div.bud-alert", { role: "status" }, [icon("alert-triangle"), el("span", null, msg.join(" "))]));
  }
  replace(box, kids);

  // Forma: joriy override (bo'sh = standart); placeholder — global qiymat.
  const g = lim.global || {};
  const c = lim.custom || {};
  const tokIn = $("#bud-tokens"), usdIn = $("#bud-usd");
  if (tokIn) { tokIn.value = c.tokens != null ? String(c.tokens) : ""; tokIn.placeholder = g.tokens ? String(g.tokens) : "0"; }
  if (usdIn) { usdIn.value = c.usd != null ? String(c.usd) : ""; usdIn.placeholder = g.usd ? String(g.usd) : "0"; }
  const defs = [t("web.budget.defaults", {
    tokens: g.tokens ? compact(g.tokens) : t("web.budget.unlimited"),
    usd: g.usd ? usd(g.usd) : t("web.budget.unlimited"),
  })];
  if (lim.ceiling) {
    defs.push(t("web.budget.ceiling", {
      tokens: lim.ceiling.tokens ? compact(lim.ceiling.tokens) : t("web.budget.unlimited"),
      usd: lim.ceiling.usd ? usd(lim.ceiling.usd) : t("web.budget.unlimited"),
    }));
  }
  const defBox = $("#bud-defaults");
  if (defBox) defBox.textContent = defs.join(" ");
  const reset = $("#bud-reset");
  if (reset) reset.disabled = lim.source !== "custom";

  const win = $("#bud-boost-window");
  if (win) win.textContent = t("web.budget.boost_note", { tz: d.tz || "Asia/Tashkent" });
  const act = $("#bud-boost-active");
  if (act) {
    if (d.boost) {
      replace(act, el("div.bud-boost-active", null, [
        icon("zap"),
        el("span", null, t("web.budget.boost_active", {
          tokens: compact(d.boost.tokens || 0),
          usd: usd(d.boost.usd || 0),
          time: dateTime(d.boost.until),
        })),
        el("button.btn.ghost.small", { type: "button", onclick: cancelBoost }, t("web.budget.boost_cancel")),
      ]));
      act.hidden = false;
    } else {
      clear(act);
      act.hidden = true;
    }
  }
}

async function loadBudget() {
  try { paintBudget(await get(`/api/settings/budget${budQuery()}`)); } catch (err) {
    const box = $("#bud-usage");
    if (box) replace(box, el("p.set-note", null, err.message || t("error.generic")));
  }
}

/** Bo'sh input — `null` (standart); aks holda manfiy bo'lmagan son yoki `NaN`. */
function numOrNull(id) {
  const raw = String($(id)?.value ?? "").trim();
  if (raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : NaN;
}

async function saveBudget(e) {
  e.preventDefault();
  const tokens = numOrNull("#bud-tokens");
  const cost = numOrNull("#bud-usd");
  if (Number.isNaN(tokens) || Number.isNaN(cost)) { budNote(t("web.budget.err_invalid")); return; }
  try {
    paintBudget(await put(`/api/settings/budget${budQuery()}`, {
      token_budget: tokens == null ? null : Math.round(tokens),
      cost_budget_usd: cost,
    }));
    budNote(t("web.budget.saved"));
  } catch (err) { budNote(err.message || t("error.generic")); }
}

async function resetBudget() {
  try {
    paintBudget(await put(`/api/settings/budget${budQuery()}`, { token_budget: null, cost_budget_usd: null }));
    budNote(t("web.budget.reset_done"));
  } catch (err) { budNote(err.message || t("error.generic")); }
}

async function addBoost(tokens, cost) {
  try {
    paintBudget(await post("/api/settings/budget/boost", {
      tokens: tokens || 0,
      usd: cost || 0,
      account_id: state.account ? state.account.id : null,
    }));
    toast(t("web.budget.boost_added"), { kind: "ok" });
    return true;
  } catch (err) { toastError(err); return false; }
}

async function cancelBoost() {
  try {
    paintBudget(await del(`/api/settings/budget/boost${budQuery()}`));
    toast(t("web.budget.boost_cleared"), { kind: "ok" });
  } catch (err) { toastError(err); }
}

async function submitBoost(e) {
  e.preventDefault();
  const tokens = numOrNull("#bud-boost-tokens");
  const cost = numOrNull("#bud-boost-usd");
  if (Number.isNaN(tokens) || Number.isNaN(cost)) { toast(t("web.budget.err_invalid"), { kind: "err" }); return; }
  if (!tokens && !cost) { toast(t("web.budget.err_boost_empty"), { kind: "err" }); return; }
  if (await addBoost(tokens ? Math.round(tokens) : 0, cost || 0)) {
    $("#bud-boost-tokens").value = "";
    $("#bud-boost-usd").value = "";
  }
}

/* ── Kirish nuqtasi ──────────────────────────────────────────────────── */

async function main() {
  try { state.me = await get("/api/me"); } catch (err) { toastError(err); return; }
  state.prefs = (state.me.user && state.me.user.prefs) || {};
  $("#set-name").value = displayValue(state.me.user);
  paintAccounts();

  // Ko'rinish
  segment("#set-theme", "mode", state.prefs.theme || themeMod.mode(), (v) => {
    themeMod.set(v);
    savePrefs({ theme: v });
  });
  segment("#set-density", "density", state.prefs.density || "comfortable", (v) => {
    document.documentElement.dataset.density = v;
    ui.set("density", v);
    savePrefs({ density: v });
  });
  segment("#set-lang", "lang", (window.TGAI || {}).lang || "uz", async (v) => {
    try { await put("/api/settings/prefs", { locale: v }); } catch { /* cookie yo'li baribir ishlaydi */ }
    const u = new URL(window.location.href);
    u.searchParams.set("lang", v);
    window.location.href = u.toString();
  });

  // AI standartlari
  const mode = $("#set-mode"), strategy = $("#set-strategy");
  if (mode) { mode.value = state.prefs.default_mode || "auto"; mode.addEventListener("change", () => savePrefs({ default_mode: mode.value })); }
  if (strategy) { strategy.value = state.prefs.default_strategy || "auto"; strategy.addEventListener("change", () => savePrefs({ default_strategy: strategy.value })); }
  switchBtn("#set-deep", state.prefs.deep_default, (v) => savePrefs({ deep_default: v }));

  state.account = (state.me.accounts || []).find((a) => a.status === "active") || (state.me.accounts || [])[0] || null;
  if (state.account) {
    try {
      state.chats = (await get(`/api/accounts/${state.account.id}/chats?limit=500`)).items || [];
    } catch (err) { toastError(err); }
  }
  paintWriteModes();
  fillArSelect();
  await Promise.allSettled([loadBriefing(), loadSystem(), loadBudget()]);
}

ready(() => {
  $("#ar-form")?.addEventListener("submit", saveAutoReply);
  $("#brief-form")?.addEventListener("submit", saveBriefing);
  $("#bud-form")?.addEventListener("submit", saveBudget);
  $("#bud-reset")?.addEventListener("click", resetBudget);
  $("#bud-boost-form")?.addEventListener("submit", submitBoost);
  for (const chip of $("#bud-chips [data-tokens], #bud-chips [data-usd]")) {
    chip.addEventListener("click", () => addBoost(Number(chip.dataset.tokens || 0), Number(chip.dataset.usd || 0)));
  }
  $("#unlink-yes")?.addEventListener("click", doUnlink);
  for (const b of $$("[data-sheet-cancel]")) b.addEventListener("click", closeUnlink);
  $("#unlink-sheet")?.addEventListener("mousedown", (e) => { if (e.target.id === "unlink-sheet") closeUnlink(); });
  $("#set-logout")?.addEventListener("click", async () => {
    try { await post("/api/auth/logout"); } catch { /* cookie baribir tozalanadi */ }
    window.location.href = "/login";
  });

  // Desktop sub-nav: skroll bilan faol bo'lim belgilanadi.
  //
  // M4: bitta `entries` chaqiruvida faqat **o'zgargan** bo'limlar keladi
  // (IntersectionObserver shartnomasi) — avvalgi kodi shu batch ichidagi
  // har bir yozuv uchun HAMMA link'ni qayta belgilardi, ya'ni bir vaqtda
  // ikki bo'lim ko'rinib tursa (yoki eski holat yangilanmasa) ikkalasi
  // ham "faol" bo'lib qolardi. Endi ko'rinib turgan bo'limlar to'plami
  // saqlanadi va har chaqiruvda **faqat bittasi** (sahifa tartibida
  // birinchisi) faol deb belgilanadi.
  const links = $$(".set-nav a");
  const sections = $$(".set-section");
  if (links.length && sections.length && "IntersectionObserver" in window) {
    const visible = new Set();
    const paintActive = () => {
      const active = sections.find((s) => visible.has(s.id));
      for (const a of links) {
        a.setAttribute("aria-current", String(Boolean(active) && a.getAttribute("href") === `#${active.id}`));
      }
    };
    const obs = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) visible.add(e.target.id);
        else visible.delete(e.target.id);
      }
      paintActive();
    }, { rootMargin: "-20% 0px -70% 0px" });
    for (const s of sections) obs.observe(s);
  }

  main();
});
