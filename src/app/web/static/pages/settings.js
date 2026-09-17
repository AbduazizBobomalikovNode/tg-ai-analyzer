/* pages/settings.js — `/settings`.
 *
 * Yetti bo'lim, bitta sahifa. Har bo'lim **mustaqil** yuklanadi va o'z
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
import { compact, usd } from "/static/lib/fmt.js";
import { peerClass, initials } from "/static/lib/palette.js";
import { toast, toastError } from "/static/lib/toast.js";
import * as themeMod from "/static/lib/theme.js";
import { ui } from "/static/lib/store.js";

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
  if (a.status !== "active") {
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
  await Promise.allSettled([loadBriefing(), loadSystem()]);
}

ready(() => {
  $("#ar-form")?.addEventListener("submit", saveAutoReply);
  $("#brief-form")?.addEventListener("submit", saveBriefing);
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
