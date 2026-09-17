/* pages/today.js — `/today` uy ekrani.
 *
 * Bir ekranda to'rt manba: kunlik briefing, alert lentasi, "so'rang" maydoni
 * va sync/plitalar. Har biri **mustaqil** yuklanadi — bittasi yiqilsa qolgani
 * ko'rinaveradi (`Promise.allSettled`).
 *
 * Xavfsizlik: chat sarlavhasi, alert snippet'i va briefing matni —
 * Telegram'dan kelgan **ishonchsiz** matn. Ular yo `el()`/`textContent`
 * orqali, yoki briefing uchun yagona renderer `md()` orqali chiqadi;
 * `innerHTML` bu faylda faqat `md()` chiqishi uchun ishlatiladi.
 */

import { $, el, replace, clear, ready } from "/static/lib/dom.js";
import { md, hydrate } from "/static/lib/md.js";
import { t, has } from "/static/lib/i18n.js";
import { get } from "/static/lib/api.js";
import { icon } from "/static/lib/icons.js";
import { compact, usd, relTime, plural, weekdayDate } from "/static/lib/fmt.js";
import { peerClass, initials } from "/static/lib/palette.js";
import { toastError } from "/static/lib/toast.js";

const R = () => (window.TGAI && window.TGAI.routes) || {};
const askUrl = (q) => `${R().ask || "/ask"}${q ? `?q=${encodeURIComponent(q)}` : ""}`;

/** `YYYY-MM-DD` — foydalanuvchi mahalliy kunida (UTC'ga surilmasin). */
function dayKey(offset = 0) {
  const d = new Date();
  d.setDate(d.getDate() + offset);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/* ── Salomlashish va sana ────────────────────────────────────────────── */

/**
 * Haqiqiy ism — `/api/me` `display_name` ustuni hozircha Telegram
 * profilida alohida ism bo'lmasa `username` (yoki `id<tg_user_id>`) ga
 * tushadi. Ular bilan bir xil bo'lsa **ism yo'q** deb hisoblanadi — hech
 * qachon username/id ism sifatida "to'qib chiqarilmaydi" (L1).
 */
function realName(user) {
  if (!user) return "";
  const dn = String(user.display_name || "").trim();
  if (!dn) return "";
  const uname = String(user.username || "").trim();
  if (dn === uname) return "";
  if (user.tg_user_id != null && dn === `id${user.tg_user_id}`) return "";
  return dn;
}

function paintGreeting(name) {
  const h = new Date().getHours();
  const part = h < 5 ? "night" : h < 12 ? "morning" : h < 18 ? "day" : "evening";
  const node = $("#today-greeting");
  if (node) node.textContent = name ? t(`web.today.greet_${part}_name`, { name }) : t(`web.today.greet_${part}`);
  const dt = $("#today-date");
  // `weekdayDate()` — Chromium'ning ba'zi ICU to'plamlarida `uz-UZ` uchun
  // oy/kun nomi yo'q (`Intl` "M09 15, Tue" qaytaradi); `fmt.js` shu holatni
  // `web.fmt.*` lug'atidan qo'lda tuzatadi (H2).
  if (dt) dt.textContent = weekdayDate(new Date()) || dayKey();
}

/* ── Briefing kartasi ────────────────────────────────────────────────── */

function briefingEmpty() {
  return el("div.empty-v2", null, [
    el("span.empty-ico", null, icon("inbox")),
    el("h2", null, t("web.today.brief_none_title")),
    el("p", null, t("web.today.brief_building")),
  ]);
}

/** `md()` chiqishi — yagona renderer, escape-first (invariant). */
function mdBlock(cls, text) {
  const node = el(`div.${cls}`);
  node.innerHTML = md(String(text || ""));
  hydrate(node);
  return node;
}

function briefingCard(row, { stale }) {
  const structured = row.structured || {};
  const chats = Array.isArray(structured.chats) ? structured.chats : [];
  const out = [];

  if (stale) {
    out.push(el("p.field-note", null, t("web.today.brief_yesterday", { day: row.day })));
  }
  out.push(mdBlock("brief-tldr", structured.tldr || row.tldr || row.text || ""));

  if (chats.length) {
    const sections = el("div.brief-sections");
    for (const c of chats.slice(0, 8)) {
      const title = String(c.title || c.chat || "").trim() || t("web.today.brief_chat");
      const body = String(c.summary || c.text || c.tldr || "");
      if (!body) continue;
      sections.appendChild(
        el("details", null, [
          el("summary", null, [icon("chevron-down"), title]),
          mdBlock("brief-body", body),
        ]),
      );
    }
    if (sections.childElementCount) out.push(sections);
  }

  out.push(
    el("div.brief-foot", null, [
      el("a.btn.ghost.small", { href: `/briefings/${encodeURIComponent(row.day)}` }, [
        icon("external-link"), t("web.today.brief_open"),
      ]),
    ]),
  );
  return out;
}

async function loadBriefing(accountId) {
  const box = $("#brief-card");
  if (!box) return;
  const badge = $("#brief-badge");
  const q = accountId ? `?account_id=${accountId}` : "";

  for (const [offset, stale] of [[0, false], [-1, true]]) {
    const day = dayKey(offset);
    try {
      const row = await get(`/api/briefings/${day}${q}`);
      replace(box, briefingCard(row, { stale }));
      if (badge) {
        badge.hidden = false;
        badge.textContent = stale ? t("web.today.brief_stale") : t("web.today.brief_fresh");
        badge.className = stale ? "pill warn" : "pill ok";
      }
      return;
    } catch (err) {
      if (err && err.status !== 404) { replace(box, briefingEmpty()); return; }
    }
  }
  replace(box, briefingEmpty());
}

/* ── Alert lentasi ───────────────────────────────────────────────────── */

const FEED_RULES = 6;
const FEED_ITEMS = 12;

function feedEmpty() {
  return el("div.empty-v2", null, [
    el("span.empty-ico", null, icon("bell")),
    el("h2", null, t("web.today.feed_empty_title")),
    el("p", null, t("web.today.feed_empty_text")),
    el("div.empty-actions", null, [
      el("a.btn.primary", { href: "/alerts" }, t("web.today.feed_empty_cta")),
    ]),
  ]);
}

function feedRow(ev, ruleLabel) {
  const body = el("div.row-body", null, [
    el("div.row-title", null, ev.chat_title || `#${ev.chat_id}`),
    el("div.feed-snippet", null, ev.text || ""),
    el("div.row-sub", null, `${ruleLabel} · ${relTime(ev.created_at)}`),
  ]);
  const avatar = el(`span.avatar.sm.${peerClass(ev.chat_id)}`, { "aria-hidden": "true" }, initials(ev.chat_title || "?"));
  if (ev.link) {
    return el("a.row-item.feed-item", { href: ev.link, target: "_blank", rel: "noopener" }, [avatar, body, icon("external-link")]);
  }
  return el("div.row-item.feed-item", null, [avatar, body]);
}

async function loadFeed() {
  const box = $("#today-feed");
  if (!box) return;
  let rules = [];
  try {
    rules = (await get("/api/alerts")).items || [];
  } catch { replace(box, feedEmpty()); return; }
  if (!rules.length) { replace(box, feedEmpty()); return; }

  const enabled = rules.filter((r) => r.enabled).slice(0, FEED_RULES);
  const settled = await Promise.allSettled(
    enabled.map((r) => get(`/api/alerts/${r.id}/events?limit=5`)),
  );
  const rows = [];
  settled.forEach((res, i) => {
    if (res.status !== "fulfilled") return;
    const rule = enabled[i];
    for (const ev of res.value.items || []) rows.push({ ev, label: rule.label || rule.pattern });
  });
  rows.sort((a, b) => String(b.ev.created_at || "").localeCompare(String(a.ev.created_at || "")));
  if (!rows.length) { replace(box, feedEmpty()); return; }
  replace(box, el("div.feed", null, rows.slice(0, FEED_ITEMS).map(({ ev, label }) => feedRow(ev, label))));
}

/* ── Sync holati va uzilgan akkaunt ──────────────────────────────────── */

async function loadSync(account) {
  const box = $("#today-sync");
  if (!box || !account) return;

  if (account.status && account.status !== "active") {
    box.hidden = false;
    replace(box, el("div.banner.err", null, [
      icon("alert-triangle"),
      el("span", null, t("web.today.sync_revoked")),
      el("span.spacer"),
      el("a.btn.small", { href: "/onboarding" }, t("web.today.sync_relogin")),
    ]));
    return;
  }

  let st;
  try { st = await get(`/api/accounts/${account.id}/sync/status`); } catch { return; }
  const counts = st.counts || {};
  const running = Number(counts.running || 0);
  const done = Number(counts.done || 0);
  const failed = Number(counts.failed || 0);

  const line = el("div.sync-line");
  if (st.running) {
    line.appendChild(icon("refresh-cw", { cls: "spin" }));
    line.appendChild(el("span", null, t("web.today.sync_running", { n: running })));
  } else if (failed) {
    line.appendChild(icon("alert-triangle"));
    line.appendChild(el("span", null, t("web.today.sync_failed", { n: failed })));
  } else if (done) {
    line.appendChild(icon("check-check"));
    line.appendChild(el("span", null, t("web.today.sync_all")));
  } else {
    return;  // aytadigan yangilik yo'q — band ko'rsatilmaydi
  }

  const card = el("div.card-v2.sync-strip", null, [line]);
  if (st.running) {
    const chats = (st.chats || []).filter((c) => c.state === "running").slice(0, 3);
    for (const c of chats) {
      card.appendChild(el("div.progress", null, el("i", { style: { inlineSize: `${c.progress || 0}%` } })));
    }
  }
  box.hidden = false;
  replace(box, card);
}

/* ── Plitalar ────────────────────────────────────────────────────────── */

function tile(href, iconName, label, value, sub) {
  // L8: yorliq **bitta qatorda** (ellipsis) + `title` — to'liq matn hover'da
  // ko'rinsin ("KUTAYOTGAN" / "AMALLAR" ikki qatorga bo'linib ketmasin).
  const kids = [
    el("span.tile-k", { title: label }, [icon(iconName), el("span", null, label)]),
    el("span.tile-v", null, value),
    sub ? el("span.tile-s", null, sub) : null,
  ];
  return href ? el("a.tile-v2", { href }, kids) : el("div.tile-v2", null, kids);
}

async function loadTiles(me) {
  const box = $("#today-tiles");
  if (!box) return;
  const accounts = me.accounts || [];
  const chats = accounts.reduce((a, x) => a + Number(x.chats_total || 0), 0);
  const synced = accounts.reduce((a, x) => a + Number(x.chats_synced || 0), 0);

  const [actionsRes, statsRes] = await Promise.allSettled([
    get("/api/actions?status=proposed&limit=200"),
    get("/api/stats/overview?days=1"),
  ]);
  const pending = actionsRes.status === "fulfilled" ? (actionsRes.value.items || []).length : 0;
  const totals = statsRes.status === "fulfilled" ? statsRes.value.totals || {} : {};

  replace(box, [
    tile("/actions", "check-check", t("web.today.tile_actions"), String(pending),
      pending ? t("web.today.tile_actions_sub") : ""),
    tile(R().chats || null, "messages-square", t("web.today.tile_chats"), compact(chats),
      plural("web.n.chats", synced)),
    tile("/dashboard", "bar-chart-3", t("web.today.tile_cost"), usd(totals.cost_usd || 0),
      compact((totals.tokens_in || 0) + (totals.tokens_out || 0))),
  ]);

  const badge = document.querySelector("[data-actions-badge]");
  if (badge) { badge.textContent = String(pending); badge.hidden = pending === 0; }
}

/* ── So'rang ─────────────────────────────────────────────────────────── */

function initAsk(hasChats) {
  const form = $("#today-ask");
  const input = $("#today-q");
  if (form && input) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = input.value.trim();
      window.location.href = askUrl(q);
    });
  }
  const chips = $("#today-suggests");
  if (!chips) return;
  clear(chips);
  const keys = hasChats
    ? ["web.today.q1", "web.today.q2", "web.today.q3"]
    : ["web.today.q1", "web.today.q3"];
  for (const k of keys) {
    if (!has(k)) continue;
    const text = t(k);
    chips.appendChild(el("button.chip-btn", {
      type: "button",
      onclick: () => { window.location.href = askUrl(text); },
    }, text));
  }
}

/* ── Kirish nuqtasi ──────────────────────────────────────────────────── */

async function main() {
  let me = { user: {}, accounts: [] };
  try {
    me = await get("/api/me");
  } catch (err) {
    toastError(err);
  }
  paintGreeting(realName(me.user));
  const account = (me.accounts || []).find((a) => a.status === "active") || (me.accounts || [])[0] || null;
  initAsk(Boolean(account && account.chats_synced));

  await Promise.allSettled([
    loadBriefing(account ? account.id : null),
    loadFeed(),
    loadSync(account),
    loadTiles(me),
  ]);
}

ready(() => { main(); });
