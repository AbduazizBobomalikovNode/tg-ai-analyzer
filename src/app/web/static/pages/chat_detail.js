/* pages/chat_detail.js — `/chats/{id}` tafsiloti (R2-A).
 *
 * Bo'limlar: sarlavha (avatar · bio clamp · sync pill) → tez amallar →
 * Overview (KPI + sparkline) · Activity (7×24 heatmap) · Top postlar ·
 * Digest'lar · A'zolar (o'rin egallovchi karta).
 *
 * Nozik joylar:
 *   * **Heatmap taxminiy.** `/stats` faqat ikkita chekka taqsimot beradi
 *     (`by_weekday`, `by_hour`), 7×24 matritsa emas. Biz ularning tashqi
 *     ko'paytmasini chizamiz va buni kartada **ochiq aytamiz** — aks holda
 *     foydalanuvchi o'lchangan ma'lumot deb o'ylaydi.
 *   * `views` — oxirgi ma'lum qiymat, o'sish emas; o'sish faqat
 *     `views_delta_period` (snapshot qatoridan).
 *   * Digest matni yagona istisno: `md()` chiqishi `innerHTML` ga tushadi
 *     (renderer escape-first, `docs/ui-v2.md` § Frontend).
 */

import { $, el, clear, replace } from "../lib/dom.js";
import { icon } from "../lib/icons.js";
import { t } from "../lib/i18n.js";
import { get, post, patch as apiPatch } from "../lib/api.js";
import { toast, toastError } from "../lib/toast.js";
import { md, hydrate } from "../lib/md.js";
import { num, compact, relTime, date, plural } from "../lib/fmt.js";
import { isMobile } from "../lib/store.js";
import { backButton } from "../lib/tg.js";
import {
  me, pickAccount, rememberAccount, pushRecentChat,
  avatarEl, typeLabel, flagNodes, syncLabel,
  askHref, alertsHref, WRITE_MODES, writeModeLabel,
  buildChatMenu, openChatMenu,
} from "./chat_shared.js";

const RANGES = [14, 30, 90];
const DIGEST_DAYS = 14;
const WD_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

const state = { accountId: 0, chatId: 0, chat: null, days: 30, stats: null, es: null };

let root = null;

/* ── akkauntni topish ──────────────────────────────────────────────── */

/**
 * Chat qaysi akkauntga tegishli — oldindan noma'lum. Avval oxirgi
 * tanlangani sinaladi, u 404 bersa qolganlari. Shu tufayli ikkinchi
 * akkauntdagi chatga to'g'ridan-to'g'ri havola ham ochiladi.
 */
async function resolveAccount(accounts) {
  const first = pickAccount(accounts);
  const order = [first, ...accounts.filter((a) => a !== first)].filter(Boolean);
  for (const acc of order) {
    try {
      const chat = await get(`/api/accounts/${acc.id}/chats/${state.chatId}`);
      return { accountId: Number(acc.id), chat };
    } catch (err) {
      if (err && err.status !== 404) throw err;
    }
  }
  return null;
}

/* ── sarlavha ──────────────────────────────────────────────────────── */

function bioNode(about) {
  if (!about) return null;
  const p = el("p.clamped", null, String(about));
  const btn = el("button.btn.link.small", {
    type: "button",
    "aria-expanded": "false",
    onclick: () => {
      const open = p.classList.toggle("clamped") === false;
      btn.setAttribute("aria-expanded", String(open));
      btn.textContent = open ? t("web.chats.less_bio") : t("web.chats.more_bio");
    },
  }, t("web.chats.more_bio"));
  return el("div.cd-bio", null, [p, btn]);
}

/**
 * M8: server/klient soat farqi (yoki eskirgan keshdan kelgan sana) tufayli
 * `last_sync_at` "kelajakda" ko'rinishi mumkin — shunda `relTime` "N soniyadan
 * keyin" deb ko'rsatardi, garchi sync allaqachon tugagan bo'lsa ham. O'tgan
 * vaqt **manfiy bo'lmasin** (`Math.max`) — pill hech qachon "keyin" demaydi.
 */
function syncedAgo(v) {
  const t0 = new Date(v).getTime();
  if (Number.isNaN(t0)) return relTime(v);
  const elapsedMs = Math.max(0, Date.now() - t0);
  return relTime(Date.now() - elapsedMs);
}

function syncNode(chat) {
  const stateName = chat.sync_state || "idle";
  const cls = stateName === "running" ? "running" : stateName === "failed" ? "failed"
    : (stateName === "done" || (chat.synced_total || 0) > 0) ? "done" : "";
  const bits = [syncLabel(chat)];
  if ((chat.synced_total || 0) > 0) bits.push(plural("web.n.messages", chat.synced_total));
  if (chat.last_sync_at) bits.push(syncedAgo(chat.last_sync_at));

  const row = el("div.cd-sync", { id: "cd-sync" }, [
    el(`span.sync-pill${cls ? `.${cls}` : ""}`, null, bits.join(" · ")),
  ]);
  if (stateName === "running") {
    row.appendChild(el("span.progress", { role: "progressbar", "aria-valuenow": String(chat.progress ?? 0), "aria-valuemin": "0", "aria-valuemax": "100" },
      el("i", { style: { inlineSize: `${chat.progress ?? 0}%` } })));
  }
  if (!(chat.synced_total || 0) && stateName !== "running") {
    row.appendChild(el("button.btn.primary.small", { type: "button", onclick: startSync }, [icon("refresh-cw"), " ", t("web.chats.sync")]));
  }
  return row;
}

function writeModeSelect(chat) {
  const sel = el("select", {
    id: "cd-write-mode",
    "aria-label": t("web.chats.write_mode"),
    onchange: async () => {
      const before = chat.write_mode;
      try {
        await apiPatch(`/api/accounts/${state.accountId}/chats/${chat.id}/write_mode`, { mode: sel.value });
        chat.write_mode = sel.value;
        toast(t("web.chats.wm_saved", { mode: writeModeLabel(sel.value) }), { kind: "ok" });
      } catch (err) { sel.value = before; toastError(err); }
    },
  });
  for (const mode of WRITE_MODES) {
    sel.appendChild(el("option", { value: mode, selected: chat.write_mode === mode }, writeModeLabel(mode)));
  }
  return el("label.cd-wm", null, [el("span.sr-only", null, t("web.chats.write_mode")), sel]);
}

function headerNode(chat) {
  const meta = [el("span.type-badge", null, typeLabel(chat.type))];
  if (chat.username) meta.push(el("span", null, `@${chat.username}`));
  if (chat.participants_count) meta.push(el("span", null, plural("web.n.members", chat.participants_count)));
  meta.push(...flagNodes(chat, { verifiedIcon: false }));

  const alerts = alertsHref(chat.id);
  return el("header", null, [
    el("div.cd-cover", null, [
      avatarEl(chat, "lg"),
      el("div.cd-id", null, [
        el("h1", null, [
          chat.title || `#${chat.id}`,
          ...(chat.is_verified ? [icon("shield-check", { cls: "verified", label: t("web.chats.verified") })] : []),
        ]),
        el("div.cd-meta", null, meta),
      ]),
    ]),
    bioNode(chat.about),
    syncNode(chat),
    el("div.cd-actions", null, [
      el("a.btn.primary.small", { href: askHref([chat.id]) }, [icon("sparkles"), " ", t("web.chats.ask")]),
      el("button.btn.small", { type: "button", onclick: startSync }, [icon("refresh-cw"), " ", t("web.chats.sync")]),
      alerts ? el("a.btn.small", { href: alerts }, [icon("bell"), " ", t("web.chats.alerts")]) : null,
      writeModeSelect(chat),
    ]),
  ]);
}

/* ── amallar ───────────────────────────────────────────────────────── */

async function startSync() {
  try {
    await post(`/api/accounts/${state.accountId}/chats/${state.chatId}/sync`);
    toast(t("web.chats.sync_started"), { kind: "ok" });
    watchSync();
  } catch (err) { toastError(err); }
}

/**
 * Sync progressi — akkaunt darajasidagi SSE (`sync/stream`). `GET`,
 * shuning uchun oddiy `EventSource` yetarli. Oqim tugaganda chat qayta
 * o'qiladi va sahifa yangilanadi.
 */
function watchSync() {
  if (state.es || !("EventSource" in window)) return;
  const es = new EventSource(`/api/accounts/${state.accountId}/sync/stream`);
  state.es = es;
  const stop = () => { try { es.close(); } catch { /* yopiq */ } state.es = null; };
  es.addEventListener("status", (ev) => {
    let payload = null;
    try { payload = JSON.parse(ev.data); } catch { return; }
    const mine = (payload.chats || []).find((c) => Number(c.chat_id) === state.chatId);
    if (!mine) return;
    state.chat.sync_state = mine.state;
    state.chat.progress = mine.progress;
    state.chat.synced_total = mine.synced_total;
    const node = $("#cd-sync");
    if (node) node.replaceWith(syncNode(state.chat));
  });
  es.addEventListener("done", async () => {
    stop();
    try {
      state.chat = await get(`/api/accounts/${state.accountId}/chats/${state.chatId}`);
      render();
    } catch { /* keyingi ochilishda ko'rinadi */ }
  });
  es.onerror = stop;
}

/* ── Overview ──────────────────────────────────────────────────────── */

function kpi(label, value, sub) {
  return el("div.kpi", null, [
    el("div.kpi-label", null, label),
    el("div.kpi-value", null, value),
    sub ? el("div.kpi-sub", null, sub) : null,
  ]);
}

function sparkNode(series) {
  const vals = (series || []).map((d) => Number(d.posts) || 0);
  const max = Math.max(1, ...vals);
  return el("div.spark", { role: "img", "aria-label": t("web.chats.spark_alt", { n: vals.length }) },
    vals.map((v) => el("i", { style: { "--h": `${Math.max(2, Math.round((v / max) * 100))}%` } })));
}

function overviewCard(s) {
  const grid = el("div.kpi-grid", null, [
    kpi(t("web.chats.kpi_posts"), num(s.posts), t("web.chats.kpi_per_day", { n: (s.posts_per_day ?? 0).toFixed(1) })),
    kpi(t("web.chats.kpi_views"), compact(s.views_sum), s.views_avg != null ? t("web.chats.kpi_avg", { n: compact(Math.round(s.views_avg)) }) : ""),
    kpi(t("web.chats.kpi_reactions"), compact(s.reactions_sum)),
    kpi(t("web.chats.kpi_forwards"), compact(s.forwards_sum)),
    kpi(t("web.chats.kpi_replies"), compact(s.replies_sum)),
  ]);
  // M8: snapshot yo'q bo'lsa sparkline hammasi ~0% balandlikda — tekis,
  // chiziqli chizilgan bo'lib ko'rinadi va "ma'lumot bor, lekin kichik" deb
  // o'qiladi. Buning o'rniga aniq bo'sh holat: `.empty-v2`.
  const noSnapshots = !s.snapshot_coverage;
  const delta = el("div.kpi.wide", null, [
    el("div.kpi-label", null, t("web.chats.kpi_views_delta")),
    el("div.kpi-value", null, s.views_delta_period == null ? "—" : compact(s.views_delta_period)),
    noSnapshots
      ? el("div.empty-v2", null, [
          el("span.empty-ico", { "aria-hidden": "true" }, icon("bar-chart-3")),
          el("p", null, t("web.chats.no_snapshots")),
        ])
      : sparkNode(s.series),
    noSnapshots ? null : el("div.kpi-sub", null, t("web.chats.kpi_delta_note", { n: num(s.snapshot_coverage || 0) })),
  ]);
  grid.appendChild(delta);
  return card("overview", t("web.chats.tab_overview"), [grid], true);
}

/* ── Activity (heatmap) ────────────────────────────────────────────── */

function heatCells(byWeekday, byHour) {
  const total = byWeekday.reduce((a, b) => a + b, 0) || 1;
  const maxHour = Math.max(1, ...byHour);
  const nodes = [];
  for (let d = 0; d < 7; d++) {
    nodes.push(el("span.heat-lab", null, t(`web.chats.wd_${WD_KEYS[d]}`)));
    const share = byWeekday[d] / total;
    for (let h = 0; h < 24; h++) {
      const v = Math.round(((byHour[h] / maxHour) * share * 7) * 100);
      nodes.push(el("span.heat-cell", {
        style: { "--v": String(Math.max(0, Math.min(100, v))) },
        title: `${t(`web.chats.wd_${WD_KEYS[d]}`)} ${String(h).padStart(2, "0")}:00`,
      }));
    }
  }
  return nodes;
}

function activityCard(s) {
  const byHour = (s.by_hour || []).length === 24 ? s.by_hour.map(Number) : new Array(24).fill(0);
  const byWeekday = (s.by_weekday || []).length === 7 ? s.by_weekday.map(Number) : new Array(7).fill(0);

  const hours = el("div.heat-hours", null, [
    el("span", null, ""),
    ...Array.from({ length: 24 }, (_, h) => el("span", null, h % 6 === 0 ? String(h) : "")),
  ]);
  const heat = el("div.heat", { role: "img", "aria-label": t("web.chats.heatmap_alt") }, heatCells(byWeekday, byHour));

  return card("activity", t("web.chats.tab_activity"), [
    el("div.heat-wrap", null, [heat, hours]),
    el("p.note", null, t("web.chats.heatmap_note")),
  ], true, rangeSwitch());
}

function rangeSwitch() {
  return el("div.segment", { role: "radiogroup", "aria-label": t("web.chats.range") },
    RANGES.map((d) => el("button", {
      type: "button", role: "radio",
      "aria-checked": String(d === state.days),
      tabindex: d === state.days ? 0 : -1,
      onclick: () => { state.days = d; loadStats(); },
    }, t("web.chats.range_days", { n: d }))));
}

/* ── Top postlar ───────────────────────────────────────────────────── */

function topList(posts, metric) {
  if (!posts || !posts.length) return el("p.note", null, t("web.state.empty"));
  return el("ul.toplist", null, posts.map((p, i) => el("li", null, [
    el("span.rank", null, String(i + 1)),
    el("div.top-body", null, [
      p.link
        ? el("a.top-text", { href: p.link, target: "_blank", rel: "noopener noreferrer" }, p.text || `#${p.msg_id}`)
        : el("span.top-text", null, p.text || `#${p.msg_id}`),
      el("div.top-meta", null, [
        p.date ? el("span", null, date(p.date)) : null,
        el("span", null, `${t(`web.chats.m_${metric}`)}: ${compact(p[metric] ?? 0)}`),
      ]),
    ]),
  ])));
}

function topCard(s) {
  const tabs = [
    ["views", s.top_by_views],
    ["reactions", s.top_by_reactions],
    ["forwards", s.top_by_forwards],
  ];
  const body = el("div");
  const seg = el("div.segment", { role: "radiogroup", "aria-label": t("web.chats.tab_top") },
    tabs.map(([metric, posts], i) => el("button", {
      type: "button", role: "radio", "aria-checked": String(i === 0), tabindex: i === 0 ? 0 : -1,
      onclick: (e) => {
        for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-checked", "false");
        e.currentTarget.setAttribute("aria-checked", "true");
        replace(body, topList(posts, metric));
      },
    }, t(`web.chats.m_${metric}`))));
  replace(body, topList(tabs[0][1], "views"));
  return card("top", t("web.chats.tab_top"), [body], false, seg);
}

/* ── Digest'lar ────────────────────────────────────────────────────── */

async function digestsCard() {
  const host = card("digests", t("web.chats.tab_digests"), [el("p.note", null, t("web.state.loading"))], true);
  let items = [];
  try {
    items = (await get(`/api/accounts/${state.accountId}/chats/${state.chatId}/digests?days=${DIGEST_DAYS}`)).items || [];
  } catch { items = []; }
  const body = host.querySelector(".cd-card-body");
  clear(body);
  if (!items.length) { body.appendChild(el("p.note", null, t("web.chats.digests_none"))); return host; }
  for (const d of items) {
    const box = el("div.digest-body");
    // Yagona `innerHTML`: `md()` escape-first renderer chiqishi.
    box.innerHTML = md(d.digest || "");
    hydrate(box);
    body.appendChild(el("details.digest-day", null, [
      el("summary", null, [
        icon("chevron-right"),
        el("span", null, d.day ? date(d.day) : "—"),
        el("span.digest-count", null, plural("web.n.messages", d.msg_count || 0)),
      ]),
      box,
    ]));
  }
  return host;
}

/* ── karta yordamchisi ─────────────────────────────────────────────── */

function card(id, title, children, span2, headExtra) {
  const body = el("div.cd-card-body", null, children);
  return el(`section.cd-card${span2 ? ".span2" : ""}`, { id: `cd-${id}`, "aria-label": title }, [
    el("div.cd-card-head", null, [el("h2", null, title), headExtra || null]),
    body,
  ]);
}

/* ── bo'lim chiplari (mobil) ───────────────────────────────────────── */

function sectionChips(ids) {
  return el("nav.cd-tabs", { "aria-label": t("web.chats.sections") },
    ids.map(([id, label], i) => el("button.cd-tab", {
      type: "button", "aria-selected": String(i === 0),
      onclick: (e) => {
        for (const b of e.currentTarget.parentElement.children) b.setAttribute("aria-selected", "false");
        e.currentTarget.setAttribute("aria-selected", "true");
        $(`#cd-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
      },
    }, label)));
}

/* ── chizish ───────────────────────────────────────────────────────── */

/**
 * M9: `digestsCard()` asinxron — u tugaguncha `render()` yana chaqirilishi
 * mumkin (masalan `rangeSwitch` bosilishi yoki sync `done` hodisasi). Eski
 * chaqiruv o'zining (endi DOM'dan uzilgan) `sections` tuguniga
 * `insertBefore` qilishga urinsa, u global `$("#cd-members")` ORQALI
 * **yangi** render'ning tugunini topib, begona daraxtga qo'shishga
 * urinardi — `NotFoundError`. Yechim: (1) `renderGen` — faqat oxirgi
 * chaqiruv natijasi qo'llanadi; (2) qidiruv `sections` ichida (global
 * emas) — o'zining daraxtidan chiqmaydi; (3) `root.replaceChildren` —
 * bitta atomik almashtirish, oraliq holat yo'q.
 */
let renderGen = 0;

function render() {
  const gen = ++renderGen;
  const chat = state.chat;
  const s = state.stats;

  const titleEl = $("#cd-topbar-title");
  if (titleEl) titleEl.textContent = chat.title || t("web.chats.title");
  const menuBtn = $("#cd-menu");
  if (menuBtn) menuBtn.hidden = false;

  const frag = document.createDocumentFragment();
  frag.appendChild(headerNode(chat));

  const ids = [
    ["overview", t("web.chats.tab_overview")],
    ["activity", t("web.chats.tab_activity")],
    ["top", t("web.chats.tab_top")],
    ["digests", t("web.chats.tab_digests")],
    ["members", t("web.chats.tab_members")],
  ];
  frag.appendChild(sectionChips(ids));

  const sections = el("div.cd-sections", { id: "cd-sections" });
  if (s) {
    sections.append(overviewCard(s), activityCard(s), topCard(s));
  } else {
    sections.appendChild(card("overview", t("web.chats.tab_overview"), [el("p.note", null, t("web.state.loading"))], true));
  }
  sections.appendChild(card("members", t("web.chats.tab_members"), [
    el("p.note", null, chat.participants_count
      ? `${plural("web.n.members", chat.participants_count)} · ${t("web.chats.members_note")}`
      : t("web.chats.members_note")),
  ], false));
  frag.appendChild(sections);

  root.replaceChildren(frag);
  root.setAttribute("aria-busy", "false");

  digestsCard().then((node) => {
    if (gen !== renderGen) return;               // eskirgan render — tashlanadi
    const members = sections.querySelector("#cd-members");   // shu daraxt ichida, global emas
    if (members) sections.insertBefore(node, members);
    else sections.appendChild(node);
  });
}

async function loadStats() {
  try {
    state.stats = await get(`/api/accounts/${state.accountId}/chats/${state.chatId}/stats?days=${state.days}`);
  } catch (err) {
    state.stats = null;
    toastError(err);
  }
  render();
}

/* ── kirish nuqtasi ────────────────────────────────────────────────── */

/**
 * H4: topbar'dagi `‹` — brauzer tarixi (yo'q bo'lsa `/chats`). Telegram
 * ichida `backButton(true)` (pastda) allaqachon o'z tugmasini ko'rsatadi —
 * ikkalasi mustaqil, biri boshqasini almashtirmaydi.
 */
function wireTopbar() {
  $("#cd-back")?.addEventListener("click", () => {
    if (window.history.length > 1) window.history.back();
    else window.location.href = "/chats";
  });
  $("#cd-menu")?.addEventListener("click", (e) => {
    const chat = state.chat;
    if (!chat) return;
    const menu = buildChatMenu(chat, {
      alertsUrl: alertsHref(chat.id),
      onSync: () => { startSync(); close(); },
    });
    const close = openChatMenu(menu, e.currentTarget);
  });
}

/** @param {{chatId: number}} opts */
export async function initChatDetail(opts) {
  root = $("#chat-detail");
  if (!root) return;
  state.chatId = Number(opts.chatId || 0);
  if (!state.chatId) return;

  wireTopbar();
  if (isMobile()) backButton(true);

  let data;
  try { data = await me(); } catch (err) { toastError(err); return; }
  const accounts = data.accounts || [];
  if (!accounts.length) {
    replace(root, el("div.empty-v2", null, [
      el("h2", null, t("web.chats.no_accounts_title")),
      el("a.btn.primary", { href: "/login" }, t("web.chats.connect")),
    ]));
    return;
  }

  let found = null;
  try { found = await resolveAccount(accounts); } catch (err) { toastError(err); }
  if (!found) {
    replace(root, el("div.empty-v2", null, [
      el("h2", null, t("web.chats.not_found")),
      el("a.btn", { href: "/chats" }, t("web.chats.title")),
    ]));
    return;
  }

  state.accountId = found.accountId;
  state.chat = found.chat;
  rememberAccount(state.accountId);
  pushRecentChat(state.chatId);
  document.title = `${state.chat.title || t("web.chats.title")} · ${t("web.title")}`;

  render();
  await loadStats();
  if (state.chat.sync_state === "running") watchSync();
}

export { state as chatDetailState };
