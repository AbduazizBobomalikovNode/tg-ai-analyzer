/* pages/actions.js — `/actions`: tasdiq kutayotgan yozish amallari.
 *
 * Invariant 4 (`CLAUDE.md`): yozish amali **user tasdig'isiz bajarilmaydi**.
 * Bu sahifa o'sha tasdiqning yagona to'liq ko'rinishi; `/ask` ichidagi inline
 * kartalar bilan **bir xil** `.action-card` sinfini ishlatadi
 * (`css/components.css`), ya'ni ikkita joyda ikki xil ko'rinish bo'lmaydi.
 *
 * TTL: taklif `expires_at` da eskiradi. Sanoq brauzerda chiziladi, lekin
 * qaror serverda — eskirgan amalni tasdiqlashga urinish 410 qaytaradi va
 * biz ro'yxatni yangilaymiz.
 */

import { $, $$, el, replace, ready } from "/static/lib/dom.js";
import { t, has } from "/static/lib/i18n.js";
import { get, post } from "/static/lib/api.js";
import { icon } from "/static/lib/icons.js";
import { dateTime, relTime } from "/static/lib/fmt.js";
import { toast, toastError } from "/static/lib/toast.js";

const PENDING_POLL_MS = 30_000;
const state = { pending: [], history: [], filter: "" };
let ticker = 0;

/* ── Yordamchilar ────────────────────────────────────────────────────── */

/** M6: har bir tool (`services/tools.py` READ_TOOLS + yozish tool'lari)
 * uchun `web.tools.<name>` — kalit yo'q bo'lsa "_" → bo'sh joy (`search_messages`
 * → "search messages"), xom snake_case ko'rsatilmaydi. */
const toolLabel = (tool) => (has(`web.tools.${tool}`) ? t(`web.tools.${tool}`) : String(tool).replace(/_/g, " "));
const statusLabel = (s) => (has(`web.act.status_${s}`) ? t(`web.act.status_${s}`) : s);

/**
 * M6: `services/agent.py` bir xil tool ketma-ket `MAX_CALLS_PER_TOOL` marta
 * chaqirilsa uni **bajarmaydi** — bu chindan xato emas, agent boshqa yo'l
 * bilan davom etadi ("to'silgan chaqiruv", `CLAUDE.md`). Bunday yozuv
 * append-only `agent_actions`ga baribir `status=failed` + muhandislik
 * izohi (`error`) bilan tushadi — UI'da buni alohida, tinch ko'rsatamiz.
 */
const BLOCKED_NOTE_RE = /^note: /i;
const isBlockedNote = (a) => a.status === "failed" && typeof a.error === "string" && BLOCKED_NOTE_RE.test(a.error);

/** `expires_at` → "12:04 qoldi" ko'rinishidagi matn (yoki muddati o'tgan). */
function ttlText(expiresAt) {
  const left = new Date(expiresAt).getTime() - Date.now();
  if (!Number.isFinite(left)) return "";
  if (left <= 0) return t("web.actions.expired");
  const mins = Math.floor(left / 60000);
  const hh = Math.floor(mins / 60);
  const mm = mins % 60;
  return t("web.actions.expires_in", { time: hh ? `${hh}:${String(mm).padStart(2, "0")}` : `${mm}m` });
}

function badgeCount(n) {
  for (const node of $$("[data-actions-badge]")) {
    node.textContent = String(n);
    node.hidden = n === 0;
  }
  const mirror = $("#act-count");
  if (mirror) { mirror.textContent = String(n); mirror.hidden = n === 0; }
}

/* ── Karta ───────────────────────────────────────────────────────────── */

function actionCard(a, { actionable }) {
  const p = a.preview || {};
  const kids = [];
  const blocked = isBlockedNote(a);

  kids.push(el("div.action-head", null, [
    icon(a.tool === "forward_message" ? "share-2" : a.tool === "pin_message" ? "pin" : a.tool === "edit_message" ? "edit-3" : "send"),
    el("span.action-tool", null, toolLabel(a.tool)),
    el("span.action-chat", null, p.chat || (p.chat_id != null ? `#${p.chat_id}` : "")),
    el("span.spacer", { style: { flex: "1 1 auto" } }),
    el(`span.pill${blocked ? ".warn" : a.status === "executed" ? ".ok" : a.status === "proposed" ? ".accent" : a.status === "confirmed" ? ".accent" : ".err"}`,
      null, blocked ? t("web.actions.blocked_pill") : statusLabel(a.status)),
  ]));

  if (p.image_url) {
    kids.push(el("img.gen-img", { src: p.image_url, alt: t("web.act.image"), loading: "lazy", decoding: "async" }));
  }
  if (p.text) kids.push(el("div.action-text", null, p.text));

  const meta = [];
  if (a.created_at) meta.push(el("span", null, dateTime(a.created_at)));
  if (a.status === "proposed" && a.expires_at) {
    const left = new Date(a.expires_at).getTime() - Date.now();
    meta.push(el(`span.ttl${left < 15 * 60_000 ? ".is-soon" : ""}`, { dataset: { expires: a.expires_at } }, ttlText(a.expires_at)));
  }
  if (a.result_msg_id) meta.push(el("span", null, `${t("web.act.result")} #${a.result_msg_id}`));
  if (a.confirmed_at && a.status !== "proposed") meta.push(el("span", null, relTime(a.confirmed_at)));
  if (a.error) meta.push(el("span", null, blocked ? t("web.actions.blocked_note") : a.error));
  if (a.block_reason) meta.push(el("span", null, a.block_reason));
  if (meta.length) kids.push(el("div.action-meta", null, meta));

  if (actionable && a.status === "proposed") {
    kids.push(el("div.action-btns", null, [
      el("button.btn.primary.small", { type: "button", onclick: (e) => decide(a, "confirm", e.currentTarget) },
        [icon("check"), t("web.act.confirm")]),
      el("button.btn.ghost.small", { type: "button", onclick: (e) => decide(a, "reject", e.currentTarget) },
        [icon("x"), t("web.act.reject")]),
    ]));
  }

  return el(`div.action-card.${a.status}`, { dataset: { id: a.id } }, kids);
}

function emptyPending() {
  return el("div.card-v2", null, el("div.empty-v2", null, [
    el("span.empty-ico", null, icon("check-check")),
    el("h2", null, t("web.actions.empty_title")),
    el("p", null, t("web.actions.empty_text")),
  ]));
}

/* ── Amal ────────────────────────────────────────────────────────────── */

async function decide(a, what, btn) {
  if (btn) btn.disabled = true;
  try {
    const view = await post(`/api/actions/${a.id}/${what}`);
    toast(t(what === "confirm" ? "web.actions.confirmed" : "web.actions.rejected"), { kind: "ok" });
    Object.assign(a, view || {});
  } catch (err) {
    toastError(err);
  } finally {
    if (btn) btn.disabled = false;
    await load();
  }
}

/* ── Yuklash ─────────────────────────────────────────────────────────── */

function paint() {
  const pendingBox = $("#act-pending");
  if (pendingBox) {
    replace(pendingBox, state.pending.length
      ? state.pending.map((a) => actionCard(a, { actionable: true }))
      : emptyPending());
  }
  const historyBox = $("#act-history");
  if (historyBox) {
    const rows = state.history.filter((a) => (state.filter ? a.status === state.filter : true));
    replace(historyBox, rows.length
      ? rows.map((a) => actionCard(a, { actionable: false }))
      : el("p.muted", null, t("web.state.empty")));
  }
  badgeCount(state.pending.length);
}

async function load() {
  const [pendingRes, allRes] = await Promise.allSettled([
    get("/api/actions?status=proposed&limit=200"),
    get("/api/actions?limit=100"),
  ]);
  state.pending = pendingRes.status === "fulfilled" ? (pendingRes.value.items || []) : [];
  const all = allRes.status === "fulfilled" ? (allRes.value.items || []) : [];
  state.history = all.filter((a) => a.status !== "proposed");
  paint();
}

/** Har 30 s TTL matnini yangilaydi (butun ro'yxatni qayta chizmasdan). */
function startTicker() {
  clearInterval(ticker);
  ticker = setInterval(() => {
    for (const node of $$(".ttl[data-expires]")) {
      const iso = node.dataset.expires;
      node.textContent = ttlText(iso);
      node.classList.toggle("is-soon", new Date(iso).getTime() - Date.now() < 15 * 60_000);
    }
  }, PENDING_POLL_MS);
}

ready(() => {
  $("#act-refresh")?.addEventListener("click", () => load());
  for (const b of $$("#act-filters button")) {
    b.addEventListener("click", () => {
      state.filter = b.dataset.status || "";
      for (const o of $$("#act-filters button")) o.setAttribute("aria-selected", String(o === b));
      paint();
    });
  }
  startTicker();
  load();
});
