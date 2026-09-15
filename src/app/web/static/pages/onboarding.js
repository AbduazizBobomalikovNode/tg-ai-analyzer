/* pages/onboarding.js — `/onboarding`: 4 bosqichli birinchi tajriba.
 *
 *   1. QR login (telefon yo'li `/login` ga havola bo'lib qoladi)
 *   2. "nimani o'qiymiz" ishonch kartasi — faqat o'qish, media yo'q,
 *      telefon raqami yo'q, deny-by-default (`CLAUDE.md` invariantlari)
 *   3. jonli sync progressi (`sync/stream` SSE) + "hozir boshlash" chiqishi
 *   4. uchta birinchi savol → `/ask?q=`
 *
 * Oxirida `prefs.onboarding_done = true` yoziladi — `/` shundan keyin
 * `/today` ga yo'naltiradi.
 *
 * QR rasmi **server tomonda** chizilgan `data:` URI (`auth.py::_qr_data_uri`):
 * brauzerga QR kutubxonasi qo'shilmaydi.
 */

import { $, $$, el, replace, clear, ready } from "/static/lib/dom.js";
import { t, has } from "/static/lib/i18n.js";
import { get, post, put } from "/static/lib/api.js";
import { icon } from "/static/lib/icons.js";
import { toast, toastError } from "/static/lib/toast.js";
import * as themeMod from "/static/lib/theme.js";

const QR_POLL_MS = 2000;
const state = { step: 1, flowId: null, accountId: null, poll: 0, es: null, hasChat: false };

/**
 * `/api/me` — **401 ni xato deb hisoblamaydi**.
 *
 * `lib/api.js` 401 da `/login` ga yo'naltiradi, bu esa to'g'ri qoida —
 * lekin onboarding'ning 1-qadami aynan kirmagan foydalanuvchi uchun.
 * Shuning uchun bu yerda xom `fetch`: 401 = "hali kirmagan", `null`.
 */
async function probeMe() {
  try {
    const res = await fetch("/api/me", {
      credentials: "same-origin",
      headers: { "X-Requested-With": "fetch" },
    });
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

/* ── Qadam boshqaruvi ────────────────────────────────────────────────── */

function goto(step) {
  state.step = step;
  for (let i = 1; i <= 4; i++) {
    const node = $(`#onb-${i}`);
    if (node) node.hidden = i !== step;
  }
  const bar = $("#onb-steps");
  if (bar) {
    bar.setAttribute("aria-valuenow", String(step));
    $$("i", bar).forEach((n, i) => {
      n.className = i + 1 < step ? "is-done" : i + 1 === step ? "is-active" : "";
    });
  }
  if (step === 1) startQr();
  if (step !== 1) stopQr();
  if (step === 3) startSync();
  if (step !== 3) stopSync();
  if (step === 4) paintQuestions();
}

/* ── 1. QR ───────────────────────────────────────────────────────────── */

function stopQr() { clearInterval(state.poll); state.poll = 0; }

function showQr(data) {
  const img = $("#onb-qr-img");
  if (img && data.qr) { img.src = data.qr; img.hidden = false; }
  const hint = $("#onb-qr-hint");
  if (hint) hint.textContent = data.error ? t("auth.err.telegram") : t("web.login.qr_waiting");
}

async function startQr() {
  stopQr();
  try {
    const data = await post("/api/auth/qr", {});
    state.flowId = data.flow_id;
    showQr(data);
  } catch (err) { toastError(err); return; }

  state.poll = setInterval(async () => {
    if (!state.flowId) return stopQr();
    let data;
    try { data = await post("/api/auth/qr/poll", { flow_id: state.flowId }); }
    catch (err) { stopQr(); toastError(err); return; }
    showQr(data);
    if (data.status === "needs_2fa") {
      stopQr();
      toast(t("web.onboarding.needs_2fa"), { kind: "info" });
      window.location.href = "/login";
    } else if (data.status === "done") {
      stopQr();
      state.accountId = data.account_id || null;
      goto(2);
    }
  }, QR_POLL_MS);
}

/* ── 3. Sync progressi ───────────────────────────────────────────────── */

function stopSync() {
  if (state.es) { try { state.es.close(); } catch { /* yopilgan */ } state.es = null; }
}

function paintSync(payload) {
  const counts = payload.counts || {};
  const box = $("#onb-counts");
  if (box) {
    replace(box, Object.entries(counts).map(([k, v]) =>
      el(`span.pill${k === "done" ? ".ok" : k === "failed" ? ".err" : k === "running" ? ".accent" : ""}`,
        null, `${has(`web.onboarding.state_${k}`) ? t(`web.onboarding.state_${k}`) : k}: ${v}`)));
  }
  const rows = $("#onb-sync-rows");
  if (rows) {
    const running = (payload.chats || []).filter((c) => c.state === "running").slice(0, 6);
    replace(rows, running.map((c) => el("div.onb-sync-row", null, [
      el("div.onb-sync-head", null, [
        el("span", null, `#${c.chat_id}`),
        el("span", null, `${c.synced_total || 0}${c.total_estimate ? ` / ${c.total_estimate}` : ""}`),
      ]),
      el("div.progress", null, el("i", { style: { inlineSize: `${c.progress || 0}%` } })),
    ])));
  }
  const label = $("#onb-sync-state");
  if (label) label.textContent = payload.running ? t("web.onboarding.s3_running") : t("web.onboarding.s3_done");
  state.hasChat = Number(counts.done || 0) > 0 || Number(counts.running || 0) > 0;
}

async function startSync() {
  if (!state.accountId) {
    const me = await probeMe();
    const acc = me && ((me.accounts || []).find((a) => a.status === "active") || (me.accounts || [])[0]);
    state.accountId = acc ? acc.id : null;
    state.hasChat = Boolean(acc && acc.chats_synced);
  }
  if (!state.accountId) return;

  // Sync'ni ishga tushiramiz (allaqachon yugurayotgan bo'lsa server o'zi
  // takroriy job yaratmaydi — job id deterministik).
  try { await post(`/api/accounts/${state.accountId}/sync`); } catch { /* holat baribir ko'rsatiladi */ }

  stopSync();
  try {
    const es = new EventSource(`/api/accounts/${state.accountId}/sync/stream`, { withCredentials: true });
    state.es = es;
    es.addEventListener("status", (e) => { try { paintSync(JSON.parse(e.data)); } catch { /* buzuq ramka */ } });
    es.addEventListener("done", () => stopSync());
    es.onerror = () => stopSync();
  } catch { /* SSE yo'q — bir marta so'raymiz */ }
  try { paintSync(await get(`/api/accounts/${state.accountId}/sync/status`)); } catch { /* jim */ }
}

/* ── 4. Birinchi savollar ────────────────────────────────────────────── */

function askUrl(q) {
  const R = (window.TGAI && window.TGAI.routes) || {};
  return `${R.ask || "/ask"}?q=${encodeURIComponent(q)}`;
}

function paintQuestions() {
  const box = $("#onb-questions");
  if (!box) return;
  clear(box);
  const keys = state.hasChat
    ? ["web.onboarding.q1", "web.onboarding.q2", "web.onboarding.q3"]
    : ["web.onboarding.q1", "web.onboarding.q3"];
  for (const k of keys) {
    const text = t(k);
    box.appendChild(el("button.chip-btn", {
      type: "button",
      onclick: () => finish(askUrl(text)),
    }, [icon("sparkles"), text]));
  }
}

async function finish(href) {
  try { await put("/api/settings/prefs", { onboarding_done: true }); }
  catch { /* belgilash muvaffaqiyatsiz — sahifa baribir ochiladi */ }
  window.location.href = href || "/today";
}

/* ── Kirish nuqtasi ──────────────────────────────────────────────────── */

ready(async () => {
  themeMod.init();
  for (const b of $$("[data-theme-toggle]")) b.addEventListener("click", () => themeMod.cycle());
  for (const b of $$("[data-onb-next]")) b.addEventListener("click", () => goto(Number(b.dataset.onbNext)));
  $("#onb-qr-retry")?.addEventListener("click", () => startQr());
  $("#onb-finish")?.addEventListener("click", () => finish("/today"));

  // Allaqachon kirgan bo'lsa 1-qadam o'tkazib yuboriladi.
  const me = await probeMe();
  const acc = me && ((me.accounts || []).find((a) => a.status === "active") || (me.accounts || [])[0]);
  if (acc) { state.accountId = acc.id; state.hasChat = Boolean(acc.chats_synced); }
  goto(acc ? 2 : 1);
});
