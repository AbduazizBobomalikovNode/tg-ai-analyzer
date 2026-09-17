/* lib/sync-center.js — sinxronizatsiya markazi (YAGONA manba).
 *
 * Uch joyda bir xil holat obyektini ko'rsatadi:
 *   * `/today`      — ixcham band (`mountStrip`)
 *   * `/chats`, `/settings` — to'liq varaq (`openSheet`)
 *   * `/chats/{id}` — qator/pill (`liveChat` yordamchisi)
 *
 * ROSTGO'YLIK (CLAUDE.md, Ingestion): bitta akkaunt ichida chatlar
 * **birma-bir** sinxronlanadi (`worker/tasks.sync_account` sikli,
 * `WAIT_BETWEEN_BATCHES`, 100 talik batch). Akkauntlar esa parallel
 * (`max_jobs=4`). FloodWait butun akkaunt ishini pauzaga qo'yadi
 * (`SyncPaused` → `defer_by` requeue). Shuning uchun UI "N ta chat
 * sinxronlanmoqda" demaydi — u bitta joriy chatni va navbatni ko'rsatadi.
 *
 * XAVFSIZLIK: `innerHTML` bu faylda **umuman yo'q**. Chat sarlavhasi va
 * xato matni — Telegram/serverdan kelgan ishonchsiz matn, hammasi
 * `el()` / `textContent` orqali chiziladi.
 */

import { el, clear, replace, trapFocus } from "./dom.js";
import { icon } from "./icons.js";
import { t } from "./i18n.js";
import { get, post } from "./api.js";
import { num, relTime, plural } from "./fmt.js";
import { peerClass, initials } from "./palette.js";
import { toast, toastError } from "./toast.js";

/* ══ Sof yordamchilar (node testi shularni tekshiradi) ═════════════════ */

/** Halqa radiusi — SVG `viewBox="0 0 64 64"` ichida (chiziq qalinligi 6). */
export const RING_R = 27;

/**
 * Doiraviy progress uchun `stroke-dasharray` / `stroke-dashoffset`.
 *
 * @param {number} percent  0..100 (chegaradan tashqarisi qisiladi)
 * @param {number} [r]      radius
 * @returns {{c: number, dash: number, offset: number}}
 */
export function ringGeometry(percent, r = RING_R) {
  const p = Math.max(0, Math.min(100, Number(percent) || 0));
  const c = 2 * Math.PI * r;
  const dash = (c * p) / 100;
  return { c, dash, offset: c - dash };
}

/**
 * ETA birligi — matn emas, **qism** (i18n kaliti chaqiruvchida).
 * 90 soniyagacha soniya, 90 daqiqagacha daqiqa, undan keyin soat.
 *
 * @param {number|null|undefined} seconds
 * @returns {{unit: "sec"|"min"|"hour", n: number}|null}
 */
export function etaParts(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s <= 0) return null;
  if (s < 90) return { unit: "sec", n: Math.max(1, Math.round(s)) };
  if (s < 5400) return { unit: "min", n: Math.max(1, Math.round(s / 60)) };
  return { unit: "hour", n: Math.max(1, Math.round(s / 3600)) };
}

/** ETA matni (`~4 daq`); noma'lum bo'lsa `null`. */
export function etaText(seconds) {
  const p = etaParts(seconds);
  return p ? t(`web.sync.eta_${p.unit}`, { n: num(p.n) }) : null;
}

/**
 * Umumiy foiz. Server `percent_overall` beradi; u yo'q bo'lsa
 * (eski javob) chat hisobidan tiklanadi — strip hech qachon bo'sh qolmasin.
 */
export function overallPercent(status) {
  if (!status) return 0;
  if (status.percent_overall != null) return Math.max(0, Math.min(100, Number(status.percent_overall)));
  const c = status.counts || {};
  const total = Number(c.total || 0);
  return total ? Math.round((100 * Number(c.done || 0)) / total) : 0;
}

/** Joriy chat foizi (`current` bo'lmasa `null`). */
export function chatPercent(current) {
  if (!current) return null;
  const est = Number(current.total_estimate || 0);
  if (!est) return null;
  return Math.max(0, Math.min(100, Math.round((100 * Number(current.synced_total || 0)) / est)));
}

/**
 * Pauza qoldig'i: `mm:ss`. O'tib ketgan bo'lsa `null`.
 * @param {string|null} untilIso
 * @param {number} [nowMs]
 */
export function countdownText(untilIso, nowMs = Date.now()) {
  if (!untilIso) return null;
  const left = Math.round((new Date(untilIso).getTime() - nowMs) / 1000);
  if (!Number.isFinite(left) || left <= 0) return null;
  const m = Math.floor(left / 60);
  const s = left % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Davomiylik: `4:05` / `1:02:33` (o'tgan vaqt uchun). */
export function elapsedText(startedIso, nowMs = Date.now()) {
  if (!startedIso) return null;
  const s = Math.max(0, Math.round((nowMs - new Date(startedIso).getTime()) / 1000));
  if (!Number.isFinite(s)) return null;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}

/** Joriy sinxronlanayotgan chat id (yo'q bo'lsa `null`) — `/chats` qatori uchun. */
export const currentChatId = (status) =>
  status && status.current && status.current.chat_id != null ? Number(status.current.chat_id) : null;

/**
 * Bitta chat uchun jonli holat — `/chats` qatorlari va `/chats/{id}` pill'i
 * shu obyektni iste'mol qiladi (ikkinchi so'rov yo'q).
 *
 * @returns {{active: boolean, percent: number|null, phase: string|null, paused: boolean}}
 */
export function liveChat(status, chatId) {
  const cur = status && status.current;
  const active = cur != null && Number(cur.chat_id) === Number(chatId);
  return {
    active,
    percent: active ? chatPercent(cur) : null,
    phase: active ? cur.phase || null : null,
    paused: active ? Boolean(cur.paused_until) : false,
  };
}

/* ══ Harakat (prefers-reduced-motion ostida) ══════════════════════════ */

const reduced = () =>
  typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Raqamni `from` dan `to` gacha yugurtiradi (400 ms, `requestAnimationFrame`).
 * Reduced-motion'da — darhol yakuniy qiymat.
 */
export function countUp(node, to, format = (v) => num(Math.round(v)), ms = 400) {
  if (!node) return;
  const from = Number(node.dataset.v || 0);
  const target = Number(to) || 0;
  node.dataset.v = String(target);
  if (reduced() || from === target || typeof requestAnimationFrame !== "function") {
    node.textContent = format(target);
    return;
  }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    const eased = 1 - (1 - k) * (1 - k);
    node.textContent = format(from + (target - from) * eased);
    if (k < 1 && node.dataset.v === String(target)) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ══ Jonli manba: SSE + polling zaxirasi ══════════════════════════════ */

const MAX_BACKOFF = 30000;

/** Akkaunt bo'yicha **bitta** jonli ulanish, barcha obunachilar uchun. */
const shared = new Map();

/**
 * Akkaunt holatiga obuna. Bir sahifada bir necha komponent (topbar ikonkasi,
 * ro'yxat qatori, varaq) obuna bo'ladi — ular **bitta** SSE ulanishini
 * bo'lishadi: brauzer HTTP/1.1 da bitta origin uchun ~6 ta ulanish beradi va
 * ularni sync ko'rsatkichiga sarflash ahmoqlik.
 *
 * @param {number|string} accountId
 * @param {(status: object) => void} onStatus
 * @returns {{stop: () => void}}
 */
export function subscribe(accountId, onStatus) {
  const key = String(accountId);
  let entry = shared.get(key);
  if (!entry) {
    entry = { listeners: new Set(), last: null, stop: null };
    shared.set(key, entry);
    entry.stop = rawSubscribe(accountId, (status) => {
      entry.last = status;
      for (const fn of [...entry.listeners]) {
        try { fn(status); } catch (err) { console.error("sync-center", err); }
      }
    }).stop;
  }
  entry.listeners.add(onStatus);
  // Kech qo'shilgan obunachi darhol oxirgi holatni oladi (bo'sh ekran yo'q).
  if (entry.last) { try { onStatus(entry.last); } catch (err) { console.error("sync-center", err); } }
  return {
    stop() {
      entry.listeners.delete(onStatus);
      if (!entry.listeners.size) { entry.stop(); shared.delete(key); }
    },
  };
}

/** Haqiqiy ulanish: `EventSource`, ikki marta yiqilsa — polling zaxirasi. */
function rawSubscribe(accountId, onStatus) {
  let stopped = false;
  let es = null;
  let timer = 0;
  let fails = 0;
  let poll = typeof EventSource !== "function";
  let pollMs = 5000;

  const later = (ms, fn) => { timer = setTimeout(() => { if (!stopped) fn(); }, ms); };

  const take = (status) => {
    fails = 0;
    pollMs = Math.max(1000, (Number(status.poll_seconds) || 5) * 1000);
    try { onStatus(status); } catch (err) { console.error("sync-center", err); }
  };

  async function pollOnce() {
    try {
      take(await get(`/api/accounts/${accountId}/sync/status`));
    } catch {
      fails += 1;
    }
    if (!stopped) later(Math.min(MAX_BACKOFF, pollMs * (fails ? 2 ** fails : 1)), pollOnce);
  }

  function openStream() {
    if (stopped) return;
    if (poll) { pollOnce(); return; }
    es = new EventSource(`/api/accounts/${accountId}/sync/stream`);
    es.addEventListener("status", (ev) => {
      let data = null;
      try { data = JSON.parse(ev.data); } catch { return; }
      take(data);
    });
    // Server bo'sh turganda `done` yuboradi va ulanishni yopadi — biz uni
    // `poll_seconds` dan keyin qayta ochamiz (ya'ni bu holatda SSE = polling).
    es.addEventListener("done", () => { close(); later(pollMs, openStream); });
    es.onerror = () => {
      close();
      fails += 1;
      if (fails >= 3) { poll = true; later(pollMs, pollOnce); return; }
      later(Math.min(MAX_BACKOFF, 1000 * 2 ** fails), openStream);
    };
  }

  function close() {
    if (es) { try { es.close(); } catch { /* allaqachon yopiq */ } es = null; }
  }

  openStream();
  return {
    stop() { stopped = true; clearTimeout(timer); close(); },
  };
}

/* ══ Umumiy bo'laklar ═════════════════════════════════════════════════ */

const avatarFor = (item, cls = "sm") => {
  const node = el(`span.avatar.${cls}`, {
    class: peerClass(item.chat_id ?? item.id ?? 0),
    "aria-hidden": "true",
  });
  if (item.avatar_url) {
    const img = el("img", {
      src: item.avatar_url, alt: "", loading: "lazy", decoding: "async",
      onerror: () => { img.remove(); node.appendChild(document.createTextNode(initials(item.title))); },
    });
    node.appendChild(img);
  } else {
    node.appendChild(document.createTextNode(initials(item.title || "")));
  }
  return node;
};

/**
 * Progress chizig'i. `role="progressbar"` — rang yolg'iz ma'no tashimaydi,
 * foiz `aria-valuenow` da ham bor.
 */
function bar(percent, { shimmer = false, label = "" } = {}) {
  const p = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  const fill = el("i", { style: { inlineSize: `${p}%` } });
  const node = el(`div.sync-bar${shimmer && !reduced() ? ".is-live" : ""}`, {
    role: "progressbar",
    "aria-valuenow": String(p),
    "aria-valuemin": "0",
    "aria-valuemax": "100",
    "aria-label": label || t("web.sync.title"),
  }, fill);
  node.setPercent = (next) => {
    const v = Math.max(0, Math.min(100, Math.round(Number(next) || 0)));
    fill.style.inlineSize = `${v}%`;
    node.setAttribute("aria-valuenow", String(v));
  };
  return node;
}

/** `12 450 / 31 000 xabar · 85 xabar/s · ~4 daq` */
function metaText(current) {
  const bits = [];
  const done = Number(current.synced_total || 0);
  const est = Number(current.total_estimate || 0);
  bits.push(est ? t("web.sync.messages_of", { done: num(done), total: num(est) }) : plural("web.n.messages", done));
  if (current.rate_per_s) bits.push(t("web.sync.rate", { n: num(Math.round(current.rate_per_s)) }));
  const eta = etaText(current.eta_s);
  if (eta) bits.push(eta);
  return bits.join(" · ");
}

const phaseBadge = (phase) =>
  el(`span.sync-phase.is-${phase === "backfill" ? "backfill" : "new"}`, null,
    t(phase === "backfill" ? "web.sync.phase_backfill" : "web.sync.phase_new"));

/** Amber pauza banneri jonli hisoblagich bilan. `stop()` beradi. */
function pausedBanner(untilIso) {
  const left = el("b");
  const node = el("div.sync-paused", { role: "status" }, [
    icon("clock"),
    el("span", null, [t("web.sync.paused_prefix"), " "]),
    left,
    el("span", null, [" ", t("web.sync.paused_suffix")]),
  ]);
  const tick = () => {
    const txt = countdownText(untilIso);
    left.textContent = txt || "00:00";
  };
  tick();
  const id = setInterval(tick, 1000);
  node.stop = () => clearInterval(id);
  return node;
}

/* ══ Ixcham band (`/today`) ═══════════════════════════════════════════ */

/**
 * `/today` bandi. Jonli obuna o'zi ichida.
 *
 * @param {HTMLElement} host  bo'shatilib to'ldiriladi
 * @param {number|string} accountId
 * @param {{onDetails?: () => void}} [opts]
 * @returns {{update: (s: object) => void, destroy: () => void}}
 */
export function mountStrip(host, accountId, opts = {}) {
  let sub = null;
  let banner = null;
  // `aria-live` tugun **bir marta** qo'yiladi va o'rnida qoladi: uni har
  // yangilanishda qayta joylashtirish ba'zi ekran o'quvchilarida butun
  // matnni qaytadan o'qitadi (sekundiga bir marta — chidab bo'lmaydi).
  const live = el("span.sr-only", { "aria-live": "polite" });
  const shell = el("div");
  clear(host);
  host.append(live, shell);
  let announced = null;

  // Jonli holatda tugunlar **qayta qurilmaydi**: har sekundda yangi `div`
  // yasalsa `transition: inline-size 600ms` hech qachon ishlamaydi (chiziq
  // sakraydi) va raqamlar sanalmaydi. Shuning uchun chat o'zgarmaguncha
  // faqat qiymatlar yangilanadi.
  let vm = null;

  const paint = (status) => {
    const cur = status.current;
    const counts = status.counts || {};
    const paused = cur && cur.paused_until;

    if (status.running && cur && vm && vm.chatId === Number(cur.chat_id) && !paused && !vm.paused) {
      const pct = chatPercent(cur);
      const v = pct == null ? overallPercent(status) : pct;
      vm.bar.setPercent(v);
      countUp(vm.pct, v, (x) => `${Math.round(x)}%`);
      vm.meta.textContent = metaText(cur);
      vm.counters.textContent = t("web.sync.counters", {
        pending: num(counts.pending || 0), done: num(counts.done || 0), failed: num(counts.failed || 0),
      });
      return;
    }

    if (banner) { banner.stop(); banner = null; }
    vm = null;
    const kids = [];

    if (status.running && cur) {
      // Ekran o'quvchi uchun **faqat chat almashganda** xabar (har soniya emas).
      if (announced !== cur.title) { announced = cur.title; live.textContent = t("web.sync.announce", { title: cur.title || "" }); }
      const pct = chatPercent(cur);
      const overall = overallPercent(status);
      const value = pct == null ? overall : pct;
      const pctNode = el("span.sync-strip-pct");
      countUp(pctNode, value, (v) => `${Math.round(v)}%`);
      const barNode = bar(value, { shimmer: true, label: cur.title || "" });
      const metaNode = el("div.sync-strip-meta", null, metaText(cur));
      vm = {
        chatId: Number(cur.chat_id), bar: barNode, pct: pctNode,
        meta: metaNode, counters: null, paused: Boolean(paused),
      };
      kids.push(el("div.sync-strip-main", null, [
        avatarFor(cur, "sm"),
        el("div.sync-strip-body", null, [
          el("div.sync-strip-title", null, [
            el("span.sync-strip-label", null, t("web.sync.running_chat")),
            el("b", null, cur.title || ""),
          ]),
          barNode,
          metaNode,
        ]),
        pctNode,
      ]));
    } else if (status.running) {
      kids.push(el("div.sync-strip-main", null, [
        icon("refresh-cw", { cls: reduced() ? "" : "spin" }),
        el("div.sync-strip-body", null, [
          el("div.sync-strip-title", null, t("web.sync.queued")),
          bar(overallPercent(status), { shimmer: true }),
        ]),
      ]));
    } else {
      const ago = status.last_sync_at ? relTime(status.last_sync_at) : null;
      kids.push(el("div.sync-strip-main", null, [
        icon("check-check", { cls: "sync-ok" }),
        el("div.sync-strip-body", null, [
          el("div.sync-strip-title", null,
            ago ? `${t("web.sync.all_synced")} · ${t("web.sync.last_ago", { ago })}` : t("web.sync.all_synced")),
        ]),
        el("button.btn.ghost.small", {
          type: "button",
          onclick: async () => {
            try { await post(`/api/accounts/${accountId}/sync`); toast(t("web.chats.sync_started"), { kind: "ok" }); }
            catch (err) { toastError(err); }
          },
        }, [icon("refresh-cw"), " ", t("web.sync.refresh")]),
      ]));
    }

    if (paused) { banner = pausedBanner(cur.paused_until); kids.push(banner); }

    const counters = el("span.sync-counters", null, t("web.sync.counters", {
      pending: num(counts.pending || 0), done: num(counts.done || 0), failed: num(counts.failed || 0),
    }));
    if (vm) vm.counters = counters;
    kids.push(el("div.sync-strip-foot", null, [
      counters,
      el("span.spacer"),
      el("button.btn.ghost.small", {
        type: "button",
        onclick: () => (opts.onDetails ? opts.onDetails() : openSheet(accountId)),
      }, t("web.sync.details")),
    ]));

    replace(shell, el("div.card-v2.sync-strip", null, kids));
    host.hidden = false;
  };

  sub = subscribe(accountId, paint);
  return {
    update: paint,
    destroy() { if (banner) banner.stop(); sub && sub.stop(); clear(host); },
  };
}

/* ══ To'liq varaq ═════════════════════════════════════════════════════ */

let openPanel = null;

/**
 * To'liq sync markazi (modal varaq). Ikkinchi marta chaqirilsa mavjudi
 * fokuslanadi — ikkita varaq ochilib qolmasin.
 *
 * @param {number|string} accountId
 * @returns {{close: () => void}}
 */
export function openSheet(accountId) {
  if (openPanel) return openPanel;

  const body = el("div.sync-sheet-body");
  const box = el("div.sheet-box.sync-sheet", null, [
    el("div.sheet-grab", { "aria-hidden": "true" }),
    el("div.sync-sheet-head", null, [
      icon("refresh-cw"),
      el("h2", { id: "sync-sheet-h" }, t("web.sync.title")),
      el("span.spacer"),
      el("button.icon-btn-v2", { type: "button", "aria-label": t("web.sync.close"), onclick: () => close() }, icon("x")),
    ]),
    body,
  ]);
  const root = el("div.sheet.sync-sheet-root", {
    role: "dialog", "aria-modal": "true", "aria-labelledby": "sync-sheet-h",
    onmousedown: (e) => { if (e.target === root) close(); },
  }, box);

  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.body.appendChild(root);
  const untrap = trapFocus(box);
  document.addEventListener("keydown", onKey);
  box.querySelector("button")?.focus();

  let banner = null;
  // Strip'dagi kabi: tarkib o'zgarmaguncha faqat qiymatlar yangilanadi —
  // aks holda halqaning `stroke-dashoffset` o'tishi va sanoq yo'qoladi,
  // ro'yxatlar esa har sekundda qayta chiziladi (hover/scroll sakraydi).
  let vm = null;
  const shapeKey = (s) => JSON.stringify([
    Boolean(s.running),
    (s.current || {}).chat_id ?? null,
    (s.current || {}).phase ?? null,
    Boolean((s.current || {}).paused_until),
    (s.queue || []).map((q) => q.chat_id),
    (s.recent_done || []).map((r) => r.chat_id),
    (s.failed || []).map((f) => f.chat_id),
    (s.counts || {}).pending ?? 0,
    (s.ramp || {}).limited ?? false,
    (s.parallel || {}).accounts_running ?? 0,
  ]);

  const sub = subscribe(accountId, (status) => {
    const key = shapeKey(status);
    if (vm && vm.key === key) {
      const pct = overallPercent(status);
      vm.ring.setPercent(pct);
      countUp(vm.ringPct, pct, (v) => `${Math.round(v)}%`);
      if (vm.bar && status.current) {
        const p = chatPercent(status.current);
        vm.bar.setPercent(p == null ? pct : p);
        vm.meta.textContent = metaText(status.current);
      }
      return;
    }
    if (banner) { banner.stop(); banner = null; }
    vm = { key, ring: null, ringPct: null, bar: null, meta: null };
    const kids = [];
    const cur = status.current;
    const counts = status.counts || {};

    // ── Umumiy halqa ──
    kids.push(ringHeader(status, accountId, vm));

    if (cur && cur.paused_until) { banner = pausedBanner(cur.paused_until); kids.push(banner); }

    // ── Joriy chat ──
    if (status.running && cur) {
      const pct = chatPercent(cur);
      const curBar = bar(pct == null ? overallPercent(status) : pct, { shimmer: true, label: cur.title || "" });
      const curMeta = el("div.sync-strip-meta", null, metaText(cur));
      vm.bar = curBar;
      vm.meta = curMeta;
      kids.push(el("div.sync-current", null, [
        avatarFor(cur, "lg-44"),
        el("div.sync-current-body", null, [
          el("div.sync-current-head", null, [el("b", null, cur.title || ""), phaseBadge(cur.phase)]),
          curBar,
          curMeta,
          el("div.sync-faint", null, [
            elapsedText(cur.started_at) ? t("web.sync.elapsed", { d: elapsedText(cur.started_at) }) : "",
          ]),
        ]),
      ]));
    } else if (!status.running) {
      kids.push(el("p.sync-faint", null, t("web.sync.idle_note")));
    }

    // ── Parallel indikatori ──
    const par = status.parallel || {};
    kids.push(el("div.sync-parallel", { title: t("web.sync.parallel_tip", { max: num(par.max_jobs || 0) }) }, [
      icon("layers"),
      el("span", null, t("web.sync.parallel", { n: num(par.accounts_running || 0) })),
      el("span.sync-faint", null, ` · ${t("web.sync.parallel_note")}`),
    ]));

    // ── Ramp-up ──
    if (status.ramp && status.ramp.limited) {
      kids.push(el("div.sync-note", null, [
        icon("info"),
        el("span", null, t("web.sync.ramp_note", {
          messages: num(status.ramp.max_messages || 0), chats: num(status.ramp.max_chats || 0),
        })),
      ]));
    }

    // ── Navbat ──
    const queue = status.queue || [];
    if (queue.length) {
      const extra = Math.max(0, Number(counts.pending || 0) - queue.length);
      kids.push(section("list", t("web.sync.queue_title"), [
        el("ul.sync-list", null, queue.map((q) => el("li.sync-item", null, [
          avatarFor(q, "sm"),
          el("span.sync-item-title", null, q.title || ""),
          el("span.sync-item-meta", null,
            q.total_estimate ? t("web.sync.est", { n: num(q.total_estimate) }) : plural("web.n.messages", q.synced_total || 0)),
        ]))),
        extra ? el("div.sync-faint", null, t("web.sync.queue_more", { n: num(extra) })) : null,
      ]));
    }

    // ── Yaqinda tayyor ──
    const recent = status.recent_done || [];
    if (recent.length) {
      kids.push(section("check-check", t("web.sync.recent_title"), [
        el("ul.sync-list", null, recent.map((r) => el("li.sync-item.is-done", null, [
          icon("check"),
          el("span.sync-item-title", null, r.title || ""),
          el("span.sync-item-meta", null, plural("web.n.messages", r.synced_total || 0)),
        ]))),
      ]));
    }

    // ── Xatolar ──
    const failed = status.failed || [];
    if (failed.length) {
      kids.push(section("alert-triangle", t("web.sync.failed_title"), [
        el("ul.sync-list", null, failed.map((f) => el("li.sync-item.is-failed", null, [
          el("div.sync-item-body", null, [
            el("span.sync-item-title", null, f.title || ""),
            el("span.sync-error", null, f.error || ""),
          ]),
          el("button.btn.ghost.small", {
            type: "button",
            onclick: async () => {
              try { await post(`/api/accounts/${accountId}/chats/${f.chat_id}/sync`); toast(t("web.chats.sync_started"), { kind: "ok" }); }
              catch (err) { toastError(err); }
            },
          }, t("web.sync.retry")),
        ]))),
      ]));
    }

    replace(body, kids);
  });

  function close() {
    if (banner) banner.stop();
    sub.stop();
    untrap();
    document.removeEventListener("keydown", onKey);
    root.remove();
    openPanel = null;
  }

  openPanel = { close };
  return openPanel;
}

function section(iconName, title, kids) {
  return el("section.sync-section", null, [
    el("div.sync-section-head", null, [icon(iconName), el("h3", null, title)]),
    ...(Array.isArray(kids) ? kids.filter(Boolean) : [kids]),
  ]);
}

/** Sarlavha: SVG halqa (umumiy foiz) + chat hisobi + "Hammasini sinxronlash". */
function ringHeader(status, accountId, vm) {
  const pct = overallPercent(status);
  const { c, offset } = ringGeometry(pct);
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "sync-ring");
  svg.setAttribute("viewBox", "0 0 64 64");
  svg.setAttribute("aria-hidden", "true");
  let fill = null;
  for (const cls of ["track", "fill"]) {
    const circle = document.createElementNS(NS, "circle");
    circle.setAttribute("class", cls);
    circle.setAttribute("cx", "32");
    circle.setAttribute("cy", "32");
    circle.setAttribute("r", String(RING_R));
    if (cls === "fill") {
      circle.setAttribute("stroke-dasharray", String(c));
      circle.setAttribute("stroke-dashoffset", String(offset));
      fill = circle;
    }
    svg.appendChild(circle);
  }
  const pctNode = el("span.sync-ring-pct");
  countUp(pctNode, pct, (v) => `${Math.round(v)}%`);
  const wrap = el("div.sync-ring-wrap", { role: "progressbar", "aria-valuenow": String(pct), "aria-valuemin": "0", "aria-valuemax": "100", "aria-label": t("web.sync.overall") }, [svg, pctNode]);
  if (vm) {
    vm.ringPct = pctNode;
    vm.ring = {
      setPercent(next) {
        const v = Math.max(0, Math.min(100, Math.round(Number(next) || 0)));
        fill.setAttribute("stroke-dashoffset", String(ringGeometry(v).offset));
        wrap.setAttribute("aria-valuenow", String(v));
      },
    };
  }
  const job = status.job || {};
  return el("div.sync-head", null, [
    wrap,
    el("div.sync-head-body", null, [
      el("div.sync-head-title", null, t("web.sync.overall")),
      el("div.sync-faint", null, t("web.sync.chats_done", {
        done: num(job.chats_done || (status.counts || {}).done || 0),
        total: num(job.chats_total || (status.counts || {}).total || 0),
      })),
    ]),
    el("span.spacer"),
    // Yugurayotganda tugma o'chiq: ikkinchi `sync_account` ishga tushmaydi
    // (`job_id` deterministik), lekin foydalanuvchi buni bilmaydi — o'chiq
    // tugma "bosdim, hech narsa bo'lmadi" holatidan yaxshiroq.
    el("button.btn.primary.small", {
      type: "button",
      disabled: Boolean(status.running),
      onclick: async () => {
        try { await post(`/api/accounts/${accountId}/sync`); toast(t("web.chats.sync_started"), { kind: "ok" }); }
        catch (err) { toastError(err); }
      },
    }, [icon("refresh-cw"), " ", t("web.chats.sync_all")]),
  ]);
}

/**
 * Topbar ikonkasiga bog'lash (`/chats`, `/settings`): bosilsa varaq ochiladi,
 * sync ketayotganda ikonka sekin puls uradi.
 *
 * @returns {{stop: () => void}}
 */
export function bindSyncButton(button, accountId) {
  if (!button) return { stop() {} };
  button.addEventListener("click", () => openSheet(accountId));
  const sub = subscribe(accountId, (status) => {
    button.classList.toggle("is-syncing", Boolean(status.running) && !reduced());
    button.setAttribute("aria-label", status.running ? t("web.sync.title") : button.getAttribute("aria-label") || t("web.sync.title"));
  });
  return { stop: () => sub.stop() };
}

