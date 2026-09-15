/* pages/ask/deep.js — chuqur qidiruv yuzasi: bosqichlar, klasterlar,
 * manbalar tasmasi, sitata chiplari va qamrov paneli.
 *
 * Shartnoma `services/deep_search.py` bilan: `plan → search → cluster →
 * rerank → token → coverage → done | error`. Hodisa nomlari va maydonlari
 * shu yerda **bir joyda** tarjima qilinadi.
 *
 * Ikki qat'iy qoida:
 *   * javob matni faqat `md()` orqali (`messages.renderAnswer`);
 *   * sitata havolasi faqat `safeHref()` dan o'tgan `link` — token'dan
 *     URL **yasalmaydi** (server bergan `link` yoki hech narsa).
 *
 * "Ishonch foizi" ko'rsatilmaydi: qamrov — nechta chat, nechta xabar,
 * qayerda bo'shliq. Bitta raqam bu savolga yolg'on aniqlik beradi.
 */

import { el, clear, replace, $ } from "../../lib/dom.js";
import { get, post } from "../../lib/api.js";
import { icon } from "../../lib/icons.js";
import { t } from "../../lib/i18n.js";
import { compact, date as fmtDate, dateTime, usd } from "../../lib/fmt.js";
import { peerClass } from "../../lib/palette.js";
import { toast, toastError } from "../../lib/toast.js";
import { isMobile } from "../../lib/store.js";
import { S, safeHref } from "./state.js";
import { openSheet, externalLink } from "./panels.js";
import { chatAvatar } from "./messages.js";

/* ── sitata tokenlari ─────────────────────────────────────────────── */

/**
 * Sitata tokeni: `#c<chat_id>m<tg_msg_id>` (lokal) yoki `#x<n>` (tashqi).
 * Raqamlar uzunligi cheklangan — cheksiz `\d+` katastrofik qaytishga
 * (ReDoS) yo'l ochmasligi uchun emas, balki tasodifiy `#c1234…` matni
 * sitata bo'lib qolmasligi uchun.
 */
const CITE_RE = /#(c(\d{1,12})m(\d{1,12})|x\d{1,12})/g;

/**
 * Matnni sitata bo'laklariga ajratadi — **sof funksiya**, DOM'siz
 * (`tests/test_js_lib.py` shu yerni tekshiradi).
 *
 * @param {string} text
 * @returns {Array<{text: string} | {token: string, raw: string, chatId: number|null, msgId: number|null}>}
 */
export function citeParts(text) {
  const src = String(text ?? "");
  const out = [];
  let last = 0;
  let m;
  CITE_RE.lastIndex = 0;
  while ((m = CITE_RE.exec(src)) !== null) {
    if (m.index > last) out.push({ text: src.slice(last, m.index) });
    out.push({
      token: m[1],
      raw: m[0],
      chatId: m[2] ? Number(m[2]) : null,
      msgId: m[3] ? Number(m[3]) : null,
    });
    last = m.index + m[0].length;
  }
  if (last < src.length) out.push({ text: src.slice(last) });
  return out;
}

/**
 * `done.citations` dan qidiruv jadvali: token → sitata, va har chat uchun
 * havola prefiksi (`link` dan `tg_msg_id` ni olib tashlab). Prefiks kerak,
 * chunki model bir chatdagi boshqa xabarga ham havola qilishi mumkin.
 */
export function citationIndex(citations) {
  const byToken = new Map();
  const prefix = new Map();
  for (const c of citations || []) {
    if (!c) continue;
    if (c.token) byToken.set(String(c.token), c);
    const tail = String(c.id ?? "");
    if (c.chat_id && tail && typeof c.link === "string" && c.link.endsWith(tail)) {
      prefix.set(String(c.chat_id), c.link.slice(0, c.link.length - tail.length));
    }
  }
  return { byToken, prefix };
}

/** Bo'lak uchun havola: to'g'ridan-to'g'ri token, bo'lmasa chat prefiksi. */
function linkFor(part, index) {
  const hit = index.byToken.get(part.token);
  if (hit && safeHref(hit.link)) return safeHref(hit.link);
  if (part.chatId != null && part.msgId != null) {
    const pre = index.prefix.get(String(part.chatId));
    if (pre) return safeHref(pre + part.msgId);
  }
  return null;
}

/* ── hover kartasi ────────────────────────────────────────────────── */

const HOVER_DELAY_MS = 200;
const cardCache = new Map();   // "chatId:msgId" → xabar yoki null

/** Xabar tafsiloti (lokal DB). Bir marta so'raladi, keyin keshdan. */
async function fetchMessage(chatId, msgId) {
  const key = `${chatId}:${msgId}`;
  if (cardCache.has(key)) return cardCache.get(key);
  let row = null;
  try {
    row = await get(`/api/accounts/${S.accountId}/chats/${chatId}/messages/${msgId}`);
  } catch {
    row = null;   // sinxronlanmagan yoki o'chirilgan — karta o'rniga faqat havola
  }
  cardCache.set(key, row);
  return row;
}

/** Karta mazmuni (hover'da ham, mobil varaqda ham bir xil). */
function cardBody(row, citation, href) {
  const chat = row && row.chat_id != null ? S.byId.get(Number(row.chat_id)) : null;
  const title = (row && row.chat_title) || (citation && citation.chat_title) || "";
  const when = (row && row.date) || (citation && citation.date) || null;
  return el("div.cite-card", null, [
    el("div.cite-card-head", null, [
      chatAvatar(chat || { title, id: row?.chat_id || 0 }),
      el("span.cite-card-title", null, title || t("web.ask.cite_unknown")),
      when ? el("span.muted", null, dateTime(when)) : null,
    ]),
    row && row.text
      ? el("p.cite-card-text", null, row.text)
      : el("p.muted", null, t("web.ask.cite_not_synced")),
    href ? externalLink(href, t("web.ask.open_in_telegram")) : null,
  ]);
}

let hoverTimer = 0;
let hoverCard = null;

function hideHover() {
  clearTimeout(hoverTimer);
  hoverCard?.remove();
  hoverCard = null;
}

/** Chip ustida hover kartasini ko'rsatadi (desktop). */
function showHover(chip, part, index, href) {
  hideHover();
  hoverTimer = setTimeout(async () => {
    const citation = index.byToken.get(part.token) || null;
    const row = part.chatId != null && part.msgId != null
      ? await fetchMessage(part.chatId, part.msgId)
      : null;
    if (!chip.isConnected) return;
    hoverCard = el("div.cite-pop", { role: "tooltip" }, cardBody(row, citation, href));
    document.body.appendChild(hoverCard);
    const r = chip.getBoundingClientRect();
    const top = r.bottom + 8;
    hoverCard.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - hoverCard.offsetWidth - 8))}px`;
    hoverCard.style.top = `${Math.min(top, window.innerHeight - hoverCard.offsetHeight - 8)}px`;
  }, HOVER_DELAY_MS);
}

/** Mobil: karta pastki varaqda (hover yo'q). */
async function openCiteSheet(part, index, href) {
  const citation = index.byToken.get(part.token) || null;
  openSheet(t("web.ask.source"), el("p.muted", null, "…"));
  const row = part.chatId != null && part.msgId != null
    ? await fetchMessage(part.chatId, part.msgId)
    : null;
  const body = $("#ask-sheet-body");
  if (body) replace(body, cardBody(row, citation, href));
}

/* ── sitata → chip ────────────────────────────────────────────────── */

/**
 * Chizilgan javob ichidagi sitata tokenlarini ustki indeks chipiga
 * aylantiradi.
 *
 * **DOM matn tugunlari bo'ylab** yuriladi, HTML qayta parse qilinmaydi —
 * ya'ni escape-first invarianti buzilmaydi. `a`, `code`, `pre` ichidagi
 * matnga tegilmaydi.
 */
export function linkifyCitations(root, citations) {
  if (!root) return;
  const index = citationIndex(citations);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const targets = [];
  while (walker.nextNode()) {
    const n = walker.currentNode;
    if (!n.parentElement || n.parentElement.closest("a, code, pre")) continue;
    if (/#(?:c\d|x\d)/.test(n.nodeValue || "")) targets.push(n);
  }
  let seq = 0;
  for (const n of targets) {
    const parts = citeParts(n.nodeValue);
    if (parts.length < 2 && !parts.some((p) => p.token)) continue;
    const frag = document.createDocumentFragment();
    for (const part of parts) {
      if (part.text !== undefined) { frag.appendChild(document.createTextNode(part.text)); continue; }
      seq += 1;
      frag.appendChild(citeChip(part, index, seq));
    }
    n.parentNode.replaceChild(frag, n);
  }
}

/** Bitta sitata chipi (`sup`). Havola bo'lmasa ham chip qoladi — raqam ma'noli. */
function citeChip(part, index, seq) {
  const href = linkFor(part, index);
  const citation = index.byToken.get(part.token) || null;
  const chip = el("sup.cite-chip", {
    tabindex: "0",
    role: "button",
    dataset: { token: part.token, chat: part.chatId ?? "", cite: citation?.chat_title || "" },
    "aria-label": t("web.ask.source_n", { n: seq }),
    title: citation?.chat_title || part.raw,
  }, String(seq));

  const open = (e) => { e.preventDefault(); openCiteSheet(part, index, href); };
  chip.addEventListener("click", open);
  chip.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") open(e); });
  if (!isMobile()) {
    chip.addEventListener("mouseenter", () => showHover(chip, part, index, href));
    chip.addEventListener("mouseleave", hideHover);
    chip.addEventListener("focus", () => showHover(chip, part, index, href));
    chip.addEventListener("blur", hideHover);
  }
  return chip;
}

/* ── bosqich timeline ─────────────────────────────────────────────── */

const STAGES = [
  ["plan", "brain", "web.deep.step_plan"],
  ["search", "search", "web.ask.stage_search"],
  ["cluster", "layers", "web.ask.stage_cluster"],
  ["rerank", "filter", "web.ask.stage_rerank"],
  ["write", "edit-3", "web.deep.step_write"],
];

/** Vertikal (desktop) / gorizontal scroll-snap (mobil) bosqich reyki. */
function stageRail() {
  const items = new Map();
  const rail = el("ol.stage-rail", { "aria-label": t("web.ask.stages") });
  for (const [key, ico, label] of STAGES) {
    const detail = el("span.stage-detail");
    const body = el("div.stage-body", null, [el("span.stage-label", null, t(label)), detail]);
    const li = el("li.stage-item", { dataset: { state: "pending", stage: key } }, [icon(ico), body]);
    rail.appendChild(li);
    items.set(key, { li, detail });
  }
  return { rail, items };
}

/* ── klaster chiplari ─────────────────────────────────────────────── */

const CLUSTER_LIMIT = 12;

function clusterChips(clusters, answerEl) {
  const wrap = el("div.cluster-chips", { role: "group", "aria-label": t("web.deep.clusters") });
  let active = null;
  for (const c of (clusters || []).slice(0, CLUSTER_LIMIT)) {
    const when = [c.first_seen, c.last_seen].filter(Boolean).map((d) => fmtDate(d)).join(" … ");
    const chip = el("button.chip.cluster", {
      type: "button",
      class: peerClass(c.index + 1),
      "aria-pressed": "false",
      title: [(c.chats || []).join(", "), when].filter(Boolean).join(" · "),
    }, [el("span", null, c.label || "…"), el("b", null, String(c.size))]);
    chip.addEventListener("click", () => {
      const on = chip.getAttribute("aria-pressed") === "true";
      if (active && active !== chip) active.setAttribute("aria-pressed", "false");
      chip.setAttribute("aria-pressed", on ? "false" : "true");
      active = on ? null : chip;
      highlightCitations(answerEl, on ? null : new Set(c.chats || []));
    });
    wrap.appendChild(chip);
  }
  return wrap;
}

/** Klaster tanlanganda faqat o'sha chatlarning sitatalari yorqin qoladi. */
function highlightCitations(answerEl, titles) {
  if (!answerEl) return;
  answerEl.classList.toggle("cite-filtered", titles != null);
  for (const chip of answerEl.querySelectorAll(".cite-chip")) {
    const title = chip.dataset.cite || "";
    chip.classList.toggle("is-dim", titles != null && !titles.has(title));
  }
}

/* ── manbalar tasmasi ─────────────────────────────────────────────── */

/** `done.citations` dan chat bo'yicha guruhlangan gorizontal tasma. */
function sourceStrip(citations, answerEl) {
  const byChat = new Map();
  for (const c of citations || []) {
    const key = String(c.chat_id ?? c.chat_title ?? "");
    if (!key) continue;
    const row = byChat.get(key) || { chat_id: c.chat_id, title: c.chat_title, n: 0, link: c.link };
    row.n += 1;
    byChat.set(key, row);
  }
  if (!byChat.size) return null;
  const strip = el("div.source-strip", { role: "list", "aria-label": t("web.ask.sources") });
  for (const row of [...byChat.values()].sort((a, b) => b.n - a.n)) {
    const chat = row.chat_id != null ? S.byId.get(Number(row.chat_id)) : null;
    const card = el("button.source-card", {
      type: "button", role: "listitem",
      title: row.title || "",
    }, [
      chatAvatar(chat || { title: row.title || "?", id: row.chat_id || 0 }),
      el("span.source-title", null, row.title || t("web.ask.cite_unknown")),
      el("span.source-count", null, t("web.ask.n_citations", { n: row.n })),
    ]);
    card.addEventListener("click", () => {
      const on = card.getAttribute("aria-pressed") === "true";
      for (const other of strip.children) other.setAttribute("aria-pressed", "false");
      card.setAttribute("aria-pressed", on ? "false" : "true");
      highlightCitations(answerEl, on ? null : new Set([row.title]));
    });
    strip.appendChild(card);
  }
  return el("div.source-strip-wrap", null, [
    el("div.source-strip-head", null, t("web.ask.sources")), strip,
  ]);
}

/* ── qamrov paneli ────────────────────────────────────────────────── */

/**
 * Qamrov — yig'ilgan `<details>`. **Ishonch foizi yo'q**: nechta chat
 * qidirildi, nechta xabar ko'rildi, qaysi chatda tarix to'liq emas.
 */
function coveragePanel(cov, costUsd) {
  if (!cov) return null;
  const summary = el("summary", null, [
    icon("shield-check"),
    el("span", null, t("web.ask.coverage_sum", {
      chats: cov.chats_searched ?? 0,
      msgs: compact(cov.messages_scanned ?? 0),
    })),
  ]);
  const rows = el("div.cov-rows");

  // Har chat uchun ikki chiziq: qidirilgan ulush va sinxronlangan ulush.
  const total = Math.max(1, cov.chats_total || cov.chats_searched || 1);
  rows.appendChild(bar(t("web.deep.cov_chats"),
    `${cov.chats_searched ?? 0}/${cov.chats_total ?? 0}`, (cov.chats_searched || 0) / total));

  const gapSet = new Set(cov.gaps || []);
  for (const title of [...gapSet].slice(0, 8)) {
    const chat = S.chats.find((c) => c.title === title);
    const line = el("div.cov-gap", null, [
      icon("alert-triangle"),
      el("span.cov-gap-title", null, title),
      el("span.muted", null, chat ? `${chat.progress ?? 0}%` : t("web.deep.gaps")),
    ]);
    if (chat) {
      line.appendChild(el("button.btn.ghost.small", {
        type: "button",
        onclick: async (e) => {
          e.currentTarget.disabled = true;
          try {
            await post(`/api/accounts/${S.accountId}/chats/${chat.id}/sync`, {});
            toast(t("web.chat.sync_queued"), { kind: "ok" });
          } catch (err) { toastError(err); e.currentTarget.disabled = false; }
        },
      }, t("web.chat.sync")));
    }
    rows.appendChild(line);
  }
  if (gapSet.size > 8) rows.appendChild(el("div.muted", null, `+${gapSet.size - 8}`));

  const r = cov.range || {};
  const facts = [
    `${t("web.deep.cov_scanned")}: ${compact(cov.messages_scanned ?? 0)}`,
    `${t("web.deep.cov_rounds")}: ${cov.rounds ?? 1}`,
  ];
  if (r.from && r.to) facts.push(`${t("web.deep.cov_range")}: ${fmtDate(r.from)} … ${fmtDate(r.to)}`);
  if (costUsd != null) facts.push(`${t("web.deep.cov_cost")}: ${usd(costUsd)}`);
  rows.appendChild(el("div.cov-facts", null, facts.join(" · ")));

  if (cov.server_rungs_used) {
    const srv = [
      `${t("web.deep.cov_server_calls")}: ${cov.server_calls || 0}`,
      `${t("web.deep.cov_server_hits")}: ${cov.server_hits || 0}`,
    ];
    if (cov.external_hits) srv.push(`${t("web.deep.cov_external")}: ${cov.external_hits}`);
    rows.appendChild(el("div.cov-facts", null, `${t("web.deep.cov_server")} — ${srv.join(" · ")}`));
  }
  if (cov.server_skipped && cov.server_skipped.length) {
    rows.appendChild(el("div.cov-gap", null, [
      icon("alert-triangle"),
      el("span", null, `${t("web.deep.server_skipped")}: ${cov.server_skipped.join(", ")}`),
    ]));
  }
  if (cov.truncated) {
    rows.appendChild(el("div.cov-gap", null, [icon("alert-triangle"), el("span", null, t("web.deep.truncated"))]));
  }
  return el("details.cov-box", null, [summary, rows]);
}

function bar(label, value, ratio) {
  const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));
  return el("div.cov-bar", null, [
    el("span.cov-bar-label", null, label),
    el("span.cov-bar-track", null, el("i", { style: { inlineSize: `${pct}%` } })),
    el("span.cov-bar-value", null, value),
  ]);
}

/* ── deep paneli ──────────────────────────────────────────────────── */

/**
 * Oqim uchun deep paneli. Qaytadi: DOM tugun + oqim hodisalarini qabul
 * qiluvchi `onEvent`, va yakuniy bezak uchun `finish`.
 */
export function createDeepPanel(answerEl) {
  const { rail, items } = stageRail();
  const plan = el("div.plan-card", { hidden: true });
  const clusters = el("div.cluster-wrap", { hidden: true });
  const sources = el("div.sources-wrap", { hidden: true });
  const coverage = el("div.coverage-wrap", { hidden: true });
  const node = el("section.deep-panel", { "aria-label": t("web.ask.deep") },
    [rail, plan, clusters, sources, coverage]);

  let searched = 0, served = 0, planned = 0, clusterRows = [];

  const setStage = (key, state, detail) => {
    const item = items.get(key);
    if (!item) return;
    item.li.dataset.state = state;
    if (detail != null) item.detail.textContent = detail;
    // Tugagan bosqich yig'iladi, lekin `<li>` DOM'da qoladi — ochib ko'rish mumkin.
    if (state === "done") item.li.classList.add("is-collapsed");
    if (state === "active") item.li.classList.remove("is-collapsed");
  };
  const done = (key) => setStage(key, "done");

  const onEvent = (kind, d) => {
    if (kind === "plan") {
      planned = (d.subqueries || []).length;
      renderPlan(plan, d);
      plan.hidden = false;
      done("plan");
      setStage("search", "active", t("web.deep.step_search", { n: 0, m: planned }));
    } else if (kind === "search") {
      if (d.level === "server") {
        served += 1;
        setStage("search", "active", t("web.deep.step_server", { n: served }));
      } else {
        searched += 1;
        setStage("search", "active", t("web.deep.step_search", { n: searched, m: planned || searched }));
      }
    } else if (kind === "cluster") {
      done("search");
      clusterRows = d.clusters || [];
      replace(clusters, clusterChips(clusterRows, answerEl));
      clusters.hidden = clusterRows.length === 0;
      setStage("cluster", "done", t("web.deep.step_cluster", { n: clusterRows.length }));
    } else if (kind === "rerank") {
      setStage("rerank", "done", t("web.deep.step_rank", { n: d.kept || 0 }));
    } else if (kind === "token") {
      done("rerank");
      setStage("write", "active");
    } else if (kind === "coverage") {
      node.dataset.coverage = "1";
      replace(coverage, coveragePanel(d, null));
      coverage.hidden = false;
    } else if (kind === "error") {
      for (const [key, item] of items) {
        if (item.li.dataset.state === "active") setStage(key, "failed");
      }
    }
  };

  /** `done` kelganda: bosqichlar yopiladi, manbalar va qamrov to'ldiriladi. */
  const finish = (d) => {
    for (const [key] of items) {
      const item = items.get(key);
      if (item.li.dataset.state !== "failed") setStage(key, "done");
    }
    const strip = sourceStrip(d.citations, answerEl);
    if (strip) { replace(sources, strip); sources.hidden = false; }
    linkifyCitations(answerEl, d.citations);
    if (node.dataset.coverage === "1" && d.cost_usd != null) {
      const facts = coverage.querySelector(".cov-facts");
      if (facts) facts.textContent += ` · ${t("web.deep.cov_cost")}: ${usd(d.cost_usd)}`;
    }
    return clusterRows;
  };

  return { node, onEvent, finish, clusters: () => clusterRows };
}

/** Reja kartasi: subquery chiplari, oyna, chat soni. */
function renderPlan(box, plan) {
  clear(box);
  const chips = (plan.subqueries || []).slice(0, 8).map((q) => el("span.chip.tiny", null, q));
  const facts = [];
  if (plan.since || plan.until) {
    facts.push(`${plan.since ? fmtDate(plan.since) : "…"} … ${plan.until ? fmtDate(plan.until) : "…"}`);
  }
  facts.push(Array.isArray(plan.chats) && plan.chats.length
    ? t("web.ask.plan_chats", { n: plan.chats.length })
    : t("web.ask.plan_all_chats"));
  box.appendChild(el("div.plan-head", null, [icon("brain"), el("b", null, t("web.ask.plan"))]));
  if (chips.length) box.appendChild(el("div.plan-chips", null, chips));
  box.appendChild(el("div.plan-facts", null, facts.join(" · ")));
}

/**
 * Saqlangan (tarixdagi) deep javobi uchun panel — qamrov va manbalar
 * `context` ichida bo'lsa chiziladi, aks holda `null`.
 */
export function storedDeepBlock(m, shell) {
  const ctx = m.context || {};
  if (!ctx.coverage && !Array.isArray(ctx.citations)) return null;
  const wrap = el("section.deep-panel.is-stored");
  const strip = sourceStrip(ctx.citations, shell.answer);
  if (strip) wrap.appendChild(strip);
  const cov = coveragePanel(ctx.coverage, m.cost_usd);
  if (cov) wrap.appendChild(cov);
  return wrap.childElementCount ? wrap : null;
}
