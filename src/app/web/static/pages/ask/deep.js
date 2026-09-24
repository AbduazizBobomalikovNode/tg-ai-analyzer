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
import { compact, num, pct, ms as fmtMs, date as fmtDate, dateTime, usd } from "../../lib/fmt.js";
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

/* ── bosqich modeli (sof funksiya) ────────────────────────────────── */

/**
 * Bosqichlar **yagona tartibi**. `fast` (klassik) oqim `plan → search →
 * cluster → rerank → write` ni ishlatadi; korpus ko'lami (`deep`,
 * `exhaustive`, `services/scale_search.py`) esa `estimate → buckets →
 * pull → filter → select → map → reduce → write` ni. Ikkalasi bitta
 * ro'yxatda turadi, chunki deep oqimi ikkalasini ham aralashtirib
 * yuborishi mumkin (rejadan keyin klaster ham keladi).
 */
export const STAGE_ORDER = [
  "plan", "estimate", "buckets", "pull", "filter", "select",
  "search", "map", "reduce", "cluster", "rerank", "write",
];

/**
 * Rejim bo'yicha **zaxira** ro'yxat: server `stages` hodisasini yubormasa
 * (eski versiya) shu ishlatiladi. Odatda reyk `stages` hodisasidan quriladi —
 * quvur har doim bir xil emas: "salom" turn'ida reja + yozish, xolos.
 */
const VISIBLE_STAGES = {
  fast: ["plan", "search", "cluster", "rerank", "write"],
  deep: ["plan", "estimate", "buckets", "pull", "filter", "select", "map", "reduce", "write"],
};
VISIBLE_STAGES.exhaustive = VISIBLE_STAGES.deep;

/** Shu turn uchun rejalashtirilgan bosqichlar (server aytgani yoki zaxira). */
function plannedStages(model) {
  const list = Array.isArray(model.visible) && model.visible.length ? model.visible : null;
  return list || VISIBLE_STAGES[model.mode] || VISIBLE_STAGES.deep;
}

/** Rejalashtirilgan ro'yxatda `key` dan keyingi bosqich (yo'q bo'lsa `null`). */
function nextStage(model, key) {
  const list = plannedStages(model);
  const i = list.indexOf(key);
  return i >= 0 && i + 1 < list.length ? list[i + 1] : null;
}

const STAGE_ICON = {
  plan: "brain", estimate: "activity", buckets: "layers", pull: "database",
  filter: "filter", select: "check-check", search: "search", map: "bolt",
  reduce: "layers", cluster: "layers", rerank: "filter", write: "edit-3",
};

const STAGE_LABEL = {
  plan: "web.deep.step_plan", estimate: "web.ask.stage_estimate",
  buckets: "web.ask.stage_buckets", pull: "web.ask.stage_pull",
  filter: "web.ask.stage_filter", select: "web.ask.stage_select",
  search: "web.ask.stage_search", map: "web.ask.stage_map",
  reduce: "web.ask.stage_reduce", cluster: "web.ask.stage_cluster",
  rerank: "web.ask.stage_rerank", write: "web.ask.stage_writing",
};

const numOr = (v, fallback = 0) => (Number.isFinite(Number(v)) ? Number(v) : fallback);

/** Bo'sh model — `stageModel()` va panel shundan boshlaydi. */
export function emptyModel(mode = "deep") {
  return {
    mode,
    //: Shu turn'da **haqiqatan** yuguradigan bosqichlar (`stages` hodisasi).
    //: `null` — server aytmadi, rejim bo'yicha zaxira ro'yxat ishlatiladi.
    visible: null,
    direct: false,         // qidiruvsiz turn (salom / "nimalar qila olasan")
    intent: "",
    stages: {},            // kalit → {state, detail?, startedAt, endedAt}
    active: null,          // hozir ishlayotgan bosqich (aria-live shu o'zgarganda)
    estimate: null,        // {mode, buckets, est_messages, est_cost_usd, ceiling_usd}
    needsConfirm: false,
    plan: null,
    clusters: [],
    coverage: null,
    stoppedBy: null,
    error: null,
    funnel: {
      bucketsPlanned: 0, bucketsDone: 0, candidates: 0,
      filtered: 0, filterTotal: 0, kept: 0, keepRate: null,
      scanCost: 0, cached: false,
      selected: 0, chats: 0, weeks: 0, trimmed: 0,
      chunk: 0, chunks: 0, claims: 0, claimsIn: 0, contradictions: 0,
      searched: 0, served: 0, planned: 0,
    },
  };
}

/**
 * Bitta hodisani modelga qo'shadi — **DOM yo'q, i18n yo'q, mutatsiya yo'q
 * emas**: model o'zgartiriladi va qaytariladi (panel bitta nusxa yuritadi).
 *
 * @param {object} model  `emptyModel()` natijasi
 * @param {string} kind   SSE hodisa nomi
 * @param {object} d      hodisa tanasi
 * @param {number} at     hodisa vaqti (ms) — bosqich davomiyligi uchun
 */
export function reduceEvent(model, kind, d = {}, at = 0) {
  const f = model.funnel;

  const touch = (key, state) => {
    const prev = model.stages[key] || { startedAt: at };
    const startedAt = prev.startedAt ?? at;
    model.stages[key] = {
      state,
      startedAt,
      endedAt: state === "done" || state === "failed" ? at : null,
    };
    model.active = state === "active" ? key : (model.active === key ? null : model.active);
  };
  /** `key` dan oldingi, boshlangan va hali tugamagan bosqichlarni yopadi. */
  const closeBefore = (key) => {
    const limit = STAGE_ORDER.indexOf(key);
    for (const [k, st] of Object.entries(model.stages)) {
      if (STAGE_ORDER.indexOf(k) < limit && st.state === "active") touch(k, "done");
    }
  };
  const advance = (key, next) => {
    closeBefore(key);
    touch(key, "done");
    if (next) touch(next, "active");
  };

  switch (kind) {
    // Quvur har doim bir xil emas — server shu turn'da yuguradigan
    // bosqichlarni aytadi va reyk **faqat** shulardan quriladi.
    case "stages": {
      const list = (d.stages || []).filter((k) => STAGE_ORDER.includes(k));
      if (list.length) model.visible = list;
      if (d.direct) model.direct = true;
      // Ro'yxat rejadan **keyin** kengaysa (tarmoq tanlangach), navbatdagi
      // bosqich shu yerda yonadi: `plan` allaqachon yopilgan bo'ladi.
      if (!model.active && model.stages.plan && model.stages.plan.state === "done") {
        const next = plannedStages(model).find((k) => !model.stages[k]);
        if (next) touch(next, "active");
      }
      break;
    }

    case "plan":
      model.plan = d;
      f.planned = (d.subqueries || []).length;
      advance("plan", nextStage(model, "plan"));
      break;

    case "estimate":
      model.estimate = {
        mode: d.mode || model.mode,
        buckets: numOr(d.buckets),
        est_messages: numOr(d.est_messages),
        est_cost_usd: d.est_cost_usd == null ? null : Number(d.est_cost_usd),
        est_total_usd: d.est_total_usd == null ? null : Number(d.est_total_usd),
        ceiling_usd: d.ceiling_usd == null ? null : Number(d.ceiling_usd),
        weeks: numOr(d.weeks),
        range: d.range || null,
      };
      f.bucketsPlanned = model.estimate.buckets || f.bucketsPlanned;
      f.weeks = model.estimate.weeks || f.weeks;
      model.range = d.range || model.range || null;
      advance("estimate", "buckets");
      break;

    case "buckets":
      f.bucketsPlanned = numOr(d.planned, f.bucketsPlanned);
      f.chats = numOr(d.chats, f.chats);
      f.weeks = numOr(d.weeks, f.weeks);
      model.range = d.range || model.range || null;
      advance("buckets", "pull");
      break;

    case "pull":
      // `bucket_i` — tugagan bucket raqami; oqim ≤1/s bo'lgani uchun
      // sakrab kelishi mumkin, shuning uchun **maksimum** olinadi.
      f.bucketsDone = Math.max(f.bucketsDone, numOr(d.bucket_i));
      f.bucketsPlanned = numOr(d.buckets, f.bucketsPlanned);
      f.candidates = Math.max(f.candidates, numOr(d.candidates));
      touch("pull", "active");
      break;

    case "filter":
      f.filtered = Math.max(f.filtered, numOr(d.done));
      f.filterTotal = Math.max(f.filterTotal, numOr(d.total));
      f.kept = Math.max(f.kept, numOr(d.kept));
      f.keepRate = d.keep_rate == null
        ? (f.filtered ? f.kept / f.filtered : null)
        : Number(d.keep_rate);
      if (d.cost_usd != null) f.scanCost = Math.max(f.scanCost, Number(d.cost_usd));
      if (d.cached != null) f.cached = Boolean(d.cached);
      closeBefore("filter");
      touch("filter", "active");
      break;

    case "select":
      f.selected = numOr(d.evidence, f.selected);
      f.chats = numOr(d.chats, f.chats);
      f.weeks = numOr(d.weeks, f.weeks);
      f.trimmed = numOr(d.quota_trimmed, f.trimmed);
      advance("select", "map");
      break;

    case "map":
      f.chunk = Math.max(f.chunk, numOr(d.chunk));
      f.chunks = Math.max(f.chunks, numOr(d.chunks));
      f.claims = Math.max(f.claims, numOr(d.claims));
      closeBefore("map");
      touch("map", "active");
      break;

    case "reduce":
      f.claimsIn = numOr(d.claims_in, f.claimsIn);
      f.contradictions = numOr(d.contradictions, f.contradictions);
      advance("reduce", "write");
      break;

    case "search":
      if (d.level === "server") f.served += 1;
      else f.searched += 1;
      touch("search", "active");
      break;

    case "cluster":
      model.clusters = d.clusters || [];
      advance("cluster", null);
      break;

    case "rerank":
      f.reranked = numOr(d.kept);
      advance("rerank", null);
      break;

    case "token":
      closeBefore("write");
      touch("write", "active");
      break;

    case "coverage":
      model.coverage = d;
      model.stoppedBy = d.stopped_by || model.stoppedBy;
      f.bucketsDone = Math.max(f.bucketsDone, numOr(d.buckets_scanned));
      f.bucketsPlanned = Math.max(f.bucketsPlanned, numOr(d.buckets_planned));
      f.filtered = Math.max(f.filtered, numOr(d.filtered));
      f.kept = Math.max(f.kept, numOr(d.kept));
      if (d.scan_cost_usd != null) f.scanCost = Math.max(f.scanCost, Number(d.scan_cost_usd));
      break;

    case "done":
      model.needsConfirm = Boolean(d.needs_confirm);
      if (d.direct) { model.direct = true; model.intent = String(d.intent || ""); }
      for (const [k, st] of Object.entries(model.stages)) {
        if (st.state === "active") touch(k, model.needsConfirm ? "pending" : "done");
      }
      model.active = null;
      break;

    case "error":
      model.error = d;
      for (const [k, st] of Object.entries(model.stages)) {
        if (st.state === "active") touch(k, "failed");
      }
      model.active = null;
      break;

    default:
      break;
  }
  return model;
}

/**
 * Hodisalar ketma-ketligidan yakuniy model — **sof funksiya**
 * (`tests/test_js_lib.py` shu yerni o'lchaydi).
 *
 * @param {Array<{kind: string, data?: object, t?: number}>} events
 * @param {{mode?: string}} [opts]
 */
export function stageModel(events, opts = {}) {
  const model = emptyModel(opts.mode || "deep");
  for (const ev of events || []) {
    if (!ev || !ev.kind) continue;
    reduceEvent(model, ev.kind, ev.data || {}, numOr(ev.t));
  }
  return model;
}

/* ── bosqich timeline (chizish) ───────────────────────────────────── */

/** Vertikal (desktop) / gorizontal scroll-snap (mobil) bosqich reyki. */
function stageRail() {
  const items = new Map();
  const rail = el("ol.stage-rail", { "aria-label": t("web.ask.stages") });
  for (const key of STAGE_ORDER) {
    const detail = el("span.stage-detail");
    const time = el("span.stage-time");
    const track = el("span.stage-bar", { hidden: true }, el("i", { style: { inlineSize: "0%" } }));
    const body = el("div.stage-body", null, [
      el("span.stage-label", null, t(STAGE_LABEL[key])), detail, time, track,
    ]);
    // Boshida hammasi yashirin: nima ko'rinishini `stages` hodisasi (yoki
    // kelgan hodisaning o'zi) hal qiladi — `paint()` da.
    const li = el("li.stage-item", {
      dataset: { state: "pending", stage: key },
      hidden: true,
    }, [icon(STAGE_ICON[key]), body]);
    rail.appendChild(li);
    items.set(key, { li, detail, time, track });
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

  /* Korpus ko'lami (`scale_search`): nechta bucket ko'rildi, saralash
     voronkasi va nega to'xtadi. Bu uch satr "qancha ko'rilmadi" degan
     savolga javob beradi — foiz emas, aniq sonlar. */
  if (cov.buckets_planned) {
    const done = Number(cov.buckets_scanned || 0);
    rows.appendChild(bar(t("web.ask.cov_buckets"),
      `${num(done)}/${num(cov.buckets_planned)}`, done / Math.max(1, cov.buckets_planned)));
  }
  if (cov.filtered || cov.kept) {
    rows.appendChild(el("div.cov-funnel", null, t("web.ask.cov_funnel", {
      scanned: num(cov.filtered || 0),
      kept: num(cov.kept || 0),
      analyzed: num(cov.messages_used ?? cov.evidence ?? cov.kept ?? 0),
    })));
  }
  const pullStop = cov.pull_stopped_by || null;
  const filterStop = cov.filter_stopped_by || null;
  if (pullStop && filterStop && pullStop !== filterStop) {
    // Ikki bosqich turli sabab bilan to'xtagan — ikkalasi ham ko'rsatiladi,
    // aks holda foydalanuvchi "nega saralash tugadi-yu tortish tugamadi"
    // savolini javobsiz qoladi.
    rows.appendChild(el("div.cov-stopped", null, [
      icon("info"),
      el("span", null, [
        `${t("web.ask.cov_stopped_pull")}: ${t(`web.ask.stopped_${pullStop}`)}`,
        " · ",
        `${t("web.ask.cov_stopped_filter")}: ${t(`web.ask.stopped_${filterStop}`)}`,
      ].join("")),
    ]));
  } else if (pullStop || filterStop || cov.stopped_by) {
    const why = t(`web.ask.stopped_${pullStop || filterStop || cov.stopped_by}`);
    rows.appendChild(el("div.cov-stopped", null, [
      icon("info"),
      el("span", null, `${t("web.ask.cov_stopped")}: ${why}`),
    ]));
  }

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
  if (cov.scan_cost_usd != null) facts.push(`${t("web.ask.cov_scan_cost")}: ${usd(cov.scan_cost_usd)}`);
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

/* ── baho kartasi ─────────────────────────────────────────────────── */

/**
 * Baho kartasi shablondan (`#tpl-estimate`) klonlanadi — matnlar serverda
 * tarjima qilinadi, bu yerda faqat `textContent`. Shablon bo'lmasa (eski
 * sahifa keshi) — `el()` bilan zaxira karta.
 */
function estimateCard() {
  const tpl = $("#tpl-estimate");
  const first = tpl && tpl.content && tpl.content.firstElementChild;
  if (first) return first.cloneNode(true);
  return el("div.estimate-card", { role: "group", "aria-label": t("web.ask.est_title") }, [
    el("div.est-head", null, [icon("activity"), el("b", null, t("web.ask.est_title"))]),
    el("p.est-line", { dataset: { estLine: "" } }),
    el("div.est-confirm", { hidden: true, dataset: { estConfirm: "" } }, [
      el("p.est-warn", null, t("web.ask.est_confirm")),
      el("div.est-actions", null, [
        el("button.btn.primary.small", { type: "button", dataset: { estStart: "" } }, t("web.ask.est_start")),
        el("button.btn.ghost.small", { type: "button", dataset: { estFallback: "" } }, t("web.ask.est_fallback")),
      ]),
    ]),
  ]);
}

/**
 * Baho satri: `≈ 14 200 xabar · 186 bo'lak · 4 hafta (1 yan → 1 sen) ·
 * skan ≈ $0.07, jami ≈ $0.30 (shift $0.50)`.
 */
function estimateLine(est) {
  const r = est.range || {};
  return t("web.ask.est_line", {
    msgs: num(est.est_messages),
    buckets: num(est.buckets),
    weeks: num(est.weeks),
    from: r.from ? fmtDate(r.from) : "—",
    to: r.to ? fmtDate(r.to) : "—",
    scanCost: usd(est.est_cost_usd),
    totalCost: usd(est.est_total_usd),
    ceiling: usd(est.ceiling_usd),
  });
}

/**
 * Tor oyna ogohlantirishi: `exhaustive` faqat 1-2 hafta rejalashtirsa
 * foydalanuvchi buni tasdiqlashdan oldin ko'rishi kerak — aks holda
 * "nega javob faqat oxirgi haftani qamrab oldi" degan savol qoladi.
 */
function estimateHint(est) {
  if (!est || est.mode !== "exhaustive") return null;
  if (!(est.weeks > 0 && est.weeks <= 2)) return null;
  return el("p.est-hint", { role: "note" }, [
    icon("alert-triangle"),
    el("span", null, t("web.ask.est_weeks_hint", { weeks: num(est.weeks) })),
  ]);
}

/* ── bosqich tafsiloti ────────────────────────────────────────────── */

/**
 * Bitta bosqichning o'ng tomonidagi matn. Model'dan o'qiydi, shuning
 * uchun hodisa **kechikib** kelsa ham satr izchil qoladi (har raqam
 * `lib/fmt` orqali — mingliklar locale bo'yicha ajratiladi).
 */
function stageDetail(key, model, shownCandidates) {
  const f = model.funnel;
  switch (key) {
    case "estimate":
      return model.estimate
        ? t("web.ask.step_estimate", {
          msgs: num(model.estimate.est_messages), buckets: num(model.estimate.buckets),
        })
        : "";
    case "buckets":
      return t("web.ask.step_buckets", {
        n: num(f.bucketsPlanned), chats: num(f.chats), weeks: num(f.weeks),
      });
    case "pull":
      return t("web.ask.step_pull", {
        i: num(f.bucketsDone), n: num(f.bucketsPlanned), candidates: num(shownCandidates),
      });
    case "filter":
      return t("web.ask.step_filter", {
        done: num(f.filtered), total: num(f.filterTotal), kept: num(f.kept),
        rate: pct(f.keepRate == null ? 0 : f.keepRate),
      });
    case "select": {
      const line = t("web.ask.step_select", {
        n: num(f.selected), chats: num(f.chats), weeks: num(f.weeks),
      });
      return f.trimmed ? `${line} · ${t("web.ask.step_trimmed", { n: num(f.trimmed) })}` : line;
    }
    case "map":
      return t("web.ask.step_map", { i: num(f.chunk), n: num(f.chunks), claims: num(f.claims) });
    case "reduce":
      return t("web.ask.step_reduce", { n: num(f.claimsIn), c: num(f.contradictions) });
    case "search":
      return f.served
        ? t("web.deep.step_server", { n: f.served })
        : t("web.deep.step_search", { n: f.searched, m: f.planned || f.searched });
    case "cluster":
      return t("web.deep.step_cluster", { n: (model.clusters || []).length });
    case "rerank":
      return f.reranked != null ? t("web.deep.step_rank", { n: num(f.reranked) }) : "";
    default:
      return "";
  }
}

/** Progress bar nisbati (`null` — bar ko'rsatilmaydi). */
function stageRatio(key, f) {
  if (key === "pull" && f.bucketsPlanned) return f.bucketsDone / f.bucketsPlanned;
  if (key === "filter" && f.filterTotal) return f.filtered / f.filterTotal;
  if (key === "map" && f.chunks) return f.chunk / f.chunks;
  return null;
}

const reducedMotion = () => {
  try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch { return false; }
};

/* ── deep paneli ──────────────────────────────────────────────────── */

const TICK_MS = 500;        // o'tgan vaqtni yangilash oralig'i
const COUNT_MS = 400;       // nomzodlar sanog'ining "o'sish" davomiyligi

/**
 * Oqim uchun deep paneli. Qaytadi: DOM tugun, oqim hodisalarini qabul
 * qiluvchi `onEvent`, tasdiq kartasi uchun `confirm`, `stop` va yakuniy
 * bezak uchun `finish`.
 *
 * Butun holat `stageModel` modelida — DOM faqat uning **ko'rinishi**.
 * Shu tufayli kech kelgan yoki tashlab ketilgan hodisa (oqim ≤1/s
 * siqiladi) bosqichlarni chalkashtirmaydi.
 *
 * @param {HTMLElement} answerEl
 * @param {{mode?: string}} [opts]  "fast" | "deep" | "exhaustive"
 */
export function createDeepPanel(answerEl, opts = {}) {
  const mode = opts.mode || "deep";
  const model = emptyModel(mode);
  const { rail, items } = stageRail();
  const estimate = el("div.estimate-wrap", { hidden: true });
  // aria-live **faqat bosqich almashganda** yangilanadi: har progress
  // yangilanishida e'lon qilinsa ekran o'quvchi sanoqni o'qib tugatmaydi.
  const live = el("p.stage-live.sr-only", { "aria-live": "polite", "aria-atomic": "true" });
  const plan = el("div.plan-card", { hidden: true });
  const clusters = el("div.cluster-wrap", { hidden: true });
  const sources = el("div.sources-wrap", { hidden: true });
  const coverage = el("div.coverage-wrap", { hidden: true });
  const node = el("section.deep-panel", {
    "aria-label": t("web.ask.deep"), dataset: { mode },
  }, [estimate, rail, live, plan, clusters, sources, coverage]);

  let shownCandidates = 0;    // ekrandagi (animatsiyalangan) qiymat
  let countRaf = 0;
  let ticker = 0;
  let announced = null;

  /* ── chizish ──────────────────────────────────────────────────── */

  const paint = () => {
    const now = Date.now();
    // Reyk **shu turn** uchun rejalashtirilgan bosqichlardan quriladi; ularga
    // qo'shimcha, haqiqatan kelgan bosqich ham ko'rinadi (reja kutmagan yo'l).
    const planned = new Set(plannedStages(model));
    for (const [key, item] of items) {
      const st = model.stages[key];
      if (!st) {
        item.li.hidden = !planned.has(key);
        item.li.dataset.state = "pending";
        continue;
      }
      item.li.hidden = false;
      item.li.dataset.state = st.state;
      item.li.classList.toggle("is-collapsed", st.state === "done");
      const detail = stageDetail(key, model, shownCandidates);
      if (detail) item.detail.textContent = detail;
      const elapsed = (st.endedAt ?? now) - st.startedAt;
      item.time.textContent = elapsed > 0 ? fmtMs(elapsed) : "";
      const ratio = stageRatio(key, model.funnel);
      item.track.hidden = ratio == null;
      if (ratio != null) {
        const p = Math.max(0, Math.min(100, Math.round(ratio * 100)));
        item.track.firstChild.style.inlineSize = `${p}%`;
      }
    }
    if (model.active !== announced) {
      announced = model.active;
      const item = announced ? items.get(announced) : null;
      live.textContent = announced && item
        ? [t(STAGE_LABEL[announced]), item.detail.textContent].filter(Boolean).join(" · ")
        : "";
    }
    // Faol bosqich bor ekan — o'tgan vaqt tiriklayin sanaladi.
    const busy = Object.values(model.stages).some((s) => s.state === "active");
    if (busy && !ticker) ticker = setInterval(paint, TICK_MS);
    if (!busy && ticker) { clearInterval(ticker); ticker = 0; }
  };

  /** Nomzodlar sanog'i sakramasin: qisqa o'sish (harakat kamaytirilsa — darhol). */
  const countTo = (target) => {
    if (reducedMotion() || typeof requestAnimationFrame !== "function") {
      shownCandidates = target;
      return;
    }
    const from = shownCandidates;
    const t0 = Date.now();
    cancelAnimationFrame(countRaf);
    const step = () => {
      const k = Math.min(1, (Date.now() - t0) / COUNT_MS);
      shownCandidates = Math.round(from + (target - from) * k);
      paint();
      if (k < 1) countRaf = requestAnimationFrame(step);
    };
    countRaf = requestAnimationFrame(step);
  };

  /* ── baho / tasdiq ────────────────────────────────────────────── */

  const paintEstimate = (handlers) => {
    if (!model.estimate) return;
    const card = estimateCard();
    const line = card.querySelector("[data-est-line]");
    if (line) line.textContent = estimateLine(model.estimate);
    const hint = estimateHint(model.estimate);
    if (hint && line) line.after(hint);
    else if (hint) card.appendChild(hint);
    const box = card.querySelector("[data-est-confirm]");
    if (box && handlers) {
      box.hidden = false;
      const start = card.querySelector("[data-est-start]");
      const fallback = card.querySelector("[data-est-fallback]");
      start?.addEventListener("click", () => { estimate.dataset.answered = "1"; handlers.onStart?.(); });
      fallback?.addEventListener("click", () => { estimate.dataset.answered = "1"; handlers.onFallback?.(); });
    }
    replace(estimate, card);
    estimate.hidden = false;
  };

  /* ── oqim ─────────────────────────────────────────────────────── */

  const onEvent = (kind, d) => {
    reduceEvent(model, kind, d, Date.now());
    if (kind === "estimate") paintEstimate(null);
    else if (kind === "plan") { renderPlan(plan, d); plan.hidden = false; }
    else if (kind === "pull") countTo(model.funnel.candidates);
    else if (kind === "cluster") {
      replace(clusters, clusterChips(model.clusters, answerEl));
      clusters.hidden = model.clusters.length === 0;
    } else if (kind === "coverage") {
      node.dataset.coverage = "1";
      replace(coverage, coveragePanel(d, null));
      coverage.hidden = false;
    }
    paint();
  };

  /**
   * `exhaustive` tasdiqsiz yuborilgan: server `estimate` + `done
   * {needs_confirm:true}` qaytardi. Karta shu yerda tugmalar bilan qayta
   * chiziladi — javobning o'zi hali yozilmagan.
   */
  const confirm = (handlers) => {
    if (!model.estimate) {
      model.estimate = { buckets: 0, est_messages: 0, est_cost_usd: null, ceiling_usd: null };
    }
    paintEstimate(handlers);
  };

  /** Stop bosilganda: faol bosqich "uzilgan" bo'lib qoladi, taymer to'xtaydi. */
  const stop = () => {
    for (const [key, st] of Object.entries(model.stages)) {
      if (st.state === "active") {
        model.stages[key] = { ...st, state: "failed", endedAt: Date.now() };
      }
    }
    model.active = null;
    if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(countRaf);
    paint();
    if (ticker) { clearInterval(ticker); ticker = 0; }
  };

  /** `done` kelganda: bosqichlar yopiladi, manbalar va qamrov to'ldiriladi. */
  const finish = (d) => {
    for (const [key, st] of Object.entries(model.stages)) {
      if (st.state !== "failed" && st.state !== "done") {
        model.stages[key] = { ...st, state: "done", endedAt: st.endedAt ?? Date.now() };
      }
    }
    model.active = null;
    paint();
    if (ticker) { clearInterval(ticker); ticker = 0; }
    const strip = sourceStrip(d.citations, answerEl);
    if (strip) { replace(sources, strip); sources.hidden = false; }
    linkifyCitations(answerEl, d.citations);
    if (node.dataset.coverage === "1" && d.cost_usd != null) {
      const facts = coverage.querySelector(".cov-facts");
      if (facts) facts.textContent += ` · ${t("web.deep.cov_cost")}: ${usd(d.cost_usd)}`;
    }
    return model.clusters;
  };

  return {
    node, onEvent, finish, confirm, stop,
    model: () => model,
    clusters: () => model.clusters,
  };
}

/**
 * Reja kartasi: qayta yozilgan savol, davom etish turi (badge), subquery
 * chiplari, oyna va chat soni.
 *
 * `standalone_question` — modelning savolni **qanday tushungani**; uni
 * ko'rsatish "nega bunday javob berdi" savolini oldindan yopadi.
 */
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

  const head = el("div.plan-head", null, [icon("brain"), el("b", null, t("web.ask.plan"))]);
  const intent = String(plan.intent_follow_up || "");
  if (INTENTS.has(intent)) {
    head.appendChild(el("span.chip.tiny.intent-badge", {
      dataset: { intent },
    }, t(`web.ask.intent_${intent}`)));
  }
  box.appendChild(head);
  if (plan.standalone_question) {
    box.appendChild(el("p.plan-standalone", null,
      t("web.ask.standalone", { q: String(plan.standalone_question) })));
  }
  if (chips.length) box.appendChild(el("div.plan-chips", null, chips));
  box.appendChild(el("div.plan-facts", null, facts.join(" · ")));
}

/** Tanilgan davom turlari — begona qiymat badge bo'lib chiqmaydi. */
const INTENTS = new Set([
  "deeper", "repeat", "correction", "context_switch", "reformat", "explain",
  "about_prior", "meta", "chitchat",
]);
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
