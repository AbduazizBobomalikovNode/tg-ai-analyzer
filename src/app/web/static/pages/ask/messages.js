/* pages/ask/messages.js — tred: xabar bloklari va javob yuzasi.
 *
 * Xavfsizlik chegarasi shu faylda: **faqat** `renderAnswer()` `innerHTML`
 * ga yozadi va faqat `md()` chiqishini (escape-first renderer). Boshqa har
 * bir satr — foydalanuvchi savoli, chat sarlavhasi, model nomi, xato matni
 * — `el()`/`textContent` orqali.
 *
 * `enhance()` — md'dan keyingi progressiv bezak: kod blokiga nusxa tugmasi,
 * jadvalga scroll konteyneri, rasmga lightbox. Hammasi **mavjud** DOM
 * ustida ishlaydi, HTML qayta parse qilinmaydi.
 */

import { $, el, clear, append } from "../../lib/dom.js";
import { md, hydrate } from "../../lib/md.js";
import { icon, iconButton } from "../../lib/icons.js";
import { t } from "../../lib/i18n.js";
import { compact, ms as fmtMs, usd } from "../../lib/fmt.js";
import { peerClass, initials } from "../../lib/palette.js";
import { toast } from "../../lib/toast.js";
import { S, guessLang } from "./state.js";
import { openLightbox } from "./panels.js";

/**
 * Modullararo ulagichlar. `answer.js`, `actions.js` va `deep.js` shu yerga
 * o'zini ro'yxatdan o'tkazadi — aks holda `messages.js` ularni import
 * qilishi kerak bo'lardi va sikl bog'liqlik chiqardi.
 */
export const hooks = {
  footer: null,      // (msg, node) => Node|null
  actionCards: null, // (actions) => Node|null
  deepBlock: null,   // (msg, node) => Node|null  — saqlangan deep javobi
  citations: null,   // (answerEl, citations) => void
};

export const threadEl = () => $("#thread");
export const scrollEl = () => $("#thread-scroll");

/** Foydalanuvchi pastdami (±80px). Shunda yangi matn avtomatik ergashadi. */
export function atBottom() {
  const sc = scrollEl();
  if (!sc) return true;
  return sc.scrollHeight - sc.scrollTop - sc.clientHeight < 80;
}

/** Pastga tushiradi. `force` bo'lmasa — faqat foydalanuvchi allaqachon pastda bo'lsa. */
export function scrollBottom(force = false) {
  const sc = scrollEl();
  if (!sc) return;
  if (force || atBottom()) sc.scrollTop = sc.scrollHeight;
}

/* ── avatar ───────────────────────────────────────────────────────── */

/** Chat avatari: rasm bo'lsa rasm, aks holda peer rangidagi inisiallar. */
export function chatAvatar(chat, size = "sm") {
  const node = el(`span.avatar.${size}`, {
    class: chat ? peerClass(chat.peer_id ?? chat.id) : "",
    "aria-hidden": "true",
  });
  if (chat && chat.avatar_url) {
    node.appendChild(el("img", { src: chat.avatar_url, alt: "", loading: "lazy", decoding: "async" }));
  } else {
    node.textContent = initials((chat && chat.title) || "?");
  }
  return node;
}

/* ── kontekst qatori ──────────────────────────────────────────────── */

const TOOL_LIMIT = 6;

/** Assistent javobi qanday kontekstda olinganini bir qatorda ko'rsatadi. */
export function contextLine(ctx) {
  if (!ctx) return null;
  if (ctx.mode === "agent" && Array.isArray(ctx.tools) && ctx.tools.length) {
    const counts = new Map();
    for (const name of ctx.tools) counts.set(name, (counts.get(name) || 0) + 1);
    const chips = [...counts].slice(0, TOOL_LIMIT).map(([name, n]) =>
      el("span.chip.tiny", null, n > 1 ? `${name} ×${n}` : name));
    return el("div.ctx-line", null, [
      icon("bolt"),
      ...chips,
      el("span.muted", null, t("web.ask.iterations", { n: ctx.iterations || 1 })),
    ]);
  }
  if (!ctx.title) return null;
  const parts = [t("web.chat.context_used", { title: ctx.title, n: ctx.messages ?? 0 })];
  if (ctx.source) parts.push(t(`web.chat.source_${ctx.source}`));
  if (ctx.strategy && ctx.strategy !== "agent") parts.push(t(`web.chat.strategy_${ctx.strategy}`));
  if (ctx.est_tokens) parts.push(`~${compact(ctx.est_tokens)} tok`);
  if (ctx.truncated) parts.push(t("web.ask.truncated"));
  return el("div.ctx-line", null, [icon("layers"), el("span", null, parts.join(" · "))]);
}

/* ── xabar tugunlari ──────────────────────────────────────────────── */

/** Foydalanuvchi savoli — o'ngdagi pufak. Matn **oddiy matn** (md emas). */
export function userNode(text, ctx) {
  return el("div.msg-u", null, [
    ctx ? contextLine(ctx) : null,
    el("div.bubble", null, String(text || "")),
  ]);
}

/**
 * Assistent bloki uchun bo'sh qobiq. Oqim **yakuniy tartibga** yoziladi:
 * `stage` yuqorida, `extra` (deep panel) keyin, `answer` matni, keyin
 * amal kartalari va footer — shuning uchun tugaganda hech narsa sakramaydi.
 */
export function assistantShell(mid = 0) {
  const stage = el("div.stage-strip", { hidden: true, "aria-live": "polite" });
  const extra = el("div.a-extra");
  const answer = el("div.answer.md");
  const cards = el("div.a-actions");
  const foot = el("div.a-foot");
  const node = el("article.msg-a", { dataset: mid ? { mid } : {} },
    [stage, extra, answer, cards, foot]);
  return { node, stage, extra, answer, cards, foot };
}

/* ── javob matni ──────────────────────────────────────────────────── */

/**
 * Markdown → HTML. **Yagona** `innerHTML` nuqtasi (`md()` escape-first).
 * `enhance` oqim paytida o'chiriladi: har bo'lakda butun blokni qayta
 * bezash isrof, yakunda bir marta yetadi.
 */
export function renderAnswer(answerEl, source, { enhance: doEnhance = true } = {}) {
  // Oqim paytida blok har token'da qayta chiziladi — chizilgan mermaid/chart
  // tugunlari fence hash'i bo'yicha saqlanadi va joyiga qaytariladi, aks holda
  // diagramma har bo'lakda yo'qolib, qayta chizishni talab qilardi.
  const kept = new Map();
  for (const fig of answerEl.querySelectorAll("figure[data-hash]")) {
    if (fig.dataset.done || fig.dataset.state === "done") kept.set(fig.dataset.hash, fig);
  }
  answerEl.innerHTML = md(source || "");
  answerSource.set(answerEl, String(source || ""));
  if (kept.size) {
    for (const fig of answerEl.querySelectorAll("figure[data-hash]")) {
      const old = kept.get(fig.dataset.hash);
      if (old) fig.replaceWith(old);
    }
  }
  const lang = guessLang(source);
  if (lang) answerEl.setAttribute("lang", lang);
  if (doEnhance) enhance(answerEl);
}

/** Chizilgan javobning markdown manbasi — blok bo'yicha nusxa olish uchun. */
export const answerSource = new WeakMap();

/** `data-l="a,b"` bo'yicha manba bo'lagi (blok markdown'i). */
export function blockSource(answerEl, node) {
  const range = String(node.dataset.l || "").split(",");
  const src = answerSource.get(answerEl);
  if (src == null || range.length !== 2) return node.textContent || "";
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  return lines.slice(+range[0], +range[1]).join("\n").trim() || (node.textContent || "");
}

const COPY_MIN_LINES = 2;

/**
 * Chizilgan javob ustidagi progressiv bezak.
 *
 * Har biri **idempotent** (`data-enh`): oqim tugagach yoki qayta
 * chizilganda ikkinchi nusxa tugma paydo bo'lmaydi.
 */
export function enhance(root) {
  // kod bloklari — nusxa tugmasi + yumshoq o'rash almashtirgichi.
  // Mermaid/chart manbalari bezatilmaydi: ular tugma bilan chiziladi.
  for (const pre of root.querySelectorAll("pre")) {
    if (pre.dataset.enh) continue;
    if (pre.classList.contains("md-mermaid-src") || pre.classList.contains("md-chart-src")) continue;
    pre.dataset.enh = "1";
    const code = pre.querySelector("code");
    if (!code) continue;
    const fig = pre.closest("figure.md-code");
    const tools = fig?.querySelector(".md-code-tools");
    const long = code.textContent.split("\n").length >= COPY_MIN_LINES;
    if (tools) {
      tools.appendChild(iconButton("wrap-text", t("web.md.wrap"), {
        cls: "md-code-btn",
        attrs: { "aria-pressed": "false" },
        onclick: (e) => {
          const on = fig.classList.toggle("is-wrapped");
          e.currentTarget.setAttribute("aria-pressed", String(on));
        },
      }));
      tools.appendChild(iconButton("copy", t("web.ask.copy_code"), {
        cls: "md-code-btn",
        onclick: () => copyText(code.textContent, t("web.ask.copied")),
      }));
    } else if (long) {
      pre.appendChild(iconButton("copy", t("web.ask.copy_code"), {
        cls: "code-copy",
        onclick: () => copyText(code.textContent, t("web.ask.copied")),
      }));
    }
  }
  // jadvallar — gorizontal scroll konteyneri (yopishqoq birinchi ustun CSS'da)
  for (const table of root.querySelectorAll("table")) {
    if (!table.parentElement?.classList.contains("table-wrap")) {
      const wrap = el("div.table-wrap.md-table-wrap", { tabindex: "0", role: "region", "aria-label": t("web.ask.table") });
      table.replaceWith(wrap);
      wrap.appendChild(table);
    }
    tableTools(table);
  }
  // rasmlar — faqat `/api/images/<id>` (md.js qoidasi), bosilsa lightbox
  for (const img of root.querySelectorAll("img.gen-img")) {
    if (img.dataset.enh) continue;
    img.dataset.enh = "1";
    img.style.cursor = "zoom-in";
    img.addEventListener("click", () => openLightbox(img.getAttribute("src"), img.alt));
  }
  blockCopy(root);
  hydrate(root);
}

/**
 * Nusxa tugmasi qo'yiladigan bloklar. `figure.md-code` yo'q — unda o'z
 * tugmasi bor; ro'yxat/paragraf yo'q — ular uchun bu shovqin bo'lardi
 * (va `<ul>` ichiga `<button>` qo'yish yaroqsiz HTML).
 */
const COPYABLE = ".md-table-wrap, figure.md-chart, figure.md-mermaid, blockquote, aside.md-callout, details.md-details";

/**
 * Har bir yirik blok ustida "markdown bo'lib nusxa ol" tugmasi.
 * Manba `data-l="a,b"` orqali **haqiqiy markdown** — HTML'dan tiklash emas.
 */
function blockCopy(root) {
  for (const node of root.querySelectorAll(":scope > [data-l]")) {
    if (node.dataset.bcopy || !node.matches(COPYABLE)) continue;
    node.dataset.bcopy = "1";
    const btn = iconButton("copy", t("web.md.copy_block"), {
      cls: "md-block-copy",
      onclick: () => copyText(blockSource(root, node), t("web.ask.copied")),
    });
    // Jadval o'z scroll konteynerida — tugma tashqaridagi asboblar qatoriga,
    // aks holda u gorizontal scroll bilan birga surilib ketardi.
    const bar = node.previousElementSibling;
    if (bar && bar.classList.contains("md-table-tools")) { bar.appendChild(btn); continue; }
    node.classList.add("md-block");
    node.appendChild(btn);
  }
}

/* ── jadval amallari ──────────────────────────────────────────────── */

/** `<table>` → CSV matn. Katakchalar RFC 4180 bo'yicha qo'shtirnoqlanadi. */
export function tableCsv(table) {
  const cell = (td) => {
    const v = (td.textContent || "").replace(/\s+/g, " ").trim();
    return /[",;\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
  };
  return [...table.rows].map((r) => [...r.cells].map(cell).join(",")).join("\r\n");
}

/**
 * Jadval ustidagi tugmalar: CSV yuklab olish va markdown nusxasi.
 *
 * Bu **ilova**, artefakt sandbox'i emas — shuning uchun `Blob` +
 * `a[download]` ishlaydi; havola darhol `revokeObjectURL` bilan tozalanadi.
 */
function tableTools(table) {
  const wrap = table.parentElement;
  if (!wrap || wrap.dataset.tools) return;
  wrap.dataset.tools = "1";
  const bar = el("div.md-table-tools");
  bar.appendChild(iconButton("download", t("web.md.csv"), {
    cls: "md-table-btn",
    onclick: () => {
      const blob = new Blob(["﻿" + tableCsv(table)], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = el("a", { href: url, download: "table.csv" });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
  }));
  wrap.insertAdjacentElement("beforebegin", bar);
}

/** Buferga nusxa; `navigator.clipboard` yo'q bo'lsa jimgina o'tkazib yuboradi. */
export async function copyText(text, okMessage) {
  try {
    await navigator.clipboard.writeText(String(text ?? ""));
    toast(okMessage || t("web.ask.copied"), { kind: "ok" });
    return true;
  } catch {
    toast(t("web.ask.copy_failed"), { kind: "err" });
    return false;
  }
}

/* ── meta qatori ──────────────────────────────────────────────────── */

/** Model/latency/narx — yig'ilgan `<details>` ichida (javobni shovqinlamaydi). */
export function metaLine(m) {
  if (!m || !m.model) return null;
  const bits = [
    m.provider || "",
    m.model,
    `${compact(m.tokens_in || 0)}→${compact(m.tokens_out || 0)}`,
  ].filter(Boolean);
  if (m.latency_ms) bits.push(fmtMs(m.latency_ms));
  if (m.cost_usd != null) bits.push(usd(m.cost_usd));
  const auto = m.auto && m.auto.relevance != null
    ? el("div.meta-auto", { title: m.auto.note || "" },
      t("web.ask.auto_score", { n: m.auto.relevance }) + (m.auto.grounded === false ? ` · ${t("web.ask.ungrounded")}` : ""))
    : null;
  return el("details.meta-box", null, [
    el("summary", { "aria-label": t("web.ask.details") }, [icon("info"), el("span", null, t("web.ask.details"))]),
    el("div.meta-rows", null, [el("div", null, bits.join(" · ")), auto]),
  ]);
}

/* ── to'liq xabar ─────────────────────────────────────────────────── */

/** API xabaridan tayyor blok (tarixni chizishda). */
export function messageNode(m) {
  if (m.role !== "assistant") return userNode(m.content, m.context);
  const sh = assistantShell(m.id);
  const ctx = contextLine(m.context);
  if (ctx) sh.extra.appendChild(ctx);
  const deep = hooks.deepBlock?.(m, sh);
  if (deep) sh.extra.appendChild(deep);
  renderAnswer(sh.answer, m.content);
  if (m.context && Array.isArray(m.context.citations)) {
    hooks.citations?.(sh.answer, m.context.citations);
  }
  const cards = hooks.actionCards?.(m.actions || []);
  if (cards) sh.cards.appendChild(cards);
  const foot = hooks.footer?.(m, sh);
  if (foot) sh.foot.appendChild(foot);
  return sh.node;
}

/** Bo'sh holat ko'rsatiladimi. */
export function setEmptyVisible(show) {
  const empty = $("#ask-empty");
  if (empty) empty.hidden = !show;
}

/**
 * Tredni to'liq qayta chizadi.
 *
 * Uzilgan deep yugurishning hisob qatori (`context.aborted`) tashlanadi —
 * u faqat kunlik byudjet uni sanashi uchun yozilgan, javob emas.
 */
export function renderMessages(items) {
  const thread = threadEl();
  const top = $("#thread-top");
  const empty = $("#ask-empty");
  clear(thread);
  if (top) thread.appendChild(top);
  if (empty) thread.appendChild(empty);
  const rows = (items || []).filter((m) => !(m.context && m.context.aborted));
  for (const m of rows) thread.appendChild(messageNode(m));
  setEmptyVisible(rows.length === 0);
  if (top) top.hidden = !S.nextBefore;
  scrollBottom(true);
}

/** "Eskiroq" sahifani tepaga qo'shadi va ko'rish nuqtasini saqlaydi. */
export function prependMessages(items) {
  const thread = threadEl();
  const sc = scrollEl();
  const top = $("#thread-top");
  const before = sc ? sc.scrollHeight - sc.scrollTop : 0;
  const frag = document.createDocumentFragment();
  for (const m of (items || []).filter((x) => !(x.context && x.context.aborted))) {
    frag.appendChild(messageNode(m));
  }
  thread.insertBefore(frag, top ? top.nextSibling : thread.firstChild);
  if (sc) sc.scrollTop = sc.scrollHeight - before;
  if (top) top.hidden = !S.nextBefore;
}

/** Tredga blok qo'shadi (savol yoki javob). */
export function appendNode(node) {
  setEmptyVisible(false);
  append(threadEl(), node);
  scrollBottom(true);
  return node;
}
