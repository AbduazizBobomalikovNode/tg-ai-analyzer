/* pages/ask/answer.js — bitta savol-javob aylanishi va javob futeri.
 *
 * Ikki oqim, bitta yuza:
 *   * oddiy — `POST /api/conversations/{id}/messages/stream`
 *     (`status | token | done | error`);
 *   * chuqur — `POST /api/deep/stream`
 *     (`plan | search | cluster | rerank | token | coverage | done | error`).
 *
 * Ikkalasi ham **yakuniy tartibga** yozadi (bosqich chizig'i → deep paneli →
 * javob → amal kartalari → futer), shuning uchun `done` kelganda hech narsa
 * sakramaydi: joylar allaqachon band.
 *
 * `aria-live` faqat bosqich chizig'ida: har token'da e'lon qilinsa ekran
 * o'quvchi javobni yuz marta qayta o'qiydi.
 */

import { $, el, clear, replace } from "../../lib/dom.js";
import { post, ApiError } from "../../lib/api.js";
import { stream } from "../../lib/sse.js";
import { icon, iconButton } from "../../lib/icons.js";
import { t } from "../../lib/i18n.js";
import { toastError } from "../../lib/toast.js";
import { isMobile } from "../../lib/store.js";
import { haptic } from "../../lib/tg.js";
import { S, emit, contextChatObjects, isDeepScale } from "./state.js";
import {
  hooks, assistantShell, userNode, appendNode, renderAnswer, enhance, metaLine,
  scrollBottom, copyText, setEmptyVisible, threadEl, contextLine,
} from "./messages.js";
import { createDeepPanel, storedDeepBlock, linkifyCitations } from "./deep.js";
import { refreshPending } from "./actions.js";
import { openSheet, banner, clearBanners } from "./panels.js";

/** Javob matnining markdown manbasi (nusxa/ulashish uchun). */
const sourceOf = new WeakMap();

/** Shu javobni keltirib chiqargan savol (deep'ni qayta yugurtirish uchun). */
const questionOf = new WeakMap();

/**
 * Savol matni: oqimdan yozib qo'yilgani, bo'lmasa tredda undan oldin
 * turgan foydalanuvchi pufagi (tarixdan chizilgan blok uchun).
 */
function questionFor(shell) {
  const known = questionOf.get(shell.node);
  if (known) return known;
  const prev = shell.node.previousElementSibling;
  return prev?.classList.contains("msg-u")
    ? (prev.querySelector(".bubble")?.textContent || "")
    : "";
}

/* ── so'rov tanasi ────────────────────────────────────────────────── */

/**
 * Composer holatidan so'rov tanasi.
 *
 * Bitta chat tanlangan bo'lsa u **to'g'ridan-to'g'ri kontekst** bo'ladi
 * (`context`), bir nechta bo'lsa — kontekst berilmaydi va `agent` rejimi
 * ishlaydi: agent chatlarni o'zi qidiradi. Server `ContextIn` bitta
 * `peer_id` qabul qiladi, shuning uchun "birinchisini olamiz" degan
 * jimgina yolg'on o'rniga rejim almashadi.
 */
export function requestBody(text) {
  const chats = contextChatObjects();
  const body = {
    text,
    deep: false,
    mode: chats.length > 1 && S.mode === "direct" ? "agent" : S.mode,
    account_id: S.accountId || null,
  };
  if (chats.length === 1 && S.accountId) {
    body.context = {
      account_id: S.accountId,
      peer_id: chats[0].peer_id,
      limit: S.ctxLimit,
      strategy: S.strategy,
    };
  }
  return body;
}

/* ── oqim holati ──────────────────────────────────────────────────── */

function setBusy(on) {
  S.busy = on;
  const send = $("#btn-send"), stop = $("#btn-stop");
  if (send) send.hidden = on;
  if (stop) stop.hidden = !on;
  emit("busy", on);
}

/** Oqimni uzadi (Stop tugmasi). Server allaqachon yozgan bo'lsa saqlanadi. */
export function stopStream() {
  S.abort?.abort();
}

/** Oddiy oqimning bosqich chizig'i: o'ylayapti → tool chiplari → yozmoqda. */
function statusStrip(stage) {
  const strip = el("div.status-strip");
  const dots = el("span.typing", null, [el("i"), el("i"), el("i")]);
  const label = el("span.status-label", null, t("web.chat.thinking"));
  const tools = el("span.status-tools");
  strip.append(dots, label, tools);
  stage.hidden = false;
  replace(stage, strip);
  const seen = new Set();
  return {
    update(d) {
      if (d.stage === "tool" && d.tool) {
        label.textContent = t("web.ask.stage_tools");
        if (!seen.has(d.tool)) {
          seen.add(d.tool);
          tools.appendChild(el("span.chip.tiny", null, d.tool));
        }
      } else if (d.stage === "writing") {
        label.textContent = t("web.ask.stage_writing");
      } else {
        label.textContent = t("web.chat.thinking");
      }
    },
    done() { stage.hidden = true; clear(stage); },
  };
}

/* ── xatolar ──────────────────────────────────────────────────────── */

/** `error` hodisasidan `ApiError` — kodi i18n kaliti bo'lib qoladi. */
function errorOfEvent(d) {
  const code = String(d?.code || "chat.err.internal");
  const full = code.startsWith("chat.err.") || code.includes(".") ? code : `chat.err.${code}`;
  return new ApiError(full, 0, d?.detail || null);
}

/** Banner talab qiladigan xatolar — ular kutish yoki sozlama o'zgarishini so'raydi. */
const BANNER_CODES = new Set([
  "chat.err.budget", "chat.err.rate_limited", "chat.err.disabled",
  "pool.err.session_revoked",
]);

function showError(err, shell) {
  // Bosqich chizig'i "o'ylayapti" da qotib qolmasin — oqim tugadi.
  if (shell) { shell.stage.hidden = true; clear(shell.stage); }
  const code = err?.code || "error.generic";
  if (code === "chat.err.budget") {
    // Byudjet tugadi — foydalanuvchi qayerda oshirishini bilsin.
    banner("err", err.message, { link: { href: "/settings#sec-budget", label: t("web.budget.goto") } });
  } else if (BANNER_CODES.has(code)) banner("err", err.message);
  else if (shell) replace(shell.foot, el("div.answer-error", null, [icon("alert-triangle"), el("span", null, err.message)]));
  else toastError(err);
}

/* ── oddiy oqim ───────────────────────────────────────────────────── */

async function runNormal(path, body, shell) {
  const status = statusStrip(shell.stage);
  let text = "";
  let final = null;
  let failed = null;

  await stream(path, body, (ev) => {
    const d = ev.data || {};
    if (ev.kind === "status") status.update(d);
    else if (ev.kind === "token") {
      text += d.text || "";
      renderAnswer(shell.answer, text, { enhance: false });
      status.update({ stage: "writing" });
      scrollBottom();
    } else if (ev.kind === "done") final = d;
    else if (ev.kind === "error") failed = errorOfEvent(d);
  }, { signal: S.abort.signal });

  if (failed) throw failed;
  if (!final) throw new ApiError("chat.err.internal", 0, "no done event");
  status.done();
  // `done.text` — saqlangan yakuniy matn (oqim bo'laklaridan farq qilishi mumkin).
  const body_ = final.text || text;
  renderAnswer(shell.answer, body_);
  sourceOf.set(shell.node, body_);
  return final;
}

/* ── chuqur oqim ──────────────────────────────────────────────────── */

/** Hozirgi deep paneli — `Stop` bosilganda bosqichni uzilgan deb belgilash uchun. */
let livePanel = null;

/**
 * Korpus oqimi (`deep` va `exhaustive`).
 *
 * `exhaustive` **tasdiqsiz** yuborilsa server faqat `estimate` va
 * `done {needs_confirm:true}` qaytaradi — javob yozilmaydi. Shunda
 * `null` qaytadi va karta tugmalari yangi aylanishni boshlaydi
 * (`confirm:true` bilan yoki `deep` rejimida).
 *
 * @param {string} question
 * @param {object} shell
 * @param {{mode: string, confirm?: boolean}} opts
 */
async function runDeep(question, shell, opts) {
  const { mode, confirm = false } = opts;
  const panel = createDeepPanel(shell.answer, { mode });
  livePanel = panel;
  shell.extra.appendChild(panel.node);
  let text = "";
  let final = null;
  let failed = null;

  await stream("/api/deep/stream", {
    conversation_id: S.convId, question, deep: true, mode, confirm,
  }, (ev) => {
    const d = ev.data || {};
    if (ev.kind === "token") {
      text += d.text || "";
      renderAnswer(shell.answer, text, { enhance: false });
      scrollBottom();
    } else if (ev.kind === "done") final = d;
    else if (ev.kind === "error") failed = errorOfEvent(d);
    panel.onEvent(ev.kind, d);
  }, { signal: S.abort.signal });

  if (failed) throw failed;
  if (!final) throw new ApiError("chat.err.internal", 0, "no done event");
  if (final.needs_confirm) {
    panel.confirm({
      onStart: () => retryDeep(shell, question, { mode, confirm: true }),
      onFallback: () => retryDeep(shell, question, { mode: "deep", confirm: false }),
    });
    scrollBottom();
    return null;
  }
  renderAnswer(shell.answer, text);
  sourceOf.set(shell.node, text);
  const clusters = panel.finish(final);
  final.text = text;
  final.followups = (clusters || []).map((c) => c.label).filter(Boolean).slice(0, 4);
  return final;
}

/**
 * Xuddi shu blok ichida deep oqimini qaytadan yugurtiradi (tasdiq
 * kartasidagi "Boshlash" / "Chuqur rejimda" tugmalari).
 */
async function retryDeep(shell, question, opts) {
  if (S.busy || !question) return;
  clearBanners();
  clear(shell.answer); clear(shell.foot); clear(shell.cards); clear(shell.extra);
  S.abort = new AbortController();
  setBusy(true);
  try {
    const final = await runDeep(question, shell, opts);
    if (final) finishBlock(shell, final, { deep: true });
  } catch (err) {
    if (err?.name === "AbortError") stopped(shell);
    else showError(err, shell);
  } finally {
    setBusy(false);
    S.abort = null;
    livePanel = null;
    scrollBottom();
  }
}

/** Oqim uzilganda: faol bosqich belgilanadi va futerga qisqa izoh. */
function stopped(shell) {
  livePanel?.stop();
  shell.stage.hidden = true;
  replace(shell.foot, el("div.muted", null, t("web.ask.stopped")));
}

/* ── umumiy aylanish ──────────────────────────────────────────────── */

/** Suhbat yo'q bo'lsa yaratadi (akkaunt bilan — deep shuni talab qiladi). */
async function ensureConversation() {
  if (S.convId) return S.convId;
  const conv = await post("/api/conversations", { account_id: S.accountId });
  S.convId = conv.id;
  S.convTitle = conv.title || "";
  emit("conversation-created", conv);
  return S.convId;
}

/**
 * Savol yuboradi va javobni oqim bilan chizadi.
 * @param {string} text
 */
export async function ask(text) {
  if (!text || S.busy) return;
  clearBanners();
  setEmptyVisible(false);
  const chats = contextChatObjects();
  appendNode(userNode(text, chats.length === 1
    ? { title: chats[0].title, messages: S.ctxLimit }
    : null));
  const shell = assistantShell();
  appendNode(shell.node);
  questionOf.set(shell.node, text);

  S.abort = new AbortController();
  setBusy(true);
  // `fast` — oddiy oqim (yaqin kontekst), `deep`/`exhaustive` — korpus quvuri.
  const deep = isDeepScale();
  try {
    await ensureConversation();
    const final = deep
      ? await runDeep(text, shell, { mode: S.scaleMode, confirm: false })
      : await runNormal(`/api/conversations/${S.convId}/messages/stream`, requestBody(text), shell);
    // `null` — `exhaustive` tasdiq kutmoqda, javob hali yo'q.
    if (final) {
      finishBlock(shell, final, { deep });
      haptic.notify("success");
    }
  } catch (err) {
    if (err?.name === "AbortError") stopped(shell);
    else showError(err, shell);
  } finally {
    setBusy(false);
    S.abort = null;
    livePanel = null;
    emit("conversations-dirty");
    scrollBottom();
  }
}

/**
 * `done.truncated_answer` — model chiqish chegarasiga yetib to'xtagan
 * (javob semantik jihatdan tugallanmagan, xato emas). Javob ostiga qisqa
 * ogohlantirish va davom ettirish chipi qo'yiladi — chip savol qatoriga
 * yozib qo'ymaydi, **darhol** yuboradi, chunki bu javobning davomi, yangi
 * savol emas.
 */
function truncatedNotice(shell) {
  return el("div.answer-truncated", null, [
    icon("alert-triangle"),
    el("span", null, t("web.ask.truncated_answer")),
    el("button.chip.suggest", {
      type: "button",
      onclick: () => ask(t("web.ask.continue_answer")),
    }, t("web.ask.continue_answer")),
  ]);
}

/** `done` dan keyingi umumiy bezak: amal kartalari, futer, sitatalar. */
function finishBlock(shell, final, { deep = false } = {}) {
  if (final.assistant_message_id) shell.node.dataset.mid = final.assistant_message_id;
  enhance(shell.answer);
  const ctx = contextLine(final.context);
  if (ctx) shell.extra.insertBefore(ctx, shell.extra.firstChild);
  if (!deep && Array.isArray(final.citations)) linkifyCitations(shell.answer, final.citations);
  if (final.truncated_answer) shell.node.insertBefore(truncatedNotice(shell), shell.cards);
  const cards = hooks.actionCards?.(final.actions || []);
  if (cards) { clear(shell.cards); shell.cards.appendChild(cards); }
  if (final.actions && final.actions.length) refreshPending();
  replace(shell.foot, buildFooter({
    id: final.assistant_message_id,
    content: final.text || "",
    model: final.model, provider: final.provider,
    tokens_in: final.tokens_in, tokens_out: final.tokens_out,
    latency_ms: final.latency_ms, cost_usd: final.cost_usd,
    followups: final.followups,
  }, shell));
  scrollBottom();
}

/* ── qayta yugurtirish ────────────────────────────────────────────── */

/** Oxirgi savolni qayta yugurtiradi; javob **shu blok** ichida almashadi. */
async function regenerate(mid, shell) {
  if (S.busy || !mid || !S.convId) return;
  const question = questionFor(shell);
  // Deep quvuri savolni tanada kutadi (`/api/deep/stream`), oddiy yo'l esa
  // uni **saqlangan qatordan** oladi (`take_for_regenerate`) — shuning uchun
  // deep uchun savol matni bo'lmasa qayta yugurtirmaymiz.
  if (isDeepScale() && !question) { toastError(new ApiError("chat.err.not_found", 0, null)); return; }
  clearBanners();
  clear(shell.answer); clear(shell.foot); clear(shell.cards); clear(shell.extra);
  S.abort = new AbortController();
  setBusy(true);
  const deep = isDeepScale();
  try {
    const body = { ...requestBody(""), stream: true };
    delete body.text;
    const final = deep
      ? await runDeep(question, shell, { mode: S.scaleMode, confirm: false })
      : await runNormal(`/api/conversations/${S.convId}/messages/${mid}/regenerate`, body, shell);
    if (final) finishBlock(shell, final, { deep });
  } catch (err) {
    if (err?.name === "AbortError") stopped(shell);
    else showError(err, shell);
  } finally {
    setBusy(false);
    S.abort = null;
    livePanel = null;
  }
}

/* ── baho ─────────────────────────────────────────────────────────── */

/** 👍/👎 — optimistik; xato bo'lsa holat qaytariladi va toast chiqadi. */
function rateButtons(mid, current) {
  const wrap = el("div.rate", { role: "group", "aria-label": t("web.ask.rate") });
  let value = current || 0;
  const make = (score, ico, label) => {
    const btn = iconButton(ico, label, {
      cls: "rate-btn",
      attrs: { "aria-pressed": String(value === score) },
    });
    btn.addEventListener("click", async () => {
      const prev = value;
      value = value === score ? 0 : score;
      paint();
      try {
        await post(`/api/conversations/${S.convId}/messages/${mid}/rate`, { rating: value });
        haptic.select();
      } catch (err) {
        value = prev; paint();
        toastError(err);
      }
    });
    return btn;
  };
  const up = make(1, "thumbs-up", t("web.chat.rate_up"));
  const down = make(-1, "thumbs-down", t("web.chat.rate_down"));
  function paint() {
    up.setAttribute("aria-pressed", String(value === 1));
    down.setAttribute("aria-pressed", String(value === -1));
  }
  wrap.append(up, down);
  return wrap;
}

/* ── mundarija ────────────────────────────────────────────────────── */

const TOC_MIN_WORDS = 1500;

/** Uzun javob uchun mundarija (desktop o'ng reyka, mobil varaq). */
function buildToc(shell) {
  const src = sourceOf.get(shell.node) || "";
  if (src.split(/\s+/).length < TOC_MIN_WORDS) return null;
  // `md.js` `#` → h3 … `######` → h6 chizadi va har biriga barqaror id beradi.
  const heads = [...shell.answer.querySelectorAll("h1, h2, h3, h4")];
  if (heads.length < 3) return null;
  heads.forEach((h, i) => { if (!h.id) h.id = `sec-${Date.now().toString(36)}-${i}`; });

  const list = () => el("nav.toc-list", { "aria-label": t("web.ask.contents") },
    heads.map((h) => el(`a.toc-link.lv-${h.tagName.toLowerCase()}`, {
      href: `#${h.id}`,
      onclick: (e) => { e.preventDefault(); h.scrollIntoView({ block: "start", behavior: "smooth" }); },
    }, h.textContent)));

  if (!isMobile()) {
    const rail = el("aside.toc-rail", null, [el("div.toc-head", null, t("web.ask.contents")), list()]);
    shell.node.appendChild(rail);
    shell.node.classList.add("has-toc");
    return null;
  }
  return el("button.chip.tool", {
    type: "button",
    onclick: () => openSheet(t("web.ask.contents"), list()),
  }, [icon("list"), el("span", null, t("web.ask.contents"))]);
}

/* ── javob chrome: oddiy matn, "hammasini ochish" ─────────────────── */

/** Tugmalar/ikonkalar matnга kirmasin — nusxa toza chiqsin. */
const NOT_TEXT = ".md-block-copy, .md-code-btn, .md-table-tools, .code-copy, .md-mermaid-bar, .cite-chip, svg";

/**
 * Chizilgan javobning **oddiy matni**: markdown belgilarisiz, jadval
 * katakchalari tab bilan, blok orasida bo'sh qator.
 */
function plainText(answerEl) {
  if (!answerEl) return "";
  const copy = answerEl.cloneNode(true);
  for (const junk of copy.querySelectorAll(NOT_TEXT)) junk.remove();
  for (const tr of copy.querySelectorAll("tr")) {
    tr.textContent = [...tr.children].map((c) => c.textContent.trim()).join("\t");
  }
  const out = [];
  for (const block of copy.children) {
    const text = (block.textContent || "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
    if (text) out.push(text);
  }
  return out.join("\n\n");
}

const EXPAND_MIN = 2;

/** Javobda bir nechta yig'ilgan blok bo'lsa — "hammasini ochish" tugmasi. */
function expandAll(shell) {
  const boxes = () => [...shell.answer.querySelectorAll("details.md-details")];
  if (boxes().length < EXPAND_MIN) return null;
  const btn = iconButton("chevrons-up-down", t("web.md.expand_all"), {
    attrs: { "aria-pressed": "false" },
  });
  btn.addEventListener("click", () => {
    const list = boxes();
    const open = !list.every((d) => d.open);
    for (const d of list) d.open = open;
    btn.setAttribute("aria-pressed", String(open));
    btn.title = open ? t("web.md.collapse_all") : t("web.md.expand_all");
    btn.setAttribute("aria-label", btn.title);
  });
  return btn;
}

/* ── futer ────────────────────────────────────────────────────────── */

/**
 * Javob futeri: nusxa · ulashish · qayta · 👍/👎 · meta · davom chiplari.
 *
 * @param {object} m   xabar (`_msg_out` yoki `done`)
 * @param {object} shell  `assistantShell()` natijasi
 */
export function buildFooter(m, shell) {
  if (!sourceOf.has(shell.node) && m.content) sourceOf.set(shell.node, m.content);
  const src = () => sourceOf.get(shell.node) || m.content || "";
  const row = el("div.answer-actions");

  row.appendChild(iconButton("copy", t("web.ask.copy"), {
    onclick: () => copyText(src(), t("web.ask.copied")),
  }));

  // "Oddiy matn bo'lib nusxa" — markdown belgilari yo'q variant (Telegram,
  // xat, hujjatga tashlash uchun). Chizilgan DOM'dan olinadi, qayta parse yo'q.
  row.appendChild(iconButton("align-left", t("web.md.copy_text"), {
    onclick: () => copyText(plainText(shell.answer) || src(), t("web.ask.copied")),
  }));

  const expand = expandAll(shell);
  if (expand) row.appendChild(expand);

  row.appendChild(iconButton("share-2", t("web.ask.share"), {
    onclick: async () => {
      const url = m.id && S.convId ? `${location.origin}/ask/${S.convId}#m${m.id}` : location.href;
      // Web Share API — faqat foydalanuvchi harakatidan; bo'lmasa havola nusxasi.
      if (navigator.share) {
        try { await navigator.share({ title: S.convTitle || t("web.ask.title"), url }); return; }
        catch { /* bekor qilindi yoki qo'llab-quvvatlanmaydi */ }
      }
      copyText(url, t("web.ask.link_copied"));
    },
  }));

  if (m.id) {
    row.appendChild(iconButton("refresh-cw", t("web.ask.regenerate"), {
      onclick: () => regenerate(m.id, shell),
    }));
    row.appendChild(rateButtons(m.id, m.rating));
  }

  const toc = buildToc(shell);
  if (toc) row.appendChild(toc);

  const meta = metaLine(m);
  const foot = el("div.foot-inner", null, [row, meta]);

  const chips = (m.followups || []).filter(Boolean);
  if (chips.length) {
    foot.appendChild(el("div.followups", { role: "group", "aria-label": t("web.ask.followups") },
      chips.map((label) => el("button.chip.suggest", {
        type: "button",
        onclick: () => {
          const input = $("#ask-input");
          if (!input) return;
          input.value = label;
          input.focus();
          input.dispatchEvent(new Event("input"));
        },
      }, label))));
  }
  return foot;
}

/* ── tarixdagi xabarlar uchun ulagichlar ──────────────────────────── */

/** `messages.js` shu yerdan futer va deep blokini oladi (sikl import yo'q). */
export function initAnswer() {
  hooks.footer = (m, shell) => buildFooter(m, shell);
  hooks.deepBlock = storedDeepBlock;
  hooks.citations = linkifyCitations;
  $("#btn-stop")?.addEventListener("click", stopStream);
}

/** Tred bo'sh bo'lganda taklif chiplari composer'ga yozadi. */
export function initSuggests() {
  threadEl()?.addEventListener("click", (e) => {
    const chip = e.target.closest(".suggest");
    if (!chip) return;
    const input = $("#ask-input");
    if (!input) return;
    input.value = chip.textContent.trim();
    input.focus();
    input.dispatchEvent(new Event("input"));
  });
}
