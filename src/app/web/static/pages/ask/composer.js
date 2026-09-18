/* pages/ask/composer.js — savol maydoni, kontekst chiplari va sozlamalar.
 *
 * Klaviatura: `Enter` yuboradi, `Shift+Enter` — yangi qator. Mobil
 * klaviatura ochilganda composer ostida qolmasin deb balandlik `--kb`
 * (`lib/theme.js::watchKeyboard`) va `dvh` bilan hisoblanadi — `vh`
 * iOS'da klaviatura bilan **kichraymaydi**.
 *
 * Kontekst chiplari — savol qaysi chatlar ustida ishlashini ko'rsatuvchi
 * yagona joy. Ular URL'dan (`?chats=1,2`) ham to'ldiriladi, shuning uchun
 * `/chats` sahifasidan "shu chatlar haqida so'rash" havolasi ishlaydi.
 */

import { $, el, clear, debounce } from "../../lib/dom.js";
import { get } from "../../lib/api.js";
import { icon, iconButton } from "../../lib/icons.js";
import { t } from "../../lib/i18n.js";
import { rank } from "../../lib/palette.js";
import { compact } from "../../lib/fmt.js";
import {
  S, cfg, on, emit, savePref, setChats, contextChatObjects, toggleContextChat, clearContextChats,
  SCALE_MODES, setScaleMode,
} from "./state.js";
import { openSheet, closeSheet } from "./panels.js";
import { chatAvatar } from "./messages.js";
import { ask } from "./answer.js";

/* ── textarea ─────────────────────────────────────────────────────── */

const MAX_ROWS_FRACTION = 0.4;   // ekranning ko'pi bilan 40% i

/** Balandlikni mazmunga moslaydi; `dvh` — klaviatura bilan ham to'g'ri. */
function autosize(input) {
  input.style.height = "auto";
  const cap = Math.round(window.innerHeight * MAX_ROWS_FRACTION);
  input.style.height = `${Math.min(input.scrollHeight, cap)}px`;
}

/* ── kontekst chiplari ────────────────────────────────────────────── */

function paintChips() {
  const wrap = $("#ctx-chips");
  const summary = $("#ctx-summary");
  if (!wrap) return;
  const chats = contextChatObjects();
  clear(wrap);
  for (const c of chats) {
    const chip = el("span.chip.ctx-chip", null, [
      chatAvatar(c),
      el("span.ctx-chip-title", null, c.title || "?"),
    ]);
    chip.appendChild(iconButton("x", t("web.ask.remove_context", { title: c.title || "" }), {
      cls: "chip-x",
      onclick: () => toggleContextChat(c.id),
    }));
    wrap.appendChild(chip);
  }
  if (chats.length > 1) {
    wrap.appendChild(el("button.chip.ghost", {
      type: "button", onclick: clearContextChats,
    }, t("web.ask.clear_context")));
    wrap.appendChild(el("span.ctx-note", null, t("web.ask.multi_context_note")));
  }
  wrap.hidden = chats.length === 0;
  if (summary) {
    summary.textContent = chats.length
      ? t("web.ask.ctx_summary", { n: chats.length, limit: S.ctxLimit })
      : "";
    summary.hidden = chats.length === 0;
  }
}

/* ── chat tanlash varag'i ─────────────────────────────────────────── */

const PICKER_LIMIT = 40;

/**
 * Chat tanlash: fuzzy (`lib/palette.rank`, uz-fold) — mahalliy ro'yxat
 * ustida, so'rovsiz. Ro'yxat bo'sh bo'lsa serverdan bir marta olinadi.
 */
export function openChatPicker() {
  const input = el("input.picker-input", {
    type: "search", autocomplete: "off", spellcheck: "false",
    placeholder: t("web.chat.search_dialogs"),
    "aria-label": t("web.chat.search_dialogs"),
  });
  const list = el("ul.ask-picker-list", { role: "listbox", "aria-label": t("web.ask.context") });
  const body = el("div.ask-picker", null, [
    el("label.field-icon", null, [icon("search"), input]),
    list,
  ]);

  const paint = () => {
    const q = input.value.trim();
    const rows = q
      ? rank(q, S.chats, (c) => `${c.title || ""} ${c.username || ""}`, PICKER_LIMIT).map((r) => r.item)
      : S.chats.slice(0, PICKER_LIMIT);
    clear(list);
    if (!rows.length) { list.appendChild(el("li.ask-picker-empty", null, t("web.cmd.empty"))); return; }
    for (const c of rows) {
      const selected = S.contextChats.includes(Number(c.id));
      const row = el("li", null, el("button.ask-picker-row", {
        type: "button", role: "option", "aria-selected": String(selected),
        onclick: () => { toggleContextChat(c.id); paint(); },
      }, [
        chatAvatar(c),
        el("span.ask-picker-body", null, [
          el("span.ask-picker-title", null, c.title || "?"),
          el("span.ask-picker-sub", null, [
            t(`web.chat.kind_${c.type || "chat"}`),
            c.synced_total ? t("web.chat.synced", { n: compact(c.synced_total) }) : t("web.chat.not_synced"),
          ].join(" · ")),
        ]),
        selected ? icon("check") : null,
      ]));
      list.appendChild(row);
    }
  };

  input.addEventListener("input", debounce(paint, 60));
  openSheet(t("web.ask.context"), body);
  paint();
  if (!S.chats.length) loadChats().then(paint);
}

/** Chatlar ro'yxatini serverdan oladi (registry bo'lmasa jimgina bo'sh qoladi). */
export async function loadChats() {
  if (!S.accountId) return;
  try {
    const data = await get(`/api/accounts/${S.accountId}/chats`);
    setChats(data.items || []);
  } catch { /* registry hali yo'q — chip tanlash o'chiq qoladi */ }
}

/* ── expert varag'i ───────────────────────────────────────────────── */

const MODES = ["auto", "agent", "direct"];
const STRATEGIES = ["auto", "search", "window", "recent"];

/** Rejim · strategiya · kontekst chegarasi — kamdan-kam tegiladigan sozlamalar. */
export function openExpertSheet() {
  const segment = (name, values, current, onPick, labelKey) => {
    const seg = el("div.segment", { role: "group", "aria-label": t(labelKey) });
    for (const v of values) {
      const btn = el("button", {
        type: "button", "aria-selected": String(v === current),
        onclick: () => {
          for (const b of seg.children) b.setAttribute("aria-selected", "false");
          btn.setAttribute("aria-selected", "true");
          onPick(v);
        },
      }, t(`web.chat.${name}_${v}`));
      seg.appendChild(btn);
    }
    return el("div.expert-row", null, [el("span.expert-label", null, t(labelKey)), seg]);
  };

  const limit = el("input", {
    type: "range", min: "10", max: String(cfg.contextMax), step: "10",
    value: String(S.ctxLimit), "aria-label": t("web.chat.context_last", { n: S.ctxLimit }),
  });
  const limitOut = el("output.expert-value", null, String(S.ctxLimit));
  limit.addEventListener("input", () => {
    S.ctxLimit = Number(limit.value);
    limitOut.textContent = String(S.ctxLimit);
    paintChips();
  });

  openSheet(t("web.ask.expert"), [
    segment("mode", MODES, S.mode, (v) => { S.mode = v; savePref("default_mode", v); }, "web.chat.mode"),
    segment("strategy", STRATEGIES, S.strategy, (v) => { S.strategy = v; savePref("default_strategy", v); }, "web.chat.strategy"),
    el("div.expert-row", null, [
      el("span.expert-label", null, t("web.ask.ctx_limit")),
      el("div.expert-slider", null, [limit, limitOut]),
    ]),
    // Bu uchtasi **`Tez` rejimda** ishlaydi: `Chuqur`/`To'liq` da kontekstni
    // korpus quvuri o'zi tanlaydi (`scale_search`).
    el("p.form-note", null, t("web.ask.expert_note")),
    el("p.form-note", null, `${t("web.ask.mode_fast")}: ${t("web.ask.mode_fast_desc")}. ${t("web.ask.mode_local_note")}`),
    el("div.sheet-actions", null,
      el("button.btn.primary", { type: "button", onclick: closeSheet }, t("web.close"))),
  ]);
}

/* ── rejim tanlagich (ko'lam) ─────────────────────────────────────── */

/**
 * Uch bosqichli segment: `Tez · Chuqur · To'liq`. Tanlangan bosqichning
 * tavsifi va taxminiy narxi/vaqti pastdagi bitta satrda ko'rinadi —
 * har tugmaning ostiga yozilsa composer ikki barobar balandlashadi.
 */
function paintMode() {
  const picker = $("#mode-picker");
  const hint = $("#mode-hint");
  if (picker) {
    for (const btn of picker.querySelectorAll("[data-mode]")) {
      const on = btn.dataset.mode === S.scaleMode;
      btn.setAttribute("aria-checked", String(on));
      btn.tabIndex = on ? 0 : -1;
    }
  }
  if (hint) {
    hint.textContent = `${t(`web.ask.mode_${S.scaleMode}_desc`)} · ${t(`web.ask.mode_${S.scaleMode}_cost`)}`;
  }
}

/** Radiogroup klaviaturasi: strelkalar tanlovni ko'chiradi (WAI-ARIA). */
function initModePicker() {
  const picker = $("#mode-picker");
  if (!picker) return;
  picker.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-mode]");
    if (!btn) return;
    setScaleMode(btn.dataset.mode);
  });
  picker.addEventListener("keydown", (e) => {
    const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1
      : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const i = SCALE_MODES.indexOf(S.scaleMode);
    const next = SCALE_MODES[(i + step + SCALE_MODES.length) % SCALE_MODES.length];
    setScaleMode(next);
    picker.querySelector(`[data-mode="${next}"]`)?.focus();
  });
}

/* ── URL'dan kontekst ─────────────────────────────────────────────── */

/**
 * `?chats=1,2,3` — `/chats` sahifasidan kelgan tanlov. Faqat **mavjud va
 * o'zimizga tegishli** id'lar qoladi (ro'yxat serverdan kelgan).
 */
export function contextFromQuery() {
  const raw = new URLSearchParams(location.search).get("chats");
  if (!raw) return;
  const ids = raw.split(",").map((x) => Number(x.trim())).filter((n) => Number.isInteger(n) && n > 0);
  S.contextChats = ids.filter((id) => S.byId.has(id)).slice(0, 12);
  emit("context");
}

/* ── init ─────────────────────────────────────────────────────────── */

export function initComposer() {
  const form = $("#composer");
  const input = $("#ask-input");
  if (!form || !input) return;

  const submit = () => {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    autosize(input);
    ask(text);
  };

  form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  input.addEventListener("input", () => autosize(input));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
  });

  $("#btn-pick-chats")?.addEventListener("click", openChatPicker);
  $("#btn-expert")?.addEventListener("click", openExpertSheet);
  initModePicker();

  on("context", paintChips);
  on("chats", paintChips);
  on("prefs", () => { paintMode(); });
  on("busy", (busy) => { input.disabled = false; form.classList.toggle("is-busy", busy); });

  S.ctxLimit = cfg.contextDefault || S.ctxLimit;
  paintChips();
  paintMode();
  autosize(input);
}
