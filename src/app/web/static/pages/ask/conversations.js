/* pages/ask/conversations.js — suhbatlar paneli.
 *
 * Desktop: `pane-list` (300px). Mobil: **o'sha DOM tuguni** varaqqa
 * ko'chiriladi (`#conv-sheet`) — ro'yxat ikki nusxada chizilmaydi va
 * tanlangan holat ajralib ketmaydi.
 *
 * O'chirish — **kechiktirilgan**: qator darhol yo'qoladi, `DELETE` esa 5
 * soniyadan keyin ketadi. Shu oraliqda "Bekor qilish" bosilsa server hech
 * narsa ko'rmaydi. Undo'ni toast'dan keyin so'rash mumkin emas: suhbat
 * allaqachon o'chgan bo'lardi.
 */

import { $, el, clear, replace, debounce } from "../../lib/dom.js";
import { get, post, patch, del } from "../../lib/api.js";
import { icon as iconOf, iconButton } from "../../lib/icons.js";
import { t } from "../../lib/i18n.js";
import { listDate } from "../../lib/fmt.js";
import { toast, toastError } from "../../lib/toast.js";
import { mobileQuery } from "../../lib/store.js";
import { S, on, emit, contextChatObjects } from "./state.js";
import { renderMessages, prependMessages, threadEl, scrollBottom } from "./messages.js";
import {
  confirmSheet, openSheet, closeSheet,
  openWriteModeSheet, openAutoReplySheet, openScheduledSheet,
} from "./panels.js";

const UNDO_MS = 5000;
const PAGE = 200;

/** Kechiktirilgan o'chirishlar: conv_id → timeout id. */
const pendingDeletes = new Map();

let filter = "";

/* ── ro'yxat ──────────────────────────────────────────────────────── */

/** Akkaunt yorlig'i (`@username (status)` — status "active" bo'lsa qisqa). */
function accountLabel(id) {
  const acc = S.accounts.find((a) => a.id === id);
  if (!acc) return "";
  return `${acc.label || acc.tg_account_id}${acc.status !== "active" ? ` (${acc.status})` : ""}`;
}

function paintList() {
  const list = $("#conv-list");
  if (!list) return;
  const q = filter.toLowerCase();
  const rows = S.conversations.filter((c) =>
    !pendingDeletes.has(c.id) && (!q || String(c.title || "").toLowerCase().includes(q)));
  clear(list);
  if (!rows.length) {
    list.appendChild(el("li.conv-empty", null, t("web.chat.no_conversations")));
    return;
  }
  for (const c of rows) list.appendChild(convRow(c));
}

function convRow(c) {
  const active = c.id === S.convId;
  const title = el("span.conv-title", null, c.title || t("web.ask.untitled"));
  const li = el("li.conv-row", {
    dataset: { conv: c.id },
    class: active ? "is-active" : "",
    "aria-current": active ? "true" : null,
  }, [
    el("button.conv-open", {
      type: "button",
      onclick: () => { openConversation(c.id); closeConvSheet(); },
    }, [
      title,
      el("span.conv-meta", null, [
        c.updated_at ? listDate(c.updated_at) : "",
        accountLabel(c.account_id),
      ].filter(Boolean).join(" · ")),
    ]),
    el("span.conv-actions", null, [
      iconButton("edit-3", t("web.ask.rename"), {
        cls: "conv-act",
        onclick: () => startRename(li, c, title),
      }),
      iconButton("trash-2", t("web.chat.delete"), {
        cls: "conv-act danger",
        onclick: () => deferredDelete(c),
      }),
    ]),
  ]);
  return li;
}

/* ── nom o'zgartirish ─────────────────────────────────────────────── */

/** Joyida tahrirlash: `Enter` saqlaydi, `Esc` bekor qiladi. */
function startRename(li, conv, titleEl) {
  if (li.querySelector(".conv-rename")) return;
  const input = el("input.conv-rename", {
    type: "text", value: conv.title || "", maxlength: "120",
    "aria-label": t("web.ask.rename"),
  });
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  const finish = async (save) => {
    const value = input.value.trim();
    input.replaceWith(titleEl);
    if (!save || value === (conv.title || "")) return;
    try {
      const updated = await patch(`/api/conversations/${conv.id}`, { title: value });
      conv.title = updated.title;
      titleEl.textContent = updated.title || t("web.ask.untitled");
      if (conv.id === S.convId) setTitle(updated.title);
    } catch (err) { toastError(err); }
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

/* ── o'chirish (undo bilan) ───────────────────────────────────────── */

function deferredDelete(conv) {
  if (pendingDeletes.has(conv.id)) return;
  const timer = setTimeout(async () => {
    pendingDeletes.delete(conv.id);
    try {
      await del(`/api/conversations/${conv.id}`);
      S.conversations = S.conversations.filter((c) => c.id !== conv.id);
    } catch (err) { toastError(err); }
    paintList();
  }, UNDO_MS);
  pendingDeletes.set(conv.id, timer);
  paintList();
  if (conv.id === S.convId) openConversation(0);
  toast(t("web.ask.deleted", { title: conv.title || t("web.ask.untitled") }), {
    ms: UNDO_MS,
    action: {
      label: t("web.ask.undo"),
      run: () => {
        clearTimeout(pendingDeletes.get(conv.id));
        pendingDeletes.delete(conv.id);
        paintList();
      },
    },
  });
}

/* ── ochish ───────────────────────────────────────────────────────── */

function setTitle(title) {
  S.convTitle = title || "";
  const h1 = $("#conv-title");
  if (h1) h1.textContent = title || t("web.ask.title");
}

/**
 * Suhbatni ochadi (`0` — yangi, bo'sh tred).
 *
 * URL `history.replaceState` bilan yangilanadi: sahifa qayta yuklanmaydi,
 * lekin havolani nusxalash ishlaydi va `?chats=` saqlanadi.
 */
export async function openConversation(id) {
  S.convId = Number(id) || 0;
  S.nextBefore = null;
  paintList();
  if (!S.convId) {
    setTitle("");
    renderMessages([]);
    replaceUrl("/ask");
    return;
  }
  replaceUrl(`/ask/${S.convId}`);
  try {
    const data = await get(`/api/conversations/${S.convId}/messages?limit=${PAGE}`);
    S.nextBefore = data.next_before || null;
    setTitle(data.conversation?.title || "");
    renderMessages(data.items);
    jumpToHash();
  } catch (err) {
    toastError(err);
    S.convId = 0;
    renderMessages([]);
  }
}

function replaceUrl(path) {
  const search = location.search || "";
  history.replaceState(null, "", `${path}${search}`);
}

/** `#m<id>` — ulashilgan havola aynan o'sha javobga tushadi. */
function jumpToHash() {
  const m = /^#m(\d+)$/.exec(location.hash || "");
  if (!m) return;
  const node = threadEl()?.querySelector(`[data-mid="${m[1]}"]`);
  if (node) node.scrollIntoView({ block: "start" });
  else scrollBottom(true);
}

/** "Eskiroq" sahifa (kursor — eng eski qator id). */
async function loadEarlier() {
  if (!S.convId || !S.nextBefore) return;
  const btn = $("#btn-earlier");
  if (btn) btn.disabled = true;
  try {
    const data = await get(`/api/conversations/${S.convId}/messages?limit=${PAGE}&before=${S.nextBefore}`);
    S.nextBefore = data.next_before || null;
    prependMessages(data.items);
  } catch (err) { toastError(err); }
  if (btn) btn.disabled = false;
}

/* ── ro'yxatni yuklash ────────────────────────────────────────────── */

export async function loadConversations() {
  try {
    const data = await get("/api/conversations?limit=50");
    S.conversations = data.items || [];
  } catch { S.conversations = []; }
  paintList();
}

/* ── mobil varaq ──────────────────────────────────────────────────── */

const pane = () => $("#convs-pane");

function openConvSheet() {
  const sheet = $("#conv-sheet");
  const body = $("#conv-sheet-body");
  if (!sheet || !body || !pane()) return;
  body.appendChild(pane());
  sheet.hidden = false;
  $("#conv-search")?.focus();
}

function closeConvSheet() {
  const sheet = $("#conv-sheet");
  if (!sheet || sheet.hidden) return;
  sheet.hidden = true;
  restorePane();
}

/** Panelni asl joyiga (shell-body ichiga, tred oldiga) qaytaradi. */
function restorePane() {
  const p = pane();
  const host = $("#main-content");
  if (!p || !host || p.parentElement === host) return;
  host.insertBefore(p, host.firstChild);
}

/* ── init ─────────────────────────────────────────────────────────── */

export function initConversations() {
  $("#conv-search")?.addEventListener("input", debounce((e) => {
    filter = e.target.value.trim();
    paintList();
  }, 80));

  $("#btn-new-conv")?.addEventListener("click", () => {
    openConversation(0);
    closeConvSheet();
    $("#ask-input")?.focus();
  });
  $("#btn-earlier")?.addEventListener("click", loadEarlier);
  $("#btn-convs")?.addEventListener("click", openConvSheet);
  $("#conv-sheet")?.addEventListener("mousedown", (e) => {
    if (e.target.id === "conv-sheet") closeConvSheet();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#conv-sheet")?.hidden) closeConvSheet();
  });

  // Mobildan desktopga o'tganda varaq yopilib panel joyiga qaytsin.
  mobileQuery().addEventListener("change", (e) => { if (!e.matches) closeConvSheet(); });

  $("#btn-conv-menu")?.addEventListener("click", openConvMenu);
  $("#account-select")?.addEventListener("change", (e) => {
    S.accountId = Number(e.target.value) || null;
    S.contextChats = [];
    e.target.title = e.target.selectedOptions[0]?.textContent || "";
    emit("account");
  });

  on("conversations-dirty", debounce(async () => {
    await loadConversations();
    const cur = S.conversations.find((c) => c.id === S.convId);
    if (cur) setTitle(cur.title);
  }, 300));
  on("conversation-created", () => { loadConversations(); });
}

/**
 * Suhbat menyusi. Kontekstga **bitta** chat tanlangan bo'lsa o'sha chatning
 * sozlamalari ham shu yerda: yozish rejimi, auto-reply, rejalashtirilganlar
 * (eski `/chat` sahifasidagi top bar tugmalarining o'rni).
 */
function openConvMenu() {
  const conv = S.conversations.find((c) => c.id === S.convId);
  const chats = contextChatObjects();
  const chat = chats.length === 1 ? chats[0] : null;
  const items = [];

  if (conv) {
    items.push(el("button.sheet-item", {
      type: "button",
      onclick: () => {
        closeSheet();
        $(`.conv-row[data-conv="${conv.id}"] .conv-act`)?.click();
      },
    }, [iconOf("edit-3"), t("web.ask.rename")]));
    items.push(el("button.sheet-item", {
      type: "button",
      onclick: async () => {
        closeSheet();
        const ok = await confirmSheet(
          conv.title || t("web.ask.untitled"),
          t("web.chat.delete_confirm"), t("web.chat.delete"), { danger: true },
        );
        if (ok) deferredDelete(conv);
      },
    }, [iconOf("trash-2"), t("web.chat.delete")]));
  }

  if (chat && S.accountId) {
    items.push(el("div.sheet-sep", null, chat.title || ""));
    items.push(el("button.sheet-item", {
      type: "button", onclick: () => openWriteModeSheet(chat, S.accountId),
    }, [iconOf("edit-3"), t("web.chat.write_mode")]));
    items.push(el("button.sheet-item", {
      type: "button", onclick: () => openAutoReplySheet(chat, S.accountId),
    }, [iconOf("bolt"), t("web.ar.title")]));
    items.push(el("button.sheet-item", {
      type: "button", onclick: () => openScheduledSheet(chat, S.accountId),
    }, [iconOf("clock"), t("web.sched.title")]));
    items.push(el("button.sheet-item", {
      type: "button",
      onclick: async () => {
        try {
          await post(`/api/accounts/${S.accountId}/chats/${chat.id}/sync`, {});
          toast(t("web.chat.sync_queued"), { kind: "ok" });
          closeSheet();
        } catch (err) { toastError(err); }
      },
    }, [iconOf("refresh-cw"), t("web.chat.sync")]));
  }

  if (!items.length) { toast(t("web.ask.no_conversation")); return; }
  openSheet(conv?.title || t("web.ask.title"), items);
}

/**
 * Akkaunt tanlagich (ro'yxat `/api/me` dan keladi).
 *
 * L4: yopiq `<select>` uzun nomni kesib ko'rsatadi (`ask.css` ellipsis) —
 * to'liq nom `title` orqali hover'da ko'rinadi, `<option title>` yopiq
 * qutida ko'rinmaydi.
 */
export function paintAccounts() {
  const sel = $("#account-select");
  if (!sel) return;
  replace(sel, S.accounts.map((a) => el("option", {
    value: String(a.id),
    selected: a.id === S.accountId,
    title: accountLabel(a.id),
  }, accountLabel(a.id))));
  if (!S.accounts.length) {
    sel.appendChild(el("option", { value: "" }, t("web.chat.no_accounts")));
  }
  sel.title = S.accountId ? accountLabel(S.accountId) : "";
}
