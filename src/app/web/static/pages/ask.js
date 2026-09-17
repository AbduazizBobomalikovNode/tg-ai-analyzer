/* pages/ask.js — `/ask` sahifasining kirish nuqtasi.
 *
 * Vazifasi faqat **ulash**: modullarni init qilish, boshlang'ich
 * ma'lumotni yuklash va suhbatni ochish. Hech qanday chizish mantiqi bu
 * yerda yo'q — u `ask/*.js` ichida.
 *
 * Yuklash tartibi ataylab shunday:
 *   1. `initAnswer()` — `messages.hooks` to'ldiriladi (tarixni chizishdan
 *      **oldin**, aks holda birinchi sahifada futer bo'lmaydi);
 *   2. `/api/me` — akkaunt va prefs (composer sozlamalari shundan);
 *   3. chatlar + suhbatlar **parallel**;
 *   4. URL'dagi suhbat ochiladi.
 */

import { ready, $ } from "../lib/dom.js";
import { get } from "../lib/api.js";
import { toastError } from "../lib/toast.js";
import { t } from "../lib/i18n.js";
import { S, cfg, on, applyPrefs } from "./ask/state.js";
import { initSheet, banner } from "./ask/panels.js";
import { initAnswer, initSuggests } from "./ask/answer.js";
import { initActions } from "./ask/actions.js";
import { initComposer, loadChats, contextFromQuery } from "./ask/composer.js";
import {
  initConversations, loadConversations, openConversation, paintAccounts,
} from "./ask/conversations.js";

/** Sahifadagi yagona eksport — shablon shuni chaqiradi. */
export function initAsk(options = {}) {
  Object.assign(cfg, options);
  ready(() => { boot().catch((err) => toastError(err)); });
}

async function boot() {
  initSheet();
  initAnswer();
  initComposer();
  initConversations();
  initActions();
  initSuggests();

  const ok = await loadMe();
  if (!ok) return;

  await Promise.all([loadChats(), loadConversations()]);
  contextFromQuery();

  // Akkaunt almashsa chatlar va kontekst qaytadan yuklanadi.
  on("account", async () => { await loadChats(); });

  const fromUrl = conversationFromUrl();
  if (fromUrl) await openConversation(fromUrl);
  else $("#ask-input")?.focus();
}

/** `/api/me`: akkauntlar + prefs. Akkaunt yo'q bo'lsa — banner, oqim to'xtaydi. */
async function loadMe() {
  let me;
  try {
    me = await get("/api/me");
  } catch (err) {
    toastError(err);
    return false;
  }
  S.accounts = me.accounts || [];
  const active = S.accounts.find((a) => a.status === "active") || S.accounts[0];
  S.accountId = active ? active.id : null;
  applyPrefs(me.user?.prefs);
  paintAccounts();
  if (!S.accounts.length) {
    banner("warn", t("web.chat.no_accounts"), {
      action: { label: t("web.chat.add_account"), run: () => { location.href = "/login"; } },
    });
  }
  return true;
}

/** `/ask/<id>` yoki eski `#c<id>` — ikkalasi ham ishlaydi. */
function conversationFromUrl() {
  const path = /^\/ask\/(\d+)\/?$/.exec(location.pathname);
  if (path) return Number(path[1]);
  if (cfg.conversationId) return Number(cfg.conversationId);
  const hash = /^#c(\d+)$/.exec(location.hash || "");
  return hash ? Number(hash[1]) : 0;
}

// M6/coordinator: "More" varag'idagi chiqish tugmasi endi `lib/shell.js::initLogout()`
// tomonidan global boshqariladi — bu yerda alohida ulash **qo'shimcha** logout
// so'roviga olib kelardi.
