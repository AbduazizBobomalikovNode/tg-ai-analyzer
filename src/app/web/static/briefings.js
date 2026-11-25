/* briefings.js — `/briefings` (M2: v2 karkasga ko'chirildi).
 *
 * Briefing matni markdown — `lib/md.js` (yagona renderer) orqali chiziladi.
 * Boshqa har qanday matn `el()`/`textContent` orqali — `innerHTML` faqat
 * `md()` chiqishi uchun (escape-first, `docs/ui-v2.md` § Frontend).
 *
 * `lib/api.js` orqali — 401'da avtomatik `/login`ga, xato kodlari i18n
 * kalit bo'lib qaytadi (ilgari bu fayl o'z `fetch` qobig'ini yozardi).
 */

import { $, el, clear, replace } from "./lib/dom.js";
import { t } from "./lib/i18n.js";
import { get, put } from "./lib/api.js";
import { md, hydrate } from "./lib/md.js";
import { icon } from "./lib/icons.js";
import { dateTime } from "./lib/fmt.js";
import { toastError } from "./lib/toast.js";

/** M2: brief hali yo'q — "Bugungi nashr" paneli tekis matn o'rniga `.empty-v2`. */
function emptyToday() {
  replace($("#brief-body"), el("div.empty-v2", null, [
    el("span.empty-ico", { "aria-hidden": "true" }, icon("inbox")),
    el("h2", null, t("web.brief.empty_title")),
    el("p", null, t("web.brief.none")),
  ]));
  $("#brief-meta").textContent = "";
}

function renderBody(row) {
  $("#brief-day").textContent = row.day;
  $("#brief-body").innerHTML = md(row.text || "");
  hydrate($("#brief-body"));
  const bits = [];
  if (row.model) bits.push(row.model);
  if (row.tokens) bits.push(`${row.tokens} tok`);
  if (row.cost_usd) bits.push(`$${Number(row.cost_usd).toFixed(4)}`);
  bits.push(row.delivered_at ? t("web.brief.delivered") : t("web.brief.not_delivered"));
  $("#brief-meta").textContent = bits.join(" · ");
}

async function loadOne(day) {
  try {
    renderBody(await get(`/api/briefings/${encodeURIComponent(day)}`));
  } catch {
    emptyToday();
  }
}

async function loadList() {
  const data = await get("/api/briefings?limit=14");
  const list = $("#brief-list");
  clear(list);
  if (!data.items.length) {
    list.appendChild(el("li.muted", null, t("web.brief.none")));
    emptyToday();
    return;
  }
  for (const it of data.items) {
    const a = el("a", {
      href: `/briefings/${it.day}`,
      onclick: (ev) => { ev.preventDefault(); history.pushState({}, "", a.href); loadOne(it.day); },
    }, it.day);
    list.appendChild(el("li", null, [a, el("span.muted.small", null, ` ${(it.tldr || "").slice(0, 120)}`)]));
  }
  if (!window.TGAI_BRIEF_DAY) loadOne(data.items[0].day);
}

async function loadSettings() {
  let data;
  try { data = await get("/api/settings/briefing"); } catch { return; }
  const s = data.settings;
  $("#b-enabled").checked = !!s.enabled;
  $("#b-hour").value = s.hour;
  $("#b-tz").value = s.tz;
  if (s.quiet_from) $("#b-qfrom").value = s.quiet_from;
  if (s.quiet_to) $("#b-qto").value = s.quiet_to;
  $("#brief-settings").dataset.accountId = s.account_id;
}

$("#brief-settings").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const accountId = Number($("#brief-settings").dataset.accountId || 0);
  if (!accountId) return;
  const body = {
    account_id: accountId,
    tz: $("#b-tz").value.trim(),
    hour: Number($("#b-hour").value),
    enabled: $("#b-enabled").checked,
    quiet_from: $("#b-qfrom").value || null,
    quiet_to: $("#b-qto").value || null,
    chats: null,
  };
  try {
    await put("/api/settings/briefing", body);
    $("#brief-saved").textContent = t("web.brief.saved");
  } catch (err) {
    $("#brief-saved").textContent = t("error.generic");
    toastError(err);
  }
});

loadSettings();
loadList();
if (window.TGAI_BRIEF_DAY) loadOne(window.TGAI_BRIEF_DAY);
