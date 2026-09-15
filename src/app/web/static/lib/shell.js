/* lib/shell.js — karkas kirish nuqtasi (har sahifada bir marta).
 *
 * Vazifasi: tema, Telegram Mini App, ⌘K palitrasi, mobil "More" varag'i,
 * tema tugmasi va `Esc`. Sahifaga xos mantiq bu yerda **yo'q**.
 */

import { $, $$, trapFocus } from "./dom.js";
import * as themeMod from "./theme.js";
import * as tg from "./tg.js";
import * as cmdk from "./palette-cmd.js";
import { bind } from "./keys.js";
import { t } from "./i18n.js";

function initMoreSheet() {
  const sheet = $("#more-sheet");
  if (!sheet) return;
  let untrap = null, last = null;
  const open = () => {
    last = document.activeElement;
    sheet.hidden = false;
    untrap = trapFocus(sheet);
    $(".sheet-item", sheet)?.focus();
    tg.haptic.impact("light");
  };
  const close = () => {
    if (sheet.hidden) return;
    sheet.hidden = true;
    untrap?.(); untrap = null;
    last?.focus?.();
  };
  for (const b of $$("[data-sheet-open]")) b.addEventListener("click", open);
  for (const b of $$("[data-sheet-close]", sheet)) b.addEventListener("click", close);
  sheet.addEventListener("mousedown", (e) => { if (e.target === sheet) close(); });
  bind("escape", () => { if (sheet.hidden) return false; close(); return true; });
}

function initThemeButtons() {
  const paint = () => {
    const m = themeMod.mode();
    for (const b of $$("[data-theme-toggle]")) {
      const lbl = `${t("web.theme.title")}: ${t(`web.theme.${m}`)}`;
      b.setAttribute("aria-label", lbl);
      b.title = lbl;
      const txt = b.querySelector("[data-theme-label]");
      if (txt) txt.textContent = t(`web.theme.${m}`);
      const use = b.querySelector("use");
      if (use) {
        const name = m === "dark" ? "moon" : m === "light" ? "sun" : "monitor";
        const href = `${(window.TGAI && window.TGAI.sprite) || "/static/icons.svg"}#${name}`;
        use.setAttribute("href", href);
      }
    }
  };
  for (const b of $$("[data-theme-toggle]")) {
    b.addEventListener("click", () => { themeMod.cycle(); tg.haptic.select(); });
  }
  window.addEventListener("tgai:theme", paint);
  paint();
}

/** Karkasni ishga tushiradi. Sahifa skripti buni birinchi chaqiradi. */
export function initShell() {
  themeMod.init();
  tg.init();
  // Tafsilot sahifasida Telegram orqaga tugmasi = tarix.
  tg.backButton(document.body.dataset.detail === "1");
  cmdk.init();
  initMoreSheet();
  initThemeButtons();
}

export { cmdk, themeMod as theme, tg };
