/* lib/tg.js — Telegram Mini App SDK ustidagi yupqa qobiq.
 *
 * Telegram tashqarisida **hamma narsa no-op**: `inTelegram()` false, haptik
 * jim, BackButton yo'q. Shu sababli sahifa kodi hech qayerda
 * `if (window.Telegram)` yozmaydi.
 *
 * SDK vendor qilinadi (`/static/vendor/telegram-web-app.js`, R1b) — CDN
 * yo'q. Fayl bo'lmasa ham bu modul ishlaydi: `WebApp` topilmaydi va hamma
 * chaqiruv no-op bo'ladi.
 *
 * `initData` **ishonchsiz** — imzosi server tomonda tekshirilmaguncha
 * (keyingi bosqich) undan foydalanilmaydi; bu yerda umuman o'qilmaydi.
 */

import * as theme from "./theme.js";

/** Telegram SDK obyekti yoki null. */
export const webApp = () => {
  try { return (window.Telegram && window.Telegram.WebApp) || null; } catch { return null; }
};

/** Haqiqatan Mini App ichidamizmi (SDK bor va platforma "unknown" emas). */
export function inTelegram() {
  const w = webApp();
  return !!(w && w.platform && w.platform !== "unknown");
}

/** Mini App CSS o'zgaruvchilari uchun `<html class="in-telegram">`. */
function markRoot() {
  document.documentElement.classList.toggle("in-telegram", inTelegram());
}

/**
 * Telegram mavzusini token'larga bog'laydi.
 * SDK 6.10+ o'zi `--tg-theme-*` larni `<html>` ga qo'yadi; eski
 * mijozlarda `themeParams` dan qo'lda yozamiz.
 */
function applyThemeParams() {
  const w = webApp();
  if (!w) return;
  const root = document.documentElement;
  for (const [k, v] of Object.entries(w.themeParams || {})) {
    if (typeof v === "string" && /^#[0-9a-f]{3,8}$/i.test(v)) {
      root.style.setProperty(`--tg-theme-${k.replace(/_/g, "-")}`, v);
    }
  }
  theme.apply();
}

let backHandler = null;

/**
 * Orqaga tugmasi ↔ brauzer tarixi. `show` bo'lsa ko'rsatadi va bosilganda
 * `history.back()` qiladi.
 */
export function backButton(show, onBack) {
  const w = webApp();
  if (!w || !w.BackButton) return;
  try {
    if (backHandler) { w.BackButton.offClick(backHandler); backHandler = null; }
    if (show) {
      backHandler = () => (onBack ? onBack() : history.back());
      w.BackButton.onClick(backHandler);
      w.BackButton.show();
    } else {
      w.BackButton.hide();
    }
  } catch { /* eski mijoz */ }
}

/** Haptik: `impact` (light/medium/heavy), `notification` (success/…), `selection`. */
export const haptic = {
  impact(style = "light") { try { webApp()?.HapticFeedback?.impactOccurred(style); } catch { /* no-op */ } },
  notify(type = "success") { try { webApp()?.HapticFeedback?.notificationOccurred(type); } catch { /* no-op */ } },
  select() { try { webApp()?.HapticFeedback?.selectionChanged(); } catch { /* no-op */ } },
};

/** Tashqi havolani Telegram ichida to'g'ri ochadi (tg:// ham). */
export function openLink(url) {
  const w = webApp();
  if (!w) { window.open(url, "_blank", "noopener"); return; }
  try {
    if (/^tg:\/\//i.test(url)) w.openTelegramLink(url);
    else w.openLink(url);
  } catch { window.open(url, "_blank", "noopener"); }
}

let started = false;

/**
 * Bir marta: `ready()` + `expand()` + tema + `theme_changed`.
 * `expand()` shart — usiz Mini App yarim balandlikda ochilib "buzilgan"
 * ko'rinadi (tadqiqot § Telegram Mini App).
 */
export function init() {
  if (started) return inTelegram();
  started = true;
  markRoot();
  const w = webApp();
  if (!w) return false;
  try {
    w.ready();
    w.expand();
    if (w.disableVerticalSwipes) w.disableVerticalSwipes();
    applyThemeParams();
    w.onEvent("themeChanged", applyThemeParams);
    w.onEvent("viewportChanged", () => {
      if (w.viewportStableHeight) {
        document.documentElement.style.setProperty("--tg-viewport-stable-height", `${w.viewportStableHeight}px`);
      }
    });
  } catch { /* eski mijoz — asosiy UI baribir ishlaydi */ }
  return true;
}
