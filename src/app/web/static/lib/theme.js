/* lib/theme.js — tema (system | light | dark), `--kb` va mermaid re-init.
 *
 * `<html data-theme>` ni **paint'dan oldin** `base.html` dagi kichik inline
 * skript qo'yadi (aks holda qorong'u temada oq chaqnash bo'ladi). Bu modul
 * o'sha holatni boshqaradi: almashtirish, `<meta name=theme-color>`,
 * Telegram `theme_changed`, va `tgai:theme` maxsus hodisasi (mermaid shu
 * hodisada qayta chiziladi).
 */

import { ui } from "./store.js";

export const MODES = ["system", "light", "dark"];
const KEY = "theme";

/** Saqlangan rejim (`system` default). */
export const mode = () => (MODES.includes(ui.get(KEY)) ? ui.get(KEY) : "system");

/** Hozir amalda qaysi tema ko'rinyapti (`light` | `dark`). */
export function effective() {
  const m = mode();
  if (m !== "system") return m;
  try { return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"; }
  catch { return "light"; }
}

function paintMeta(theme) {
  const root = document.documentElement;
  const bg = getComputedStyle(root).getPropertyValue("--bg").trim() || (theme === "dark" ? "#0b1219" : "#eef1f5");
  let meta = document.querySelector('meta[name="theme-color"]');
  if (!meta) { meta = document.createElement("meta"); meta.name = "theme-color"; document.head.appendChild(meta); }
  meta.setAttribute("content", bg);
}

/** Rejimni qo'llaydi (DOM + meta + hodisa). */
export function apply(m = mode()) {
  const root = document.documentElement;
  if (m === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", m);
  const eff = effective();
  root.style.colorScheme = eff;
  paintMeta(eff);
  window.dispatchEvent(new CustomEvent("tgai:theme", { detail: { mode: m, effective: eff } }));
  return eff;
}

/** Rejimni o'rnatadi va saqlaydi. */
export function set(m) {
  ui.set(KEY, MODES.includes(m) ? m : "system");
  return apply();
}

/** Keyingi rejimga aylantiradi: system → light → dark → system. */
export const cycle = () => set(MODES[(MODES.indexOf(mode()) + 1) % MODES.length]);

/**
 * Klaviatura ochilganda pastki panellar ko'rinib tursin.
 *
 * WebKit xatosi 217754: klaviatura ochilganda `env(safe-area-inset-bottom)`
 * kichraymaydi, ya'ni composer klaviatura ostida qolib ketadi. Yechim —
 * `visualViewport` dan hisoblangan `--kb`; CSS `max(safe-area, --kb)`
 * ishlatadi (`--pad-bottom`).
 */
function watchKeyboard() {
  const vv = window.visualViewport;
  if (!vv) return;
  const onResize = () => {
    const kb = Math.max(0, window.innerHeight - vv.height - vv.offsetTop);
    document.documentElement.style.setProperty("--kb", `${Math.round(kb)}px`);
  };
  vv.addEventListener("resize", onResize);
  vv.addEventListener("scroll", onResize);
  onResize();
}

let started = false;

/** Bir marta chaqiriladi (sahifa kirish nuqtasidan). */
export function init() {
  if (started) return;
  started = true;
  apply();
  try {
    window.matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", () => { if (mode() === "system") apply(); });
  } catch { /* eski brauzer */ }
  watchKeyboard();

  // Mermaid temani bir marta oladi — tema almashsa qayta chizish kerak.
  window.addEventListener("tgai:theme", () => {
    if (!window.mermaid) return;
    try {
      window.mermaid.initialize({
        startOnLoad: false, securityLevel: "strict", fontFamily: "inherit",
        theme: effective() === "dark" ? "dark" : "default",
      });
    } catch { /* mermaid yuklanmagan */ }
  });
}
