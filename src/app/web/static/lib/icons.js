/* lib/icons.js — Lucide sprite (ISC, `static/icons.svg`) ustidagi yordamchi.
 *
 * Nega sprite: CDN yo'q, 67 ikonka bitta ~20KB fayl bo'lib bir marta
 * keshlanadi va har `<use>` faqat DOM tugun. Chiziqlar `currentColor` —
 * rang matndan meros, shuning uchun tema almashganda hech narsa
 * qilinmaydi.
 *
 * Emoji ikonkalar bu bilan almashtirildi: emoji ekran o'quvchida
 * "ohladed face" deb o'qiladi va platformalarda har xil ko'rinadi.
 */

const SPRITE = "/static/icons.svg";
const NS = "http://www.w3.org/2000/svg";
const XLINK = "http://www.w3.org/1999/xlink";

/** Sprite manzili — `asset()` bilan versiyalangan bo'lsa shu yerdan o'qiladi. */
const href = () =>
  (typeof window !== "undefined" && window.TGAI && window.TGAI.sprite) || SPRITE;

/**
 * Ikonka elementi.
 *
 *   icon("search")                       → bezak (aria-hidden)
 *   icon("bell", { label: t("...") })    → ma'noli (role=img + <title>)
 *
 * @param {string} name  sprite'dagi `<symbol id>`
 * @param {{label?: string, cls?: string, size?: number|string}} [opts]
 * @returns {SVGElement}
 */
export function icon(name, opts = {}) {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", ["icon", opts.cls].filter(Boolean).join(" "));
  if (opts.size) { svg.style.inlineSize = typeof opts.size === "number" ? `${opts.size}px` : opts.size; svg.style.blockSize = svg.style.inlineSize; }
  if (opts.label) {
    svg.setAttribute("role", "img");
    const title = document.createElementNS(NS, "title");
    title.textContent = opts.label;        // textContent — escape shart emas
    svg.appendChild(title);
  } else {
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
  }
  const use = document.createElementNS(NS, "use");
  const u = `${href()}#${name}`;
  use.setAttribute("href", u);
  use.setAttributeNS(XLINK, "xlink:href", u);  // eski Safari uchun
  svg.appendChild(use);
  return svg;
}

/**
 * Shablonlar/`innerHTML` uchun HTML matn. `name` **oq ro'yxat naqshi**
 * bilan cheklanadi (faqat `a-z0-9-`), ya'ni atributdan chiqib ketish
 * mumkin emas.
 */
export function iconHTML(name, opts = {}) {
  const safe = String(name).replace(/[^a-z0-9-]/gi, "");
  const cls = String(opts.cls || "").replace(/[^a-z0-9 _-]/gi, "");
  const u = `${href()}#${safe}`;
  return `<svg class="icon ${cls}" aria-hidden="true" focusable="false"><use href="${u}"></use></svg>`;
}

/** Ikonkali tugma (`aria-label` majburiy — ikonka yolg'iz ma'no tashimaydi). */
export function iconButton(name, label, props = {}) {
  const btn = document.createElement("button");
  btn.type = props.type || "button";
  btn.className = ["icon-btn-v2", props.cls].filter(Boolean).join(" ");
  btn.setAttribute("aria-label", label);
  btn.title = label;
  for (const [k, v] of Object.entries(props.attrs || {})) btn.setAttribute(k, String(v));
  if (props.onclick) btn.addEventListener("click", props.onclick);
  btn.appendChild(icon(name));
  return btn;
}
