/* pages/dashboard.js — `/dashboard` (ui/R4).
 *
 * Maqsad: texnik bo'lmagan odam **10 soniyada** tushunsin. Shuning uchun
 * har son uchta narsa bilan keladi: oddiy tilda yorliq, birlik va bitta
 * kontekst qatori (delta / "n=" / "byudjetdan"). Xom metrika nomlari
 * (`tgai_*`, "p50", "auto-baho (moslik)") sahifada ko'rinmaydi — ular
 * ⓘ tugmasi ostidagi bir gapli izohga ko'chdi.
 *
 * Tuzilma (yuqoridan pastga):
 *   a) davr almashtirgichi (`?days=` — URL holati)
 *   b) 4 ta bosh KPI kartasi (delta, sparkline, byudjet chizig'i)
 *   c) sync salomatligi — `lib/sync-center.js::mountStrip` (yagona manba)
 *   d) ikkita grafik (savollar+xarajat · javob vaqti)
 *   e) modellar · f) eng faol chatlar · g) ko'rib chiqish · h) tizim
 *
 * Qoidalar:
 *   * `innerHTML` **yo'q** — har tugun `lib/dom.js::el()` yoki `svgEl()`.
 *   * Grafiklar qo'lda SVG (`viewBox`), kutubxonasiz; konteyner kengligi
 *     `ResizeObserver` bilan kuzatiladi, CSS tomonda `container-type`.
 *   * Har grafikning `<details>` ichida jadval muqobili bor (a11y).
 */

import { $, $$, el, clear, replace, debounce, ready } from "../lib/dom.js";
import { icon } from "../lib/icons.js";
import { t, has } from "../lib/i18n.js";
import { get } from "../lib/api.js";
import { compact, num, usd, ms, pct, dateTime } from "../lib/fmt.js";
import { peerClass, initials } from "../lib/palette.js";
import { toastError } from "../lib/toast.js";
import { mountStrip } from "../lib/sync-center.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Ruxsat etilgan davrlar — URL'dan kelgan qiymat shulardan biri bo'lishi shart. */
export const PERIODS = [1, 7, 30];
/** KPI sparkline'ida nechta kun ko'rsatiladi. */
export const SPARK_DAYS = 14;

/* ══ Sof yordamchilar (node testlari shularni import qiladi) ═══════════ */

/**
 * O'q uchun "chiroyli" maksimum: 1/2/2.5/5/10 × 10ⁿ.
 * Usiz o'q yorliqlari `3,7k` / `7,4k` kabi o'qib bo'lmaydigan sonlar bo'ladi.
 */
export function niceMax(value) {
  const v = Number(value);
  if (!(v > 0)) return 1;
  const exp = Math.floor(Math.log10(v));
  const base = 10 ** exp;
  const n = v / base;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
  return step * base;
}

const NICE_STEPS = [1, 2, 2.5, 5, 10];

/** Qadamning o'zi ham "chiroyli" bo'lsin: `1 / 2 / 2.5 / 5 × 10ⁿ`. */
function isNiceStep(step) {
  if (!(step > 0)) return false;
  const m = step / 10 ** Math.floor(Math.log10(step));
  return NICE_STEPS.some((n) => Math.abs(m - n) < 1e-9);
}

/** `[0 … top]` oralig'ida `count` ta teng nuqta (o'q yorliqlari uchun). */
export function spread(top, count) {
  const out = [];
  for (let i = 0; i < count; i++) out.push((top * i) / (count - 1));
  return out;
}

/**
 * Y o'qi qiymatlari. Chiziqlar soni **qadam chiroyli chiqadigan** qilib
 * tanlanadi: usiz "0 · 1,7 · 3,3 · 5" kabi kasrli yorliqlar chiqadi va
 * o'qni o'qib bo'lmaydi. `integer` — savollar kabi butun sonli o'qlar uchun.
 */
export function tickValues(max, count = 4, { integer = false } = {}) {
  const top = niceMax(max);
  const fits = (c) => {
    const step = top / (c - 1);
    return integer ? Number.isInteger(step) : isNiceStep(step);
  };
  const chosen = [count, 5, 6, 3, 2].find((c) => c >= 2 && fits(c)) || count;
  return spread(top, chosen);
}

/**
 * X o'qi yorliqlari **siyraklashtiriladi**: ustunlar orasidagi masofa
 * yorliq kengligidan kichik bo'lsa matnlar bir-birining ustiga chiqadi.
 * `maxLabels = floor(width / 64)`; "oxirgisi majburiy" mantiqi **yo'q**
 * (u qo'shni yorliq bilan to'qnashardi — `test_web.py` M7).
 */
export function tickIndices(n, maxLabels) {
  const count = Math.max(2, Math.floor(maxLabels));
  if (n <= 0) return [];
  const step = Math.max(1, Math.ceil(n / count));
  const out = [];
  for (let i = 0; i < n; i += step) out.push(i);
  return out;
}

/**
 * Nisbiy o'zgarish: `0.25` = +25%. Oldingi davr nol bo'lsa — `null`
 * ("cheksiz o'sish" degan son ko'rsatilmaydi, matn bilan aytiladi).
 */
export function deltaPct(current, previous) {
  const cur = Number(current || 0);
  const prev = Number(previous || 0);
  if (!prev) return null;
  return (cur - prev) / prev;
}

/**
 * Nuqtalardan `path` `d` atributi. `null` element — ma'lumot yo'q kun:
 * chiziq uziladi (nolga tushgandek ko'rinmasin).
 */
export function linePath(points) {
  let out = "";
  let pen = false;
  for (const p of points) {
    if (!p) { pen = false; continue; }
    out += `${pen ? "L" : "M"}${p.x.toFixed(2)},${p.y.toFixed(2)}`;
    pen = true;
  }
  return out;
}

/* ══ SVG quruvchi ═════════════════════════════════════════════════════ */

function svgEl(tag, attrs, children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    node.setAttribute(k, String(v));
  }
  for (const c of Array.isArray(children) ? children : [children]) {
    if (c == null || c === false) continue;
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const withTitle = (node, text) => (node.appendChild(svgEl("title", null, text)), node);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ══ Holat ════════════════════════════════════════════════════════════ */

const state = {
  days: 30,
  overview: null,
  sync: null,
  redraw: null,   // grafiklarni qayta chizish (tema/o'lcham o'zgarsa)
};

function readUrl() {
  const raw = Number(new URLSearchParams(location.search).get("days"));
  state.days = PERIODS.includes(raw) ? raw : 30;
}

function writeUrl(push) {
  const url = `${location.pathname}?days=${state.days}`;
  if (url === `${location.pathname}${location.search}`) return;
  if (push) history.pushState({ dash: true }, "", url);
  else history.replaceState({ dash: true }, "", url);
}

/* ══ Umumiy bo'laklar ═════════════════════════════════════════════════ */

/** Bir gapli izoh — texnik bo'lmagan foydalanuvchi uchun (ⓘ). */
function hint(key) {
  if (!has(key)) return null;
  const text = t(key);
  const pop = el("p.hint-pop", { hidden: true }, text);
  const btn = el("button.hint-btn", {
    type: "button",
    "aria-label": t("web.dash.what_is_this"),
    "aria-expanded": "false",
    title: text,
    onclick: () => {
      const open = pop.hidden;
      pop.hidden = !open;
      btn.setAttribute("aria-expanded", String(open));
    },
  }, icon("info"));
  return el("span.hint", null, [btn, pop]);
}

/** Sarlavhalardagi `data-hint` slotlarini to'ldiradi (shablon → modul). */
function mountHints(root = document) {
  for (const slot of $$(".hint-slot", root)) {
    if (slot.childElementCount) continue;
    const node = hint(slot.dataset.hint || "");
    if (node) slot.appendChild(node);
  }
}

/** Delta qatori: o'q + foiz + "oldingi N kunga nisbatan". */
function deltaLine(current, previous, { lowerIsBetter = false } = {}) {
  const d = deltaPct(current, previous);
  const period = state.days === 1 ? t("web.dash.vs_prev_day") : t("web.dash.vs_prev", { n: state.days });
  if (d == null) {
    return el("span.kpi-ctx.is-flat", null, t("web.dash.no_prev"));
  }
  const up = d > 0;
  const flat = Math.abs(d) < 0.005;
  const good = flat ? null : (lowerIsBetter ? !up : up);
  const cls = flat ? "is-flat" : good ? "is-good" : "is-bad";
  return el(`span.kpi-ctx.${cls}`, null, [
    flat ? null : el(`span.delta-arrow${up ? "" : ".is-down"}`, { "aria-hidden": "true" }, icon("arrow-up")),
    el("b", null, flat ? t("web.dash.no_change") : `${up ? "+" : "−"}${pct(Math.abs(d))}`),
    " ",
    period,
  ]);
}

/** 14 kunlik mayda chiziq — trend "his qilinsin" (o'qsiz, yorliqsiz). */
function sparkline(values, color) {
  const W = 88, H = 24, pad = 2;
  const vals = values.slice(-SPARK_DAYS);
  if (vals.length < 2) return null;
  const max = Math.max(1, ...vals);
  const step = (W - pad * 2) / (vals.length - 1);
  const pts = vals.map((v, i) => ({
    x: pad + i * step,
    y: H - pad - ((H - pad * 2) * (Number(v) || 0)) / max,
  }));
  return svgEl("svg", {
    class: "kpi-spark", viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true",
    preserveAspectRatio: "none", focusable: "false",
  }, [
    svgEl("path", { d: linePath(pts), fill: "none", stroke: color, "stroke-width": 1.5, "stroke-linejoin": "round" }),
  ]);
}

/** KPI kartasi: yorliq + ⓘ · katta son + birlik · kontekst qatori. */
function kpiCard({ iconName, label, hintKey, value, unit, context, extra }) {
  return el("article.kpi", null, [
    el("header.kpi-head", null, [
      el("span.kpi-ico", { "aria-hidden": "true" }, icon(iconName)),
      el("h3.kpi-label", null, label),
      el("span.spacer"),
      hint(hintKey),
    ]),
    el("p.kpi-value", null, [
      el("b", null, value),
      unit ? el("span.kpi-unit", null, unit) : null,
    ]),
    context || null,
    extra || null,
  ]);
}

/** Byudjet chizig'i: "X / Y kunlik byudjet" (yoki "byudjet o'rnatilmagan"). */
function budgetBar(budget) {
  const total = budget && budget.daily_usd;
  const used = Number((budget && budget.used_usd) || 0);
  if (!total) return el("span.kpi-ctx.is-flat", null, t("web.dash.budget_none"));
  const share = Math.max(0, Math.min(1, used / total));
  return el("div.kpi-budget", null, [
    el("div.meter", {
      role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
      "aria-valuenow": String(Math.round(share * 100)),
      "aria-label": t("web.dash.budget_of", { used: usd(used), total: usd(total) }),
    }, el(`i${share >= 0.9 ? ".is-hot" : ""}`, { style: { inlineSize: `${share * 100}%` } })),
    el("span.kpi-ctx", null, t("web.dash.budget_of", { used: usd(used), total: usd(total) })),
  ]);
}

/** "Yorliq → qiymat" juftligi (sifat kartasidagi ikki qator). */
function kpiKv(label, value) {
  return el("span.kpi-ctx.kpi-kv.is-flat", null, [
    el("b", null, label),
    el("span", null, value),
  ]);
}

function emptyState(iconName, title, text, cta) {
  return el("div.empty-v2", null, [
    el("span.empty-ico", null, icon(iconName)),
    el("h2", null, title),
    text ? el("p", null, text) : null,
    cta ? el("div.empty-actions", null, [el("a.btn.primary", { href: cta.href }, cta.label)]) : null,
  ]);
}

/** Skeleton — har bir yuklash uchun (bo'sh joy "sakramasin"). */
function skeletonLines(n, host) {
  replace(host, Array.from({ length: n }, () => el("div.skeleton-row", null, [
    el("span.skeleton.circle"),
    el("div.skeleton-lines", null, [el("span.skeleton.l1"), el("span.skeleton.l2")]),
  ])));
}

function skeletonKpis(host) {
  replace(host, Array.from({ length: 4 }, () => el("article.kpi.is-loading", null, [
    el("span.skeleton.l2"), el("span.skeleton.l1"), el("span.skeleton.l2"),
  ])));
}

/* ══ (b) KPI kartalari ════════════════════════════════════════════════ */

function paintKpis(ov) {
  const host = $("#dash-kpis");
  if (!host) return;
  const series = ov.series || ov.daily || [];
  const totals = ov.totals || {};
  const prev = ov.prev || {};
  const lat = ov.latency || {};
  const q = ov.quality || {};
  const questions = totals.questions ?? totals.requests ?? 0;

  if (!questions && !(prev.questions || 0)) {
    replace(host, el("div.kpi-empty", null, emptyState(
      "sparkles",
      t("web.dash.empty_title"),
      t("web.dash.empty_text"),
      { href: (window.TGAI && window.TGAI.routes && window.TGAI.routes.ask) || "/ask", label: t("web.dash.empty_cta") },
    )));
    return;
  }

  const upShare = q.rated ? q.up / q.rated : null;
  const prevShare = prev.rated ? prev.up / prev.rated : null;

  replace(host, [
    kpiCard({
      iconName: "sparkles",
      label: t("web.dash.kpi_questions"),
      hintKey: "web.dash.kpi_questions_hint",
      value: num(questions),
      unit: t("web.dash.unit_questions"),
      context: deltaLine(questions, prev.questions),
      extra: sparkline(series.map((d) => d.questions ?? d.requests ?? 0), cssVar("--series-1")),
    }),
    kpiCard({
      iconName: "bolt",
      label: t("web.dash.kpi_cost"),
      hintKey: "web.dash.kpi_cost_hint",
      value: usd(totals.cost_usd || 0),
      unit: "",
      context: deltaLine(totals.cost_usd, prev.cost_usd, { lowerIsBetter: true }),
      extra: budgetBar(ov.budget),
    }),
    kpiCard({
      iconName: "clock",
      label: t("web.dash.kpi_speed"),
      hintKey: "web.dash.kpi_speed_hint",
      value: lat.p50_ms == null ? "—" : ms(lat.p50_ms),
      unit: "",
      context: deltaLine(lat.p50_ms, prev.p50_ms, { lowerIsBetter: true }),
      extra: el("span.kpi-ctx.is-flat", null,
        lat.p95_ms == null ? t("web.dash.no_rating") : t("web.dash.slowest", { v: ms(lat.p95_ms) })),
    }),
    kpiCard({
      iconName: "thumbs-up",
      label: t("web.dash.kpi_quality"),
      hintKey: "web.dash.kpi_quality_hint",
      value: upShare == null ? "—" : pct(upShare),
      unit: "",
      // Ikki qator: "kim baholagan" → "qancha". Ajratgich belgisi yo'q —
      // tor kartada u qator boshiga tushib qolardi.
      context: kpiKv(
        t("web.dash.user_rating"),
        q.rated ? t("web.dash.rated_n", { n: num(q.rated) }) : t("web.dash.no_rating"),
      ),
      extra: kpiKv(
        t("web.dash.auto_rating"),
        q.auto_n
          ? t("web.dash.auto_score", { v: String(q.auto_avg ?? "—"), n: num(q.auto_n) })
          : t("web.dash.auto_none_plain"),
      ),
    }),
  ]);
  // Delta sifat kartasi uchun alohida qator sifatida qo'shiladi (n= bilan
  // birga ikkita kontekst chizig'i chiqmasin — kartani uzaytiradi).
  if (prevShare != null && upShare != null) {
    const card = host.lastElementChild;
    if (card) card.appendChild(deltaLine(upShare, prevShare));
  }
  mountHints(host);
}

/* ══ (d) Grafiklar ════════════════════════════════════════════════════ */

const chartGeom = (width) => ({
  W: Math.max(280, Math.round(width || 560)),
  H: width && width < 420 ? 170 : 210,
  padL: 40, padR: 44, padT: 14, padB: 26,
});

/** Grafik atrofidagi umumiy qism: o'qlar, panjara, jadval muqobili. */
function chartFrame(host, { W, H, padL, padR, padT, padB }, ticks, fmtTick) {
  clear(host);
  const svg = svgEl("svg", {
    class: "chart-svg", viewBox: `0 0 ${W} ${H}`, role: "img", focusable: "false",
    preserveAspectRatio: "xMidYMid meet",
  });
  const plot = { x0: padL, x1: W - padR, y0: padT, y1: H - padB };
  const border = cssVar("--border");
  const muted = cssVar("--muted");
  for (let i = 0; i < ticks.length; i++) {
    const y = plot.y1 - ((plot.y1 - plot.y0) * i) / (ticks.length - 1);
    svg.appendChild(svgEl("line", {
      x1: plot.x0, x2: plot.x1, y1: y, y2: y, stroke: border, "stroke-width": 1,
    }));
    svg.appendChild(svgEl("text", {
      x: plot.x0 - 6, y: y + 4, "text-anchor": "end", "font-size": 10, fill: muted,
    }, fmtTick(ticks[i])));
  }
  host.appendChild(svg);
  return { svg, plot };
}

function xLabels(svg, days, plot, H) {
  const muted = cssVar("--muted");
  const slot = (plot.x1 - plot.x0) / Math.max(1, days.length);
  for (const i of tickIndices(days.length, (plot.x1 - plot.x0) / 64)) {
    const d = days[i];
    if (!d) continue;
    svg.appendChild(svgEl("text", {
      x: plot.x0 + i * slot + slot / 2, y: H - 8,
      "text-anchor": "middle", "font-size": 10, fill: muted,
    }, String(d.day || "").slice(5).replace("-", ".")));
  }
}

/** A11y muqobili: har grafik ostida yig'ilgan jadval (`<details>`). */
function chartTable(host, days, cols) {
  const rows = days.filter((d) => cols.some((c) => c.get(d)));
  const table = el("table", null, [
    el("thead", null, el("tr", null, [
      el("th", null, t("web.dash.day")),
      ...cols.map((c) => el("th", null, c.label)),
    ])),
    el("tbody", null, rows.map((d) => el("tr", null, [
      el("td", null, d.day),
      ...cols.map((c) => el("td.num", null, c.fmt(c.get(d)))),
    ]))),
  ]);
  host.appendChild(el("details.chart-table", null, [
    el("summary", null, t("web.dash.table_view")),
    rows.length ? table : el("p.muted.small", null, t("web.dash.no_data")),
  ]));
}

function legend(items) {
  return el("div.chart-legend", null, items.map((it) =>
    el("span", null, [el("i.sw", { style: { background: it.color } }), it.label])));
}

/** "Savollar va xarajat kunlar bo'yicha" — ustunlar + chiziq (ikki o'q). */
function drawVolume(host, days) {
  const geom = chartGeom(host.clientWidth);
  const { H, padR } = geom;
  const qMax = niceMax(Math.max(1, ...days.map((d) => d.questions ?? d.requests ?? 0)));
  const cMax = niceMax(Math.max(0.0001, ...days.map((d) => d.cost_usd || 0)));
  const hasCost = days.some((d) => (d.cost_usd || 0) > 0);
  host.appendChild(legend([
    { color: cssVar("--series-1"), label: t("web.dash.legend_questions") },
    ...(hasCost ? [{ color: cssVar("--series-2"), label: t("web.dash.legend_cost") }] : []),
  ]));
  const box = el("div.chart-canvas");
  host.appendChild(box);
  const qTicks = tickValues(qMax, 4, { integer: true });
  const { svg, plot } = chartFrame(box, geom, qTicks, (v) => compact(v));

  const ih = plot.y1 - plot.y0;
  const slot = (plot.x1 - plot.x0) / Math.max(1, days.length);
  const bw = Math.max(2, Math.min(20, slot * 0.62));
  const c1 = cssVar("--series-1"), c2 = cssVar("--series-2");

  days.forEach((d, i) => {
    const v = d.questions ?? d.requests ?? 0;
    const cx = plot.x0 + i * slot + slot / 2;
    const tip = `${d.day}\n${t("web.dash.legend_questions")}: ${num(v)}\n${t("web.dash.legend_cost")}: ${usd(d.cost_usd || 0)}`;
    if (v > 0) {
      const h = (ih * v) / qMax;
      svg.appendChild(withTitle(svgEl("rect", {
        x: cx - bw / 2, y: plot.y1 - h, width: bw, height: Math.max(h, 1.5), rx: 3, fill: c1,
      }), tip));
    }
    // Ko'rinmas "hover" zonasi — nol kunlarda ham tooltip bo'lsin.
    svg.appendChild(withTitle(svgEl("rect", {
      x: plot.x0 + i * slot, y: plot.y0, width: slot, height: ih, fill: "transparent",
    }), tip));
  });

  if (hasCost) {
    const pts = days.map((d, i) => ({
      x: plot.x0 + i * slot + slot / 2,
      y: plot.y1 - (ih * (d.cost_usd || 0)) / cMax,
    }));
    svg.appendChild(svgEl("path", {
      d: linePath(pts), fill: "none", stroke: c2, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
    // O'ng o'q — xarajat shkalasi (chapdagisi savollar uchun). Nuqtalar
    // soni chap o'q bilan **bir xil**: aks holda yorliqlar panjara
    // chiziqlariga tushmay qoladi.
    const muted = cssVar("--muted");
    spread(cMax, qTicks.length).forEach((v, i, arr) => {
      const y = plot.y1 - (ih * i) / (arr.length - 1);
      svg.appendChild(svgEl("text", {
        x: plot.x1 + 6, y: y + 4, "text-anchor": "start", "font-size": 10, fill: muted,
      }, usd(v)));
    });
  } else {
    svg.setAttribute("viewBox", `0 0 ${geom.W - padR + 10} ${H}`);
  }

  xLabels(svg, days, plot, H);
  svg.appendChild(svgEl("title", null, t("web.dash.chart_volume")));
  chartTable(host, days, [
    { label: t("web.dash.legend_questions"), get: (d) => d.questions ?? d.requests ?? 0, fmt: (v) => num(v) },
    { label: t("web.dash.legend_cost"), get: (d) => d.cost_usd || 0, fmt: (v) => usd(v) },
  ]);
}

/** "Javob vaqti" — kunlik median va eng sekin 5%. */
function drawLatency(host, days) {
  const geom = chartGeom(host.clientWidth);
  const { H } = geom;
  const all = days.flatMap((d) => [d.p50_ms, d.p95_ms]).filter((v) => v != null);
  if (!all.length) {
    clear(host);
    host.appendChild(emptyState("clock", t("web.dash.no_data"), t("web.dash.latency_empty")));
    return;
  }
  const max = niceMax(Math.max(...all));
  host.appendChild(legend([
    { color: cssVar("--series-1"), label: t("web.dash.legend_p50") },
    { color: cssVar("--series-3"), label: t("web.dash.legend_p95") },
  ]));
  const box = el("div.chart-canvas");
  host.appendChild(box);
  const { svg, plot } = chartFrame(box, { ...geom, padR: 14 }, tickValues(max), (v) => ms(v));
  const ih = plot.y1 - plot.y0;
  const slot = (plot.x1 - plot.x0) / Math.max(1, days.length);
  const at = (v, i) => (v == null ? null : {
    x: plot.x0 + i * slot + slot / 2,
    y: plot.y1 - (ih * v) / max,
  });

  for (const [key, color] of [["p95_ms", cssVar("--series-3")], ["p50_ms", cssVar("--series-1")]]) {
    const pts = days.map((d, i) => at(d[key], i));
    svg.appendChild(svgEl("path", {
      d: linePath(pts), fill: "none", stroke: color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
    for (let i = 0; i < pts.length; i++) {
      if (!pts[i]) continue;
      svg.appendChild(withTitle(
        svgEl("circle", { cx: pts[i].x, cy: pts[i].y, r: 2.5, fill: color }),
        `${days[i].day}\n${key === "p50_ms" ? t("web.dash.legend_p50") : t("web.dash.legend_p95")}: ${ms(days[i][key])}`,
      ));
    }
  }
  xLabels(svg, days, plot, H);
  svg.appendChild(svgEl("title", null, t("web.dash.chart_latency")));
  chartTable(host, days, [
    { label: t("web.dash.legend_p50"), get: (d) => d.p50_ms, fmt: (v) => ms(v) },
    { label: t("web.dash.legend_p95"), get: (d) => d.p95_ms, fmt: (v) => ms(v) },
  ]);
}

function paintCharts(ov) {
  const days = ov.series || ov.daily || [];
  const draw = () => {
    const a = $("#chart-volume"), b = $("#chart-latency");
    if (a) { clear(a); drawVolume(a, days); }
    if (b) { clear(b); drawLatency(b, days); }
  };
  state.redraw = draw;
  draw();
}

/* ══ (e) Modellar ═════════════════════════════════════════════════════ */

function paintModels(ov) {
  const host = $("#dash-models");
  if (!host) return;
  const rows = ov.models || [];
  if (!rows.length) {
    replace(host, el("p.muted.small", null, t("web.dash.no_data")));
    return;
  }
  const total = rows.reduce((a, r) => a + (r.cost_usd || 0), 0);
  // Mobilda jadval emas — qatorlar (640px dan pastda CSS kartaga o'tkazadi).
  replace(host, el("ul.model-list", null, rows.map((r) => {
    const share = total > 0 ? (r.cost_usd || 0) / total : 0;
    return el("li.model-row", null, [
      el("div.model-main", null, [
        el("span.model-name", { title: r.model || "" }, r.model || "—"),
        el("span.model-sub", null, r.provider || ""),
      ]),
      el("div.model-nums", null, [
        el("span", null, [el("b", null, num(r.questions ?? r.requests ?? 0)), " ", el("small", null, t("web.dash.unit_questions"))]),
        el("span", null, [el("b", null, usd(r.cost_usd || 0))]),
        el("span", null, [el("b", null, (r.avg_ms ?? r.latency_avg_ms) ? ms(r.avg_ms ?? r.latency_avg_ms) : "—")]),
      ]),
      el("div.meter.sm", {
        role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
        "aria-valuenow": String(Math.round(share * 100)),
        "aria-label": t("web.dash.share_of_cost", { p: pct(share) }),
        title: t("web.dash.share_of_cost", { p: pct(share) }),
      }, el("i", { style: { inlineSize: `${share * 100}%` } })),
    ]);
  })));
}

/* ══ (f) Eng faol chatlar ═════════════════════════════════════════════ */

function chatAvatar(row) {
  const node = el("span.avatar.sm", {
    class: peerClass(row.chat_id || row.peer_id || 0), "aria-hidden": "true",
  });
  if (row.avatar_url) {
    const img = el("img", {
      src: row.avatar_url, alt: "", loading: "lazy", decoding: "async",
      onerror: () => { img.remove(); node.appendChild(document.createTextNode(initials(row.title || "?"))); },
    });
    node.appendChild(img);
  } else {
    node.appendChild(document.createTextNode(initials(row.title || "?")));
  }
  return node;
}

function paintTopChats(ov) {
  const host = $("#dash-chats");
  if (!host) return;
  const rows = ov.top_chats || [];
  if (!rows.length) {
    replace(host, el("p.muted.small", null, t("web.dash.chats_empty")));
    return;
  }
  const max = Math.max(1, ...rows.map((r) => r.questions ?? r.requests ?? 0));
  replace(host, el("div.chat-list", null, rows.map((r) => {
    const n = r.questions ?? r.requests ?? 0;
    const body = el("div.row-body", null, [
      el("div.row-title", null, r.title || "—"),
      el("div.meter.sm", { "aria-hidden": "true" }, el("i", { style: { inlineSize: `${(n / max) * 100}%` } })),
    ]);
    const aside = el("div.row-aside", null, [
      el("b", null, num(n)), " ", t("web.dash.unit_questions"),
    ]);
    const kids = [chatAvatar(r), body, aside];
    return r.chat_id
      ? el("a.row-item", { href: `/chats/${r.chat_id}` }, kids)
      : el("div.row-item", null, kids);
  })));
}

/* ══ (g) Ko'rib chiqish kerak ═════════════════════════════════════════ */

function paintReview(ov) {
  const host = $("#dash-review");
  if (!host) return;
  const rows = ov.review || [];
  if (!rows.length) {
    replace(host, el("p.muted.small", null, t("web.dash.review_empty")));
    return;
  }
  replace(host, el("div.review-list", null, rows.map((r) => {
    const badges = [];
    if (r.rating === -1) {
      badges.push(el("span.pill.err", null, [icon("thumbs-down"), t("web.dash.thumb_down")]));
    }
    if (r.auto_score != null) {
      badges.push(el("span.pill", null, t("web.dash.auto_score_short", { v: String(r.auto_score) })));
    }
    return el("a.review-card", { href: `/ask/${r.conversation_id}` }, [
      el("div.review-head", null, [...badges, el("span.spacer"), el("time", null, dateTime(r.created_at))]),
      el("p.review-q", null, r.question || r.excerpt || ""),
      el("span.review-open", null, [t("web.dash.review_open"), icon("chevron-right")]),
    ]);
  })));
}

/* ══ (h) Tizim ════════════════════════════════════════════════════════ */

function kvRow(key, value) {
  return el("div.kv-row", null, [el("span", null, key), el("b", null, value)]);
}

function paintSystem(sys, ing) {
  const host = $("#dash-system");
  if (!host) return;
  const rows = [];
  const okNode = (o) => (o && o.ok
    ? el("span.pill.ok", null, [icon("check"), `${t("web.dash.ok")} · ${o.ms} ms`])
    : el("span.pill.err", null, [icon("alert-triangle"), t("web.dash.fail")]));
  if (sys) {
    rows.push(el("div.kv-row", null, [el("span", null, t("web.dash.db")), okNode(sys.db)]));
    rows.push(el("div.kv-row", null, [el("span", null, t("web.dash.redis")), okNode(sys.redis)]));
    const b = sys.budget || {};
    rows.push(kvRow(t("web.dash.budget"),
      `${usd(b.cost_usd || 0)}${b.cost_limit ? ` / ${usd(b.cost_limit)}` : ""} · ${compact(b.tokens || 0)} tok`));
    const hb = sys.heartbeats && !sys.heartbeats._error ? Object.entries(sys.heartbeats) : [];
    rows.push(el("div.kv-row", null, [
      el("span", null, t("web.dash.heartbeats")),
      hb.length
        ? el("div.hb-list", null, hb.map(([k, v]) => el("span", null, `${k}: ${dateTime((v && v.at) || "")}`)))
        : el("b", null, t("web.dash.no_heartbeats")),
    ]));
    rows.push(kvRow(t("web.dash.version"), `${sys.version || ""} · ${sys.env || ""}`));
  }
  if (ing) {
    rows.push(kvRow(t("web.dash.ing_chats"), `${num(ing.synced_chats)} / ${num(ing.chats)}`));
    rows.push(kvRow(t("web.dash.ing_messages"), compact(ing.messages)));
    rows.push(kvRow(t("web.dash.ing_embedded"),
      `${compact(ing.embedded)} (${ing.messages ? Math.round((100 * ing.embedded) / ing.messages) : 0}%)`));
    rows.push(kvRow(t("web.dash.ing_snapshots"), compact(ing.snapshots_7d)));
  }
  replace(host, el("div.kv", null, rows));
}

/* ══ (c) Sync ═════════════════════════════════════════════════════════ */

async function mountSync() {
  const card = $("#dash-sync-card");
  const box = $("#dash-sync");
  if (!card || !box) return;
  let me = null;
  try { me = await get("/api/me"); } catch { return; }
  const account = (me.accounts || []).find((a) => a.status === "active") || (me.accounts || [])[0];
  if (!account) return;
  card.hidden = false;
  state.sync = mountStrip(box, account.id);
  window.addEventListener("pagehide", () => state.sync && state.sync.destroy(), { once: true });
}

/* ══ Yuklash ══════════════════════════════════════════════════════════ */

function paintPeriod() {
  for (const btn of $$("#dash-period button")) {
    btn.setAttribute("aria-selected", String(Number(btn.dataset.days) === state.days));
  }
}

async function loadOverview() {
  skeletonKpis($("#dash-kpis"));
  for (const id of ["#dash-models", "#dash-chats", "#dash-review"]) {
    const node = $(id);
    if (node) skeletonLines(3, node);
  }
  let ov;
  try {
    ov = await get(`/api/stats/overview?days=${state.days}`);
  } catch (err) {
    toastError(err);
    replace($("#dash-kpis"), el("p.muted.small", null, t("error.generic")));
    return;
  }
  state.overview = ov;
  paintKpis(ov);
  paintCharts(ov);
  paintModels(ov);
  paintTopChats(ov);
  paintReview(ov);
}

async function loadSystem() {
  const [sys, ing] = await Promise.allSettled([
    get("/api/stats/system"),
    get("/api/stats/ingestion"),
  ]);
  paintSystem(
    sys.status === "fulfilled" ? sys.value : null,
    ing.status === "fulfilled" ? ing.value : null,
  );
}

function bind() {
  const seg = $("#dash-period");
  if (seg) {
    seg.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-days]");
      if (!btn) return;
      const days = Number(btn.dataset.days);
      if (!PERIODS.includes(days) || days === state.days) return;
      state.days = days;
      paintPeriod();
      writeUrl(true);
      loadOverview();
    });
  }
  window.addEventListener("popstate", () => {
    readUrl();
    paintPeriod();
    loadOverview();
  });
  // Grafiklar konteyner kengligiga bog'liq — o'lcham va tema o'zgarsa qayta chizamiz.
  const redraw = debounce(() => state.redraw && state.redraw(), 150);
  if (window.ResizeObserver) {
    // **Faqat kenglik** o'zgarganda: qayta chizish balandlikni o'zgartiradi,
    // balandlik esa yana kuzatuvchini uyg'otadi — cheksiz sikl bo'lardi.
    let lastW = 0;
    const ro = new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w === lastW) return;
      lastW = w;
      redraw();
    });
    const node = $("#chart-volume");
    if (node) ro.observe(node);
  } else {
    window.addEventListener("resize", redraw);
  }
  window.addEventListener("tgai:theme", redraw);
}

function main() {
  readUrl();
  writeUrl(false);
  paintPeriod();
  mountHints();
  bind();
  loadOverview();
  mountSync();
  const sysBox = $("#dash-system-box");
  if (sysBox) sysBox.addEventListener("toggle", () => { if (sysBox.open) loadSystem(); }, { once: true });
}

/** Kirish nuqtasi. Guard — modul node testlarida ham import qilinadi
 *  (sof funksiyalar: `niceMax`, `tickIndices`, `linePath`), u yerda DOM yo'q. */
export function initDashboard() { main(); }

if (typeof document !== "undefined") ready(main);
