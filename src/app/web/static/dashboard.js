/* Dashboard — SVG chart'lar (kutubxonasiz), tooltip, jadval ko'rinishi. */
(function () {
  const S = window.TGAI.strings;
  const t = (k, vars) => {
    let v = S[k] || k;
    if (vars) for (const [key, val] of Object.entries(vars)) v = v.replaceAll(`{${key}}`, val);
    return v;
  };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n) => n == null ? "—" : n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e4 ? (n / 1e3).toFixed(0) + "k" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(Math.round(n));
  const money = (v) => v == null ? "—" : "$" + (v < 1 ? v.toFixed(3) : v.toFixed(2));
  const ms = (v) => v == null || v === 0 ? "—" : v >= 1000 ? (v / 1000).toFixed(1) + " s" : v + " ms";
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  async function api(path) {
    const res = await fetch(path, { credentials: "same-origin", headers: { "X-Requested-With": "fetch" } });
    if (res.status === 401) { window.location.href = "/login"; throw new Error("unauthorized"); }
    if (!res.ok) throw new Error(t("error.generic"));
    return res.json();
  }

  // ── umumiy SVG bar chart (kunlik) ────────────────────────────────────────
  // series: [{key, color, label}] — stacked=true bo'lsa ustma-ust
  function barChart(el, days, series, { stacked = false, valueFmt = fmt } = {}) {
    el.innerHTML = "";
    const W = el.clientWidth || 520, H = 180, padL = 36, padB = 22, padT = 8;
    const iw = W - padL - 6, ih = H - padB - padT;
    const n = days.length;
    const totals = days.map((d) => stacked ? series.reduce((a, s) => a + (d[s.key] || 0), 0) : Math.max(...series.map((s) => d[s.key] || 0)));
    const max = Math.max(1, ...totals);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%"); svg.setAttribute("height", H);
    svg.setAttribute("role", "img");
    const g = (tag, attrs) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
    // gridlines (recessive)
    for (let i = 0; i <= 3; i++) {
      const y = padT + ih - (ih * i) / 3;
      svg.appendChild(g("line", { x1: padL, x2: W - 6, y1: y, y2: y, stroke: cssVar("--border"), "stroke-width": 1 }));
      const lbl = g("text", { x: padL - 6, y: y + 4, "text-anchor": "end", "font-size": 10, fill: cssVar("--muted") });
      lbl.textContent = fmt((max * i) / 3); svg.appendChild(lbl);
    }
    const slot = iw / n, bw = Math.max(2, Math.min(18, slot * 0.7));
    days.forEach((d, i) => {
      const x = padL + i * slot + (slot - bw) / 2;
      let yBase = padT + ih;
      const parts = [];
      series.forEach((s, si) => {
        const v = d[s.key] || 0;
        if (!v) return;
        const h = (ih * v) / max;
        let bx = x, w = bw;
        if (!stacked && series.length > 1) { w = bw / series.length - 1; bx = x + si * (bw / series.length); yBase = padT + ih; }
        const y = yBase - h;
        const r = g("rect", { x: bx, y, width: w, height: Math.max(h, 1), rx: 3, fill: s.color });
        r.appendChild(g("title", {})).textContent = `${d.day}\n${series.map((ss) => `${ss.label}: ${valueFmt(d[ss.key] || 0)}`).join("\n")}`;
        svg.appendChild(r);
        // 2px surface gap between stacked segments
        if (stacked) yBase = y - 2;
        parts.push(v);
      });
      if (n <= 31 && (i % Math.ceil(n / 8) === 0 || i === n - 1)) {
        const tx = g("text", { x: x + bw / 2, y: H - 6, "text-anchor": "middle", "font-size": 10, fill: cssVar("--muted") });
        tx.textContent = d.day.slice(5); svg.appendChild(tx);
      }
    });
    el.appendChild(svg);
    // jadval ko'rinishi (a11y)
    const det = document.createElement("details"); det.className = "chart-table";
    det.innerHTML = `<summary>${esc(t("web.dash.table_view"))}</summary><table><tr><th>${esc(t("web.dash.day"))}</th>${series.map((s) => `<th>${esc(s.label)}</th>`).join("")}</tr>${days.filter((d) => series.some((s) => d[s.key])).map((d) => `<tr><td>${d.day}</td>${series.map((s) => `<td>${valueFmt(d[s.key] || 0)}</td>`).join("")}</tr>`).join("")}</table>`;
    el.appendChild(det);
  }

  function table(el, cols, rows) {
    if (!rows.length) { el.innerHTML = `<tr><td class="muted">${esc(t("web.dash.no_data"))}</td></tr>`; return; }
    el.innerHTML = `<tr>${cols.map((c) => `<th>${esc(c.label)}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr>${cols.map((c) => `<td class="${c.num ? "num" : ""}">${esc(c.fmt ? c.fmt(r[c.key], r) : r[c.key] ?? "—")}</td>`).join("")}</tr>`).join("");
  }

  const stars = (v) => v == null ? "—" : "★".repeat(Math.round(v)) + "☆".repeat(5 - Math.round(v)) + ` ${Number(v).toFixed(1)}`;

  let last = null;
  function drawCharts(ov) {
    const c1 = cssVar("--series-1"), c2 = cssVar("--series-2"), good = cssVar("--good"), bad = cssVar("--bad");
    barChart($("chart-requests"), ov.daily, [{ key: "requests", color: c1, label: t("web.dash.requests") }]);
    barChart($("chart-tokens"), ov.daily, [{ key: "tokens_in", color: c1, label: t("web.dash.tokens_in") }, { key: "tokens_out", color: c2, label: t("web.dash.tokens_out") }], { stacked: true });
    barChart($("chart-ratings"), ov.daily, [{ key: "up", color: good, label: "👍" }, { key: "down", color: bad, label: "👎" }], { stacked: false });
  }
  async function load() {
    const days = +$("days").value;
    const [ov, ing] = await Promise.all([api(`/api/stats/overview?days=${days}`), api("/api/stats/ingestion")]);
    last = ov;
    const T = ov.totals;
    const set = (k, v) => { const e = document.querySelector(`[data-k="${k}"]`); if (e) e.textContent = v; };
    set("requests", fmt(T.requests));
    set("tokens", fmt(T.tokens_in + T.tokens_out));
    set("tokens_split", `${fmt(T.tokens_in)} in · ${fmt(T.tokens_out)} out`);
    set("cost", money(T.cost_usd));
    set("latency", ms(T.latency_p50_ms));
    set("latency_sub", `${t("web.dash.avg")} ${ms(T.latency_avg_ms)}`);
    set("satisfaction", T.satisfaction == null ? "—" : T.satisfaction + "%");
    set("satisfaction_sub", `👍 ${T.up} · 👎 ${T.down} · ${t("web.dash.rated", { n: T.rated, total: T.requests })}`);
    set("auto", T.auto_relevance == null ? "—" : `${T.auto_relevance.toFixed(1)} / 5`);
    set("auto_sub", T.auto_relevance == null ? t("web.dash.auto_none") : `${t("web.dash.usefulness")} ${T.auto_usefulness?.toFixed(1)} · ${t("web.dash.ungrounded")} ${T.ungrounded}/${T.auto_evaluated}`);

    drawCharts(ov);

    $("ingestion").innerHTML = [
      [t("web.dash.ing_accounts"), ing.accounts],
      [t("web.dash.ing_chats"), `${ing.synced_chats} / ${ing.chats}`],
      [t("web.dash.ing_messages"), fmt(ing.messages)],
      [t("web.dash.ing_embedded"), `${fmt(ing.embedded)} (${ing.messages ? Math.round(100 * ing.embedded / ing.messages) : 0}%)`],
      [t("web.dash.ing_snapshots"), fmt(ing.snapshots_7d)],
      [t("web.dash.ing_running"), ing.running],
      [t("web.dash.actions"), Object.keys(ov.actions || {}).length ? Object.entries(ov.actions).map(([k, v]) => `${k}: ${v}`).join(" · ") : "—"],
    ].map(([k, v]) => `<div class="kv-row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");

    table($("tbl-models"), [
      { key: "provider", label: t("web.dash.provider") },
      { key: "model", label: t("web.dash.model") },
      { key: "requests", label: t("web.dash.requests"), num: true, fmt: fmt },
      { key: "tokens_in", label: t("web.dash.tokens_in"), num: true, fmt: fmt },
      { key: "tokens_out", label: t("web.dash.tokens_out"), num: true, fmt: fmt },
      { key: "cost_usd", label: t("web.dash.cost"), num: true, fmt: money },
      { key: "latency_avg_ms", label: t("web.dash.latency"), num: true, fmt: ms },
      { key: "auto_relevance", label: t("web.dash.relevance"), num: true, fmt: stars },
    ], ov.models);

    table($("tbl-strategies"), [
      { key: "strategy", label: t("web.dash.strategy"), fmt: (v) => t(`web.dash.strategy_${v}`) === `web.dash.strategy_${v}` ? v : t(`web.dash.strategy_${v}`) },
      { key: "source", label: t("web.dash.source") },
      { key: "requests", label: t("web.dash.requests"), num: true, fmt: fmt },
      { key: "avg_ctx_tokens", label: t("web.dash.ctx_tokens"), num: true, fmt: fmt },
      { key: "auto_relevance", label: t("web.dash.relevance"), num: true, fmt: stars },
    ], ov.strategies);
    $("strategy-note").textContent = t("web.dash.strategy_note");

    table($("tbl-chats"), [
      { key: "title", label: t("web.dash.chat") },
      { key: "requests", label: t("web.dash.requests"), num: true, fmt: fmt },
      { key: "auto_usefulness", label: t("web.dash.usefulness"), num: true, fmt: stars },
    ], ov.top_chats);

    $("review").innerHTML = ov.review.length ? ov.review.map((r) => `
      <li><a href="/chat#c${r.conversation_id}"><b>${r.rating === -1 ? "👎" : ""}${r.auto_relevance != null ? ` ★${r.auto_relevance}` : ""}</b> ${esc(r.excerpt)}</a>
      <div class="muted small">${esc(r.model || "")} · ${esc((r.created_at || "").slice(0, 16).replace("T", " "))}${r.comment ? ` · “${esc(r.comment)}”` : ""}${r.auto_note ? ` · ${esc(r.auto_note)}` : ""}</div></li>`).join("")
      : `<li class="muted">${esc(t("web.dash.review_empty"))}</li>`;
  }

  $("days").addEventListener("change", load);
  window.addEventListener("resize", () => { clearTimeout(window.__rz); window.__rz = setTimeout(() => { if (last) drawCharts(last); }, 150); });
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (last) drawCharts(last); });
  load().catch((e) => { console.error(e); });
})();
