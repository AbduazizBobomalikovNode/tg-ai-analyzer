/* Xavfsiz Markdown renderer (GFM+ qism-to'plami) — AI javoblari uchun.
 *
 * Nega o'zimizniki: LLM javobi ichida begona chat matni bo'lishi mumkin —
 * HTML "by construction" xavfsiz: **avval hamma narsa escape qilinadi, keyin
 * faqat bizning teglar qo'shiladi**. Xom HTML hech qachon o'tmaydi, havolalar
 * faqat http(s)/mailto/tg, rasmlar faqat `/api/images/<id>`.
 *
 * Qo'llab-quvvatlaydi:
 *   blok  — sarlavha (id bilan, TOC uchun), paragraf, ro'yxat (ichma-ich,
 *           tartibli/tartibsiz, `- [ ]` vazifa), > blockquote,
 *           `> [!NOTE|TIP|WARN|IMPORTANT|CAUTION]` callout, --- hr,
 *           GFM jadval (tekislash, raqamli ustun avtomat o'ngga, `Table:`
 *           sarlavhasi), ``` fence (til yorlig'i bilan), ```mermaid →
 *           "Diagrammani chizish" kartasi (lazy), ```chart → inline SVG,
 *           `:::details Sarlavha` … `:::` yig'iladigan blok, `[^1]:` izoh.
 *   inline — **bold**, *italic*, ~~strike~~, ==mark==, `code`, [havola](url),
 *           avtolink, `[^1]` izoh havolasi, `![](/api/images/<id>)`,
 *           qator uzilishi.
 *
 * Sitata tokenlari (`#c1023m42`) **matn bo'lib qoladi** — ularni
 * `pages/ask/deep.js::linkifyCitations` TreeWalker bilan chipga aylantiradi.
 *
 * window.TGAI.md(text)      → HTML (sof funksiya, deterministik)
 * window.TGAI.hydrate(root) → mermaid tugmasi + chart SVG + tema kuzatuvi
 * window.TGAI.chartSvg(obj) → chart spetsifikatsiyasidan SVG (sof funksiya)
 */
(function () {
  "use strict";

  const ESC_MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC_MAP[c]);

  const safeUrl = (u) => {
    const t = String(u).trim();
    if (/^(https?:\/\/|mailto:)/i.test(t)) return t.replace(/"/g, "%22");
    if (/^tg:\/\//i.test(t)) return t.replace(/"/g, "%22");
    return null;
  };

  /** Lug'at (`window.TGAI.strings`) — yo'q bo'lsa zaxira matn. */
  const L = (key, fallback) => {
    const d = (typeof window !== "undefined" && window.TGAI && window.TGAI.strings) || null;
    const v = d && d[key];
    return typeof v === "string" && v ? v : fallback;
  };

  const spriteHref = () =>
    (typeof window !== "undefined" && window.TGAI && window.TGAI.sprite) || "/static/icons.svg";

  /** Sprite ikonkasi HTML matn sifatida. `name` oq ro'yxat bilan cheklangan. */
  function iconSvg(name, cls) {
    const safe = String(name).replace(/[^a-z0-9-]/gi, "");
    const c = String(cls || "").replace(/[^a-z0-9 _-]/gi, "");
    const cn = c ? `icon ${c}` : "icon";
    return `<svg class="${cn}" aria-hidden="true" focusable="false"><use href="${esc(spriteHref())}#${safe}"></use></svg>`;
  }

  /** FNV-1a 32-bit — barqaror (deterministik) id prefiksi va fence keshi uchun. */
  function hash(s) {
    let h = 0x811c9dc5;
    const str = String(s);
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return h.toString(36);
  }

  /** Sarlavha matnidan id bo'lagi (unicode harflar qoladi — kirill ham). */
  function slug(text) {
    let s = String(text).toLowerCase().replace(/<[^>]*>/g, " ");
    try {
      s = s.replace(/[^\p{L}\p{N}]+/gu, "-");
    } catch (_) {
      s = s.replace(/[^a-z0-9]+/g, "-");
    }
    s = s.replace(/^-+|-+$/g, "").slice(0, 48);
    return s || "x";
  }

  /* ── render holati ────────────────────────────────────────────────────────
   * `md()` sof funksiya bo'lib qolishi uchun holat har chaqiruvda yangilanadi
   * va id'lar manba matnining hash'idan olinadi (oqim paytida ham barqaror). */
  let ST = null;

  const MAX_DEPTH = 6;          // blockquote / details / list ichma-ichligi
  const MAX_INPUT = 400000;     // uzun javob himoyasi
  const MAX_TABLE_ROWS = 400;
  const MAX_TABLE_COLS = 24;

  /* ── inline ──────────────────────────────────────────────────────────────*/

  /**
   * Bitta havola tegi. `href` allaqachon `safeUrl` dan o'tgan.
   * Tashqi havolaga ikonka, `t.me` ga Telegram chipi.
   */
  function anchor(href, innerHtml) {
    const isTg = /^https?:\/\/(?:www\.)?t\.me\//i.test(href) || /^tg:\/\//i.test(href);
    const isExternal = /^https?:\/\//i.test(href);
    const cls = isTg ? "md-link md-tglink" : "md-link";
    const tail = isTg ? iconSvg("send", "md-link-ico") : isExternal ? iconSvg("external-link", "md-link-ico") : "";
    return `<a class="${cls}" href="${esc(href)}" target="_blank" rel="noopener noreferrer">${innerHtml}${tail}</a>`;
  }

  function inline(src) {
    // `code` bo'laklari escape'gacha olinadi — ichidagi `*`/`_` formatlanmasin
    const codes = [];
    let s = String(src).replace(/`([^`\n]+)`/g, (_, c) => {
      codes.push(`<code class="md-code-inline">${esc(c)}</code>`);
      return `${codes.length - 1}`;
    });

    // ⚠️ shu qatordan keyin matnda xom `<`, `&`, `"` yo'q — hamma tег bizniki
    s = esc(s);

    // rasmlar — faqat o'zimizniki (`/api/images/<id>`), boshqasi matn bo'lib qoladi
    s = s.replace(
      /!\[([^\]\n]{0,300})\]\((\/api\/images\/[A-Za-z0-9-]{8,40})\)/g,
      (m, alt, url) => `<img class="gen-img" src="${url}" alt="${alt}" loading="lazy" decoding="async">`
    );

    // izoh havolasi `[^1]` — ta'rif bo'lsa raqamli sup, bo'lmasa matn
    s = s.replace(/\[\^([^\]\s]{1,32})\]/g, (m, key) => {
      const fn = ST && ST.fn.get(key);
      if (!fn) return m;
      fn.used = true;
      return `<sup class="md-fn-ref" id="${ST.uid}-fr-${fn.n}"><a href="#${ST.uid}-fn-${fn.n}">${fn.n}</a></sup>`;
    });

    // havolalar [matn](url "sarlavha") — ichma-ich qavs YO'Q (backtracking xavfi)
    s = s.replace(
      /\[([^\]\n]{1,400})\]\(([^()\s]{0,400})(\([^()\s]{0,100}\))?([^()\s]{0,200})(?:\s+&quot;[^&\n]{0,200}&quot;)?\)/g,
      (m, text, a, b, c) => {
        const u = safeUrl(((a || "") + (b || "") + (c || "")).replace(/&amp;/g, "&"));
        return u ? anchor(u, text) : text;
      }
    );

    // avtolink — havola ichidagilar allaqachon tegda, ularга tegilmaydi
    s = s.replace(
      /(^|[\s(])(https?:\/\/[^\s<]{3,400}[^\s<.,;:!?)"'])/g,
      (m, pre, url) => `${pre}${anchor(url.replace(/&amp;/g, "&"), url)}`
    );

    s = s
      .replace(/\*\*([^*\n]{1,500}?)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]{1,500}?)__/g, "<strong>$1</strong>")
      .replace(/(^|[^\w*])\*([^*\n]{1,500}?)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/(^|[^\w_])_([^_\n]{1,500}?)_(?!\w)/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]{1,500}?)~~/g, "<del>$1</del>")
      .replace(/==([^=\n]{1,500}?)==/g, "<mark>$1</mark>");

    s = s.replace(/ {2,}\n|\\\n/g, "<br>");
    return s.replace(/(\d+)/g, (_, i) => codes[+i]);
  }

  /* ── blok naqshlari ──────────────────────────────────────────────────────*/

  const reFence = /^\s{0,3}(```|~~~)\s*([\w+#.-]*)\s*$/;
  const reFenceEnd = /^\s{0,3}(```|~~~)\s*$/;
  const reHead = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
  const reHr = /^\s*([-*_])(\s*\1){2,}\s*$/;
  const reUl = /^(\s*)([-*+•])\s+(.*)$/;
  const reOl = /^(\s*)(\d{1,3})[.)]\s+(.*)$/;
  const reTask = /^\[([ xX])\]\s+(.*)$/;
  const reQuote = /^\s*>\s?(.*)$/;
  // GFM ajratuvchisi: `---`, `:--`, `--:`, `:-:` (oxirgisi bitta chiziq bilan ham)
  const _SEP = ":-+:|:-+|-+:|-{2,}";
  const reTableSep = new RegExp(`^\\s*\\|?\\s*(?:${_SEP})\\s*(?:\\|\\s*(?:${_SEP})\\s*)*\\|?\\s*$`);
  const reFnDef = /^\s{0,3}\[\^([^\]\s]{1,32})\]:\s*(.*)$/;
  const reDetailsOpen = /^\s*:::\s*(details|note|tip|warn|warning|info)?\s*(.*)$/;
  const reDetailsEnd = /^\s*:::\s*$/;
  const reCaption = /^\s*(?:Table|Jadval|Таблица)\s*:\s*(.{1,200})$/i;
  const reCallout = /^\s*\[!(NOTE|TIP|WARN|WARNING|IMPORTANT|CAUTION)\]\s*(.*)$/i;

  const CALLOUTS = {
    note: { cls: "note", icon: "info", key: "web.md.note", label: "Eslatma" },
    tip: { cls: "tip", icon: "sparkles", key: "web.md.tip", label: "Maslahat" },
    warn: { cls: "warn", icon: "alert-triangle", key: "web.md.warn", label: "Diqqat" },
    warning: { cls: "warn", icon: "alert-triangle", key: "web.md.warn", label: "Diqqat" },
    important: { cls: "important", icon: "bolt", key: "web.md.important", label: "Muhim" },
    caution: { cls: "caution", icon: "shield-check", key: "web.md.caution", label: "Ehtiyot" },
  };

  /** `| a | b |` → ["a", "b"] (ekranlangan `\|` saqlanadi). */
  function splitRow(line) {
    let l = String(line).trim();
    if (l.startsWith("|")) l = l.slice(1);
    if (l.endsWith("|")) l = l.slice(0, -1);
    return l.split(/(?<!\\)\|/, MAX_TABLE_COLS).map((c) => c.replace(/\\\|/g, "|").trim());
  }

  const reNumeric = /^[-+(]?\s*[\d][\d\s.,'’]*\s*[%°]?\s*(?:[kKmMbB]|mln|mlrd|тыс|млн)?\s*[$€₽¢)]?$/;
  /** Ustun raqamlimi (kamida bitta qiymat va ≥60% mos) — o'ngga tekislash uchun. */
  function numericColumn(rows, k) {
    let seen = 0;
    let num = 0;
    for (const r of rows) {
      const v = (r[k] || "").trim();
      if (!v || v === "—" || v === "-") continue;
      seen++;
      if (reNumeric.test(v)) num++;
    }
    return seen >= 2 && num / seen >= 0.6;
  }

  /* ── chart (```chart) ────────────────────────────────────────────────────*/

  const CH_W = 640;
  const CH_H = 260;
  const CH_PAD = { t: 18, r: 14, b: 34, l: 46 };
  const CH_COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--accent)", "var(--ok)", "var(--warn-text)"];

  /** Raqamni qisqa yorliqqa (1 200 → 1.2k). */
  function chNum(v) {
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1).replace(/\.0$/, "") + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(1).replace(/\.0$/, "") + "k";
    return String(Math.round(v * 100) / 100);
  }

  /**
   * Chart spetsifikatsiyasidan inline SVG. Kutubxona yo'q, ranglar CSS
   * o'zgaruvchilari — tema almashganda qayta chizish shart emas.
   *
   * @param {{type?: string, labels?: string[], series?: {name?: string, values?: number[]}[], title?: string}} spec
   * @returns {string|null} SVG HTML yoki `null` (yaroqsiz spetsifikatsiya)
   */
  function chartSvg(spec) {
    if (!spec || typeof spec !== "object") return null;
    const labels = (Array.isArray(spec.labels) ? spec.labels : []).slice(0, 60).map((x) => String(x));
    const series = (Array.isArray(spec.series) ? spec.series : [])
      .slice(0, 6)
      .map((s, i) => ({
        name: String((s && s.name) || `S${i + 1}`),
        values: (Array.isArray(s && s.values) ? s.values : []).slice(0, 60).map((v) => (Number.isFinite(+v) ? +v : 0)),
      }))
      .filter((s) => s.values.length);
    if (!labels.length || !series.length) return null;
    // `pie` qo'llab-quvvatlanmaydi — bar bo'lib chiziladi (o'qilishi oson)
    const type = spec.type === "line" ? "line" : "bar";
    const n = Math.min(labels.length, Math.max(...series.map((s) => s.values.length)));
    if (!n) return null;

    let max = 0;
    let min = 0;
    for (const s of series) for (let i = 0; i < n; i++) {
      const v = s.values[i] ?? 0;
      if (v > max) max = v;
      if (v < min) min = v;
    }
    if (max === min) max = min + 1;
    max += (max - min) * 0.08;
    // "Chiroyli" bo'linma: 0 / 5k / 10k … — hisoblangan 5.7k / 11.3k emas.
    // O'qilishi uchun muhim: o'q yorliqlari birinchi qaraladigan narsa.
    const raw = (max - min) / 4;
    const exp = Math.pow(10, Math.floor(Math.log10(raw > 0 ? raw : 1)));
    const f = raw / exp;
    const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * exp;
    min = Math.floor(min / step) * step;
    max = Math.ceil(max / step) * step;
    const ticks = Math.min(8, Math.max(2, Math.round((max - min) / step)));

    const x0 = CH_PAD.l;
    const x1 = CH_W - CH_PAD.r;
    const y0 = CH_H - CH_PAD.b;
    const y1 = CH_PAD.t;
    const sy = (v) => y0 - ((v - min) / (max - min)) * (y0 - y1);
    const band = (x1 - x0) / n;

    const parts = [];
    // to'r + y yorliqlari
    for (let g = 0; g <= ticks; g++) {
      const v = min + ((max - min) * g) / ticks;
      const y = sy(v);
      parts.push(`<line class="md-chart-grid" x1="${x0}" x2="${x1}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"></line>`);
      parts.push(`<text class="md-chart-tick" x="${x0 - 6}" y="${(y + 3.5).toFixed(1)}" text-anchor="end">${esc(chNum(v))}</text>`);
    }

    if (type === "bar") {
      const gw = band * 0.72;
      const bw = Math.max(2, gw / series.length);
      for (let si = 0; si < series.length; si++) {
        for (let i = 0; i < n; i++) {
          const v = series[si].values[i] ?? 0;
          const x = x0 + band * i + (band - gw) / 2 + bw * si;
          const yv = sy(v);
          const yz = sy(Math.max(min, 0));
          const top = Math.min(yv, yz);
          const h = Math.max(1, Math.abs(yz - yv));
          parts.push(
            `<rect class="md-chart-bar" x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${CH_COLORS[si % CH_COLORS.length]}"><title>${esc(series[si].name)}: ${esc(chNum(v))}</title></rect>`
          );
        }
      }
    } else {
      for (let si = 0; si < series.length; si++) {
        const pts = [];
        for (let i = 0; i < n; i++) {
          const v = series[si].values[i] ?? 0;
          pts.push(`${(x0 + band * i + band / 2).toFixed(1)},${sy(v).toFixed(1)}`);
        }
        parts.push(
          `<polyline class="md-chart-line" points="${pts.join(" ")}" fill="none" stroke="${CH_COLORS[si % CH_COLORS.length]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"></polyline>`
        );
        for (let i = 0; i < n; i++) {
          const v = series[si].values[i] ?? 0;
          parts.push(
            `<circle class="md-chart-dot" cx="${(x0 + band * i + band / 2).toFixed(1)}" cy="${sy(v).toFixed(1)}" r="2.6" fill="${CH_COLORS[si % CH_COLORS.length]}"><title>${esc(series[si].name)}: ${esc(chNum(v))}</title></circle>`
          );
        }
      }
    }

    // x yorliqlari — sig'maganda har `step` inchisi
    const xstep = Math.ceil(n / 12);
    for (let i = 0; i < n; i++) {
      if (i % xstep) continue;
      parts.push(
        `<text class="md-chart-tick" x="${(x0 + band * i + band / 2).toFixed(1)}" y="${y0 + 16}" text-anchor="middle">${esc(labels[i] || "")}</text>`
      );
    }

    const legend =
      series.length > 1
        ? `<div class="md-chart-legend">${series
            .map((s, i) => `<span class="md-chart-key"><i style="background:${CH_COLORS[i % CH_COLORS.length]}"></i>${esc(s.name)}</span>`)
            .join("")}</div>`
        : "";
    const title = spec.title ? `<div class="md-chart-title">${esc(String(spec.title))}</div>` : "";
    return (
      `${title}<svg class="md-chart-svg" viewBox="0 0 ${CH_W} ${CH_H}" role="img" ` +
      `aria-label="${esc(String(spec.title || L("web.md.chart", "Diagramma")))}" preserveAspectRatio="xMidYMid meet">` +
      `${parts.join("")}</svg>${legend}`
    );
  }

  /* ── blok renderer ───────────────────────────────────────────────────────*/

  function renderBlocks(src, depth) {
    if (depth > MAX_DEPTH) return `<p>${inline(src)}</p>`;
    const lines = String(src == null ? "" : src).replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;
    let buf = [];
    // Oxirgi bir qatorli paragraf — jadval `Table:` sarlavhasi bo'lishi mumkin
    // (u bo'sh qator bilan ajratilgan bo'lsa allaqachon chiqarilgan bo'ladi).
    let lastPara = null;

    let bufFrom = 0;

    /**
     * Yuqori darajadagi blokka manba qatorlari oralig'i (`data-l="a,b"`).
     * Shu tufayli UI "bu blokni markdown bo'lib nusxa ol" qila oladi —
     * HTML'dan markdown tiklash (fragile) shart emas.
     */
    const stamp = (html, from, to) =>
      depth === 0 ? html.replace(/^(<[a-zA-Z][\w-]*\b[^>]*?)(\/?>)/, `$1 data-l="${from},${to}"$2`) : html;

    const flush = () => {
      if (!buf.length) return;
      lastPara = buf.length === 1 ? { at: out.length, text: buf[0] } : null;
      out.push(stamp(`<p>${inline(buf.join("\n"))}</p>`, bufFrom, i));
      buf = [];
    };

    /** Bitta element matni — `- [ ]` bo'lsa checkbox bilan. */
    function itemHtml(text) {
      const task = String(text).match(reTask);
      if (!task) return inline(text);
      const on = task[1].toLowerCase() === "x";
      return (
        `<input type="checkbox" class="md-task-box" disabled${on ? " checked" : ""} aria-hidden="true">` +
        `<span class="md-task-text">${inline(task[2])}</span>`
      );
    }

    function list(startIndent, ordered, d) {
      const tag = ordered ? "ol" : "ul";
      const cls = ordered ? "md-ol" : "md-ul";
      let html = "";
      let open = false;
      let hasTask = false;
      while (i < lines.length) {
        const m = ordered ? lines[i].match(reOl) : lines[i].match(reUl);
        const other = ordered ? lines[i].match(reUl) : lines[i].match(reOl);
        if (m && m[1].length === startIndent) {
          if (open) html += "</li>";
          const isTask = reTask.test(m[3]);
          if (isTask) hasTask = true;
          html += `<li${isTask ? ' class="md-task"' : ""}>${itemHtml(m[3])}`;
          open = true;
          i++;
          continue;
        }
        const anyItem = lines[i].match(reUl) || lines[i].match(reOl);
        if (anyItem && anyItem[1].length > startIndent && open && d < MAX_DEPTH) {
          html += list(anyItem[1].length, !!lines[i].match(reOl) && !lines[i].match(reUl), d + 1);
          continue;
        }
        if (other && other[1].length === startIndent) break; // tur o'zgardi
        if (open && lines[i].trim() !== "" && /^\s+/.test(lines[i]) && !anyItem) {
          html += `<br>${inline(lines[i].trim())}`;
          i++;
          continue;
        }
        break;
      }
      if (open) html += "</li>";
      return `<${tag} class="${cls}${hasTask ? " md-tasklist" : ""}">${html}</${tag}>`;
    }

    while (i < lines.length) {
      const from = i;
      const mark = out.length;
      step();
      for (let k = mark; k < out.length; k++) {
        if (out[k].indexOf(' data-l="') < 0) out[k] = stamp(out[k], from, i);
      }
    }
    flush();
    return out.join("\n");

    /** Bitta blokni o'qiydi va `out` ga qo'shadi (`i` ni oldinga suradi). */
    function step() {
      const line = lines[i];

      // ── fence ──────────────────────────────────────────────────────────
      const f = line.match(reFence);
      if (f) {
        flush();
        const lang = (f[2] || "").toLowerCase();
        const body = [];
        i++;
        while (i < lines.length && !reFenceEnd.test(lines[i])) body.push(lines[i++]);
        i++; // yopuvchi
        const code = body.join("\n");
        const h = hash(lang + "\n" + code);
        if (lang === "mermaid") {
          out.push(
            `<figure class="md-mermaid" data-hash="${h}">` +
              `<pre class="md-mermaid-src">${esc(code)}</pre>` +
              `<div class="md-mermaid-bar">` +
              `<button type="button" class="btn md-mermaid-draw">${iconSvg("bar-chart-3", "")}` +
              `<span>${esc(L("web.md.draw_diagram", "Diagrammani chizish"))}</span></button>` +
              `</div></figure>`
          );
          return;
        }
        if (lang === "chart") {
          let spec = null;
          try {
            spec = JSON.parse(code);
          } catch (_) {
            spec = null;
          }
          if (spec && chartSvg(spec)) {
            out.push(
              `<figure class="md-chart" data-hash="${h}">` +
                `<pre class="md-chart-src" hidden>${esc(code)}</pre>` +
                `</figure>`
            );
            return;
          }
          // yaroqsiz JSON — oddiy kod bloki bo'lib qoladi (ma'lumot yo'qolmasin)
        }
        out.push(
          `<figure class="md-code"${lang ? ` data-lang="${esc(lang)}"` : ""}>` +
            `<figcaption class="md-code-head"><span class="md-code-lang">${esc(lang)}</span>` +
            `<span class="md-code-tools"></span></figcaption>` +
            `<pre><code${lang ? ` class="lang-${esc(lang)}"` : ""}>${esc(code)}</code></pre></figure>`
        );
        return;
      }

      // ── `:::details Sarlavha` … `:::` ──────────────────────────────────
      const dOpen = line.match(reDetailsOpen);
      if (dOpen && !reDetailsEnd.test(line) && depth < MAX_DEPTH) {
        flush();
        const kind = (dOpen[1] || "details").toLowerCase();
        const title = (dOpen[2] || "").trim();
        const body = [];
        let level = 1;
        i++;
        while (i < lines.length) {
          if (reDetailsEnd.test(lines[i])) {
            level--;
            if (!level) { i++; break; }
          } else if (reDetailsOpen.test(lines[i]) && !reDetailsEnd.test(lines[i])) {
            level++;
          }
          body.push(lines[i++]);
        }
        const sum = title ? inline(title) : esc(L("web.md.details", "Tafsilot"));
        out.push(
          `<details class="md-details" data-kind="${esc(kind)}">` +
            `<summary class="md-details-sum">${sum}</summary>` +
            `<div class="md-details-body">${renderBlocks(body.join("\n"), depth + 1)}</div></details>`
        );
        return;
      }

      // ── bo'sh qator ────────────────────────────────────────────────────
      if (line.trim() === "") { flush(); i++; return; }

      // ── izoh ta'rifi `[^1]: matn` ──────────────────────────────────────
      const fd = line.match(reFnDef);
      if (fd && ST && ST.fn.has(fd[1])) {
        flush();
        i++;
        const extra = [fd[2]];
        while (i < lines.length && /^\s{2,}\S/.test(lines[i])) extra.push(lines[i++].trim());
        ST.fn.get(fd[1]).text = extra.join(" ").trim();
        return;
      }

      // ── sarlavha ───────────────────────────────────────────────────────
      const h = line.match(reHead);
      if (h) {
        flush();
        const lvl = Math.min(6, h[1].length + 2);
        const html = inline(h[2]);
        const base = ST ? `${ST.uid}-${slug(h[2])}` : slug(h[2]);
        let id = base;
        if (ST) {
          const seen = ST.ids.get(base) || 0;
          ST.ids.set(base, seen + 1);
          if (seen) id = `${base}-${seen + 1}`;
        }
        out.push(`<h${lvl} class="md-h" id="${esc(id)}">${html}</h${lvl}>`);
        i++;
        return;
      }

      // ── hr ─────────────────────────────────────────────────────────────
      if (reHr.test(line) && !line.match(reUl)) { flush(); out.push('<hr class="md-hr">'); i++; return; }

      // ── jadval ─────────────────────────────────────────────────────────
      if (line.includes("|") && i + 1 < lines.length && reTableSep.test(lines[i + 1])) {
        // sarlavha: jadval oldidagi `Table: …` qatori
        let caption = "";
        if (buf.length) {
          const cm = buf[buf.length - 1].match(reCaption);
          if (cm) { caption = cm[1].trim(); buf.pop(); }
        }
        flush();
        if (!caption && lastPara && lastPara.at === out.length - 1) {
          const cm = lastPara.text.match(reCaption);
          if (cm) { caption = cm[1].trim(); out.pop(); }
        }
        lastPara = null;
        const head = splitRow(line);
        const aligns = splitRow(lines[i + 1]).map((c) =>
          c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : ""
        );
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "" && rows.length < MAX_TABLE_ROWS) {
          rows.push(splitRow(lines[i]));
          i++;
        }
        const cols = head.map((_, k) => aligns[k] || (numericColumn(rows, k) ? "right" : ""));
        const th = head
          .map((c, k) => `<th scope="col"${cols[k] ? ` style="text-align:${cols[k]}"` : ""}>${inline(c)}</th>`)
          .join("");
        const tb = rows
          .map(
            (r) =>
              `<tr>${head
                .map((_, k) => `<td${cols[k] ? ` style="text-align:${cols[k]}"` : ""}>${inline(r[k] == null ? "" : r[k])}</td>`)
                .join("")}</tr>`
          )
          .join("");
        out.push(
          `<div class="md-table-wrap table-wrap" tabindex="0" role="region"` +
            `${caption ? ` aria-label="${esc(caption)}"` : ""}>` +
            `<table class="md-table">${caption ? `<caption class="md-table-cap">${inline(caption)}</caption>` : ""}` +
            `<thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table></div>`
        );
        return;
      }

      // ── blockquote / callout ───────────────────────────────────────────
      if (reQuote.test(line)) {
        flush();
        const q = [];
        while (i < lines.length && reQuote.test(lines[i])) { q.push(lines[i].match(reQuote)[1]); i++; }
        const co = q.length ? q[0].match(reCallout) : null;
        if (co) {
          const kind = CALLOUTS[co[1].toLowerCase()] || CALLOUTS.note;
          const title = (co[2] || "").trim();
          const body = q.slice(1).join("\n");
          out.push(
            `<aside class="md-callout md-callout--${kind.cls}" role="note">` +
              `<div class="md-callout-head">${iconSvg(kind.icon, "md-callout-ico")}` +
              `<span class="md-callout-title">${title ? inline(title) : esc(L(kind.key, kind.label))}</span></div>` +
              `<div class="md-callout-body">${renderBlocks(body, depth + 1)}</div></aside>`
          );
          return;
        }
        out.push(`<blockquote class="md-quote">${renderBlocks(q.join("\n"), depth + 1)}</blockquote>`);
        return;
      }

      // ── ro'yxat ────────────────────────────────────────────────────────
      const ul = line.match(reUl);
      const ol = line.match(reOl);
      if (ul || ol) {
        flush();
        const m = ul || ol;
        out.push(list(m[1].length, !!ol && !ul, depth + 1));
        return;
      }

      if (!buf.length) bufFrom = i;
      buf.push(line);
      i++;
    }
  }

  /* ── ommaviy `md()` ──────────────────────────────────────────────────────*/

  /**
   * Markdown → xavfsiz HTML. Sof funksiya: bir xil matn → bir xil HTML
   * (id'lar manba hash'idan olinadi, hisoblagichdan emas).
   * @param {string} src
   * @returns {string}
   */
  function render(src) {
    let text = String(src == null ? "" : src).replace(/\r\n?/g, "\n");
    if (text.length > MAX_INPUT) text = text.slice(0, MAX_INPUT);

    // 1-o'tish: izoh ta'riflari (ular matnning oxirida turadi, havolalar esa
    // oldinda) — shuning uchun ularni oldindan bilib olamiz.
    const fn = new Map();
    let n = 0;
    for (const line of text.split("\n")) {
      const m = line.match(reFnDef);
      if (m && !fn.has(m[1])) fn.set(m[1], { n: ++n, text: "", used: false });
    }
    ST = { uid: "md" + hash(text), fn, ids: new Map() };
    let html = renderBlocks(text, 0);

    const notes = [...fn.values()].filter((f) => f.used);
    if (notes.length) {
      html +=
        `\n<section class="md-footnotes"><h6 class="md-fn-head">${esc(L("web.md.footnotes", "Izohlar"))}</h6><ol class="md-fn-list">` +
        notes
          .map(
            (f) =>
              `<li id="${ST.uid}-fn-${f.n}">${inline(f.text)} ` +
              `<a class="md-fn-back" href="#${ST.uid}-fr-${f.n}" aria-label="${esc(L("web.md.back", "Orqaga"))}">↩</a></li>`
          )
          .join("") +
        `</ol></section>`;
    }
    ST = null;
    return html;
  }

  /* ── mermaid (lazy, strict) ─────────────────────────────────────────────*/

  let mermaidLoading = null;
  function themeName() {
    if (typeof document !== "undefined" && document.documentElement) {
      const attr = document.documentElement.getAttribute("data-theme");
      if (attr === "dark") return "dark";
      if (attr === "light") return "default";
    }
    const dark = typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    return dark ? "dark" : "default";
  }

  function loadMermaid() {
    if (typeof window !== "undefined" && window.mermaid) return Promise.resolve(window.mermaid);
    if (mermaidLoading) return mermaidLoading;
    mermaidLoading = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "/static/vendor/mermaid.min.js";
      s.onload = () => {
        window.mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: themeName(),
          fontFamily: "inherit",
        });
        resolve(window.mermaid);
      };
      s.onerror = () => reject(new Error("mermaid load failed"));
      document.head.appendChild(s);
    });
    return mermaidLoading;
  }

  let seq = 0;

  /** Bitta `figure.md-mermaid` ni chizadi (manba `data-src` da saqlanadi). */
  async function drawMermaid(fig) {
    const src = fig.dataset.src || (fig.querySelector(".md-mermaid-src") || {}).textContent || "";
    if (!src.trim()) return;
    fig.dataset.src = src;
    let mm;
    fig.dataset.state = "loading";
    try {
      mm = await loadMermaid();
    } catch (_) {
      fig.dataset.state = "failed";
      return; // manba `<pre>` bo'lib qoladi — ma'lumot yo'qolmaydi
    }
    const id = `mmd-${Date.now().toString(36)}-${seq++}`;
    try {
      if (window.mermaid && window.mermaid.initialize) {
        window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: themeName(), fontFamily: "inherit" });
      }
      const { svg } = await mm.render(id, src);
      let holder = fig.querySelector(".md-mermaid-svg");
      if (!holder) {
        holder = document.createElement("div");
        holder.className = "md-mermaid-svg mermaid-svg";
        fig.insertBefore(holder, fig.firstChild);
      }
      holder.innerHTML = svg; // mermaid securityLevel="strict" → sanitizatsiya qilingan
      fig.dataset.state = "done";
    } catch (e) {
      fig.dataset.state = "failed";
      fig.title = String((e && e.message) || e);
      const bad = document.getElementById("d" + id);
      if (bad) bad.remove();
    }
  }

  /* ── hydrate ────────────────────────────────────────────────────────────*/

  /**
   * Chizilgan javob ustidagi progressiv bezak:
   *   * `figure.md-chart` → inline SVG (kutubxonasiz), fence hash bo'yicha kesh;
   *   * `figure.md-mermaid` → "Diagrammani chizish" tugmasi (lazy, 2.5MB vendor
   *     fayli **faqat bosilganda** yuklanadi).
   * Har ikkalasi ham idempotent: oqim paytida qayta chaqirilsa allaqachon
   * chizilgan tugunlar qayta chizilmaydi.
   */
  function hydrate(root) {
    if (!root || !root.querySelectorAll) return Promise.resolve();

    for (const fig of root.querySelectorAll("figure.md-chart")) {
      if (fig.dataset.done === fig.dataset.hash) continue;
      const pre = fig.querySelector(".md-chart-src");
      let spec = null;
      try {
        spec = JSON.parse((pre && pre.textContent) || "");
      } catch (_) {
        spec = null;
      }
      const svg = spec ? chartSvg(spec) : null;
      if (!svg) continue;
      let holder = fig.querySelector(".md-chart-box");
      if (!holder) {
        holder = document.createElement("div");
        holder.className = "md-chart-box";
        fig.insertBefore(holder, fig.firstChild);
      }
      holder.innerHTML = svg; // faqat `chartSvg` chiqishi — escape qilingan
      fig.dataset.done = fig.dataset.hash || "1";
    }

    for (const fig of root.querySelectorAll("figure.md-mermaid")) {
      if (fig.dataset.bound === fig.dataset.hash) continue;
      fig.dataset.bound = fig.dataset.hash || "1";
      const pre = fig.querySelector(".md-mermaid-src");
      if (pre) fig.dataset.src = pre.textContent;
      const btn = fig.querySelector(".md-mermaid-draw");
      if (btn) {
        btn.addEventListener("click", () => {
          btn.disabled = true;
          drawMermaid(fig).finally(() => { btn.disabled = false; });
        });
      }
    }
    return Promise.resolve();
  }

  // Tema almashsa — **chizilgan** diagrammalar qayta chiziladi (chizilmaganiga
  // tegilmaydi: vendor fayli hamon yuklanmagan bo'lishi kerak).
  if (typeof window !== "undefined" && window.addEventListener) {
    window.addEventListener("tgai:theme", () => {
      if (!window.mermaid || typeof document === "undefined") return;
      for (const fig of document.querySelectorAll('figure.md-mermaid[data-state="done"]')) drawMermaid(fig);
    });
  }

  window.TGAI = window.TGAI || {};
  window.TGAI.md = render;
  window.TGAI.hydrate = hydrate;
  window.TGAI.esc = esc;
  window.TGAI.chartSvg = chartSvg;
  window.TGAI.mdHash = hash;
})();
