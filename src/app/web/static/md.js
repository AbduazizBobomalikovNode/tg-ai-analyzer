/* Xavfsiz Markdown renderer (GFM qism-to'plami) — AI javoblari uchun.
 *
 * Nega o'zimizniki: LLM javobi ichida begona chat matni bo'lishi mumkin —
 * HTML "by construction" xavfsiz: avval hamma narsa escape qilinadi, keyin
 * faqat bizning teglar qo'shiladi. Havolalar faqat http(s)/mailto.
 *
 * Qo'llab-quvvatlaydi: sarlavha, paragraf, **bold**, *italic*, ~~strike~~,
 * `code`, ``` fence (til bilan), ```mermaid → diagramma (lazy mermaid.js),
 * ro'yxatlar (ichma-ich, - * + 1. 1)), > blockquote, --- hr, GFM jadval
 * (| a | b |, tekislash), [havola](url), avtolink, qator uzilishi.
 *
 * window.TGAI.md(text) → HTML;  window.TGAI.hydrate(rootEl) → mermaid render.
 */
(function () {
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const safeUrl = (u) => {
    const t = String(u).trim();
    if (/^(https?:\/\/|mailto:)/i.test(t)) return t.replace(/"/g, "%22");
    if (/^tg:\/\//i.test(t)) return t.replace(/"/g, "%22");
    return null;
  };

  // ── inline ────────────────────────────────────────────────────────────────
  function inline(src) {
    // matn allaqachon escape qilinmagan — bu yerda escape qilamiz, kod bo'laklarini saqlab
    const codes = [];
    let s = src.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(`<code>${esc(c)}</code>`); return `\uE000${codes.length - 1}\uE001`; });
    s = esc(s);
    // havolalar [text](url "title")
    s = s.replace(/\[([^\]]+)\]\(((?:[^()\s]|\([^()\s]*\))+)(?:\s+&quot;[^&]*&quot;)?\)/g, (m, text, url) => {
      const u = safeUrl(url.replace(/&amp;/g, "&"));
      return u ? `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${text}</a>` : text;
    });
    // avtolink (escape'dan keyin & → &amp; bo'lgan; havola ichidagini o'tkazib yuboramiz)
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<]+[^\s<.,;:!?)"'])/g, (m, pre, url) => `${pre}<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
    s = s.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+?)__/g, "<strong>$1</strong>")
      .replace(/(^|[^\w*])\*([^*\n]+?)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/(^|[^\w_])_([^_\n]+?)_(?!\w)/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]+?)~~/g, "<del>$1</del>");
    s = s.replace(/ {2,}\n|\\\n/g, "<br>");
    return s.replace(/\uE000(\d+)\uE001/g, (_, i) => codes[+i]);
  }

  // ── block ─────────────────────────────────────────────────────────────────
  const reFence = /^\s*(```|~~~)\s*([\w+-]*)\s*$/;
  const reHead = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
  const reHr = /^\s*([-*_])(\s*\1){2,}\s*$/;
  const reUl = /^(\s*)([-*+•])\s+(.*)$/;
  const reOl = /^(\s*)(\d{1,3})[.)]\s+(.*)$/;
  const reQuote = /^\s*>\s?(.*)$/;
  const reTableSep = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

  function splitRow(line) {
    let l = line.trim();
    if (l.startsWith("|")) l = l.slice(1);
    if (l.endsWith("|")) l = l.slice(0, -1);
    return l.split(/(?<!\\)\|/).map((c) => c.replace(/\\\|/g, "|").trim());
  }

  function render(src) {
    const lines = String(src || "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;
    let mermaidCount = 0;

    function paragraph(buf) { if (buf.length) out.push(`<p>${inline(buf.join("\n"))}</p>`); }

    function list(startIndent, ordered) {
      // qaytaradi: HTML; i ni yangilaydi
      const tag = ordered ? "ol" : "ul";
      let html = `<${tag}>`;
      let open = false;
      while (i < lines.length) {
        const m = ordered ? lines[i].match(reOl) : lines[i].match(reUl);
        const other = ordered ? lines[i].match(reUl) : lines[i].match(reOl);
        if (m && m[1].length === startIndent) {
          if (open) html += "</li>";
          html += `<li>${inline(m[3])}`; open = true; i++;
          continue;
        }
        const anyItem = lines[i].match(reUl) || lines[i].match(reOl);
        if (anyItem && anyItem[1].length > startIndent && open) {
          html += list(anyItem[1].length, !!lines[i].match(reOl) && !lines[i].match(reUl));
          continue;
        }
        if (other && other[1].length === startIndent) break; // tur o'zgardi — tashqi sikl ochadi
        // ro'yxat elementining davomi (bo'sh qatorsiz, indent bilan)
        if (open && lines[i].trim() !== "" && /^\s+/.test(lines[i]) && !anyItem) {
          html += `<br>${inline(lines[i].trim())}`; i++; continue;
        }
        break;
      }
      if (open) html += "</li>";
      return html + `</${tag}>`;
    }

    let buf = [];
    while (i < lines.length) {
      const line = lines[i];
      // fence
      const f = line.match(reFence);
      if (f) {
        paragraph(buf); buf = [];
        const lang = (f[2] || "").toLowerCase();
        const body = [];
        i++;
        while (i < lines.length && !lines[i].match(reFence)) body.push(lines[i++]);
        i++; // yopuvchi
        const code = body.join("\n");
        if (lang === "mermaid") {
          out.push(`<div class="mermaid-wrap"><pre class="mermaid-src" data-idx="${mermaidCount++}">${esc(code)}</pre></div>`);
        } else {
          out.push(`<pre><code${lang ? ` class="lang-${esc(lang)}"` : ""}>${esc(code)}</code></pre>`);
        }
        continue;
      }
      // bo'sh qator
      if (line.trim() === "") { paragraph(buf); buf = []; i++; continue; }
      // sarlavha
      const h = line.match(reHead);
      if (h) { paragraph(buf); buf = []; const lvl = Math.min(6, h[1].length + 2); out.push(`<h${lvl}>${inline(h[2])}</h${lvl}>`); i++; continue; }
      // hr
      if (reHr.test(line) && !line.match(reUl)) { paragraph(buf); buf = []; out.push("<hr>"); i++; continue; }
      // jadval: joriy qator | bor va keyingisi ajratuvchi
      if (line.includes("|") && i + 1 < lines.length && reTableSep.test(lines[i + 1])) {
        paragraph(buf); buf = [];
        const head = splitRow(line);
        const aligns = splitRow(lines[i + 1]).map((c) => c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : "");
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "") { rows.push(splitRow(lines[i])); i++; }
        const th = head.map((c, k) => `<th${aligns[k] ? ` style="text-align:${aligns[k]}"` : ""}>${inline(c)}</th>`).join("");
        const tb = rows.map((r) => `<tr>${head.map((_, k) => `<td${aligns[k] ? ` style="text-align:${aligns[k]}"` : ""}>${inline(r[k] ?? "")}</td>`).join("")}</tr>`).join("");
        out.push(`<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table></div>`);
        continue;
      }
      // blockquote
      if (reQuote.test(line)) {
        paragraph(buf); buf = [];
        const q = [];
        while (i < lines.length && reQuote.test(lines[i])) q.push(lines[i].match(reQuote)[1]), i++;
        out.push(`<blockquote>${render(q.join("\n"))}</blockquote>`);
        continue;
      }
      // ro'yxat
      const ul = line.match(reUl), ol = line.match(reOl);
      if (ul || ol) {
        paragraph(buf); buf = [];
        const m = ul || ol;
        out.push(list(m[1].length, !!ol && !ul));
        continue;
      }
      buf.push(line); i++;
    }
    paragraph(buf);
    return out.join("\n");
  }

  // ── mermaid (lazy) ────────────────────────────────────────────────────────
  let mermaidLoading = null;
  function loadMermaid() {
    if (window.mermaid) return Promise.resolve(window.mermaid);
    if (mermaidLoading) return mermaidLoading;
    mermaidLoading = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "/static/vendor/mermaid.min.js";
      s.onload = () => {
        const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "default", fontFamily: "inherit" });
        resolve(window.mermaid);
      };
      s.onerror = () => reject(new Error("mermaid load failed"));
      document.head.appendChild(s);
    });
    return mermaidLoading;
  }
  let seq = 0;
  async function hydrate(root) {
    const blocks = root.querySelectorAll("pre.mermaid-src:not([data-done])");
    if (!blocks.length) return;
    let mm;
    try { mm = await loadMermaid(); } catch (_) { return; } // diagramma o'rniga manba qoladi
    for (const pre of blocks) {
      pre.dataset.done = "1";
      const code = pre.textContent;
      const id = `mmd-${Date.now()}-${seq++}`;
      try {
        const { svg } = await mm.render(id, code);
        const holder = document.createElement("div");
        holder.className = "mermaid-svg";
        holder.innerHTML = svg; // mermaid securityLevel=strict → sanitized
        pre.replaceWith(holder);
      } catch (e) {
        pre.classList.add("mermaid-error");
        pre.title = String(e && e.message || e);
        const bad = document.getElementById("d" + id); if (bad) bad.remove();
      }
    }
  }

  window.TGAI = window.TGAI || {};
  window.TGAI.md = render;
  window.TGAI.hydrate = hydrate;
  window.TGAI.esc = esc;
})();
