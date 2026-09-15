/* AI chat: akkaunt → (ixtiyoriy) Telegram chat konteksti → savol → javob. */
(function () {
  const S = window.TGAI.strings;
  const t = (k, vars) => {
    let v = S[k] || k;
    if (vars) for (const [key, val] of Object.entries(vars)) v = v.replaceAll(`{${key}}`, val);
    return v;
  };
  const $ = (id) => document.getElementById(id);
  const el = {
    sidebar: $("sidebar"), accountSelect: $("account-select"), accountWarn: $("account-warn"),
    dialogSearch: $("dialog-search"), dialogList: $("dialog-list"), convList: $("conv-list"),
    messages: $("messages"), empty: $("empty-state"), composer: $("composer"), input: $("input"),
    send: $("btn-send"), deep: $("deep"), strategy: $("strategy"), mode: $("mode"), ctxLimit: $("ctx-limit"), ctxLimitLabel: $("ctx-limit-label"),
    ctxLimitWrap: $("ctx-limit-wrap"), convTitle: $("conv-title"), ctxBadge: $("ctx-badge"),
    del: $("btn-delete"), wmWrap: $("wm-wrap"), writeMode: $("write-mode"), pending: $("pending-badge"),
    arBtn: $("btn-autoreply"), schedBtn: $("btn-scheduled"),
    panelBackdrop: $("panel-backdrop"), panelTitle: $("panel-title"), panelBody: $("panel-body"), panelClose: $("panel-close"),
  };

  const state = {
    accounts: [], accountId: null, dialogs: [], chatsByPeer: {}, peerId: 0, peerTitle: "",
    conversations: [], convId: null, busy: false, pollTimer: null,
  };

  // ── API ──────────────────────────────────────────────────────────────────
  async function api(method, path, body) {
    const res = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json", "X-Requested-With": "fetch" },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
    });
    if (res.status === 401) { window.location.href = "/login"; throw new Error("unauthorized"); }
    let data = null;
    try { data = await res.json(); } catch (_) { /* bo'sh */ }
    if (!res.ok) {
      const d = (data && data.detail) || {};
      const err = new Error(t(d.code || "error.generic") + (d.detail ? ` (${d.detail})` : ""));
      err.code = d.code; throw err;
    }
    return data;
  }

  // ── markdown (md.js: GFM jadval, ro'yxat, kod, mermaid) ──────────────────
  const esc = window.TGAI.esc;
  const md = (src) => window.TGAI.md(src);

  // ── render ───────────────────────────────────────────────────────────────
  function initials(s) { return (s || "?").trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase() || "?"; }
  // Sprite ikonkalari (Lucide) — emoji o'rniga: platformada bir xil
  // ko'rinadi va ekran o'quvchi uni ism deb o'qimaydi (aria-hidden).
  const SPRITE = (window.TGAI && window.TGAI.sprite) || "/static/icons.svg";
  const ico = (name) => `<svg class="icon" aria-hidden="true" focusable="false"><use href="${SPRITE}#${name}"></use></svg>`;
  const kindIcon = { channel: ico("hash"), group: ico("users"), user: ico("user") };

  function renderAccounts() {
    el.accountSelect.innerHTML = "";
    for (const a of state.accounts) {
      const o = document.createElement("option");
      o.value = a.id; o.textContent = `${a.label || a.tg_account_id}${a.status !== "active" ? ` (${a.status})` : ""}`;
      el.accountSelect.appendChild(o);
    }
    if (!state.accounts.length) { el.accountWarn.textContent = t("web.chat.no_accounts"); el.accountWarn.hidden = false; }
    else el.accountWarn.hidden = true;
    if (state.accountId) el.accountSelect.value = state.accountId;
  }

  function fmtN(n) { return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k" : String(n); }
  function syncBadge(c) {
    if (!c) return `<span class="dialog-sync">${esc(t("web.chat.not_synced"))}</span>`;
    const running = c.sync_state === "running";
    const pct = c.progress || 0;
    const label = running ? t("web.chat.sync_running") : t("web.chat.synced", { n: fmtN(c.synced_total || 0) });
    return `<span class="dialog-sync" title="${esc(c.sync_error || "")}"><span class="bar${running ? " running" : ""}"><i style="width:${pct}%"></i></span>${esc(label)}${c.sync_error ? " ⚠️" : ""}</span>`;
  }

  function renderDialogs() {
    const q = el.dialogSearch.value.trim().toLowerCase();
    const items = state.dialogs.filter((d) => !q || d.title.toLowerCase().includes(q) || (d.username || "").toLowerCase().includes(q));
    const noneSel = state.peerId === 0 ? " selected" : "";
    let html = `<li class="dialog-item${noneSel}" data-peer="0"><span class="dialog-avatar">${ico("layers")}</span><span class="dialog-body"><span class="dialog-title">${esc(t("web.chat.context_none"))}</span></span></li>`;
    for (const d of items) {
      const c = state.chatsByPeer[d.peer_id];
      html += `<li class="dialog-item${d.peer_id === state.peerId ? " selected" : ""}" data-peer="${d.peer_id}" data-title="${esc(d.title)}" data-chat="${c ? c.id : ""}">
        <span class="dialog-avatar">${esc(initials(d.title))}</span>
        <span class="dialog-body"><span class="dialog-title">${kindIcon[d.kind] || ""} ${esc(d.title)}</span>
        <span class="dialog-sub">${esc(d.last_message_text || "")}</span>${syncBadge(c)}</span>
        ${c ? `<button class="sync-btn" data-sync="${c.id}" aria-label="${esc(t("web.chat.sync"))}" title="${esc(t("web.chat.sync"))}">${ico("refresh-cw")}</button>` : ""}</li>`;
    }
    el.dialogList.innerHTML = html;
    updateCtxUI();
  }

  function renderConvs() {
    if (!state.conversations.length) { el.convList.innerHTML = `<li class="dialog-sub" style="padding:.4rem .6rem">${esc(t("web.chat.no_conversations"))}</li>`; return; }
    el.convList.innerHTML = state.conversations.map((c) =>
      `<li class="conv-item${c.id === state.convId ? " selected" : ""}" data-conv="${c.id}"><span class="conv-title">${esc(c.title || "…")}</span></li>`
    ).join("");
  }

  function toolsBadge(c) {
    if (!c || !c.tools || !c.tools.length) return "";
    const counts = {};
    for (const t of c.tools) counts[t] = (counts[t] || 0) + 1;
    const title = (c.tool_calls || []).map((x) => `${x.tool}(${JSON.stringify(x.args || {})}) ${x.ok ? "✓" : "✗"} ${x.ms || 0}ms`).join("\n");
    return `<span class="ctx" title="${esc(title)}">🔧 ${esc(Object.entries(counts).map(([k, v]) => v > 1 ? `${k}×${v}` : k).join(", "))} · ${c.iterations || 1} it.</span>`;
  }
  function stratLabel(c) {
    if (!c || !c.strategy || c.strategy === "agent") return "";
    const key = `web.dash.strategy_${c.strategy}`;
    const lbl = t(key) === key ? c.strategy : t(key);
    return ` · ${lbl}${c.est_tokens ? ` ~${fmtN(c.est_tokens)} tok` : ""}${c.truncated ? " ✂" : ""}`;
  }
  function autoBadge(a) {
    if (!a || a.relevance == null) return "";
    return `<span class="auto" title="${esc(a.note || "")}">🤖 ${a.relevance}/5${a.grounded === false ? " ⚠️" : ""}</span>`;
  }
  function rateButtons(m) {
    if (m.role !== "assistant" || !m.id) return "";
    return `<span class="rate" data-mid="${m.id}"><button data-r="1" class="${m.rating === 1 ? "on" : ""}" title="${esc(t("web.chat.rate_up"))}">👍</button><button data-r="-1" class="${m.rating === -1 ? "on" : ""}" title="${esc(t("web.chat.rate_down"))}">👎</button></span>`;
  }
  function actionCard(a) {
    const st = a.status;
    const label = t(`web.act.tool_${a.tool}`) === `web.act.tool_${a.tool}` ? a.tool : t(`web.act.tool_${a.tool}`);
    const p = a.preview || {};
    let body = "";
    // Rasm faqat o'zimizning endpoint'dan (`/api/images/<id>`) — `md.js`
    // dagi qoida bilan bir xil. Boshqa sxema (`javascript:`, `data:`)
    // umuman chizilmaydi.
    if (/^\/api\/images\/[A-Za-z0-9-]{8,40}$/.test(String(p.image_url || ""))) {
      body += `<img class="gen-img" src="${esc(p.image_url)}" alt="" loading="lazy">`;
    }
    if (p.text) body += `<div class="action-text">${esc(p.text)}</div>`;
    const meta = [];
    if (p.chat) meta.push(`→ ${esc(p.chat)}`);
    if (p.message_id) meta.push(`#${p.message_id}`);
    if (p.reply_to) meta.push(`reply #${p.reply_to}`);
    if (p.schedule_at) meta.push(`${ico("clock")} ${esc(String(p.schedule_at).slice(0, 16).replace("T", " "))}`);
    if (p.drop_author) meta.push("copy");
    if (p.image_id) meta.push(`🖼 ${esc(t("web.act.image"))}`);
    if (a.result_msg_id) meta.push(t("web.act.result", { id: a.result_msg_id }));
    if (a.error) meta.push(`⚠️ ${esc(a.error)}`);
    if (a.block_reason) meta.push(`⛔ ${esc(a.block_reason)}`);
    if (st === "proposed" && a.expires_at) meta.push(t("web.act.expires", { t: String(a.expires_at).slice(0, 16).replace("T", " ") }));
    const btns = st === "proposed"
      ? `<div class="action-btns"><button class="btn ok" data-act="confirm" data-aid="${a.id}">${esc(t("web.act.confirm"))}</button><button class="btn" data-act="reject" data-aid="${a.id}">${esc(t("web.act.reject"))}</button></div>`
      : "";
    return `<div class="action-card ${esc(st)}" data-aid="${a.id}"><div class="action-head"><b>✍️ ${esc(label)}</b><span class="action-status">${esc(t(`web.act.status_${st}`))}</span></div>${body}<div class="action-meta">${meta.join(" · ")}</div>${btns}</div>`;
  }
  function msgNode(m) {
    const div = document.createElement("div");
    div.className = `msg ${m.role}`;
    if (m.id) div.dataset.mid = m.id;
    let inner = "";
    if (m.context && m.context.mode === "agent") inner += toolsBadge(m.context);
    else if (m.context && m.context.title) inner += `<span class="ctx">📎 ${esc(t("web.chat.context_used", { title: m.context.title, n: m.context.messages }))}${m.context.source ? ` · ${esc(t(`web.chat.source_${m.context.source}`))}` : ""}${esc(stratLabel(m.context))}</span>`;
    inner += m.role === "assistant" ? md(m.content) : `<p>${esc(m.content).replace(/\n/g, "<br>")}</p>`;
    if (m.role === "assistant" && m.actions && m.actions.length) inner += m.actions.map(actionCard).join("");
    if (m.role === "assistant" && m.model) {
      const cost = m.cost_usd != null ? ` · $${m.cost_usd < 0.01 ? m.cost_usd.toFixed(4) : m.cost_usd.toFixed(3)}` : "";
      const lat = m.latency_ms ? ` · ${(m.latency_ms / 1000).toFixed(1)}s` : "";
      inner += `<span class="meta">${esc(m.provider || "")} · ${esc(m.model)} · ${fmtN(m.tokens_in || 0)}→${fmtN(m.tokens_out || 0)}${lat}${cost}${autoBadge(m.auto)}${rateButtons(m)}</span>`;
    }
    div.innerHTML = inner;
    return div;
  }
  el.messages.addEventListener("click", async (e) => {
    const ab = e.target.closest("[data-act]");
    if (ab) {
      const aid = +ab.dataset.aid, act = ab.dataset.act;
      const card = ab.closest(".action-card");
      card.querySelectorAll("button").forEach((x) => { x.disabled = true; });
      if (act === "confirm") ab.textContent = t("web.act.confirming");
      try {
        const a = await api("POST", `/api/actions/${aid}/${act}`, {});
        card.outerHTML = actionCard(a);
      } catch (err) {
        card.querySelectorAll("button").forEach((x) => { x.disabled = false; });
        ab.textContent = act === "confirm" ? t("web.act.confirm") : t("web.act.reject");
        alert(err.message);
      }
      loadPending();
      return;
    }
    const b = e.target.closest(".rate button"); if (!b) return;
    const wrap = b.closest(".rate"); const mid = +wrap.dataset.mid; const r = +b.dataset.r;
    const already = b.classList.contains("on");
    try {
      await api("POST", `/api/conversations/${state.convId}/messages/${mid}/rate`, { rating: already ? 0 : r });
      wrap.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
      if (!already) b.classList.add("on");
    } catch (err) { console.warn(err); }
  });

  // ── deep search (SSE) ────────────────────────────────────────────────────
  // `EventSource` POST qila olmaydi → `fetch` + ReadableStream. Javob matni
  // faqat md.js orqali chiziladi (yagona renderer), qolgan hamma narsa
  // `textContent` yoki `esc()` bilan.
  const safeHref = (u) => (typeof u === "string" && /^(https?:\/\/|tg:\/\/)/i.test(u) ? u : null);

  function deepNode() {
    const div = document.createElement("div");
    div.className = "msg assistant deep";
    div.innerHTML = `<div class="deep-status"><span class="typing"><i></i><i></i><i></i></span> <span class="deep-step"></span></div>
      <div class="deep-chips" hidden></div>
      <div class="deep-answer"></div>
      <div class="deep-coverage" hidden></div>
      <span class="meta" hidden></span>`;
    return div;
  }

  function renderChips(wrap, clusters) {
    if (!clusters || !clusters.length) return;
    wrap.innerHTML = clusters.slice(0, 12).map((c) => {
      const when = [c.first_seen, c.last_seen].filter(Boolean).map((d) => String(d).slice(0, 10)).join(" … ");
      const title = [(c.chats || []).join(", "), when].filter(Boolean).join(" · ");
      return `<span class="chip" title="${esc(title)}">${esc(c.label || "…")} <b>${c.size}</b></span>`;
    }).join("");
    wrap.hidden = false;
  }

  function renderCoverage(wrap, cov, cost) {
    const r = cov.range || {};
    const rows = [
      `${esc(t("web.deep.cov_chats"))}: <b>${cov.chats_searched}/${cov.chats_total}</b>`,
      `${esc(t("web.deep.cov_scanned"))}: <b>${cov.messages_scanned}</b>`,
      `${esc(t("web.deep.cov_rounds"))}: <b>${cov.rounds}</b>`,
      `${esc(t("web.deep.cov_confidence"))}: <b>${esc(t(`web.deep.conf_${cov.confidence}`))}</b>`,
    ];
    if (r.from && r.to) rows.splice(2, 0, `${esc(t("web.deep.cov_range"))}: <b>${esc(String(r.from).slice(0, 10))} … ${esc(String(r.to).slice(0, 10))}</b>`);
    if (cost != null) rows.push(`${esc(t("web.deep.cov_cost"))}: <b>$${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(3)}</b>`);
    let html = `<div class="cov-head">${esc(t("web.deep.coverage"))}</div><div class="cov-rows">${rows.join(" · ")}</div>`;
    if (cov.gaps && cov.gaps.length) {
      html += `<div class="cov-gaps">⚠️ ${esc(t("web.deep.gaps"))}: ${esc(cov.gaps.slice(0, 8).join(", "))}${cov.gaps.length > 8 ? ` (+${cov.gaps.length - 8})` : ""}</div>`;
    }
    // Server pog'onalari — halol hisob: nechta so'rov ketdi, nechta topildi,
    // qaysi shift to'ldi. Tekshirilmagan (tashqi) topilmalar alohida.
    if (cov.server_rungs_used) {
      const srv = [
        `${esc(t("web.deep.cov_server_calls"))}: <b>${cov.server_calls || 0}</b>`,
        `${esc(t("web.deep.cov_server_hits"))}: <b>${cov.server_hits || 0}</b>`,
      ];
      if (cov.external_hits) srv.push(`${esc(t("web.deep.cov_external"))}: <b>${cov.external_hits}</b>`);
      html += `<div class="cov-rows">${esc(t("web.deep.cov_server"))} — ${srv.join(" · ")}</div>`;
    }
    if (cov.server_skipped && cov.server_skipped.length) {
      html += `<div class="cov-gaps">⚠️ ${esc(t("web.deep.server_skipped"))}: ${esc(cov.server_skipped.join(", "))}</div>`;
    }
    if (cov.truncated) html += `<div class="cov-gaps">⚠️ ${esc(t("web.deep.truncated"))}</div>`;
    wrap.innerHTML = html;
    wrap.hidden = false;
  }

  // Sitatalar (#c<chat>m<msg> / #x<n>) → havola. DOM matn tugunlari bo'ylab
  // yuriladi — HTML qayta parse qilinmaydi, ya'ni escape-first invarianti
  // buzilmaydi. Kalit chat bilan birga: bir xil `tg_msg_id` li ikki chat
  // bir-birining havolasini olmaydi.
  function linkifyCitations(root, links) {
    if (!links || !Object.keys(links).length) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const targets = [];
    while (walker.nextNode()) {
      const n = walker.currentNode;
      if (!n.parentElement || n.parentElement.closest("a, code, pre")) continue;
      if (/#(?:c\d|x\d)/.test(n.nodeValue || "")) targets.push(n);
    }
    for (const n of targets) {
      const s = n.nodeValue;
      const frag = document.createDocumentFragment();
      let last = 0;
      const re = /#(c(\d{1,12})m(\d{1,12})|x\d{1,12})/g;
      let m;
      while ((m = re.exec(s)) !== null) {
        frag.appendChild(document.createTextNode(s.slice(last, m.index)));
        // Tokendan to'g'ridan-to'g'ri, bo'lmasa o'sha chatning havola prefiksidan.
        let href = links[m[1]];
        if (!href && m[2] && links.prefix && links.prefix[m[2]]) href = links.prefix[m[2]] + m[3];
        const url = safeHref(href);
        if (url) {
          const a = document.createElement("a");
          a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.className = "cite";
          a.textContent = m[0];
          frag.appendChild(a);
        } else frag.appendChild(document.createTextNode(m[0]));
        last = m.index + m[0].length;
      }
      frag.appendChild(document.createTextNode(s.slice(last)));
      n.parentNode.replaceChild(frag, n);
    }
  }

  // `done` hodisasidagi sitatalardan havola jadvali: token → URL, va har chat
  // uchun havola prefiksi (`link` dan `tg_msg_id` ni olib tashlab).
  function citationLinks(citations) {
    const links = { prefix: {} };
    for (const c of citations || []) {
      if (!c || !c.link) continue;
      if (c.token) links[c.token] = c.link;
      const tail = String(c.id == null ? "" : c.id);
      if (c.chat_id && tail && c.link.endsWith(tail)) {
        links.prefix[String(c.chat_id)] = c.link.slice(0, c.link.length - tail.length);
      }
    }
    return links;
  }

  const SSE_MAX_BUFFER = 2 * 1024 * 1024; // bitta ramka shundan oshsa — xato

  function parseFrame(frame) {
    let kind = "message";
    const data = [];
    // SSE: bir hodisada bir nechta `data:` satri bo'lishi mumkin va ular
    // **`\n` bilan** birlashtiriladi (spetsifikatsiya). Ilgari ular yopishtirib
    // yuborilardi — ko'p satrli JSON buzilardi.
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) kind = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    const body = data.join("\n").trim();
    if (!body) return null;
    try { return { kind, data: JSON.parse(body) }; } catch (_) { return null; }
  }

  async function sendDeep(text, node) {
    const stepEl = node.querySelector(".deep-step");
    const chips = node.querySelector(".deep-chips");
    const answerEl = node.querySelector(".deep-answer");
    const covEl = node.querySelector(".deep-coverage");
    const metaEl = node.querySelector(".meta");
    const status = node.querySelector(".deep-status");
    const step = (k, vars) => { stepEl.textContent = t(k, vars); };
    step("web.deep.step_plan");

    const res = await fetch("/api/deep/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Requested-With": "fetch" },
      body: JSON.stringify({ conversation_id: state.convId, question: text, deep: true }),
      credentials: "same-origin",
    });
    if (res.status === 401) { window.location.href = "/login"; throw new Error("unauthorized"); }
    if (!res.ok || !res.body) {
      let d = {};
      try { d = (await res.json()).detail || {}; } catch (_) { /* bo'sh */ }
      const err = new Error(t(d.code || "error.generic") + (d.detail ? ` (${d.detail})` : ""));
      err.code = d.code; throw err;
    }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", answer = "", total = 0, searched = 0, served = 0, failed = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let m;
      // Ramka chegarasi — `\r\n\r\n` ham bo'lishi mumkin (proxy CRLF qo'shadi).
      while ((m = /\r?\n\r?\n/.exec(buf)) !== null) {
        const ev = parseFrame(buf.slice(0, m.index));
        buf = buf.slice(m.index + m[0].length);
        if (!ev) continue;
        const d = ev.data || {};
        if (ev.kind === "plan") { total = (d.subqueries || []).length; step("web.deep.step_search", { n: 0, m: total }); }
        else if (ev.kind === "search" && d.level === "server") { served++; step("web.deep.step_server", { n: served }); }
        else if (ev.kind === "search") { searched++; step("web.deep.step_search", { n: searched, m: total || searched }); }
        else if (ev.kind === "cluster") { renderChips(chips, d.clusters); step("web.deep.step_cluster", { n: (d.clusters || []).length }); }
        else if (ev.kind === "rerank") { step("web.deep.step_rank", { n: d.kept || 0 }); }
        else if (ev.kind === "token") { answer += d.text || ""; answerEl.innerHTML = md(answer); step("web.deep.step_write"); scrollBottom(); }
        else if (ev.kind === "coverage") { renderCoverage(covEl, d, null); }
        else if (ev.kind === "done") {
          status.hidden = true;
          linkifyCitations(answerEl, citationLinks(d.citations));
          if (d.cost_usd != null) { const rows = covEl.querySelector(".cov-rows"); if (rows) rows.insertAdjacentHTML("beforeend", ` · ${esc(t("web.deep.cov_cost"))}: <b>$${d.cost_usd < 0.01 ? d.cost_usd.toFixed(4) : d.cost_usd.toFixed(3)}</b>`); }
          metaEl.textContent = `${d.provider || ""} · ${d.model || ""} · ${fmtN(d.tokens_in || 0)}→${fmtN(d.tokens_out || 0)}${d.latency_ms ? ` · ${(d.latency_ms / 1000).toFixed(1)}s` : ""}`;
          metaEl.hidden = false;
          if (d.assistant_message_id) node.dataset.mid = d.assistant_message_id;
        } else if (ev.kind === "error") {
          failed = new Error(t(`chat.err.${d.code}`) === `chat.err.${d.code}` ? (d.detail || d.code) : t(`chat.err.${d.code}`));
          failed.code = `chat.err.${d.code}`;
        }
      }
      // Ramka chegarasiz cheksiz o'sish — xotirani yeydi. Qopqoq: 2 MB.
      if (buf.length > SSE_MAX_BUFFER) {
        try { await reader.cancel(); } catch (_) { /* allaqachon yopiq */ }
        const err = new Error(t("chat.err.internal"));
        err.code = "chat.err.internal";
        throw err;
      }
    }
    if (failed) throw failed;
    window.TGAI.hydrate(node).then(scrollBottom);
    await loadConvs();
    const cur = state.conversations.find((c) => c.id === state.convId);
    if (cur) el.convTitle.textContent = cur.title || t("web.chat.title");
    el.del.hidden = false;
  }

  function renderMessages(all) {
    el.messages.innerHTML = "";
    // Uzilgan deep yugurishning hisob qatori (`context.aborted`) — u faqat
    // kunlik byudjet uni sanashi uchun yoziladi, javob emas (H2).
    const items = (all || []).filter((m) => !(m.context && m.context.aborted));
    if (!items.length) { el.messages.appendChild(el.empty); el.empty.hidden = false; return; }
    for (const m of items) el.messages.appendChild(msgNode(m));
    window.TGAI.hydrate(el.messages).then(scrollBottom);
    scrollBottom();
  }
  function scrollBottom() { el.messages.scrollTop = el.messages.scrollHeight; }

  function updateWriteModeUI() {
    const c = state.peerId ? state.chatsByPeer[state.peerId] : null;
    if (!c) { el.wmWrap.hidden = true; el.arBtn.hidden = true; el.schedBtn.hidden = true; return; }
    el.wmWrap.hidden = false;
    el.writeMode.value = c.write_mode || "read_only";
    el.writeMode.disabled = false;
    el.arBtn.hidden = false;
    el.schedBtn.hidden = false;
  }

  // ── panel ────────────────────────────────────────────────────────────────
  function openPanel(title, html) { el.panelTitle.textContent = title; el.panelBody.innerHTML = html; el.panelBackdrop.hidden = false; }
  function closePanel() { el.panelBackdrop.hidden = true; el.panelBody.innerHTML = ""; }
  el.panelClose.addEventListener("click", closePanel);
  el.panelBackdrop.addEventListener("click", (e) => { if (e.target === el.panelBackdrop) closePanel(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !el.panelBackdrop.hidden) closePanel(); });

  async function openPendingPanel() {
    const d = await api("GET", "/api/actions?status=proposed&limit=50");
    openPanel(t("web.pending.title"), d.items.length ? d.items.map(actionCard).join("") : `<p class="muted">${esc(t("web.pending.empty"))}</p>`);
  }
  el.pending.addEventListener("click", (e) => { e.preventDefault(); openPendingPanel(); });
  el.panelBody.addEventListener("click", async (e) => {
    const ab = e.target.closest("[data-act]"); if (!ab) return;
    const aid = +ab.dataset.aid, act = ab.dataset.act;
    const card = ab.closest(".action-card");
    card.querySelectorAll("button").forEach((x) => { x.disabled = true; });
    try { const a = await api("POST", `/api/actions/${aid}/${act}`, {}); card.outerHTML = actionCard(a); }
    catch (err) { card.querySelectorAll("button").forEach((x) => { x.disabled = false; }); alert(err.message); }
    loadPending();
  });

  async function openAutoReplyPanel() {
    const c = state.peerId ? state.chatsByPeer[state.peerId] : null; if (!c) return;
    let rule = null;
    try { rule = (await api("GET", `/api/accounts/${state.accountId}/chats/${c.id}/autoreply`)).rule; } catch (_) { /* yo'q */ }
    const r = rule || { enabled: false, trigger: "questions", keywords: "", instructions: "", max_per_hour: 5, quiet_from: null, quiet_to: null };
    const opt = (v, label) => `<option value="${v}" ${r.trigger === v ? "selected" : ""}>${esc(label)}</option>`;
    const needsWrite = (c.write_mode || "read_only") === "read_only";
    openPanel(`${t("web.ar.title")} · ${esc(c.title)}`, `
      ${needsWrite ? `<div class="warn">${esc(t("web.ar.needs_write"))}</div>` : ""}
      <form id="ar-form" class="form-grid">
        <label><span>${esc(t("web.ar.enabled"))}</span><input type="checkbox" name="enabled" ${r.enabled ? "checked" : ""}></label>
        <label><span>${esc(t("web.ar.trigger"))}</span><select name="trigger">${opt("questions", t("web.ar.trig_questions"))}${opt("mentions", t("web.ar.trig_mentions"))}${opt("keywords", t("web.ar.trig_keywords"))}${opt("all", t("web.ar.trig_all"))}</select></label>
        <label class="full"><span>${esc(t("web.ar.keywords"))}</span><input type="text" name="keywords" value="${esc(r.keywords || "")}"></label>
        <label class="full"><span>${esc(t("web.ar.instructions"))}</span><textarea name="instructions" placeholder="${esc(t("web.ar.instructions_ph"))}">${esc(r.instructions || "")}</textarea></label>
        <label><span>${esc(t("web.ar.max_per_hour"))}</span><input type="number" name="max_per_hour" min="1" max="60" value="${r.max_per_hour || 5}"></label>
        <label><span>${esc(t("web.ar.quiet"))}</span><span class="row"><input type="number" name="quiet_from" min="0" max="23" placeholder="22" value="${r.quiet_from ?? ""}"><input type="number" name="quiet_to" min="0" max="23" placeholder="7" value="${r.quiet_to ?? ""}"></span></label>
        <p class="form-note full">${esc(t("web.ar.note"))}</p>
        <div class="full"><button type="submit" class="btn primary" style="width:auto">${esc(t("web.ar.save"))}</button></div>
      </form>`);
    $("ar-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = new FormData(e.target);
      const body = {
        enabled: f.get("enabled") === "on", trigger: f.get("trigger"), keywords: f.get("keywords") || "",
        instructions: f.get("instructions") || "", max_per_hour: +f.get("max_per_hour") || 5,
        quiet_from: f.get("quiet_from") === "" ? null : +f.get("quiet_from"),
        quiet_to: f.get("quiet_to") === "" ? null : +f.get("quiet_to"),
      };
      try {
        await api("PUT", `/api/accounts/${state.accountId}/chats/${c.id}/autoreply`, body);
        const note = $("sync-note"); note.textContent = t("web.ar.saved"); note.hidden = false; setTimeout(() => { note.hidden = true; }, 2500);
        closePanel();
      } catch (err) { alert(err.message); }
    });
  }
  el.arBtn.addEventListener("click", openAutoReplyPanel);

  async function openScheduledPanel() {
    const c = state.peerId ? state.chatsByPeer[state.peerId] : null; if (!c) return;
    openPanel(`${t("web.sched.title")} · ${esc(c.title)}`, `<p class="muted">…</p>`);
    try {
      const d = await api("GET", `/api/accounts/${state.accountId}/chats/${c.id}/scheduled`);
      el.panelBody.innerHTML = d.items.length
        ? d.items.map((m) => `<div class="sched-item"><div class="muted">${ico("clock")} ${esc((m.date || "").slice(0, 16).replace("T", " "))} · #${m.msg_id}${m.has_media ? ` · ${ico("image")}` : ""}</div>${esc(m.text || "")}</div>`).join("")
        : `<p class="muted">${esc(t("web.sched.empty"))}</p>`;
    } catch (err) { el.panelBody.innerHTML = `<p class="warn">${esc(err.message)}</p>`; }
  }
  el.schedBtn.addEventListener("click", openScheduledPanel);
  async function loadPending() {
    try {
      const d = await api("GET", "/api/actions?status=proposed&limit=50");
      const n = d.items.length;
      el.pending.hidden = n === 0;
      el.pending.textContent = t("web.act.pending", { n });
      el.pending.dataset.items = JSON.stringify(d.items.map((a) => a.id));
    } catch (_) { /* ignore */ }
  }
  el.writeMode.addEventListener("change", async () => {
    const c = state.peerId ? state.chatsByPeer[state.peerId] : null; if (!c) return;
    const mode = el.writeMode.value;
    if (mode === "autonomous" && !confirm(t("web.chat.wm_autonomous_warn"))) { el.writeMode.value = c.write_mode; return; }
    try {
      const r = await api("PATCH", `/api/accounts/${state.accountId}/chats/${c.id}/write_mode`, { mode });
      c.write_mode = r.write_mode;
      const note = $("sync-note"); note.textContent = t("web.chat.wm_saved"); note.hidden = false; setTimeout(() => { note.hidden = true; }, 2500);
    } catch (err) { alert(err.message); el.writeMode.value = c.write_mode; }
  });
  function updateCtxUI() {
    updateWriteModeUI();
    const n = +el.ctxLimit.value;
    const liveMax = +(el.ctxLimit.dataset.liveMax || 200);
    const src = n > liveMax ? ` · ${t("web.chat.source_db")}` : "";
    el.ctxLimitLabel.textContent = t("web.chat.context_last", { n }) + src;
    el.ctxLimitWrap.style.opacity = state.peerId ? 1 : 0.45;
    if (state.peerId) { el.ctxBadge.textContent = `📎 ${state.peerTitle} · ${n}`; el.ctxBadge.hidden = false; }
    else el.ctxBadge.hidden = true;
  }

  // ── yuklash ──────────────────────────────────────────────────────────────
  async function loadMe() {
    const me = await api("GET", "/api/me");
    state.accounts = me.accounts;
    const active = me.accounts.find((a) => a.status === "active") || me.accounts[0];
    state.accountId = active ? active.id : null;
    renderAccounts();
  }
  async function loadChats() {
    if (!state.accountId) { state.chatsByPeer = {}; return; }
    try {
      const data = await api("GET", `/api/accounts/${state.accountId}/chats`);
      state.chatsByPeer = {};
      for (const c of data.items) state.chatsByPeer[c.peer_id] = c;
      // ⌘K palitrasi shu ro'yxatdan oziqlanadi — qayta so'rov yo'q.
      window.TGAI.chats = data.items;
      window.dispatchEvent(new CustomEvent("tgai:chats", { detail: data.items }));
      // jonli dialoglar bo'lmasa (sessiya yo'q) — DB registry'dan ro'yxat
      if (!state.dialogs.length && data.items.length) {
        state.dialogs = data.items.map((c) => ({ peer_id: c.peer_id, title: c.title, kind: c.type === "private" ? "user" : c.type === "channel" ? "channel" : "group", username: c.username, last_message_text: "" }));
      }
      clearTimeout(state.pollTimer);
      if (data.running > 0) state.pollTimer = setTimeout(async () => { await loadChats(); renderDialogs(); }, 15000);
    } catch (_) { /* registry hali yo'q */ }
  }
  async function loadDialogs(refresh) {
    if (!state.accountId) { state.dialogs = []; renderDialogs(); return; }
    try {
      const data = await api("GET", `/api/accounts/${state.accountId}/dialogs${refresh ? "?refresh=1" : ""}`);
      state.dialogs = data.items; el.accountWarn.hidden = true;
    } catch (err) {
      state.dialogs = [];
      el.accountWarn.textContent = err.message; el.accountWarn.hidden = false;
    }
    await loadChats();
    renderDialogs();
  }
  async function startSync(chatId) {
    const note = $("sync-note");
    try {
      const url = chatId ? `/api/accounts/${state.accountId}/chats/${chatId}/sync` : `/api/accounts/${state.accountId}/sync`;
      await api("POST", url, {});
      note.textContent = t("web.chat.sync_queued"); note.hidden = false;
      setTimeout(() => { note.hidden = true; }, 6000);
      setTimeout(async () => { await loadChats(); renderDialogs(); }, 3000);
    } catch (err) { note.textContent = err.message; note.hidden = false; }
  }
  async function loadConvs() {
    const data = await api("GET", "/api/conversations");
    state.conversations = data.items; renderConvs();
  }
  async function openConv(id) {
    state.convId = id; renderConvs();
    if (!id) { el.convTitle.textContent = t("web.chat.title"); el.del.hidden = true; renderMessages([]); return; }
    const data = await api("GET", `/api/conversations/${id}/messages`);
    el.convTitle.textContent = data.conversation.title || t("web.chat.title");
    el.del.hidden = false;
    renderMessages(data.items);
  }

  // ── yuborish ─────────────────────────────────────────────────────────────
  async function send() {
    const text = el.input.value.trim();
    if (!text || state.busy) return;
    state.busy = true; el.send.disabled = true;
    el.empty.hidden = true;
    el.messages.appendChild(msgNode({ role: "user", content: text, context: state.peerId ? { title: state.peerTitle, messages: el.ctxLimit.value } : null }));
    const typing = document.createElement("div");
    typing.className = "msg assistant"; typing.innerHTML = `<span class="typing"><i></i><i></i><i></i></span> <span class="muted">${esc(t("web.chat.thinking"))}</span>`;
    el.messages.appendChild(typing); scrollBottom();
    let live = typing;
    el.input.value = ""; autosize();
    try {
      if (!state.convId) {
        const c = await api("POST", "/api/conversations", { account_id: state.accountId });
        state.convId = c.id;
      }
      if (el.deep.checked) {
        // "Chuqur tahlil" — alohida quvur (/api/deep/stream), oqimli javob.
        const dn = deepNode();
        typing.replaceWith(dn);
        live = dn;
        await sendDeep(text, dn);
        return;
      }
      const body = { text, deep: false, mode: el.mode.value, account_id: state.accountId || null };
      if (state.peerId && state.accountId) body.context = { account_id: state.accountId, peer_id: state.peerId, limit: +el.ctxLimit.value, strategy: el.strategy.value };
      const r = await api("POST", `/api/conversations/${state.convId}/messages`, body);
      const node = msgNode({ id: r.assistant_message_id, role: "assistant", content: r.text, model: r.model, provider: r.provider, tokens_in: r.tokens_in, tokens_out: r.tokens_out, context: r.context, latency_ms: r.latency_ms, cost_usd: r.cost_usd, actions: r.actions || [] });
      if (r.actions && r.actions.length) loadPending();
      typing.replaceWith(node);
      window.TGAI.hydrate(node).then(scrollBottom);
      await loadConvs();
      const cur = state.conversations.find((c) => c.id === state.convId);
      if (cur) el.convTitle.textContent = cur.title || t("web.chat.title");
      el.del.hidden = false;
    } catch (err) {
      live.className = "msg error"; live.textContent = err.message;
      if (err.code === "pool.err.session_revoked") { el.accountWarn.textContent = t("web.chat.session_revoked"); el.accountWarn.hidden = false; }
    } finally {
      state.busy = false; el.send.disabled = false; scrollBottom(); el.input.focus();
    }
  }

  function autosize() { el.input.style.height = "auto"; el.input.style.height = Math.min(el.input.scrollHeight, window.innerHeight * 0.4) + "px"; }

  // ── hodisalar ────────────────────────────────────────────────────────────
  el.composer.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  el.input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
  el.input.addEventListener("input", autosize);
  el.ctxLimit.addEventListener("input", updateCtxUI);
  el.dialogSearch.addEventListener("input", renderDialogs);
  el.dialogList.addEventListener("click", (e) => {
    const sb = e.target.closest(".sync-btn");
    if (sb) { e.stopPropagation(); startSync(+sb.dataset.sync); return; }
    const li = e.target.closest(".dialog-item"); if (!li) return;
    state.peerId = +li.dataset.peer; state.peerTitle = li.dataset.title || "";
    renderDialogs(); closeSidebarMobile();
  });
  el.convList.addEventListener("click", (e) => {
    const li = e.target.closest(".conv-item"); if (!li) return;
    openConv(+li.dataset.conv); closeSidebarMobile();
  });
  el.accountSelect.addEventListener("change", () => { state.accountId = +el.accountSelect.value; state.peerId = 0; loadDialogs(false); });
  $("btn-sync-all").addEventListener("click", () => startSync(null));
  $("btn-new").addEventListener("click", () => { openConv(null); closeSidebarMobile(); el.input.focus(); });
  el.del.addEventListener("click", async () => {
    if (!state.convId || !confirm(t("web.chat.delete_confirm"))) return;
    await api("DELETE", `/api/conversations/${state.convId}`);
    await loadConvs(); openConv(null);
  });
  $("btn-logout").addEventListener("click", async () => { await api("POST", "/api/auth/logout", {}); window.location.href = "/login"; });
  $("btn-open-sidebar").addEventListener("click", () => el.sidebar.classList.add("open"));
  $("btn-close-sidebar").addEventListener("click", closeSidebarMobile);
  function closeSidebarMobile() { if (window.innerWidth <= 860) el.sidebar.classList.remove("open"); }

  // Bo'sh holatdagi taklif chiplari — savolni kiritish maydoniga qo'yadi.
  // Delegatsiya: chiplar shablonda, lekin bo'sh holat qayta chizilishi mumkin.
  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".suggest");
    if (!chip) return;
    el.input.value = chip.textContent.trim();
    el.input.focus();
    el.input.dispatchEvent(new Event("input"));  // textarea balandligi moslashsin
  });

  // ── start ────────────────────────────────────────────────────────────────
  (async () => {
    updateCtxUI();
    try { await loadMe(); } catch (_) { return; }
    await Promise.all([loadDialogs(false), loadConvs(), loadPending()]);
    const m = location.hash.match(/^#c(\d+)$/);
    openConv(m ? +m[1] : null);
  })();
})();
