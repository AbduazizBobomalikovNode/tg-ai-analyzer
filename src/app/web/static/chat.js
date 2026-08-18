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
    del: $("btn-delete"),
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
  const kindIcon = { channel: "📣", group: "👥", user: "👤" };

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
    let html = `<li class="dialog-item${noneSel}" data-peer="0"><span class="dialog-avatar">∅</span><span class="dialog-body"><span class="dialog-title">${esc(t("web.chat.context_none"))}</span></span></li>`;
    for (const d of items) {
      const c = state.chatsByPeer[d.peer_id];
      html += `<li class="dialog-item${d.peer_id === state.peerId ? " selected" : ""}" data-peer="${d.peer_id}" data-title="${esc(d.title)}" data-chat="${c ? c.id : ""}">
        <span class="dialog-avatar">${esc(initials(d.title))}</span>
        <span class="dialog-body"><span class="dialog-title">${kindIcon[d.kind] || ""} ${esc(d.title)}</span>
        <span class="dialog-sub">${esc(d.last_message_text || "")}</span>${syncBadge(c)}</span>
        ${c ? `<button class="sync-btn" data-sync="${c.id}" title="${esc(t("web.chat.sync"))}">⟳</button>` : ""}</li>`;
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
  function msgNode(m) {
    const div = document.createElement("div");
    div.className = `msg ${m.role}`;
    if (m.id) div.dataset.mid = m.id;
    let inner = "";
    if (m.context && m.context.mode === "agent") inner += toolsBadge(m.context);
    else if (m.context && m.context.title) inner += `<span class="ctx">📎 ${esc(t("web.chat.context_used", { title: m.context.title, n: m.context.messages }))}${m.context.source ? ` · ${esc(t(`web.chat.source_${m.context.source}`))}` : ""}${esc(stratLabel(m.context))}</span>`;
    inner += m.role === "assistant" ? md(m.content) : `<p>${esc(m.content).replace(/\n/g, "<br>")}</p>`;
    if (m.role === "assistant" && m.model) {
      const cost = m.cost_usd != null ? ` · $${m.cost_usd < 0.01 ? m.cost_usd.toFixed(4) : m.cost_usd.toFixed(3)}` : "";
      const lat = m.latency_ms ? ` · ${(m.latency_ms / 1000).toFixed(1)}s` : "";
      inner += `<span class="meta">${esc(m.provider || "")} · ${esc(m.model)} · ${fmtN(m.tokens_in || 0)}→${fmtN(m.tokens_out || 0)}${lat}${cost}${autoBadge(m.auto)}${rateButtons(m)}</span>`;
    }
    div.innerHTML = inner;
    return div;
  }
  el.messages.addEventListener("click", async (e) => {
    const b = e.target.closest(".rate button"); if (!b) return;
    const wrap = b.closest(".rate"); const mid = +wrap.dataset.mid; const r = +b.dataset.r;
    const already = b.classList.contains("on");
    try {
      await api("POST", `/api/conversations/${state.convId}/messages/${mid}/rate`, { rating: already ? 0 : r });
      wrap.querySelectorAll("button").forEach((x) => x.classList.remove("on"));
      if (!already) b.classList.add("on");
    } catch (err) { console.warn(err); }
  });

  function renderMessages(items) {
    el.messages.innerHTML = "";
    if (!items.length) { el.messages.appendChild(el.empty); el.empty.hidden = false; return; }
    for (const m of items) el.messages.appendChild(msgNode(m));
    window.TGAI.hydrate(el.messages).then(scrollBottom);
    scrollBottom();
  }
  function scrollBottom() { el.messages.scrollTop = el.messages.scrollHeight; }

  function updateCtxUI() {
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
    el.input.value = ""; autosize();
    try {
      if (!state.convId) {
        const c = await api("POST", "/api/conversations", { account_id: state.accountId });
        state.convId = c.id;
      }
      const body = { text, deep: el.deep.checked, mode: el.mode.value, account_id: state.accountId || null };
      if (state.peerId && state.accountId) body.context = { account_id: state.accountId, peer_id: state.peerId, limit: +el.ctxLimit.value, strategy: el.strategy.value };
      const r = await api("POST", `/api/conversations/${state.convId}/messages`, body);
      const node = msgNode({ id: r.assistant_message_id, role: "assistant", content: r.text, model: r.model, provider: r.provider, tokens_in: r.tokens_in, tokens_out: r.tokens_out, context: r.context, latency_ms: r.latency_ms, cost_usd: r.cost_usd });
      typing.replaceWith(node);
      window.TGAI.hydrate(node).then(scrollBottom);
      await loadConvs();
      const cur = state.conversations.find((c) => c.id === state.convId);
      if (cur) el.convTitle.textContent = cur.title || t("web.chat.title");
      el.del.hidden = false;
    } catch (err) {
      typing.className = "msg error"; typing.textContent = err.message;
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

  // ── start ────────────────────────────────────────────────────────────────
  (async () => {
    updateCtxUI();
    try { await loadMe(); } catch (_) { return; }
    await Promise.all([loadDialogs(false), loadConvs()]);
    const m = location.hash.match(/^#c(\d+)$/);
    openConv(m ? +m[1] : null);
  })();
})();
