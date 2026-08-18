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
    send: $("btn-send"), deep: $("deep"), ctxLimit: $("ctx-limit"), ctxLimitLabel: $("ctx-limit-label"),
    ctxLimitWrap: $("ctx-limit-wrap"), convTitle: $("conv-title"), ctxBadge: $("ctx-badge"),
    del: $("btn-delete"),
  };

  const state = {
    accounts: [], accountId: null, dialogs: [], peerId: 0, peerTitle: "",
    conversations: [], convId: null, busy: false,
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

  // ── minimal, xavfsiz markdown ────────────────────────────────────────────
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function md(src) {
    const codeBlocks = [];
    let text = esc(src).replace(/```([\s\S]*?)```/g, (_, code) => {
      codeBlocks.push(`<pre><code>${code.replace(/^\w+\n/, "")}</code></pre>`);
      return `@@CODEBLOCK${codeBlocks.length - 1}@@`;
    });
    text = text
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/(^|[\s(])_([^_\n]+)_/g, "$1<em>$2</em>");
    const lines = text.split("\n");
    const out = []; let list = null;
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of lines) {
      const line = raw.trimEnd();
      let m;
      if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; } out.push(`<li>${m[1]}</li>`); continue; }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; } out.push(`<li>${m[1]}</li>`); continue; }
      closeList();
      if ((m = line.match(/^#{1,6}\s+(.*)$/))) { out.push(`<p><strong>${m[1]}</strong></p>`); continue; }
      if (line.trim() === "") { out.push(""); continue; }
      out.push(`<p>${line}</p>`);
    }
    closeList();
    return out.join("\n").replace(/(?:<p>)?@@CODEBLOCK(\d+)@@(?:<\/p>)?/g, (_, i) => codeBlocks[+i]);
  }

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

  function renderDialogs() {
    const q = el.dialogSearch.value.trim().toLowerCase();
    const items = state.dialogs.filter((d) => !q || d.title.toLowerCase().includes(q) || (d.username || "").toLowerCase().includes(q));
    const noneSel = state.peerId === 0 ? " selected" : "";
    let html = `<li class="dialog-item${noneSel}" data-peer="0"><span class="dialog-avatar">∅</span><span class="dialog-body"><span class="dialog-title">${esc(t("web.chat.context_none"))}</span></span></li>`;
    for (const d of items) {
      html += `<li class="dialog-item${d.peer_id === state.peerId ? " selected" : ""}" data-peer="${d.peer_id}" data-title="${esc(d.title)}">
        <span class="dialog-avatar">${esc(initials(d.title))}</span>
        <span class="dialog-body"><span class="dialog-title">${kindIcon[d.kind] || ""} ${esc(d.title)}</span>
        <span class="dialog-sub">${esc(d.last_message_text || "")}</span></span></li>`;
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

  function msgNode(m) {
    const div = document.createElement("div");
    div.className = `msg ${m.role}`;
    let inner = "";
    if (m.context && m.context.title) inner += `<span class="ctx">📎 ${esc(t("web.chat.context_used", { title: m.context.title, n: m.context.messages }))}</span>`;
    inner += m.role === "assistant" ? md(m.content) : `<p>${esc(m.content).replace(/\n/g, "<br>")}</p>`;
    if (m.role === "assistant" && m.model) inner += `<span class="meta">${esc(m.provider || "")} · ${esc(m.model)} · ${m.tokens_in || 0}→${m.tokens_out || 0}</span>`;
    div.innerHTML = inner;
    return div;
  }

  function renderMessages(items) {
    el.messages.innerHTML = "";
    if (!items.length) { el.messages.appendChild(el.empty); el.empty.hidden = false; return; }
    for (const m of items) el.messages.appendChild(msgNode(m));
    scrollBottom();
  }
  function scrollBottom() { el.messages.scrollTop = el.messages.scrollHeight; }

  function updateCtxUI() {
    const n = el.ctxLimit.value;
    el.ctxLimitLabel.textContent = t("web.chat.context_last", { n });
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
  async function loadDialogs(refresh) {
    if (!state.accountId) { state.dialogs = []; renderDialogs(); return; }
    try {
      const data = await api("GET", `/api/accounts/${state.accountId}/dialogs${refresh ? "?refresh=1" : ""}`);
      state.dialogs = data.items; el.accountWarn.hidden = true;
    } catch (err) {
      state.dialogs = [];
      el.accountWarn.textContent = err.message; el.accountWarn.hidden = false;
    }
    renderDialogs();
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
      const body = { text, deep: el.deep.checked };
      if (state.peerId && state.accountId) body.context = { account_id: state.accountId, peer_id: state.peerId, limit: +el.ctxLimit.value };
      const r = await api("POST", `/api/conversations/${state.convId}/messages`, body);
      typing.replaceWith(msgNode({ role: "assistant", content: r.text, model: r.model, provider: r.provider, tokens_in: r.tokens_in, tokens_out: r.tokens_out, context: r.context }));
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
    const li = e.target.closest(".dialog-item"); if (!li) return;
    state.peerId = +li.dataset.peer; state.peerTitle = li.dataset.title || "";
    renderDialogs(); closeSidebarMobile();
  });
  el.convList.addEventListener("click", (e) => {
    const li = e.target.closest(".conv-item"); if (!li) return;
    openConv(+li.dataset.conv); closeSidebarMobile();
  });
  el.accountSelect.addEventListener("change", () => { state.accountId = +el.accountSelect.value; state.peerId = 0; loadDialogs(false); });
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
    openConv(null);
  })();
})();
