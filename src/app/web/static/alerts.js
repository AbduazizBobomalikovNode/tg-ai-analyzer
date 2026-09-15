/* Alertlar (phase3/C) — dashboard'dagi minimal CRUD bo'limi.
 *
 * Barcha matn `esc()` dan o'tadi: server allaqachon `untrusted.neutralize` +
 * `html.escape` qilgan, lekin ikkinchi qatlam bepul va `md.js` dagi bilan bir
 * xil tamoyil — "matn avval escape, teglar keyin".
 */
(function () {
  const S = (window.TGAI && window.TGAI.strings) || {};
  const t = (k, vars) => {
    let v = S[k] || k;
    if (vars) for (const [key, val] of Object.entries(vars)) v = v.replaceAll(`{${key}}`, val);
    return v;
  };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const root = $("alerts");
  if (!root) return;

  const H = { "X-Requested-With": "fetch", "Content-Type": "application/json" };

  async function api(path, opts) {
    const res = await fetch(path, { credentials: "same-origin", headers: H, ...(opts || {}) });
    if (res.status === 401) { window.location.href = "/login"; throw new Error("unauthorized"); }
    const body = res.status === 204 ? {} : await res.json().catch(() => ({}));
    if (!res.ok) {
      const code = body && body.detail && body.detail.code;
      throw new Error(t(code || "error.generic"));
    }
    return body;
  }

  function note(msg) {
    const el = $("alerts-note");
    if (el) el.textContent = msg || "";
  }

  function ruleRow(r) {
    const chat = r.chat_id == null ? t("web.alerts.chat_all") : "#" + r.chat_id;
    return `<tr data-id="${r.id}">
      <td><b>${esc(r.label || r.pattern)}</b><div class="muted small">${esc(r.pattern)}</div></td>
      <td>${esc(t("web.alerts.kind_" + r.kind))}</td>
      <td>${esc(chat)}</td>
      <td>${r.kind === "semantic" ? r.threshold.toFixed(2) : "—"}</td>
      <td>${r.cooldown_minutes}</td>
      <td>
        <button class="btn ghost small" data-act="toggle">${esc(r.enabled ? t("web.alerts.off") : t("web.alerts.on"))}</button>
        <button class="btn ghost small" data-act="events">${esc(t("web.alerts.events"))}</button>
        <button class="btn ghost small" data-act="del">${esc(t("web.alerts.delete"))}</button>
      </td>
    </tr>`;
  }

  function eventRow(e) {
    const link = e.link ? `<a href="${esc(e.link)}" target="_blank" rel="noopener">#${e.tg_msg_id}</a>` : `#${e.tg_msg_id}`;
    const when = (e.created_at || "").replace("T", " ").slice(0, 16);
    return `<li><span class="muted small">${esc(when)}</span> <b>${esc(e.chat_title)}</b> ${link} — ${esc(e.text)}</li>`;
  }

  async function load() {
    const tbl = $("alerts-tbl");
    try {
      const data = await api("/api/alerts");
      const items = data.items || [];
      tbl.innerHTML = items.length
        ? `<thead><tr>
             <th>${esc(t("web.alerts.pattern"))}</th><th>${esc(t("web.alerts.kind"))}</th>
             <th>${esc(t("web.alerts.chat"))}</th><th>${esc(t("web.alerts.threshold"))}</th>
             <th>${esc(t("web.alerts.cooldown"))}</th><th></th>
           </tr></thead><tbody>${items.map(ruleRow).join("")}</tbody>`
        : `<tbody><tr><td class="muted">${esc(t("web.alerts.no_rules"))}</td></tr></tbody>`;
      root.dataset.rules = JSON.stringify(items.map((r) => [r.id, r.enabled]));
    } catch (err) {
      note(err.message);
    }
  }

  async function showEvents(id) {
    const ul = $("alerts-events");
    try {
      const data = await api(`/api/alerts/${id}/events?limit=20`);
      const items = data.items || [];
      ul.innerHTML = items.length
        ? items.map(eventRow).join("")
        : `<li class="muted">${esc(t("web.alerts.no_events"))}</li>`;
    } catch (err) {
      note(err.message);
    }
  }

  $("alerts-tbl").addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const id = Number(btn.closest("tr").dataset.id);
    const act = btn.dataset.act;
    note("");
    try {
      if (act === "events") return void (await showEvents(id));
      if (act === "del") {
        if (!window.confirm(t("web.alerts.confirm_delete"))) return;
        await api(`/api/alerts/${id}`, { method: "DELETE" });
      } else if (act === "toggle") {
        const map = new Map(JSON.parse(root.dataset.rules || "[]"));
        await api(`/api/alerts/${id}`, { method: "PUT", body: JSON.stringify({ enabled: !map.get(id) }) });
      }
      await load();
    } catch (err) {
      note(err.message);
    }
  });

  $("alerts-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    note("");
    const chat = $("al-chat").value.trim();
    const payload = {
      account_id: Number($("al-account").value || 0),
      chat_id: chat ? Number(chat) : null,
      kind: $("al-kind").value,
      label: $("al-label").value.trim(),
      pattern: $("al-pattern").value.trim(),
      threshold: Number($("al-threshold").value || 0.8),
      cooldown_minutes: Number($("al-cooldown").value || 30),
      enabled: true,
    };
    try {
      await api("/api/alerts", { method: "POST", body: JSON.stringify(payload) });
      $("al-pattern").value = "";
      $("al-label").value = "";
      await load();
    } catch (err) {
      note(err.message);
    }
  });

  load();
})();
