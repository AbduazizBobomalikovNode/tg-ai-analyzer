/* Shaxsiy briefing sahifasi (phase3/B).
 *
 * Briefing matni markdown — `window.TGAI.md` (yagona renderer, md.js) orqali
 * chiziladi. Boshqa har qanday matn `TGAI.esc` bilan escape qilinadi;
 * `innerHTML` ga faqat shu ikkitasining chiqishi tushadi.
 */
(function () {
  const S = window.TGAI.strings;
  const t = (k) => S[k] || k;
  const $ = (id) => document.getElementById(id);
  const esc = window.TGAI.esc;

  async function api(path, opts) {
    const res = await fetch(path, {
      credentials: "same-origin",
      headers: { "X-Requested-With": "fetch", ...(opts && opts.body ? { "Content-Type": "application/json" } : {}) },
      ...opts,
    });
    if (res.status === 401) { window.location.href = "/login"; throw new Error("unauthorized"); }
    if (!res.ok) throw new Error(String(res.status));
    return res.json();
  }

  function renderBody(row) {
    $("brief-day").textContent = row.day;
    $("brief-body").innerHTML = window.TGAI.md(row.text || "");
    window.TGAI.hydrate($("brief-body"));
    const bits = [];
    if (row.model) bits.push(esc(row.model));
    if (row.tokens) bits.push(row.tokens + " tok");
    if (row.cost_usd) bits.push("$" + Number(row.cost_usd).toFixed(4));
    bits.push(row.delivered_at ? t("web.brief.delivered") : t("web.brief.not_delivered"));
    $("brief-meta").innerHTML = bits.join(" · ");
  }

  async function loadOne(day) {
    try {
      renderBody(await api("/api/briefings/" + encodeURIComponent(day)));
    } catch (e) {
      $("brief-body").innerHTML = "<p class='muted'>" + esc(t("web.brief.none")) + "</p>";
      $("brief-meta").textContent = "";
    }
  }

  async function loadList() {
    const data = await api("/api/briefings?limit=14");
    const list = $("brief-list");
    list.innerHTML = "";
    if (!data.items.length) {
      list.innerHTML = "<li class='muted'>" + esc(t("web.brief.none")) + "</li>";
      return;
    }
    for (const it of data.items) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = "/briefings/" + it.day;
      a.textContent = it.day;
      a.addEventListener("click", (ev) => { ev.preventDefault(); history.pushState({}, "", a.href); loadOne(it.day); });
      li.appendChild(a);
      const span = document.createElement("span");
      span.className = "muted small";
      span.textContent = " " + (it.tldr || "").slice(0, 120);
      li.appendChild(span);
      list.appendChild(li);
    }
    if (!window.TGAI_BRIEF_DAY) loadOne(data.items[0].day);
  }

  async function loadSettings() {
    let data;
    try { data = await api("/api/settings/briefing"); } catch (e) { return; }
    const s = data.settings;
    $("b-enabled").checked = !!s.enabled;
    $("b-hour").value = s.hour;
    $("b-tz").value = s.tz;
    if (s.quiet_from) $("b-qfrom").value = s.quiet_from;
    if (s.quiet_to) $("b-qto").value = s.quiet_to;
    $("brief-settings").dataset.accountId = s.account_id;
  }

  $("brief-settings").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const accountId = Number($("brief-settings").dataset.accountId || 0);
    if (!accountId) return;
    const body = {
      account_id: accountId,
      tz: $("b-tz").value.trim(),
      hour: Number($("b-hour").value),
      enabled: $("b-enabled").checked,
      quiet_from: $("b-qfrom").value || null,
      quiet_to: $("b-qto").value || null,
      chats: null,
    };
    try {
      await api("/api/settings/briefing", { method: "PUT", body: JSON.stringify(body) });
      $("brief-saved").textContent = t("web.brief.saved");
    } catch (e) {
      $("brief-saved").textContent = t("error.generic");
    }
  });

  loadSettings();
  loadList();
  if (window.TGAI_BRIEF_DAY) loadOne(window.TGAI_BRIEF_DAY);
})();
