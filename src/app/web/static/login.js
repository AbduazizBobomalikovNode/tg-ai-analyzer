/* Telefon → kod → 2FA. Kod/parol faqat shu HTTPS API'ga ketadi. */
(function () {
  const S = window.TGAI.strings;
  const t = (k, vars) => {
    let v = S[k] || k;
    if (vars) for (const [key, val] of Object.entries(vars)) v = v.replace(`{${key}}`, val);
    return v;
  };
  const $ = (id) => document.getElementById(id);
  const steps = { phone: $("step-phone"), code: $("step-code"), password: $("step-password"), done: $("step-done") };
  const alertBox = $("alert");
  let flowId = null;
  let retryTimer = null;

  function show(step) {
    for (const [name, el] of Object.entries(steps)) el.hidden = name !== step;
    const first = steps[step].querySelector("input");
    if (first) setTimeout(() => first.focus(), 30);
  }
  function showError(msg) { alertBox.textContent = msg; alertBox.hidden = false; }
  function clearError() { alertBox.hidden = true; alertBox.textContent = ""; }

  async function api(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Requested-With": "fetch" },
      body: JSON.stringify(body || {}),
      credentials: "same-origin",
    });
    let data = null;
    try { data = await res.json(); } catch (_) { /* bo'sh */ }
    if (!res.ok) {
      const d = (data && data.detail) || {};
      const err = new Error(t(d.code || "error.generic"));
      err.code = d.code; err.retryAfter = d.retry_after; err.status = res.status;
      throw err;
    }
    return data;
  }

  function busy(btn, on) { btn.disabled = on; btn.dataset.label ||= btn.textContent; btn.textContent = on ? "…" : btn.dataset.label; }

  function handleError(err, btn) {
    showError(err.message);
    if (err.retryAfter && btn) {
      let left = err.retryAfter;
      clearInterval(retryTimer);
      btn.disabled = true;
      const tick = () => {
        btn.textContent = t("web.login.retry_in", { seconds: left });
        if (left-- <= 0) { clearInterval(retryTimer); btn.disabled = false; btn.textContent = btn.dataset.label; }
      };
      tick(); retryTimer = setInterval(tick, 1000);
    }
    if (err.code === "auth.err.flow_expired" || err.code === "auth.err.too_many_attempts" ||
        err.code === "auth.err.code_expired" || err.code === "auth.err.wrong_step") {
      flowId = null; show("phone");
    }
  }

  steps.phone.addEventListener("submit", async (e) => {
    e.preventDefault(); clearError();
    const btn = $("btn-phone"); busy(btn, true);
    try {
      const data = await api("/api/auth/phone", { phone: $("phone").value.trim() });
      flowId = data.flow_id;
      const hintKey = { app: "app", sms: "sms", call: "call", flashcall: "call" }[data.code_type] || "other";
      $("code-hint").textContent = t(`web.login.code_hint.${hintKey}`);
      $("code").value = "";
      show("code");
    } catch (err) { handleError(err, btn); }
    finally { busy(btn, false); }
  });

  steps.code.addEventListener("submit", async (e) => {
    e.preventDefault(); clearError();
    const btn = $("btn-code"); busy(btn, true);
    try {
      const data = await api("/api/auth/code", { flow_id: flowId, code: $("code").value.trim() });
      if (data.status === "needs_2fa") { $("password").value = ""; show("password"); }
      else if (data.status === "done") finish();
    } catch (err) { handleError(err, btn); }
    finally { busy(btn, false); }
  });

  steps.password.addEventListener("submit", async (e) => {
    e.preventDefault(); clearError();
    const btn = $("btn-password"); busy(btn, true);
    try {
      const data = await api("/api/auth/password", { flow_id: flowId, password: $("password").value });
      if (data.status === "done") finish();
    } catch (err) { handleError(err, btn); }
    finally { busy(btn, false); }
  });

  document.querySelectorAll("[data-back]").forEach((b) => b.addEventListener("click", async () => {
    clearError();
    if (flowId) { try { await api("/api/auth/cancel", { flow_id: flowId }); } catch (_) { /* ignore */ } }
    flowId = null; show("phone");
  }));

  function finish() {
    show("done");
    $("password").value = ""; $("code").value = "";
    setTimeout(() => { window.location.href = "/chat"; }, 700);
  }
})();
