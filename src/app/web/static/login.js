/* Telefon → kod → 2FA. Kod/parol faqat shu HTTPS API'ga ketadi. */
(function () {
  const S = window.TGAI.strings;
  const t = (k, vars) => {
    let v = S[k] || k;
    if (vars) for (const [key, val] of Object.entries(vars)) v = v.replace(`{${key}}`, val);
    return v;
  };
  const $ = (id) => document.getElementById(id);
  // QR — asosiy yo'l (`CLAUDE.md`: "QR login — asosiy yo'l"). Telefon formasi
  // QR kartasi ichidagi disclosure'da, shuning uchun u alohida qadam emas:
  // qadam mashinasida faqat qr · code · password · done bor.
  const steps = { qr: $("step-qr"), code: $("step-code"), password: $("step-password"), done: $("step-done") };
  const openPhone = () => { const d = $("phone-fallback"); if (d) d.open = true; };
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

  // ── H7: QR muvaffaqiyatsiz bo'lganda ("429/5xx/tarmoq") ──────────────────
  // Avval: rasm bo'sh/buzilgan qolardi, lekin "Skanerlash kutilmoqda…" matni
  // baribir ko'rsatilardi — foydalanuvchi mavjud bo'lmagan QR'ni kutardi.
  // Endi: QR blokining o'zi yashiriladi, o'rniga xato + "qayta urinish"
  // kartasi (backoff bilan, `err.retryAfter` bo'lsa hisoblagich).
  const qrLive = $("qr-live");
  const qrRetry = $("qr-retry");
  const qrRetryMsg = $("qr-retry-msg");
  const btnQrRetry = $("btn-qr-retry");
  let qrRetryTimer = null;

  function showQrLive() {
    clearInterval(qrRetryTimer);
    qrRetry.hidden = true;
    qrLive.hidden = false;
  }

  function showQrRetry(err) {
    stopQrPoll();
    qrLive.hidden = true;
    qrRetryMsg.textContent = err.message;
    qrRetry.hidden = false;
    clearInterval(qrRetryTimer);
    btnQrRetry.disabled = false;
    btnQrRetry.dataset.label ||= btnQrRetry.textContent;
    if (err.retryAfter) {
      let left = err.retryAfter;
      btnQrRetry.disabled = true;
      const tick = () => {
        btnQrRetry.textContent = t("web.login.retry_in", { seconds: left });
        if (left-- <= 0) {
          clearInterval(qrRetryTimer);
          btnQrRetry.disabled = false;
          btnQrRetry.textContent = btnQrRetry.dataset.label;
        }
      };
      tick();
      qrRetryTimer = setInterval(tick, 1000);
    } else {
      btnQrRetry.textContent = btnQrRetry.dataset.label;
    }
  }

  btnQrRetry?.addEventListener("click", () => startQr());

  // ── kod qadami: izoh + "qayta yuborish" tugmasi ──────────────────────────
  // Telegram faol sessiya ko'rsa kodni SMS emas, ilovaga tashlaydi
  // (code_type=app). `next_type` — qayta yuborilsa qanday kelishi; bo'sh
  // bo'lsa Telegram qayta yuborishga ruxsat bermaydi, tugma ko'rsatilmaydi.
  let resendTimer = null;

  function applyCodeStep(data) {
    const hintKey = { app: "app", sms: "sms", call: "call", flashcall: "call" }[data.code_type] || "other";
    $("code-hint").textContent = t(`web.login.code_hint.${hintKey}`);
    setupResend(data.next_type, data.timeout);
  }

  function setupResend(nextType, timeout) {
    const btn = $("btn-resend");
    clearInterval(resendTimer);
    if (!nextType) { btn.hidden = true; return; }
    btn.hidden = false;
    const label = t(`web.login.resend_as.${nextType === "sms" ? "sms" : "other"}`);
    let left = Math.max(0, Number(timeout) || 0);
    const tick = () => {
      if (left > 0) {
        btn.disabled = true;
        btn.textContent = t("web.login.resend_wait", { s: left, how: label });
        left -= 1;
      } else {
        btn.disabled = false;
        btn.textContent = label;
        clearInterval(resendTimer);
      }
    };
    tick();
    if (left > 0) resendTimer = setInterval(tick, 1000);
  }

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
      flowId = null; show("qr"); startQr();
    }
  }

  $("step-phone").addEventListener("submit", async (e) => {
    e.preventDefault(); clearError();
    const btn = $("btn-phone"); busy(btn, true);
    try {
      const data = await api("/api/auth/phone", { phone: $("phone").value.trim() });
      flowId = data.flow_id;
      applyCodeStep(data);
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

  $("btn-resend").addEventListener("click", async () => {
    clearError();
    const btn = $("btn-resend"); busy(btn, true);
    try {
      const data = await api("/api/auth/resend", { flow_id: flowId });
      applyCodeStep(data);          // yangi code_type/next_type/timeout
      $("code").value = "";
      $("code").focus();
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
    clearError(); stopQrPoll();
    if (flowId) { try { await api("/api/auth/cancel", { flow_id: flowId }); } catch (_) { /* ignore */ } }
    flowId = null; show("qr"); startQr();
  }));

  // ── QR login ─────────────────────────────────────────────────────────────
  // Telefon/kod umuman kerak emas. QR server tomonda SVG qilib chiziladi va
  // <img src=data:…> bo'lib keladi. Token ~1 daqiqada eskiradi — backend uni
  // o'zi yangilaydi, biz shunchaki holatni so'rab turamiz.
  let qrPoll = null;

  function stopQrPoll() { clearInterval(qrPoll); qrPoll = null; }

  /** QR oqimini boshlaydi (yoki yangilaydi). Xato bo'lsa retry karta ko'rsatiladi. */
  async function startQr() {
    clearError();
    const btn = $("btn-qr"); busy(btn, true);
    try {
      const data = await api("/api/auth/qr", {});
      flowId = data.flow_id;
      if (data.qr) $("qr-img").src = data.qr;
      showQrLive();
      show("qr");
      startQrPoll();
    } catch (err) {
      showQrRetry(err);
      openPhone();   // QR ishlamasa — telefon zaxira yo'li ko'rinib tursin
    } finally { busy(btn, false); }
  }

  $("btn-qr").addEventListener("click", () => startQr());

  function startQrPoll() {
    stopQrPoll();
    qrPoll = setInterval(async () => {
      if (!flowId) return stopQrPoll();
      let data;
      try {
        data = await api("/api/auth/qr/poll", { flow_id: flowId });
      } catch (err) {
        show("qr");
        showQrRetry(err);
        return;
      }
      if (data.qr) $("qr-img").src = data.qr;       // token yangilangan bo'lsa
      if (data.error) $("qr-hint").textContent = t("auth.err.telegram");
      if (data.status === "needs_2fa") {
        stopQrPoll(); $("password").value = ""; show("password");
      } else if (data.status === "done") {
        stopQrPoll(); finish();
      }
    }, 2000);
  }

  function finish() {
    show("done");
    $("password").value = ""; $("code").value = "";
    // `/` o'zi hal qiladi: onboarding tugamagan bo'lsa `/onboarding`, aks holda `/today`.
    setTimeout(() => { window.location.href = "/"; }, 700);
  }

  // QR-first: sahifa ochilishi bilan QR chiziladi, foydalanuvchi hech nima
  // bosmaydi. Telefon formasi disclosure ichida yopiq turadi.
  startQr();
})();
