/* lib/api.js — YAGONA fetch yordamchisi.
 *
 * Nega bitta: audit paytida 5 xil nusxa bor edi va ularning ba'zisida 401
 * redirect yo'q edi — sessiya tugaganda sahifa jim o'lardi.
 *
 * Shartnoma (server tomoni `web/security.py`):
 *   * `X-Requested-With: fetch` — CSRF; o'zgartiruvchi API'lar talab qiladi.
 *   * `credentials: "same-origin"` — imzolangan cookie.
 *   * `Content-Type: application/json` **faqat tana bo'lsa** — GET'da
 *     ortiqcha sarlavha preflight keltirib chiqarishi mumkin.
 *   * 401 → `/login` (bir marta; qayta yo'naltirish siklisiz).
 *   * xato tanasi `{detail: {code, detail}}` → `code` i18n kaliti.
 */

import { t } from "./i18n.js";

export class ApiError extends Error {
  constructor(code, status, detail) {
    super(detail ? `${t(code)} (${detail})` : t(code));
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.detail = detail;
  }
}

let redirecting = false;

/** 401 — sessiya tugadi; bir marta `/login` ga. */
function toLogin() {
  if (redirecting) return;
  redirecting = true;
  if (typeof window !== "undefined" && window.location.pathname !== "/login") {
    window.location.href = "/login";
  }
}

/**
 * @param {"GET"|"POST"|"PUT"|"PATCH"|"DELETE"} method
 * @param {string} path
 * @param {unknown} [body]  berilsa JSON qilib yuboriladi
 * @param {{signal?: AbortSignal, raw?: boolean}} [opts]
 *   `raw: true` → `Response` ning o'zi qaytadi (SSE/blob uchun).
 * @returns {Promise<any>}
 * @throws {ApiError}
 */
export async function api(method, path, body, opts = {}) {
  const headers = { "X-Requested-With": "fetch" };
  if (body !== undefined) headers["Content-Type"] = "application/json";

  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "same-origin",
    signal: opts.signal,
  });

  if (res.status === 401) { toLogin(); throw new ApiError("error.unauthorized", 401, null); }
  if (opts.raw) {
    if (!res.ok) throw await errorOf(res);
    return res;
  }

  let data = null;
  try { data = await res.json(); } catch { /* bo'sh tana — 204 */ }
  if (!res.ok) {
    const d = (data && data.detail) || {};
    throw new ApiError(d.code || "error.generic", res.status, d.detail || null);
  }
  return data;
}

/** `Response` dan `ApiError` yasaydi (tanani JSON deb o'qishga urinadi). */
export async function errorOf(res) {
  let d = {};
  try { d = (await res.json()).detail || {}; } catch { /* JSON emas */ }
  return new ApiError(d.code || "error.generic", res.status, d.detail || null);
}

export const get = (path, opts) => api("GET", path, undefined, opts);
export const post = (path, body, opts) => api("POST", path, body ?? {}, opts);
export const put = (path, body, opts) => api("PUT", path, body ?? {}, opts);
export const patch = (path, body, opts) => api("PATCH", path, body ?? {}, opts);
export const del = (path, opts) => api("DELETE", path, undefined, opts);
