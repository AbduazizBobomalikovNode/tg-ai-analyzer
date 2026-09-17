/* lib/sse.js — POST + `ReadableStream` ustidagi SSE o'quvchisi.
 *
 * `EventSource` ishlatib bo'lmaydi: u faqat GET va tana yubormaydi,
 * bizda esa `POST /api/deep/stream` (savol tanada, CSRF sarlavhasi bilan).
 *
 * Nozik joylar (chat.js dan ko'chirilgan, o'lchov bilan tekshirilgan):
 *   * ramka chegarasi `\r?\n\r?\n` — proxy CRLF qo'shishi mumkin;
 *   * bir hodisada bir nechta `data:` satri **`\n` bilan** birlashadi
 *     (spetsifikatsiya) — yopishtirib yuborilsa ko'p satrli JSON buziladi;
 *   * bufer qopqog'i 2 MB — chegarasiz ramka xotirani yeydi.
 */

import { errorOf, ApiError } from "./api.js";

/** Bitta ramka shundan oshsa — oqim uziladi. */
export const SSE_MAX_BUFFER = 2 * 1024 * 1024;

/**
 * SSE ramkasini tahlil qiladi.
 * @param {string} frame  `\n\n` gacha bo'lgan matn
 * @returns {{kind: string, data: any}|null}  JSON emas yoki bo'sh → null
 */
export function parseFrame(frame) {
  let kind = "message";
  const data = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) kind = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  const body = data.join("\n").trim();
  if (!body) return null;
  try { return { kind, data: JSON.parse(body) }; } catch { return null; }
}

/**
 * `path` ga POST qilib SSE hodisalarini `onEvent` ga uzatadi.
 *
 * @param {string} path
 * @param {unknown} body  JSON tana
 * @param {(ev: {kind: string, data: any}) => void} onEvent
 * @param {{signal?: AbortSignal}} [opts]
 * @returns {Promise<void>}  oqim tugaganda hal bo'ladi
 * @throws {ApiError}  HTTP xatosi, 401, yoki bufer qopqog'i oshib ketsa
 */
export async function stream(path, body, onEvent, opts = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Requested-With": "fetch" },
    body: JSON.stringify(body),
    credentials: "same-origin",
    signal: opts.signal,
  });
  if (res.status === 401) {
    if (typeof window !== "undefined") window.location.href = "/login";
    throw new ApiError("error.unauthorized", 401, null);
  }
  if (!res.ok || !res.body) throw await errorOf(res);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let m;
    while ((m = /\r?\n\r?\n/.exec(buf)) !== null) {
      const ev = parseFrame(buf.slice(0, m.index));
      buf = buf.slice(m.index + m[0].length);
      if (ev) onEvent(ev);
    }
    if (buf.length > SSE_MAX_BUFFER) {
      try { await reader.cancel(); } catch { /* allaqachon yopiq */ }
      throw new ApiError("chat.err.internal", 0, "sse buffer overflow");
    }
  }
}
