/* lib/actions-badge.js — rail/tabbar badge'i: tasdiq kutayotgan amallar soni.
 *
 * Karkas darajasidagi bitta yengil so'rov (`GET /api/actions?status=proposed`).
 * Sahifa o'zi ham sonni bilsa (`/actions`, `/today`) shu atributni to'g'ridan
 * to'g'ri yangilaydi — shuning uchun sanoq mantiqi `setActionsBadge()` da
 * yagona nuqtada turadi.
 *
 * Xato bo'lsa **jim** o'tadi: badge — bezak, sahifani yiqitmasligi kerak.
 */

const SEL = "[data-actions-badge]";

/** Badge sonini qo'yadi (0 bo'lsa yashiradi). */
export function setActionsBadge(n) {
  const count = Number(n) || 0;
  for (const node of document.querySelectorAll(SEL)) {
    node.textContent = String(count);
    node.hidden = count === 0;
  }
  return count;
}

/** Bir marta so'raydi va badge'ni to'ldiradi. */
export async function initActionsBadge() {
  if (!document.querySelector(SEL)) return 0;
  try {
    const res = await fetch("/api/actions?status=proposed&limit=200", {
      credentials: "same-origin",
      headers: { "X-Requested-With": "fetch" },
    });
    if (!res.ok) return 0;
    const data = await res.json();
    return setActionsBadge((data.items || []).length);
  } catch {
    return 0;   // tarmoq/sessiya — badge shunchaki ko'rinmaydi
  }
}
