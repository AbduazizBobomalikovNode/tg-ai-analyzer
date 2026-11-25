# Vendor fayllar (CDN yo'q)

Bu papkadagi hamma narsa **lokal** beriladi. Runtime'da tashqi CDN'ga
murojaat qilinmaydi: uchinchi tomon hosti sahifamizga skript yuklashi
(supply-chain) ham, foydalanuvchini kuzatishi ham mumkin, va offline/bloklangan
tarmoqda UI shunchaki ishlamay qolardi.

Yangilash qoidasi: faylni **tarmoqli konteynerda** yuklab oling, shu jadvalni
yangilang (sana + sha256), keyin `git diff` bilan o'zgarishni ko'zdan
kechiring. Diffni o'qimasdan minifikatsiyalangan skriptni yangilamang.

Kutubxonalar ro'yxati (versiya/litsenziya) — `README.md`. Bu yerda esa
**qayerdan va qachon** olingani, mazmun sha256'si bilan.

| Fayl | Manba | Versiya | Yuklangan | sha256 | Litsenziya |
|---|---|---|---|---|---|
| `telegram-web-app.js` | `https://telegram.org/js/telegram-web-app.js` | `webAppVersion = '6.0'` (runtime'da `isVersionAtLeast` bilan tekshiriladi) | 2026-09-16 | `3549138a7934039fe7dfd1291a4ee739bd2b705a614308053a8b08a87d85c451` | Telegram (rasmiy Mini App SDK) |

## `telegram-web-app.js`

Telegram'ning rasmiy Mini App SDK'si: `window.Telegram.WebApp` (tema
o'zgaruvchilari, `initData`, `BackButton`, `MainButton`, `expand()`,
`HapticFeedback`). Telegram'ning o'zi bu faylni CDN'dan yuklashni tavsiya
qiladi, lekin biz **lokal nusxani** beramiz:

* fayl `<script src="/static/vendor/telegram-web-app.js">` bilan ulanadi
  (fingerprint uchun `asset()` orqali) — ya'ni versiya bizning nazoratimizda;
* `initData` faqat **serverda** tekshiriladi (`web/routers/auth.py::verify_init_data`,
  HMAC-SHA256, bot token log'ga hech qachon tushmaydi) — brauzerdagi
  skriptning rostgo'yligiga ishonilmaydi.

Ichki `webAppVersion` — SDK yozilgan paytdagi bazaviy versiya; haqiqiy
imkoniyat `Telegram.WebApp.isVersionAtLeast(...)` bilan runtime'da tekshiriladi.
