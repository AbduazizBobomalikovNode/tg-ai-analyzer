# Bitta promptda to'liq funksional test

AI chatga (`/chat`) shu matnni to'liq nusxalab qo'ying. Bitta so'rovda 15 ta
tool ishga tushadi va agent o'zi hisobot yozadi.

**Talab:** `AGENT_MAX_ITERATIONS=12` (shiftli qiymat). 5 da ham ishlaydi, lekin
oxirigacha yetmasligi mumkin. Narxi ≈ 44k kirish tokeni (~$0.02 DeepSeek v4-flash).

**Chat nomlari va xabar ID'lari o'zingiznikiga moslang** — quyidagilar namuna.

---

```
TIZIM TESTI. Quyidagi 15 bandni ketma-ket bajar. Har biri uchun natijani
bitta-ikki qator qilib yoz: ✅ ishladi (nima topilgani) yoki ❌ ishlamadi
(aniq sabab). Oxirida "Ishladi: N/15" deb yakunla. Javoblarni qisqa tut —
bu funksional test, tahlil emas.

1. Sinxronlangan chatlarim ro'yxatini ko'rsat (faqat soni va eng faol 3 tasi).
2. "Honadoshlar" guruhida "kv puli" haqida qidir (5 ta natija).
3. "Honadoshlar" dagi oxirgi 5 ta xabar.
4. "Honadoshlar" dagi #1775 xabar atrofidagi kontekst (radius 3).
5. "Honadoshlar" ning so'nggi 7 kunlik kunlik digesti.
6. "backup Re-electronik" kanalining 30 kunlik statistikasi.
7. "Honadoshlar" guruhi haqida ma'lumot: a'zolar soni, media hajmi.
8. "Bobomalikov Abduaziz's blog" kanalining a'zolar ro'yxati (10 tasi yetarli).
9. "Firuz" haqida profil ma'lumoti (bio, umumiy chatlar).
10. "Honadoshlar" dagi #1784 xabarga kim reaksiya qo'ygan.
11. "Bobomalikov Abduaziz's blog" dagi #5 post ostidagi izohlar.
12. "Firuz" bilan yozishuvda Telegram serveridan "salom" so'zini qidir (5 ta).
13. Barcha chatlarim bo'ylab "backup" so'zini qidir (5 ta).
14. "Bobomalikov Abduaziz's blog" da rejalashtirilgan postlar bormi.
15. Menga "moviy fonda oq robot" rasmini generatsiya qilib ber.
```

---

## Band → tool xaritasi

| # | Tool | Manba |
|---|---|---|
| 1 | `list_chats` | DB |
| 2 | `search_messages` | DB (FTS + vektor) |
| 3 | `get_recent_messages` | DB |
| 4 | `get_message_context` | DB |
| 5 | `get_window_digest` | DB + kesh |
| 6 | `get_chat_stats` | DB + snapshotlar |
| 7 | `get_chat_info` | MTProto |
| 8 | `list_members` | MTProto |
| 9 | `get_user_info` | MTProto |
| 10 | `get_reactions` | MTProto |
| 11 | `get_post_comments` | MTProto |
| 12 | `search_telegram` | MTProto |
| 13 | `search_global` | MTProto |
| 14 | `list_scheduled_messages` | MTProto |
| 15 | `generate_image` | Gemini |

## Kutilayotgan "muvaffaqiyatsizliklar"

Bular xato emas — Telegram/provider cheklovi. ❌ chiqsa, sabab shu:

- **11 (`get_post_comments`)** — kanalga bog'langan muhokama guruhi ulanmagan
  bo'lsa izoh bo'lmaydi (`MsgIdInvalidError`).
- **15 (`generate_image`)** — Gemini **bepul tarifida rasm modeli kvotasi
  nol** (`limit: 0, model: gemini-2.5-flash-preview-image`). Pullik rejaga
  o'tmasangiz ishlamaydi; `IMAGE_GEN_ENABLED=false` qilib o'chirsangiz,
  agent behuda 3 marta urinmaydi.
- **10 (`get_reactions`)** kanalda — Telegram broadcast'da kim bosganini
  oshkor qilmaydi (`BroadcastForbiddenError`). Guruhda ishlaydi.

## Qisqa variantlar

Byudjetni tejash uchun bo'lib sinash:

- **DB yo'li:** 1–6 bandlar
- **MTProto yo'li:** 7–14 bandlar
- **Bir savolli tez tekshiruv:** `"Honadoshlar" guruhi haqida hamma narsani
  ayt: a'zolar, media, so'nggi hafta nima bo'lgan, eng faol odam kim?`
