# Ma'lumot modeli

Alembic migratsiyalar: `0001` asosiy sxema, `0002` conversations, `0003` chat sync
progress, `0004` sifat/xarajat ustunlari, `0005` chat_digests.

```
users(id, tg_user_id, username, locale)
accounts(id, user_id, tg_account_id, label, phone_hash, status,
         session_ciphertext, session_wrapped_dek, last_seen_at)
chats(id, account_id, tg_peer_id, access_hash, type, title, username,
      is_writable, is_admin, participants_count, write_mode,
      sync_state, synced_min_id, synced_max_id, synced_total,
      total_estimate, sync_error, last_message_at, last_sync_at)
messages(id, chat_id, tg_msg_id, sender_id, published_at, edited_at, text,
         media_type, reply_to_msg_id, fwd_from_id, grouped_id, is_pinned,
         views, forwards, reactions_total, replies_count)
  ix_messages_fts (GIN to_tsvector('simple')), ix_messages_trgm (gin_trgm_ops)
message_embeddings(message_id, model, vector(EMBED_DIM))  hnsw cosine
message_metric_snapshots(id, message_id, captured_at, views, forwards,
                         replies_count, reactions_total, reactions jsonb)
chat_daily_rollups(...)            (4-bosqich, hali to'ldirilmaydi)
chat_digests(id, chat_id, day, digest, msg_count, model, tokens_in, tokens_out)
conversations(id, user_id, account_id, title, updated_at)
conversation_messages(id, conversation_id, role, content, model, provider,
                      tokens_in, tokens_out, context jsonb, task, latency_ms,
                      cost_usd, rating, rating_comment, rated_at,
                      auto_relevance, auto_usefulness, auto_grounded, auto_note)
agent_runs(id, user_id, account_id, chat_id, prompt, model, tokens_in, tokens_out)
agent_actions(id, run_id, tool, args jsonb, status, block_reason,
              target_peer_id, result_msg_id, error, confirmed_at)   ← append-only trigger
scheduled_jobs(...)                (7-bosqich)
```

## Nozik nuqtalar

* **Media saqlanmaydi** — faqat `messages.media_type`.
* **`message_metric_snapshots`** — views/reactions vaqt qatori. Telegram tarixiy
  qiymat bermaydi; "bu hafta +N ko'rish" faqat shu jadvaldan. Cron o'chirilsa
  bo'shliq abadiy.
* `messages.views` — oxirgi qiymat (tez filtr), tarix emas.
* `chats.synced_min_id == 1` — tarix boshiga yetilgan sentinel; `sync_state`
  idle/running/done/failed. `write_mode` — per-chat: read_only /
  write_with_confirm / autonomous (akkauntga bitta).
* `chat_digests` — kunlik digest keshi; `msg_count` farq qilsa qayta hisoblanadi.
* `conversation_messages.context` (jsonb) — kontekst **meta**'si (chat, strategiya,
  manba, taxminiy token, tool chaqiruvlari, run_id, action_ids), kontent emas.
* `agent_actions.args` — yozish uchun normallashtirilgan argumentlar
  (`chat_id`, `peer_id`, `text`, …); UI preview shundan.
