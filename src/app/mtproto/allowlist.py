"""MTProto RPC allowlist — "o'chirish imkonsiz"ligining strukturaviy kafolati.

Prinsip: **deny by default**. Ro'yxatda yo'q har qanday TL metod bloklanadi.
Prompt'da "o'chirma" deb yozish yetarli emas — LLM'ni ko'ndirish mumkin,
bu qatlamni esa yo'q.

Uch daraja:
  1. `DENIED` — hech qachon, hech qanday sharoitda. Qattiq blok.
  2. `INTERNAL` — Telethon'ning o'z ishi (connect, update, DC migration).
  3. `READ` / `WRITE` — biznes operatsiyalari.

`AUTH` alohida: faqat login jarayonida `auth_window()` konteksti ichida ochiladi,
qolgan vaqtda yopiq — ya'ni agent hech qachon logout qila olmaydi.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. HECH QACHON. Bu ro'yxat faqat o'sadi, hech qachon qisqarmaydi.
# ─────────────────────────────────────────────────────────────────────────────
DENIED: frozenset[str] = frozenset(
    {
        # xabar/tarix o'chirish
        "messages.DeleteMessagesRequest",
        "channels.DeleteMessagesRequest",
        "messages.DeleteHistoryRequest",
        "channels.DeleteHistoryRequest",
        "messages.DeleteScheduledMessagesRequest",
        "messages.DeletePhoneCallHistoryRequest",
        "messages.DeleteRevokedExportedChatInvitesRequest",
        "messages.DeleteExportedChatInviteRequest",
        "messages.DeleteChatRequest",
        "messages.DeleteChatUserRequest",
        "channels.DeleteParticipantHistoryRequest",
        "channels.DeleteChannelRequest",
        "channels.DeleteTopicHistoryRequest",
        "folders.DeleteFolderRequest",
        "stories.DeleteStoriesRequest",
        "contacts.DeleteContactsRequest",
        "contacts.DeleteByPhonesRequest",
        "photos.DeletePhotosRequest",
        "account.DeleteAccountRequest",
        "account.DeleteSecureValueRequest",
        # chiqib ketish / sessiyani buzish
        "auth.LogOutRequest",
        "auth.ResetAuthorizationsRequest",
        "account.ResetAuthorizationRequest",
        "account.ResetWebAuthorizationRequest",
        "account.ResetWebAuthorizationsRequest",
        "channels.LeaveChannelRequest",
        # profil/xavfsizlik o'zgartirish
        "account.UpdateProfileRequest",
        "account.UpdateUsernameRequest",
        "account.UpdatePasswordSettingsRequest",
        "account.ChangePhoneRequest",
        "account.ChangeAuthorizationSettingsRequest",
        "account.UpdateEmojiStatusRequest",
        "photos.UploadProfilePhotoRequest",
        # ban / kick / huquq o'zgartirish
        "channels.EditBannedRequest",
        "channels.EditAdminRequest",
        "channels.EditCreatorRequest",
        "messages.EditChatAdminRequest",
        "messages.EditChatDefaultBannedRightsRequest",
        # pul
        "payments.SendPaymentFormRequest",
        "payments.SendStarsFormRequest",
        # ommaviy tarqatish riski
        "contacts.ImportContactsRequest",
        "channels.InviteToChannelRequest",
        "messages.AddChatUserRequest",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Telethon ichki mexanikasi — busiz klient ulanmaydi.
# ─────────────────────────────────────────────────────────────────────────────
INTERNAL: frozenset[str] = frozenset(
    {
        "functions.InvokeWithLayerRequest",
        "functions.InitConnectionRequest",
        "functions.InvokeWithoutUpdatesRequest",
        "functions.InvokeWithTakeoutRequest",
        "help.GetConfigRequest",
        "help.GetNearestDcRequest",
        "help.GetAppConfigRequest",
        "help.GetTermsOfServiceUpdateRequest",
        "help.GetCountriesListRequest",
        "updates.GetStateRequest",
        "updates.GetDifferenceRequest",
        "updates.GetChannelDifferenceRequest",
        "auth.ExportAuthorizationRequest",
        "auth.ImportAuthorizationRequest",
        "langpack.GetLangPackRequest",
        "langpack.GetDifferenceRequest",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Login — faqat auth_window() ichida ochiladi.
# ─────────────────────────────────────────────────────────────────────────────
AUTH: frozenset[str] = frozenset(
    {
        "auth.ExportLoginTokenRequest",  # QR login — asosiy yo'l
        "auth.ImportLoginTokenRequest",
        "auth.AcceptLoginTokenRequest",
        "auth.SendCodeRequest",  # Mini App fallback
        "auth.SignInRequest",
        "auth.ResendCodeRequest",
        "account.GetPasswordRequest",  # 2FA
        "auth.CheckPasswordRequest",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. O'qish.
# ─────────────────────────────────────────────────────────────────────────────
READ: frozenset[str] = frozenset(
    {
        "messages.GetHistoryRequest",
        "messages.GetMessagesRequest",
        "channels.GetMessagesRequest",
        "messages.GetRepliesRequest",
        "messages.GetDiscussionMessageRequest",
        "messages.SearchRequest",
        "messages.SearchGlobalRequest",
        "messages.GetSearchCountersRequest",
        "messages.GetDialogsRequest",
        "messages.GetPeerDialogsRequest",
        "messages.GetPinnedDialogsRequest",
        "messages.GetMessagesViewsRequest",
        "messages.GetMessageReactionsListRequest",
        "messages.GetMessageReadParticipantsRequest",
        "messages.GetScheduledHistoryRequest",
        "messages.GetScheduledMessagesRequest",
        "messages.GetFullChatRequest",
        "channels.GetFullChannelRequest",
        "channels.GetChannelsRequest",
        "channels.GetParticipantsRequest",
        "channels.GetParticipantRequest",
        "users.GetFullUserRequest",
        "users.GetUsersRequest",
        "contacts.ResolveUsernameRequest",
        "contacts.GetContactsRequest",
        "contacts.SearchRequest",
        # kanal analitikasi (faqat admin + minimal obunachi bo'lsa ishlaydi)
        "stats.GetBroadcastStatsRequest",
        "stats.GetMegagroupStatsRequest",
        "stats.GetMessageStatsRequest",
        "stats.LoadAsyncGraphRequest",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Yozish. O'chirish yo'q — ataylab.
# ─────────────────────────────────────────────────────────────────────────────
WRITE: frozenset[str] = frozenset(
    {
        "messages.SendMessageRequest",
        "messages.SendMediaRequest",
        "messages.SendMultiMediaRequest",
        "messages.EditMessageRequest",
        "messages.ForwardMessagesRequest",  # copy ham shu (drop_author=True)
        "messages.UpdatePinnedMessageRequest",
        "messages.SendReactionRequest",
        "messages.SetTypingRequest",
        "messages.ReadHistoryRequest",
        "channels.ReadHistoryRequest",
        "messages.ReadMentionsRequest",
        "messages.SendScheduledMessagesRequest",
        "messages.SaveDraftRequest",
        "upload.SaveFilePartRequest",  # generatsiya qilingan rasm yuklash
        "upload.SaveBigFilePartRequest",
    }
)

ALLOWED: frozenset[str] = INTERNAL | READ | WRITE

# Sanity: allowlist va denylist kesishmasligi shart.
assert not (ALLOWED | AUTH) & DENIED, "allowlist DENIED bilan kesishdi"


class RpcBlocked(RuntimeError):
    """Guardrail to'sgan TL so'rov."""

    def __init__(self, method: str, reason: str) -> None:
        self.method = method
        self.reason = reason
        super().__init__(f"MTProto so'rov bloklandi: {method} — {reason}")


_auth_open: ContextVar[bool] = ContextVar("mtproto_auth_window", default=False)


@contextmanager
def auth_window() -> Any:
    """Login oynasi. Faqat auth flow shu kontekst ichida ishlaydi."""
    token = _auth_open.set(True)
    try:
        yield
    finally:
        _auth_open.reset(token)


def request_key(request: object) -> str:
    """TL so'rov obyektidan `modul.KlassNomi` kalitini yasaydi.

    `telethon.tl.functions.messages.GetHistoryRequest` → `messages.GetHistoryRequest`
    `telethon.tl.functions.InvokeWithLayerRequest`     → `functions.InvokeWithLayerRequest`
    """
    cls = type(request)
    module = getattr(cls, "__module__", "") or ""
    ns = module.rsplit(".", 1)[-1] if module else "?"
    return f"{ns}.{cls.__name__}"


def check_request(request: object) -> str:
    """So'rovni tekshiradi. Ruxsat bo'lsa kalitni qaytaradi, aks holda `RpcBlocked`."""
    key = request_key(request)

    if key in DENIED:
        raise RpcBlocked(key, "qora ro'yxatda (destruktiv operatsiya)")

    if key in AUTH:
        if not _auth_open.get():
            raise RpcBlocked(key, "auth metodi login oynasidan tashqarida")
        return key

    if key not in ALLOWED:
        raise RpcBlocked(key, "oq ro'yxatda yo'q (deny-by-default)")

    return key


def is_write(key: str) -> bool:
    return key in WRITE
