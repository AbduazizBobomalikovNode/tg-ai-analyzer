"""Envelope encryption — MTProto session string'lari uchun.

Sxema:
    DEK (32 bayt, akkauntga xos, random)  →  plaintext'ni AES-256-GCM bilan shifrlaydi
    MASTER_KEY (env, 32 bayt)             →  DEK'ni AES-256-GCM bilan shifrlaydi

DB'da faqat `wrapped_dek` va `ciphertext` saqlanadi. Master key faqat env'da.
Master key rotatsiyasi: har bir DEK'ni qayta wrap qilish kifoya — session'lar
qayta shifrlanmaydi (arzon rotatsiya).

Session string DB'dan RAM'ga chiqqanda ham hech qachon log'ga yozilmaydi
(`app.logging._redact` ga qarang).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEK_SIZE = 32
NONCE_SIZE = 12


@dataclass(frozen=True, slots=True)
class SealedSecret:
    """DB'ga yoziladigan uchlik."""

    wrapped_dek: bytes  # nonce || AESGCM(master_key).encrypt(dek)
    ciphertext: bytes  # nonce || AESGCM(dek).encrypt(plaintext)
    aad: bytes  # bog'lovchi kontekst (masalan b"account:17")


def _seal(key: bytes, plaintext: bytes, aad: bytes | None = None) -> bytes:
    nonce = os.urandom(NONCE_SIZE)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def _open(key: bytes, blob: bytes, aad: bytes | None = None) -> bytes:
    nonce, body = blob[:NONCE_SIZE], blob[NONCE_SIZE:]
    return AESGCM(key).decrypt(nonce, body, aad)


def seal(master_key: bytes, plaintext: str | bytes, *, aad: bytes = b"") -> SealedSecret:
    """Yangi DEK yaratib, plaintext'ni shifrlaydi."""
    if len(master_key) != DEK_SIZE:
        raise ValueError("master_key 32 bayt bo'lishi kerak")
    data = plaintext.encode() if isinstance(plaintext, str) else plaintext
    dek = os.urandom(DEK_SIZE)
    try:
        return SealedSecret(
            wrapped_dek=_seal(master_key, dek, aad),
            ciphertext=_seal(dek, data, aad),
            aad=aad,
        )
    finally:
        del dek


def unseal(master_key: bytes, sealed: SealedSecret) -> bytes:
    """Master key bilan DEK'ni ochib, plaintext'ni qaytaradi."""
    dek = _open(master_key, sealed.wrapped_dek, sealed.aad)
    try:
        return _open(dek, sealed.ciphertext, sealed.aad)
    finally:
        del dek


def unseal_str(master_key: bytes, sealed: SealedSecret) -> str:
    return unseal(master_key, sealed).decode()


def rewrap_dek(old_master: bytes, new_master: bytes, sealed: SealedSecret) -> SealedSecret:
    """Master key rotatsiyasi — ciphertext tegilmaydi."""
    dek = _open(old_master, sealed.wrapped_dek, sealed.aad)
    try:
        return SealedSecret(
            wrapped_dek=_seal(new_master, dek, sealed.aad),
            ciphertext=sealed.ciphertext,
            aad=sealed.aad,
        )
    finally:
        del dek


def account_aad(account_id: int) -> bytes:
    """AAD akkauntga bog'laydi — bir akkauntning blob'i boshqasiga ko'chirilmasin."""
    return f"account:{account_id}".encode()
