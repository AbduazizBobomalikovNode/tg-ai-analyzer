from __future__ import annotations

import os

import pytest
from cryptography.exceptions import InvalidTag

from app.crypto.envelope import account_aad, rewrap_dek, seal, unseal, unseal_str

SESSION = "1BVtsOK4Bu0FAKESESSIONSTRINGxyz=="


def test_roundtrip(master_key: bytes) -> None:
    sealed = seal(master_key, SESSION, aad=account_aad(7))
    assert unseal_str(master_key, sealed) == SESSION


def test_ciphertext_is_not_plaintext(master_key: bytes) -> None:
    sealed = seal(master_key, SESSION)
    assert SESSION.encode() not in sealed.ciphertext
    assert SESSION.encode() not in sealed.wrapped_dek


def test_each_seal_uses_fresh_dek(master_key: bytes) -> None:
    a = seal(master_key, SESSION)
    b = seal(master_key, SESSION)
    assert a.wrapped_dek != b.wrapped_dek
    assert a.ciphertext != b.ciphertext


def test_wrong_master_key_fails(master_key: bytes) -> None:
    sealed = seal(master_key, SESSION)
    with pytest.raises(InvalidTag):
        unseal(os.urandom(32), sealed)


def test_aad_binds_to_account(master_key: bytes) -> None:
    """7-akkauntning blob'ini 8-akkaunt sifatida ocha olmaslik kerak."""
    sealed = seal(master_key, SESSION, aad=account_aad(7))
    moved = type(sealed)(sealed.wrapped_dek, sealed.ciphertext, account_aad(8))
    with pytest.raises(InvalidTag):
        unseal(master_key, moved)


def test_rewrap_keeps_ciphertext(master_key: bytes) -> None:
    new_master = os.urandom(32)
    sealed = seal(master_key, SESSION, aad=account_aad(1))
    rotated = rewrap_dek(master_key, new_master, sealed)

    assert rotated.ciphertext == sealed.ciphertext  # arzon rotatsiya
    assert rotated.wrapped_dek != sealed.wrapped_dek
    assert unseal_str(new_master, rotated) == SESSION
    with pytest.raises(InvalidTag):
        unseal(master_key, rotated)


def test_short_master_key_rejected() -> None:
    with pytest.raises(ValueError, match="32 bayt"):
        seal(b"tooshort", SESSION)
