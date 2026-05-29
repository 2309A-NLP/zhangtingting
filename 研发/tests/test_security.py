from __future__ import annotations

from app.core.security import hash_password, verify_password


def test_hash_and_verify_password_roundtrip():
    password = "demo123456"
    password_hash = hash_password(password)

    assert password_hash.startswith("$2")
    assert verify_password(password, password_hash) is True


def test_verify_password_returns_false_for_malformed_hash():
    assert verify_password("demo123456", "$2b$12$N0nPr0dHashedPasswordExample") is False
