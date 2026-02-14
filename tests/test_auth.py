"""Unit tests for authentication utilities: password hashing and JWT tokens.

These test the pure functions in app/auth.py — no database or HTTP needed.
"""

import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from app.auth import create_access_token, decode_access_token, hash_password, verify_password
from app.config import settings


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self):
        """A hashed password is verified successfully with the original."""
        password = "correct-horse-battery-staple"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_wrong_password_fails(self):
        """Verification rejects a password that doesn't match the hash."""
        hashed = hash_password("real-password")
        assert verify_password("wrong-password", hashed) is False

    def test_unique_salts_per_hash(self):
        """Hashing the same password twice produces different ciphertexts (unique salts)."""
        password = "same-input"
        h1 = hash_password(password)
        h2 = hash_password(password)
        assert h1 != h2
        assert verify_password(password, h1)
        assert verify_password(password, h2)


class TestJWTTokens:
    def test_create_and_decode_roundtrip(self):
        """Encoding then decoding a token yields the original user ID."""
        user_id = uuid.uuid4()
        token = create_access_token(user_id)
        assert decode_access_token(token) == user_id

    def test_expired_token_raises_401(self):
        """A token whose exp is in the past is rejected with 401."""
        payload = {
            "sub": str(uuid.uuid4()),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        token = pyjwt.encode(payload, settings.jwt_secret, algorithm="HS256")

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401

    def test_wrong_secret_raises_401(self):
        """A token signed with a different secret is rejected."""
        payload = {
            "sub": str(uuid.uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        token = pyjwt.encode(payload, "wrong-secret", algorithm="HS256")

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401

    def test_missing_sub_claim_raises_401(self):
        """A token without a 'sub' field triggers a 401."""
        payload = {"exp": datetime.now(timezone.utc) + timedelta(hours=1)}
        token = pyjwt.encode(payload, settings.jwt_secret, algorithm="HS256")

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401

    def test_garbage_string_raises_401(self):
        """A completely invalid string is rejected with 401."""
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("this-is-not-a-jwt")
        assert exc_info.value.status_code == 401


class TestUploadEndpointAuth:
    """Verify that POST /upload rejects unauthenticated requests."""

    @pytest.mark.asyncio
    async def test_upload_without_token_returns_401(self):
        """An anonymous POST /upload is rejected — no token means no access."""
        from httpx import ASGITransport, AsyncClient
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/upload", files={"file": ("test.png", b"fake-image-data", "image/png")}
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_upload_with_invalid_token_returns_401(self):
        """A request with a bogus Bearer token is rejected with 401."""
        from httpx import ASGITransport, AsyncClient
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/upload",
                files={"file": ("test.png", b"fake-image-data", "image/png")},
                headers={"Authorization": "Bearer invalid-token"},
            )
        assert resp.status_code == 401
