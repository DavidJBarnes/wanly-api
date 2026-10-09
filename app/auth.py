import time
from datetime import datetime, timedelta, timezone
from uuid import UUID

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import User

security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key")

#: How long a "this user exists" answer is trusted (wanly-api#434). A console page of thumbnails
#: is one GET /files per image, and every one used to run a SELECT on users just to accept the
#: token -- 150+ at once (a living dataset) took all 15 pooled connections, the rest waited 10 s
#: and failed, and on the 2 GB box the pile-up hung it. The JWT's signature and expiry are
#: checked on every request regardless; this only saves re-asking the database whether its user
#: still exists. Only a YES is cached, so a deleted user stops working within this window.
USER_EXISTS_TTL_S = 300.0
_user_seen: dict[str, float] = {}


async def _user_exists(db: AsyncSession, user_id) -> bool:
    key = str(user_id)
    now = time.monotonic()
    seen = _user_seen.get(key)
    if seen is not None and now - seen < USER_EXISTS_TTL_S:
        return True
    result = await db.execute(select(User.id).where(User.id == user_id))
    if result.scalar_one_or_none() is None:
        _user_seen.pop(key, None)
        return False
    _user_seen[key] = now
    return True


async def verify_api_key(key: str = Depends(api_key_header)):
    if not settings.api_key or key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def verify_api_key_or_bearer(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Accept either X-API-Key header (daemon) or Authorization Bearer JWT (console)."""
    api_key = request.headers.get("x-api-key")
    if api_key:
        if settings.api_key and api_key == settings.api_key:
            return
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        user_id = decode_access_token(token)
        if await _user_exists(db, user_id):
            return
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


async def verify_api_key_or_token(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Accept either X-API-Key header (daemon) or ?token= query param (browser).

    For browser media loads (<img src>, <video src>) that can't send custom headers.
    """
    # Try API key first
    api_key = request.headers.get("x-api-key")
    if api_key:
        if settings.api_key and api_key == settings.api_key:
            return
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    # Try JWT query param
    token = request.query_params.get("token")
    if token:
        user_id = decode_access_token(token)
        if await _user_exists(db, user_id):
            return
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_access_token(user_id: UUID) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiry_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> UUID:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return UUID(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    user_id = decode_access_token(credentials.credentials)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
