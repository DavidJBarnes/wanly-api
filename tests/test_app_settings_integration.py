"""Integration tests for the /settings PUT endpoint with a real database."""

import pytest
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.routes.app_settings import _DEFAULTS, _get_all_settings, _to_response


# ── Minimal in-memory models (avoid JSONB issues with SQLite) ────────────────

class _Base(DeclarativeBase):
    pass


class AppSetting(_Base):
    __tablename__ = "app_settings"
    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return create_async_engine("sqlite+aiosqlite://", echo=False)


@pytest.fixture
def _sessionmaker(engine):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(_Base.metadata.drop_all)


@pytest.fixture
async def db(_sessionmaker):
    async with _sessionmaker() as session:
        yield session


@pytest.fixture
async def seeded_db(db):
    """Seed the app_settings table with defaults."""
    for k, v in _DEFAULTS.items():
        db.add(AppSetting(key=k, value=v, updated_at=datetime.now(timezone.utc)))
    await db.commit()
    return db


# ── Override the dependency so the router uses our test DB ───────────────────

@pytest.fixture
def app_with_db(db, seeded_db):
    """Return FastAPI app with get_db overridden to use our test DB."""
    from app.main import app
    from app.database import get_db

    # Use seeded_db as the session
    async def override_get_db():
        yield seeded_db

    app.dependency_overrides[get_db] = override_get_db

    # Also override get_current_user to skip auth
    from app.auth import get_current_user
    from app.models import User
    import uuid

    async def override_get_current_user():
        return User(id=uuid.uuid4(), username="test")

    app.dependency_overrides[get_current_user] = override_get_current_user

    yield app
    app.dependency_overrides.clear()


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_put_settings_persists_negative_prompt(seeded_db):
    """PUT /settings should persist negative_prompt and return it on subsequent GET."""
    # Verify initial state
    settings_before = await _get_all_settings(seeded_db)
    assert settings_before["negative_prompt"] == _DEFAULTS["negative_prompt"]

    # Now test the raw DB logic (same as what the route does)
    from app.schemas.app_settings import AppSettingsUpdate

    body = AppSettingsUpdate(negative_prompt="custom negative text")
    updates = body.model_dump(exclude_none=True)

    now = datetime.now(timezone.utc)
    for key, value in updates.items():
        existing = await seeded_db.get(AppSetting, key)
        if existing:
            existing.value = str(value)
            existing.updated_at = now
        else:
            seeded_db.add(AppSetting(key=key, value=str(value), updated_at=now))
    await seeded_db.commit()

    # Read back
    settings_after = await _get_all_settings(seeded_db)
    assert settings_after["negative_prompt"] == "custom negative text"


@pytest.mark.asyncio
async def test_put_settings_partial_update_preserves_others(app_with_db):
    """When only negative_prompt is sent, other settings should not be overwritten."""
    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Read current
        get_resp = await client.get("/settings")
        assert get_resp.status_code == 200
        original = get_resp.json()
        assert "negative_prompt" in original

        # Update only negative_prompt
        put_resp = await client.put(
            "/settings",
            json={"negative_prompt": "some new prompt text"},
        )
        assert put_resp.status_code == 200, f"PUT failed: {put_resp.text}"
        updated = put_resp.json()
        assert updated["negative_prompt"] == "some new prompt text"
        # Other fields should be unchanged
        assert updated["cfg_high"] == original["cfg_high"]
        assert updated["cfg_low"] == original["cfg_low"]

        # Verify GET returns the new value too
        get_resp2 = await client.get("/settings")
        assert get_resp2.status_code == 200
        assert get_resp2.json()["negative_prompt"] == "some new prompt text"


@pytest.mark.asyncio
async def test_put_settings_clearing_negative_prompt(app_with_db):
    """Setting negative_prompt to empty string should persist."""
    transport = ASGITransport(app=app_with_db)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First set it to something
        await client.put("/settings", json={"negative_prompt": "initial value"})

        # Now clear it
        put_resp = await client.put("/settings", json={"negative_prompt": ""})
        assert put_resp.status_code == 200, f"PUT failed: {put_resp.text}"
        assert put_resp.json()["negative_prompt"] == ""

        # Verify GET
        get_resp = await client.get("/settings")
        assert get_resp.json()["negative_prompt"] == ""


@pytest.mark.asyncio
async def test_put_settings_all_fields(seeded_db):
    """Full update with all 5 fields should persist correctly."""
    from app.schemas.app_settings import AppSettingsUpdate

    body = AppSettingsUpdate(
        lightx2v_strength_high=3.5,
        lightx2v_strength_low=0.5,
        cfg_high=2.0,
        cfg_low=0.5,
        negative_prompt="full update test",
    )
    updates = body.model_dump(exclude_none=True)

    now = datetime.now(timezone.utc)
    for key, value in updates.items():
        existing = await seeded_db.get(AppSetting, key)
        if existing:
            existing.value = str(value)
            existing.updated_at = now
        else:
            seeded_db.add(AppSetting(key=key, value=str(value), updated_at=now))
    await seeded_db.commit()

    settings = await _get_all_settings(seeded_db)
    assert settings["lightx2v_strength_high"] == "3.5"
    assert settings["cfg_high"] == "2.0"
    assert settings["negative_prompt"] == "full update test"


@pytest.mark.asyncio
async def test_put_settings_unicode_negative_prompt(seeded_db):
    """Chinese/Unicode negative prompts should survive round-trip."""
    from app.schemas.app_settings import AppSettingsUpdate

    chinese_text = "色调艳丽，过曝，低质量"
    body = AppSettingsUpdate(negative_prompt=chinese_text)
    updates = body.model_dump(exclude_none=True)

    now = datetime.now(timezone.utc)
    for key, value in updates.items():
        existing = await seeded_db.get(AppSetting, key)
        existing.value = str(value)
    await seeded_db.commit()

    settings = await _get_all_settings(seeded_db)
    assert settings["negative_prompt"] == chinese_text
