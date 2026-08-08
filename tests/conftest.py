import os

# These must be set before any app imports, since app/config.py reads
# them at module level via pydantic-settings.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-unit-tests")


# ---------------------------------------------------------------------------
# Database fixtures (#162)
#
# CI already runs a postgres service with DATABASE_URL pointing at it — what was missing was a
# way for a test to open a session. Without one, every rule that lives in a WHERE clause could
# only be checked by compiling the statement and asserting on the SQL string, which pins the
# shape of a query and says nothing about its behaviour against real rows.
#
# The models use JSONB, so SQLite is not an option and there is no local fallback. Tests that
# need a database SKIP when one is unreachable, so the suite still runs on a laptop with no
# postgres — but CI sets REQUIRE_DB=1, which turns those skips into failures. A silently
# skipped test in CI is indistinguishable from a passing one, and that is exactly how this
# class of bug survives.
# ---------------------------------------------------------------------------

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _test_database_url() -> str:
    # A separate variable so a stray run can never point at anything real.
    return os.environ.get("TEST_DATABASE_URL") or os.environ["DATABASE_URL"]


# asyncpg connections are bound to the event loop that created them, and pytest-asyncio gives
# each test its own loop. A session-scoped engine therefore fails with "attached to a different
# loop" on the second test. The engine is per-test; only the CREATE TABLE work is done once.
_tables_created = False


@pytest_asyncio.fixture
async def db_engine():
    import pytest

    global _tables_created
    from app.models import Base

    engine = create_async_engine(_test_database_url())
    try:
        if not _tables_created:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            _tables_created = True
        else:
            # Cheap connectivity probe, so an unreachable database still skips rather than
            # failing every test with a connection error.
            async with engine.connect():
                pass
    except Exception as e:  # noqa: BLE001 - any connection problem means "no database here"
        await engine.dispose()
        if os.environ.get("REQUIRE_DB"):
            raise
        pytest.skip(f"no test database available: {type(e).__name__}: {e}")
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db(db_engine):
    """A session whose writes are rolled back at the end of the test.

    Everything runs inside one outer transaction that is never committed, so tests cannot leak
    rows into each other and order cannot matter. Code under test may call session.commit()
    freely — it commits the inner nested transaction, not the outer one.
    """
    connection = await db_engine.connect()
    transaction = await connection.begin()
    session = async_sessionmaker(bind=connection, expire_on_commit=False)()
    try:
        yield session
    finally:
        await session.close()
        await transaction.rollback()
        await connection.close()
