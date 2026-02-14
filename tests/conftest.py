import os

# These must be set before any app imports, since app/config.py reads
# them at module level via pydantic-settings.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-unit-tests")
