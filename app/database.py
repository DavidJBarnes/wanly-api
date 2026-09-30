from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

#: How long a request waits for a pooled connection before it is refused (console#559).
#:
#: SQLAlchemy's default is 30 s, which with the console's own timeouts read as a hang. A pool
#: that is empty for this long means something is holding connections it should not, and a
#: prompt 503 that says so (app.main) beats a request that sits there. Pool size stays at the
#: default 5 + 10 overflow: one user, and the fix for exhaustion is not holding connections
#: across waits (release_connection), not a bigger pool to fill.
POOL_TIMEOUT_S = 10

engine = create_async_engine(settings.database_url, pool_timeout=POOL_TIMEOUT_S)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    async with async_session() as session:
        yield session


async def release_connection(db: AsyncSession) -> None:
    """End the session's transaction so its connection goes back to the pool NOW.

    Call it before any wait that is not the database's: a turn in the caption queue, a
    captioner call. A session keeps its connection from its first query until commit,
    rollback or close -- and every request's first query is the auth lookup, so a describe
    waiting minutes for its turn held a connection the whole time, "idle in transaction".
    Sixteen of those and the pool is empty, and every other request -- a delete, a listing --
    waits on the pool behind captioning (console#559).

    Commit, not rollback: nothing is pending at these points, and a rollback would expire
    every loaded object for no reason. The session stays usable; its next query checks a
    connection out again.
    """
    await db.commit()
