# database.py
"""
Database engine and session management.
Responsibility: Engine creation, session factory, and DB initialization.
"""

import logging
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from config.settings import DATABASE_URL, DB_ECHO

logger = logging.getLogger("uvicorn.error")

__all__ = ["engine", "init_db", "check_db_connection", "get_session"]

# -----------------------------------------------------------
# ENGINE CONFIGURATION
# -----------------------------------------------------------

def _create_engine():
    """
    Creates SQLAlchemy engine with appropriate settings.
    SQLite gets a lightweight config (for local dev/testing).
    PostgreSQL gets full production pool configuration.
    """
    if DATABASE_URL.startswith("sqlite"):
        # SQLite — no connection pooling params
        logger.warning(
            "Using SQLite — not recommended for production."
        )
        return create_engine(
            DATABASE_URL,
            echo=DB_ECHO,
            future=True,
            connect_args={"check_same_thread": False},
        )

    # PostgreSQL / managed DB — full pool config
    return create_engine(
        DATABASE_URL,
        echo=DB_ECHO,
        pool_pre_ping=True,      # detects stale connections ✅
        pool_size=10,            # persistent connections in pool
        max_overflow=20,         # burst capacity beyond pool_size
        pool_timeout=30,         # seconds to wait for a connection
        pool_recycle=1800,       # recycle connections every 30 mins
        future=True,
    )


engine = _create_engine()


# -----------------------------------------------------------
# DB HEALTH CHECK
# -----------------------------------------------------------

def check_db_connection() -> bool:
    """
    Verifies the database is reachable.
    Used in lifespan startup and /health endpoint.
    Returns True if healthy, False otherwise — never raises.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database health check passed.")
        return True
    except Exception as e:
        logger.error(
            "Database health check FAILED | error=%s",
            str(e),
        )
        return False


# -----------------------------------------------------------
# DB INITIALIZATION
# -----------------------------------------------------------

def init_db() -> None:
    """
    Creates all tables defined in SQLModel metadata.
    Safe to call on every startup — skips existing tables.
    Raises on failure — startup should not continue if DB is broken.
    """
    try:
        logger.info("Running database table initialization...")

        import models  # noqa: F401 — registers SQLModel table metadata

        SQLModel.metadata.create_all(engine)

        table_count = len(SQLModel.metadata.tables)
        logger.info(
            "Database initialization complete. "
            "Tables registered: %d",
            table_count,
        )
    except Exception:
        logger.exception("Database initialization failed.")
        raise


# -----------------------------------------------------------
# SESSION FACTORY
# -----------------------------------------------------------

def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a managed DB session.

    Usage:
        @app.get("/endpoint")
        async def endpoint(session: Session = Depends(get_session)):
            ...

    Behavior:
        - Auto-commits on successful request
        - Auto-rolls back on exception
        - Always closes session in finally block
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
