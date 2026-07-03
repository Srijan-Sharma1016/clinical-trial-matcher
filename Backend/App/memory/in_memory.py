# memory/in_memory.py
"""
In-memory session store.
Responsibility: Store conversation history per session.
No Redis dependency — resets on server restart.
Production: Replace with RedisMemory.
"""

import logging
from typing import Any, Dict, List, Optional
from memory.base_memory import BaseMemory

logger = logging.getLogger("uvicorn.error")

__all__ = ["InMemoryStore"]

# Module-level store — persists for server lifetime
_store: Dict[str, Any] = {}


class InMemoryStore(BaseMemory):
    """
    Simple in-memory implementation of BaseMemory.
    Data lives in a module-level dict.
    Resets on server restart — suitable for dev/demo.
    For production: swap with RedisMemory.
    """

    async def save(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store value under key. TTL ignored in memory implementation."""
        _store[key] = value
        logger.debug("InMemoryStore.save | key=%s", key)

    async def load(self, key: str) -> Optional[Any]:
        """Retrieve value by key. Returns None if not found."""
        value = _store.get(key)
        logger.debug(
            "InMemoryStore.load | key=%s | found=%s",
            key,
            value is not None,
        )
        return value

    async def delete(self, key: str) -> None:
        """Remove a key from the store."""
        _store.pop(key, None)
        logger.debug("InMemoryStore.delete | key=%s", key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in _store

    async def clear(self, key: str) -> None:
        """Clear a specific session."""
        await self.delete(key)

    async def clear_all(self) -> None:
        """
        Clear ALL sessions.
        WARNING: Destructive — use only in testing.
        """
        _store.clear()
        logger.warning("InMemoryStore.clear_all() — all sessions cleared.")

    def __repr__(self) -> str:
        return f""
