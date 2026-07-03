# memory/redis_memory.py
"""
Redis-backed memory implementation.
Responsibility: Persist agent conversation memory in Redis.
Depends on: memory/base_memory.py, config/settings.py
"""

import logging
import pickle
from typing import Any, Optional

# import redis.asyncio as aioredis

from memory.base_memory import BaseMemory
from config.settings import REDIS_URL

logger = logging.getLogger("uvicorn.error")

__all__ = ["RedisMemory"]

# Namespace prefix — prevents key collisions with other Redis users
_KEY_PREFIX = "trial_matcher:memory:"


class RedisMemory(BaseMemory):
    """
    Redis-backed async memory store.

    Requirements:
        - redis-py >= 4.2.0 (includes redis.asyncio)
        - Running Redis instance
        - REDIS_URL in .env  e.g. redis://localhost:6379

    Serialization:
        Uses pickle for complex objects (LangChain messages etc.)
        ⚠️ Only unpickle data from trusted sources.

    Key namespacing:
        All keys prefixed with 'trial_matcher:memory:'
        to prevent collisions with other Redis consumers.
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Args:
            redis_url: Redis connection URL.
                       Defaults to REDIS_URL from settings.
        """
        self._redis_url = redis_url or REDIS_URL
        self._client: Optional[aioredis.Redis] = None

    def _get_client(self) -> aioredis.Redis:
        """
        Lazy Redis client initialization.
        Created on first use — not at import time.
        """
        if self._client is None:
            self._client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=False,  # raw bytes for pickle ✅
            )
            logger.info(
                "Redis client initialized | url=%s",
                self._redis_url,
            )
        return self._client

    def _make_key(self, key: str) -> str:
        """Applies namespace prefix to prevent key collisions."""
        return f"{_KEY_PREFIX}{key}"

    # -----------------------------------------------------------
    # BaseMemory Implementation
    # -----------------------------------------------------------

    async def save(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Serialize and persist value under namespaced key.

        Args:
            key:         Session or context identifier.
            value:       Any picklable Python object.
            ttl_seconds: Optional Redis TTL expiry in seconds.
        """
        try:
            serialized = pickle.dumps(value)
            await self._get_client().set(
                self._make_key(key),
                serialized,
                ex=ttl_seconds,
            )
            logger.debug("Memory saved | key=%s | ttl=%s", key, ttl_seconds)
        except Exception:
            logger.exception("RedisMemory.save() failed for key='%s'", key)
            raise

    async def load(self, key: str) -> Optional[Any]:
        """
        Load and deserialize value by key.

        Returns:
            Deserialized value, or None if key does not exist.
        """
        try:
            raw = await self._get_client().get(self._make_key(key))
            if raw is None:
                return None
            return pickle.loads(raw)
        except Exception:
            logger.exception("RedisMemory.load() failed for key='%s'", key)
            raise

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists without loading its value.

        Returns:
            True if key exists, False otherwise.
        """
        try:
            result = await self._get_client().exists(self._make_key(key))
            return bool(result)
        except Exception:
            logger.exception("RedisMemory.exists() failed for key='%s'", key)
            raise

    async def delete(self, key: str) -> None:
        """
        Delete a specific key from memory.
        """
        try:
            await self._get_client().delete(self._make_key(key))
            logger.debug("Memory deleted | key=%s", key)
        except Exception:
            logger.exception("RedisMemory.delete() failed for key='%s'", key)
            raise

    async def clear(self, key: str) -> None:
        """
        Clear all memory for a specific session/key.
        Alias for delete() — kept for BaseMemory contract.
        """
        await self.delete(key)

    async def clear_all(self) -> None:
        """
        Clear ALL keys with the trial_matcher:memory: prefix.

        WARNING: Destructive — use only in testing or admin ops.
        Does NOT call flushdb() — scoped to prefix only for safety.
        """
        try:
            client = self._get_client()
            keys = await client.keys(f"{_KEY_PREFIX}*")
            if keys:
                await client.delete(*keys)
                logger.info(
                    "RedisMemory.clear_all() removed %d keys.",
                    len(keys),
                )
        except Exception:
            logger.exception("RedisMemory.clear_all() failed")
            raise

    async def close(self) -> None:
        """
        Close the Redis connection cleanly.
        Call this in FastAPI lifespan shutdown.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Redis connection closed.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} url={self._redis_url}>"