# memory/base_memory.py
"""
Abstract base memory class.
Responsibility: Define the contract for all memory implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = ["BaseMemory"]


class BaseMemory(ABC):
    """
    Abstract base class for all memory backends.

    Implement this for:
        - Redis       → memory/redis_memory.py
        - In-memory   → dict-based for testing
        - DB-backed   → PostgreSQL session history

    All methods are async — memory operations are I/O bound.
    """

    @abstractmethod
    async def save(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Persist a value under the given key.

        Args:
            key:         Unique session or context identifier.
            value:       Data to store — must be serializable.
            ttl_seconds: Optional expiry in seconds.
                         None = persists indefinitely.
        """
        ...

    @abstractmethod
    async def load(self, key: str) -> Optional[Any]:
        """
        Retrieve a value by key.

        Args:
            key: Session or context identifier.

        Returns:
            Stored value, or None if key does not exist.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Remove a specific key from memory.

        Args:
            key: Session or context identifier to remove.
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in memory without loading its value.

        Args:
            key: Session or context identifier.

        Returns:
            True if key exists, False otherwise.
        """
        ...

    @abstractmethod
    async def clear(self, key: str) -> None:
        """
        Clear all memory for a specific session/key.

        Args:
            key: Session or context identifier to clear.
        """
        ...

    @abstractmethod
    async def clear_all(self) -> None:
        """
        Clear ALL stored memory across all keys.

        ⚠️ WARNING: Destructive operation.
        Use with extreme caution in production.
        Intended for testing and admin operations only.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self.__class__.__name__
