import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from googleapiclient.discovery import build

from ..config import settings
from .quota_manager import QuotaManager

logger = logging.getLogger(__name__)


@dataclass
class KeyState:
    """Holds one API key with its own QuotaManager and google client."""

    api_key: str
    quota_manager: QuotaManager = field(init=False)
    youtube: object = field(init=False)
    is_exhausted: bool = field(default=False)

    def __post_init__(self):
        key_hash = hashlib.sha256(self.api_key.encode()).hexdigest()[:12]
        storage_path = Path(f"/tmp/youtube_quota_{key_hash}.json")
        self.quota_manager = QuotaManager(
            daily_quota=settings.YOUTUBE_QUOTA_PER_DAY,
            storage_path=storage_path,
        )
        self.youtube = build("youtube", "v3", developerKey=self.api_key)


class APIKeyPool:
    """Singleton pool that manages multiple YouTube API keys with rotation."""

    _instance: Optional["APIKeyPool"] = None

    def __new__(cls):
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._initialized = False
            cls._instance = inst
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._rotation_lock = asyncio.Lock()

        keys = self._resolve_keys()
        self._keys: List[KeyState] = [KeyState(api_key=k) for k in keys]
        self._current_index = 0

        logger.info(f"APIKeyPool initialized with {len(self._keys)} key(s)")

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_keys() -> List[str]:
        """Read keys from YOUTUBE_API_KEYS (comma-separated) or fall back to YOUTUBE_API_KEY."""
        multi = getattr(settings, "YOUTUBE_API_KEYS", "")
        if multi:
            keys = [k.strip() for k in multi.split(",") if k.strip()]
            if keys:
                return keys
        single = settings.YOUTUBE_API_KEY
        if single:
            return [single]
        return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_current(self) -> KeyState:
        """Return the active KeyState, skipping exhausted keys (with daily-reset check)."""
        async with self._rotation_lock:
            # Try each key once, resetting exhausted flags if the day has rolled over
            for _ in range(len(self._keys)):
                ks = self._keys[self._current_index]

                if ks.is_exhausted:
                    # Check if daily reset has happened
                    await ks.quota_manager.check_and_reset_if_needed()
                    if ks.quota_manager.get_remaining_quota() > 0:
                        ks.is_exhausted = False
                        logger.info(
                            f"Key {self._current_index} reset after daily rollover"
                        )

                if not ks.is_exhausted:
                    return ks

                # Advance to next key
                self._current_index = (self._current_index + 1) % len(self._keys)

            # All keys exhausted
            from ..utils.exceptions import QuotaExceededError

            total_used = sum(k.quota_manager.current_usage for k in self._keys)
            total_limit = sum(k.quota_manager.daily_quota for k in self._keys)
            raise QuotaExceededError(
                message=f"All {len(self._keys)} API keys exhausted",
                quota_used=total_used,
                quota_limit=total_limit,
            )

    async def mark_exhausted(self, key_state: KeyState) -> None:
        """Flag a key as exhausted and advance the index."""
        async with self._rotation_lock:
            key_state.is_exhausted = True
            idx = self._keys.index(key_state)
            logger.warning(f"Key {idx} marked as exhausted, rotating")
            self._current_index = (idx + 1) % len(self._keys)

    async def get_pool_status(self) -> dict:
        """Aggregate quota stats across all keys."""
        per_key = []
        total_used = 0
        total_limit = 0

        for i, ks in enumerate(self._keys):
            stats = await ks.quota_manager.get_usage_stats()
            stats["key_index"] = i
            stats["is_exhausted"] = ks.is_exhausted
            per_key.append(stats)
            total_used += ks.quota_manager.current_usage
            total_limit += ks.quota_manager.daily_quota

        return {
            "total_keys": len(self._keys),
            "current_key_index": self._current_index,
            "total_usage": total_used,
            "total_quota": total_limit,
            "total_remaining": total_limit - total_used,
            "usage_percentage": (total_used / total_limit * 100) if total_limit else 0,
            "per_key": per_key,
        }

    @property
    def size(self) -> int:
        return len(self._keys)

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None
