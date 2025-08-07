import json
import logging
import zlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.exceptions import RedisError

from ..config import settings
from ..models.transcript_data import TranscriptData, TranscriptInfo

logger = logging.getLogger(__name__)


class TranscriptCache:
    """
    Redis-based caching system for transcript data.
    Implements compression for large transcripts and TTL management.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_hours: Optional[int] = None,
        enable_compression: bool = True,
        compression_threshold: int = 5000  # Compress if content > 5KB
    ):
        self.redis_url = redis_url or settings.REDIS_URL
        self.ttl_hours = ttl_hours or settings.TRANSCRIPT_CACHE_TTL_HOURS
        self.ttl_seconds = self.ttl_hours * 3600
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Redis connection pool
        self.redis_client: Optional[redis.Redis] = None
        
        # Cache key prefixes
        self.KEY_PREFIX = "transcript:"
        self.INFO_PREFIX = "transcript_info:"
        self.METADATA_PREFIX = "transcript_meta:"
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0
        
        logger.info(
            f"Initialized TranscriptCache with TTL={self.ttl_hours}h, "
            f"compression={enable_compression}"
        )
    
    async def connect(self):
        """Establish Redis connection."""
        if self.redis_client is None:
            try:
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False  # We'll handle encoding ourselves
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Successfully connected to Redis")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("Disconnected from Redis")
    
    async def ensure_connected(self):
        """Ensure Redis connection is established."""
        if self.redis_client is None:
            await self.connect()
    
    def _make_key(self, youtube_video_id: str, prefix: str = None) -> str:
        """
        Generate cache key for video.
        
        Args:
            youtube_video_id: YouTube video ID
            prefix: Key prefix to use
            
        Returns:
            Cache key
        """
        prefix = prefix or self.KEY_PREFIX
        return f"{prefix}{youtube_video_id}"
    
    def _serialize(self, data: TranscriptData) -> bytes:
        """
        Serialize transcript data for storage.
        
        Args:
            data: TranscriptData object
            
        Returns:
            Serialized bytes
        """
        # Convert to dictionary
        data_dict = {
            "content": data.content,
            "segments": [seg.dict() for seg in data.segments],
            "language": data.language,
            "language_code": data.language_code,
            "is_auto_generated": data.is_auto_generated,
            "confidence_score": data.confidence_score,
            "has_timestamps": data.has_timestamps,
            "cached_at": datetime.utcnow().isoformat()
        }
        
        # Convert to JSON string
        json_str = json.dumps(data_dict)
        json_bytes = json_str.encode('utf-8')
        
        # Compress if needed
        if self.enable_compression and len(json_bytes) > self.compression_threshold:
            compressed = zlib.compress(json_bytes, level=6)
            # Add compression marker
            return b'COMPRESSED:' + compressed
        
        return json_bytes
    
    def _deserialize(self, data: bytes) -> TranscriptData:
        """
        Deserialize transcript data from storage.
        
        Args:
            data: Serialized bytes
            
        Returns:
            TranscriptData object
        """
        # Check for compression marker
        if data.startswith(b'COMPRESSED:'):
            data = zlib.decompress(data[11:])  # Skip 'COMPRESSED:' prefix
        
        # Decode JSON
        json_str = data.decode('utf-8')
        data_dict = json.loads(json_str)
        
        # Remove cached_at timestamp (not part of TranscriptData)
        data_dict.pop('cached_at', None)
        
        # Convert segments back to objects
        from ..models.transcript_data import TranscriptSegment
        data_dict['segments'] = [
            TranscriptSegment(**seg) for seg in data_dict['segments']
        ]
        
        return TranscriptData(**data_dict)
    
    async def get(
        self,
        youtube_video_id: str
    ) -> Optional[TranscriptData]:
        """
        Retrieve transcript from cache.
        
        Args:
            youtube_video_id: YouTube video ID
            
        Returns:
            TranscriptData if found, None otherwise
        """
        await self.ensure_connected()
        
        try:
            key = self._make_key(youtube_video_id)
            data = await self.redis_client.get(key)
            
            if data is None:
                self._misses += 1
                logger.debug(f"Cache miss for video {youtube_video_id}")
                return None
            
            # Deserialize
            transcript = self._deserialize(data)
            
            self._hits += 1
            logger.debug(f"Cache hit for video {youtube_video_id}")
            
            # Update access time in metadata
            await self._update_metadata(youtube_video_id, {"last_accessed": datetime.utcnow().isoformat()})
            
            return transcript
            
        except RedisError as e:
            self._errors += 1
            logger.error(f"Redis error getting transcript for {youtube_video_id}: {e}")
            return None
        except Exception as e:
            self._errors += 1
            logger.error(f"Error deserializing transcript for {youtube_video_id}: {e}")
            # Remove corrupted entry
            await self.invalidate(youtube_video_id)
            return None
    
    async def set(
        self,
        youtube_video_id: str,
        transcript_data: TranscriptData,
        ttl_override: Optional[int] = None
    ):
        """
        Store transcript in cache.
        
        Args:
            youtube_video_id: YouTube video ID
            transcript_data: TranscriptData to cache
            ttl_override: Optional TTL override in seconds
        """
        await self.ensure_connected()
        
        try:
            key = self._make_key(youtube_video_id)
            ttl = ttl_override or self.ttl_seconds
            
            # Serialize data
            serialized = self._serialize(transcript_data)
            
            # Store in Redis with TTL
            await self.redis_client.setex(key, ttl, serialized)
            
            # Store metadata
            metadata = {
                "video_id": youtube_video_id,
                "cached_at": datetime.utcnow().isoformat(),
                "ttl_seconds": ttl,
                "size_bytes": len(serialized),
                "compressed": serialized.startswith(b'COMPRESSED:'),
                "language": transcript_data.language,
                "word_count": len(transcript_data.content.split()),
                "confidence_score": transcript_data.confidence_score
            }
            await self._set_metadata(youtube_video_id, metadata)
            
            logger.info(
                f"Cached transcript for {youtube_video_id} "
                f"(size: {len(serialized)} bytes, TTL: {ttl}s)"
            )
            
        except RedisError as e:
            self._errors += 1
            logger.error(f"Redis error setting transcript for {youtube_video_id}: {e}")
        except Exception as e:
            self._errors += 1
            logger.error(f"Error caching transcript for {youtube_video_id}: {e}")
    
    async def invalidate(self, youtube_video_id: str) -> bool:
        """
        Remove transcript from cache.
        
        Args:
            youtube_video_id: YouTube video ID
            
        Returns:
            True if removed, False otherwise
        """
        await self.ensure_connected()
        
        try:
            # Remove transcript data
            key = self._make_key(youtube_video_id)
            result = await self.redis_client.delete(key)
            
            # Remove metadata
            meta_key = self._make_key(youtube_video_id, self.METADATA_PREFIX)
            await self.redis_client.delete(meta_key)
            
            if result > 0:
                logger.info(f"Invalidated cache for video {youtube_video_id}")
                return True
            return False
            
        except RedisError as e:
            self._errors += 1
            logger.error(f"Redis error invalidating cache for {youtube_video_id}: {e}")
            return False
    
    async def invalidate_batch(self, youtube_video_ids: List[str]) -> int:
        """
        Remove multiple transcripts from cache.
        
        Args:
            youtube_video_ids: List of YouTube video IDs
            
        Returns:
            Number of entries removed
        """
        await self.ensure_connected()
        
        try:
            # Build keys
            keys = [self._make_key(vid) for vid in youtube_video_ids]
            meta_keys = [self._make_key(vid, self.METADATA_PREFIX) for vid in youtube_video_ids]
            
            # Delete in batch
            result = await self.redis_client.delete(*keys, *meta_keys)
            
            logger.info(f"Invalidated {result} cache entries")
            return result
            
        except RedisError as e:
            self._errors += 1
            logger.error(f"Redis error in batch invalidation: {e}")
            return 0
    
    async def exists(self, youtube_video_id: str) -> bool:
        """
        Check if transcript exists in cache.
        
        Args:
            youtube_video_id: YouTube video ID
            
        Returns:
            True if exists, False otherwise
        """
        await self.ensure_connected()
        
        try:
            key = self._make_key(youtube_video_id)
            return await self.redis_client.exists(key) > 0
        except RedisError as e:
            self._errors += 1
            logger.error(f"Redis error checking existence for {youtube_video_id}: {e}")
            return False
    
    async def get_ttl(self, youtube_video_id: str) -> Optional[int]:
        """
        Get remaining TTL for cached transcript.
        
        Args:
            youtube_video_id: YouTube video ID
            
        Returns:
            TTL in seconds, None if not found
        """
        await self.ensure_connected()
        
        try:
            key = self._make_key(youtube_video_id)
            ttl = await self.redis_client.ttl(key)
            return ttl if ttl > 0 else None
        except RedisError as e:
            self._errors += 1
            logger.error(f"Redis error getting TTL for {youtube_video_id}: {e}")
            return None
    
    async def extend_ttl(self, youtube_video_id: str, additional_hours: int = 24) -> bool:
        """
        Extend TTL for cached transcript.
        
        Args:
            youtube_video_id: YouTube video ID
            additional_hours: Hours to add to current TTL
            
        Returns:
            True if extended, False otherwise
        """
        await self.ensure_connected()
        
        try:
            key = self._make_key(youtube_video_id)
            additional_seconds = additional_hours * 3600
            
            # Get current TTL
            current_ttl = await self.redis_client.ttl(key)
            if current_ttl <= 0:
                return False
            
            # Set new TTL
            new_ttl = current_ttl + additional_seconds
            result = await self.redis_client.expire(key, new_ttl)
            
            if result:
                logger.info(f"Extended TTL for {youtube_video_id} by {additional_hours}h")
                await self._update_metadata(youtube_video_id, {"ttl_extended_at": datetime.utcnow().isoformat()})
            
            return result
            
        except RedisError as e:
            self._errors += 1
            logger.error(f"Redis error extending TTL for {youtube_video_id}: {e}")
            return False
    
    async def _set_metadata(self, youtube_video_id: str, metadata: Dict[str, Any]):
        """Store metadata about cached transcript."""
        try:
            key = self._make_key(youtube_video_id, self.METADATA_PREFIX)
            await self.redis_client.hset(key, mapping={
                k: str(v) for k, v in metadata.items()
            })
            await self.redis_client.expire(key, self.ttl_seconds)
        except RedisError as e:
            logger.error(f"Error setting metadata for {youtube_video_id}: {e}")
    
    async def _update_metadata(self, youtube_video_id: str, updates: Dict[str, Any]):
        """Update metadata about cached transcript."""
        try:
            key = self._make_key(youtube_video_id, self.METADATA_PREFIX)
            if await self.redis_client.exists(key):
                await self.redis_client.hset(key, mapping={
                    k: str(v) for k, v in updates.items()
                })
        except RedisError as e:
            logger.error(f"Error updating metadata for {youtube_video_id}: {e}")
    
    async def get_metadata(self, youtube_video_id: str) -> Optional[Dict[str, str]]:
        """Get metadata about cached transcript."""
        await self.ensure_connected()
        
        try:
            key = self._make_key(youtube_video_id, self.METADATA_PREFIX)
            metadata = await self.redis_client.hgetall(key)
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()} if metadata else None
        except RedisError as e:
            logger.error(f"Error getting metadata for {youtube_video_id}: {e}")
            return None
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        await self.ensure_connected()
        
        try:
            # Get Redis info
            info = await self.redis_client.info('memory')
            
            # Count cached transcripts
            pattern = f"{self.KEY_PREFIX}*"
            cursor = 0
            count = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
            
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_cached": count,
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "cache_errors": self._errors,
                "hit_rate": round(hit_rate, 2),
                "memory_used_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
                "memory_peak_mb": round(info.get('used_memory_peak', 0) / 1024 / 1024, 2)
            }
            
        except RedisError as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "cache_errors": self._errors
            }
    
    async def clear_all(self) -> int:
        """
        Clear all cached transcripts (use with caution).
        
        Returns:
            Number of entries cleared
        """
        await self.ensure_connected()
        
        try:
            # Find all transcript keys
            pattern = f"{self.KEY_PREFIX}*"
            meta_pattern = f"{self.METADATA_PREFIX}*"
            
            keys_to_delete = []
            cursor = 0
            
            # Scan for transcript keys
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                keys_to_delete.extend(keys)
                if cursor == 0:
                    break
            
            # Scan for metadata keys
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=meta_pattern, count=100)
                keys_to_delete.extend(keys)
                if cursor == 0:
                    break
            
            # Delete all keys
            if keys_to_delete:
                result = await self.redis_client.delete(*keys_to_delete)
                logger.warning(f"Cleared {result} cache entries")
                return result
            
            return 0
            
        except RedisError as e:
            logger.error(f"Error clearing cache: {e}")
            return 0