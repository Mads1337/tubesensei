"""
Redis session management for TubeSensei
"""
import redis.asyncio as redis
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID
import logging
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.exceptions import (
    ServiceUnavailableException,
    AuthenticationException,
    ValidationException
)

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisSessionManager:
    """Redis-based session management"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                decode_responses=settings.REDIS_DECODE_RESPONSES,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=settings.REDIS_CONNECTION_TIMEOUT,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.connected = True
            logger.info("Redis session manager connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            raise ServiceUnavailableException("Redis", "Session storage is unavailable")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False
            logger.info("Redis session manager disconnected")
    
    def _ensure_connected(self):
        """Ensure Redis is connected"""
        if not self.connected or not self.redis_client:
            raise ServiceUnavailableException("Redis", "Session storage is not connected")
    
    def _generate_session_id(self) -> str:
        """Generate a secure session ID"""
        return secrets.token_urlsafe(32)
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session"""
        return f"session:{session_id}"
    
    def _get_user_sessions_key(self, user_id: UUID) -> str:
        """Get Redis key for user sessions"""
        return f"user_sessions:{user_id}"
    
    def _serialize_session_data(self, data: Dict[str, Any]) -> str:
        """Serialize session data for Redis storage"""
        return json.dumps(data, default=str)
    
    def _deserialize_session_data(self, data: str) -> Dict[str, Any]:
        """Deserialize session data from Redis"""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {}
    
    async def create_session(
        self,
        user_id: UUID,
        data: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create a new session"""
        self._ensure_connected()
        
        try:
            session_id = self._generate_session_id()
            session_key = self._get_session_key(session_id)
            user_sessions_key = self._get_user_sessions_key(user_id)
            
            # Prepare session data
            session_data = {
                "session_id": session_id,
                "user_id": str(user_id),
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "ip_address": ip_address,
                "user_agent": user_agent,
                "data": data or {}
            }
            
            # Calculate TTL
            ttl_seconds = (ttl_hours or settings.security.SESSION_EXPIRE_HOURS) * 3600
            
            # Store session data
            serialized_data = self._serialize_session_data(session_data)
            await self.redis_client.setex(
                session_key,
                ttl_seconds,
                serialized_data
            )
            
            # Add session to user's session list
            await self.redis_client.sadd(user_sessions_key, session_id)
            await self.redis_client.expire(user_sessions_key, ttl_seconds)
            
            logger.info(f"Session created: {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise ServiceUnavailableException("Redis", "Failed to create session")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        self._ensure_connected()
        
        try:
            session_key = self._get_session_key(session_id)
            data = await self.redis_client.get(session_key)
            
            if not data:
                return None
            
            session_data = self._deserialize_session_data(data)
            
            # Update last accessed time
            session_data["last_accessed"] = datetime.utcnow().isoformat()
            
            # Update in Redis with original TTL
            ttl = await self.redis_client.ttl(session_key)
            if ttl > 0:
                await self.redis_client.setex(
                    session_key,
                    ttl,
                    self._serialize_session_data(session_data)
                )
            
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def update_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        extend_ttl: bool = True
    ) -> bool:
        """Update session data"""
        self._ensure_connected()
        
        try:
            session_key = self._get_session_key(session_id)
            
            # Get existing session
            existing_data = await self.redis_client.get(session_key)
            if not existing_data:
                return False
            
            session_data = self._deserialize_session_data(existing_data)
            
            # Update data
            session_data["data"].update(data)
            session_data["last_accessed"] = datetime.utcnow().isoformat()
            
            # Get current TTL
            ttl = await self.redis_client.ttl(session_key)
            if ttl <= 0:
                return False
            
            # Extend TTL if requested
            if extend_ttl:
                ttl = settings.security.SESSION_EXPIRE_HOURS * 3600
            
            # Update in Redis
            await self.redis_client.setex(
                session_key,
                ttl,
                self._serialize_session_data(session_data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        self._ensure_connected()
        
        try:
            session_key = self._get_session_key(session_id)
            
            # Get session data to find user_id
            data = await self.redis_client.get(session_key)
            if data:
                session_data = self._deserialize_session_data(data)
                user_id = session_data.get("user_id")
                
                if user_id:
                    user_sessions_key = self._get_user_sessions_key(UUID(user_id))
                    await self.redis_client.srem(user_sessions_key, session_id)
            
            # Delete the session
            result = await self.redis_client.delete(session_key)
            
            if result:
                logger.info(f"Session deleted: {session_id}")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def delete_user_sessions(self, user_id: UUID) -> int:
        """Delete all sessions for a user"""
        self._ensure_connected()
        
        try:
            user_sessions_key = self._get_user_sessions_key(user_id)
            
            # Get all session IDs for the user
            session_ids = await self.redis_client.smembers(user_sessions_key)
            
            if not session_ids:
                return 0
            
            # Delete all sessions
            session_keys = [self._get_session_key(sid) for sid in session_ids]
            deleted_count = await self.redis_client.delete(*session_keys)
            
            # Delete the user sessions set
            await self.redis_client.delete(user_sessions_key)
            
            logger.info(f"Deleted {deleted_count} sessions for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete sessions for user {user_id}: {e}")
            return 0
    
    async def get_user_sessions(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all active sessions for a user"""
        self._ensure_connected()
        
        try:
            user_sessions_key = self._get_user_sessions_key(user_id)
            session_ids = await self.redis_client.smembers(user_sessions_key)
            
            if not session_ids:
                return []
            
            sessions = []
            for session_id in session_ids:
                session_data = await self.get_session(session_id)
                if session_data:
                    sessions.append(session_data)
                else:
                    # Clean up stale session ID from user sessions
                    await self.redis_client.srem(user_sessions_key, session_id)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get sessions for user {user_id}: {e}")
            return []
    
    async def validate_session(self, session_id: str, user_id: UUID) -> bool:
        """Validate that a session exists and belongs to the user"""
        self._ensure_connected()
        
        try:
            session_data = await self.get_session(session_id)
            
            if not session_data:
                return False
            
            return session_data.get("user_id") == str(user_id)
            
        except Exception as e:
            logger.error(f"Failed to validate session {session_id}: {e}")
            return False
    
    async def extend_session(self, session_id: str, ttl_hours: Optional[int] = None) -> bool:
        """Extend session TTL"""
        self._ensure_connected()
        
        try:
            session_key = self._get_session_key(session_id)
            
            # Check if session exists
            if not await self.redis_client.exists(session_key):
                return False
            
            # Calculate new TTL
            ttl_seconds = (ttl_hours or settings.security.SESSION_EXPIRE_HOURS) * 3600
            
            # Extend TTL
            await self.redis_client.expire(session_key, ttl_seconds)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extend session {session_id}: {e}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions (maintenance task)"""
        self._ensure_connected()
        
        try:
            # Get all user session keys
            user_session_keys = await self.redis_client.keys("user_sessions:*")
            cleaned_count = 0
            
            for user_key in user_session_keys:
                # Get all session IDs for this user
                session_ids = await self.redis_client.smembers(user_key)
                
                for session_id in session_ids:
                    session_key = self._get_session_key(session_id)
                    
                    # Check if session still exists
                    if not await self.redis_client.exists(session_key):
                        # Remove from user sessions set
                        await self.redis_client.srem(user_key, session_id)
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired session references")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        self._ensure_connected()
        
        try:
            # Count total sessions
            session_keys = await self.redis_client.keys("session:*")
            total_sessions = len(session_keys)
            
            # Count users with sessions
            user_session_keys = await self.redis_client.keys("user_sessions:*")
            active_users = len(user_session_keys)
            
            return {
                "total_sessions": total_sessions,
                "active_users": active_users,
                "redis_connected": self.connected
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {
                "total_sessions": 0,
                "active_users": 0,
                "redis_connected": False,
                "error": str(e)
            }


# Global session manager instance
session_manager = RedisSessionManager()


@asynccontextmanager
async def get_session_manager():
    """Context manager for session manager"""
    try:
        if not session_manager.connected:
            await session_manager.connect()
        yield session_manager
    except Exception as e:
        logger.error(f"Session manager error: {e}")
        raise


async def get_session_manager_dependency() -> RedisSessionManager:
    """Dependency for getting session manager"""
    if not session_manager.connected:
        await session_manager.connect()
    return session_manager


# Session utilities
class SessionValidator:
    """Session validation utilities"""
    
    @staticmethod
    def validate_session_data(data: Dict[str, Any]) -> bool:
        """Validate session data structure"""
        required_fields = ["session_id", "user_id", "created_at"]
        return all(field in data for field in required_fields)
    
    @staticmethod
    def is_session_expired(session_data: Dict[str, Any], max_age_hours: int = 24) -> bool:
        """Check if session is expired based on creation time"""
        try:
            created_at = datetime.fromisoformat(session_data["created_at"])
            max_age = timedelta(hours=max_age_hours)
            return datetime.utcnow() - created_at > max_age
        except (KeyError, ValueError):
            return True
    
    @staticmethod
    def should_refresh_session(session_data: Dict[str, Any], refresh_threshold_hours: int = 1) -> bool:
        """Check if session should be refreshed based on last access time"""
        try:
            last_accessed = datetime.fromisoformat(session_data["last_accessed"])
            threshold = timedelta(hours=refresh_threshold_hours)
            return datetime.utcnow() - last_accessed > threshold
        except (KeyError, ValueError):
            return False


# Initialize session manager on module import
async def init_session_manager():
    """Initialize session manager"""
    try:
        await session_manager.connect()
        logger.info("Session manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize session manager: {e}")


async def cleanup_session_manager():
    """Cleanup session manager"""
    try:
        await session_manager.disconnect()
        logger.info("Session manager cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up session manager: {e}")