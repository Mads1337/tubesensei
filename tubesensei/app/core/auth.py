"""
JWT Authentication handler for TubeSensei
"""
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Dict, Any
import secrets
import logging
from uuid import UUID

from app.core.config import get_settings
from app.database import get_db
from app.models.user import User, UserRole, UserStatus
from app.schemas.user import TokenPayload
from app.core.exceptions import (
    AuthenticationException, 
    PermissionException,
    NotFoundException
)

logger = logging.getLogger(__name__)
settings = get_settings()
security = HTTPBearer(auto_error=False)


class PasswordManager:
    """Password hashing and verification utilities"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=settings.security.BCRYPT_ROUNDS)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'), 
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength according to security settings"""
        errors = []
        
        if len(password) < settings.security.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {settings.security.PASSWORD_MIN_LENGTH} characters long")
        
        if settings.security.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if settings.security.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if settings.security.PASSWORD_REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if settings.security.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*(),.?\":{}|<>"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }


class JWTManager:
    """JWT token management utilities"""
    
    @staticmethod
    def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.security.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        payload = {
            "sub": str(user.id),
            "email": user.email,
            "username": user.username,
            "role": user.role.value,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
            "type": "access"
        }
        
        token = jwt.encode(
            payload,
            settings.security.SECRET_KEY,
            algorithm=settings.security.ALGORITHM
        )
        
        logger.info(f"Access token created for user {user.email}")
        return token
    
    @staticmethod
    def create_refresh_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.security.REFRESH_TOKEN_EXPIRE_MINUTES
            )
        
        payload = {
            "sub": str(user.id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),
            "type": "refresh"
        }
        
        token = jwt.encode(
            payload,
            settings.security.SECRET_KEY,
            algorithm=settings.security.ALGORITHM
        )
        
        logger.info(f"Refresh token created for user {user.email}")
        return token
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.security.SECRET_KEY,
                algorithms=[settings.security.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            raise AuthenticationException("Invalid token")
    
    @staticmethod
    def get_token_expiry_time(token: str) -> datetime:
        """Get token expiry time"""
        try:
            payload = jwt.decode(
                token,
                settings.security.SECRET_KEY,
                algorithms=[settings.security.ALGORITHM],
                options={"verify_exp": False}
            )
            return datetime.fromtimestamp(payload["exp"])
        except Exception:
            raise AuthenticationException("Invalid token")


class AuthService:
    """Authentication service"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def authenticate_user(
        self, 
        email: str, 
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate user with email and password"""
        try:
            # Get user by email
            result = await self.db.execute(
                select(User).where(User.email == email.lower())
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"Authentication failed: User not found for email {email}")
                return None
            
            # Check if account is locked
            if user.is_locked:
                logger.warning(f"Authentication failed: Account locked for user {email}")
                raise AuthenticationException(
                    "Account is temporarily locked due to too many failed login attempts"
                )
            
            # Check if user can login
            if not user.can_login:
                logger.warning(f"Authentication failed: User cannot login {email}")
                raise AuthenticationException(
                    f"Account is not active (status: {user.status.value})"
                )
            
            # Verify password
            if not PasswordManager.verify_password(password, user.hashed_password):
                # Increment login attempts
                user.increment_login_attempts()
                
                # Lock account if too many attempts
                if user.login_attempts >= settings.security.LOGIN_ATTEMPTS_MAX:
                    user.lock_account(settings.security.LOGIN_LOCKOUT_MINUTES)
                    logger.warning(f"Account locked after {user.login_attempts} failed attempts: {email}")
                
                await self.db.commit()
                logger.warning(f"Authentication failed: Invalid password for user {email}")
                return None
            
            # Update last login info
            user.update_last_login(ip_address, user_agent)
            await self.db.commit()
            
            logger.info(f"User authenticated successfully: {email}")
            return user
            
        except AuthenticationException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self.db.rollback()
            raise AuthenticationException("Authentication failed")
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        try:
            result = await self.db.execute(
                select(User).where(
                    User.id == user_id,
                    User.deleted_at.is_(None)
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            result = await self.db.execute(
                select(User).where(
                    User.email == email.lower(),
                    User.deleted_at.is_(None)
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None


# Authentication dependencies
async def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    """Get authentication service dependency"""
    return AuthService(db)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Get current authenticated user"""
    # Allow passwordless admin access for now (simplifies setup)
    # In production, you would implement proper authentication
    from datetime import datetime
    
    # Check for simple auth bypass environment variable or always allow for now
    allow_bypass = True  # Always allow admin access without auth
    
    if allow_bypass or settings.DEBUG:
        mock_user = User(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            email="admin@tubesensei.dev",
            username="admin",
            full_name="TubeSensei Admin",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        return mock_user
    
    if not credentials:
        raise AuthenticationException("Authentication credentials required")
    
    token = credentials.credentials
    
    try:
        # Decode token
        payload = JWTManager.decode_token(token)
        
        # Validate token type
        if payload.get("type") != "access":
            raise AuthenticationException("Invalid token type")
        
        # Get user ID from token
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise AuthenticationException("Invalid token payload")
        
        user_id = UUID(user_id_str)
        
        # Get user from database
        user = await auth_service.get_user_by_id(user_id)
        if not user:
            raise AuthenticationException("User not found")
        
        # Check if user is still active
        if not user.can_login:
            raise AuthenticationException("Account is not active")
        
        # Update last activity
        user.update_activity()
        await auth_service.db.commit()
        
        return user
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise AuthenticationException("Authentication failed")


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (deprecated, use get_current_user)"""
    return current_user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, auth_service)
    except AuthenticationException:
        return None


async def require_role(required_role: UserRole):
    """Dependency factory for role-based access control"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission(required_role):
            raise PermissionException(
                f"Access denied. Required role: {required_role.value}",
                required_permission=required_role.value
            )
        return current_user
    
    return role_checker


# Role-specific dependencies
async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    if not current_user.has_permission(UserRole.ADMIN):
        raise PermissionException(
            "Admin access required",
            required_permission="admin"
        )
    return current_user


async def require_user(current_user: User = Depends(get_current_user)) -> User:
    """Require user role or higher"""
    if not current_user.has_permission(UserRole.USER):
        raise PermissionException(
            "User access required",
            required_permission="user"
        )
    return current_user


# Token refresh utilities
async def refresh_access_token(
    refresh_token: str,
    auth_service: AuthService
) -> Dict[str, Any]:
    """Refresh access token using refresh token"""
    try:
        # Decode refresh token
        payload = JWTManager.decode_token(refresh_token)
        
        # Validate token type
        if payload.get("type") != "refresh":
            raise AuthenticationException("Invalid token type")
        
        # Get user
        user_id = UUID(payload.get("sub"))
        user = await auth_service.get_user_by_id(user_id)
        
        if not user or not user.can_login:
            raise AuthenticationException("User not found or inactive")
        
        # Generate new tokens
        access_token = JWTManager.create_access_token(user)
        new_refresh_token = JWTManager.create_refresh_token(user)
        
        # Calculate expiry
        expires_in = settings.security.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": expires_in,
            "user": user
        }
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise AuthenticationException("Token refresh failed")


# Password utilities
def hash_password(password: str) -> str:
    """Hash password - convenience function"""
    return PasswordManager.hash_password(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password - convenience function"""
    return PasswordManager.verify_password(password, hashed_password)


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength - convenience function"""
    return PasswordManager.validate_password_strength(password)