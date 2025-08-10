"""
Authentication API endpoints for TubeSensei
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import logging
from datetime import datetime, timedelta

from app.database import get_db
from app.models.user import User, UserRole, UserStatus
from app.schemas.user import (
    UserCreate, UserLogin, UserResponse, TokenResponse, 
    PasswordChange, ResendVerification
)
from app.core.auth import (
    AuthService, JWTManager, PasswordManager,
    get_auth_service, get_current_user, get_optional_current_user,
    refresh_access_token
)
from app.core.session import get_session_manager_dependency, RedisSessionManager
from app.core.config import get_settings
from app.core.exceptions import (
    AuthenticationException, ValidationException, ConflictException,
    NotFoundException, BadRequestException
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    try:
        # Check if registration is enabled
        if not settings.FEATURES_ENABLE_REGISTRATION:
            raise ValidationException(
                {"registration": ["User registration is currently disabled"]},
                "Registration is not allowed"
            )
        
        # Validate password strength
        password_validation = PasswordManager.validate_password_strength(user_data.password)
        if not password_validation["is_valid"]:
            raise ValidationException(
                {"password": password_validation["errors"]},
                "Password does not meet security requirements"
            )
        
        # Check if user already exists
        existing_user = await db.execute(
            select(User).where(
                (User.email == user_data.email.lower()) |
                (User.username == user_data.username.lower())
            )
        )
        existing_user = existing_user.scalar_one_or_none()
        
        if existing_user:
            error_field = "email" if existing_user.email == user_data.email.lower() else "username"
            raise ConflictException(
                f"User with this {error_field} already exists",
                details={error_field: [f"This {error_field} is already registered"]}
            )
        
        # Get client info
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Create new user
        hashed_password = PasswordManager.hash_password(user_data.password)
        new_user = User(
            email=user_data.email.lower(),
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            role=user_data.role or UserRole.USER,
            status=UserStatus.ACTIVE,  # Set to active for now, can be changed to PENDING for email verification
            is_active=True,
            is_verified=False,  # Require email verification
            registration_ip=client_ip,
            user_agent=user_agent
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        logger.info(f"User registered successfully: {new_user.email}")
        
        # TODO: Send verification email in production
        # await send_verification_email(new_user.email, verification_token)
        
        return new_user
        
    except (ValidationException, ConflictException):
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    user_credentials: UserLogin,
    request: Request,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
    session_manager: RedisSessionManager = Depends(get_session_manager_dependency)
):
    """Authenticate user and return access token"""
    try:
        # Get client info
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Authenticate user
        user = await auth_service.authenticate_user(
            user_credentials.email,
            user_credentials.password,
            client_ip,
            user_agent
        )
        
        if not user:
            raise AuthenticationException("Invalid email or password")
        
        # Generate JWT tokens
        access_token = JWTManager.create_access_token(user)
        refresh_token = JWTManager.create_refresh_token(user)
        
        # Create session if remember_me is enabled
        session_id = None
        if user_credentials.remember_me:
            session_id = await session_manager.create_session(
                user_id=user.id,
                data={"remember_me": True, "login_method": "password"},
                ttl_hours=settings.security.SESSION_EXPIRE_HOURS,
                ip_address=client_ip,
                user_agent=user_agent
            )
            
            # Set session cookie
            response.set_cookie(
                key=settings.security.SESSION_COOKIE_NAME,
                value=session_id,
                max_age=settings.security.SESSION_EXPIRE_HOURS * 3600,
                httponly=settings.security.SESSION_COOKIE_HTTPONLY,
                secure=settings.security.SESSION_COOKIE_SECURE,
                samesite=settings.security.SESSION_COOKIE_SAMESITE
            )
        
        # Calculate token expiry
        expires_in = settings.security.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        
        logger.info(f"User logged in successfully: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
            user=user
        )
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    session_manager: RedisSessionManager = Depends(get_session_manager_dependency)
):
    """Logout user and invalidate session"""
    try:
        # Get session ID from cookie
        session_id = request.cookies.get(settings.security.SESSION_COOKIE_NAME)
        
        if session_id:
            # Delete session from Redis
            await session_manager.delete_session(session_id)
            
            # Clear session cookie
            response.delete_cookie(
                key=settings.security.SESSION_COOKIE_NAME,
                httponly=settings.security.SESSION_COOKIE_HTTPONLY,
                secure=settings.security.SESSION_COOKIE_SECURE,
                samesite=settings.security.SESSION_COOKIE_SAMESITE
            )
        
        logger.info(f"User logged out: {current_user.email}")
        
        return {"message": "Logout successful"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return current_user


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Refresh access token using refresh token"""
    try:
        # Get refresh token from request body or header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise AuthenticationException("Refresh token required")
        
        refresh_token = auth_header.split(" ")[1]
        
        # Refresh the token
        token_data = await refresh_access_token(refresh_token, auth_service)
        
        logger.info(f"Token refreshed for user: {token_data['user'].email}")
        
        return TokenResponse(
            access_token=token_data["access_token"],
            refresh_token=token_data["refresh_token"],
            token_type=token_data["token_type"],
            expires_in=token_data["expires_in"],
            user=token_data["user"]
        )
        
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change user password"""
    try:
        # Verify current password
        if not PasswordManager.verify_password(
            password_data.current_password, 
            current_user.hashed_password
        ):
            raise AuthenticationException("Current password is incorrect")
        
        # Validate new password strength
        password_validation = PasswordManager.validate_password_strength(
            password_data.new_password
        )
        if not password_validation["is_valid"]:
            raise ValidationException(
                {"new_password": password_validation["errors"]},
                "New password does not meet security requirements"
            )
        
        # Hash new password
        new_hashed_password = PasswordManager.hash_password(password_data.new_password)
        
        # Update user password
        current_user.hashed_password = new_hashed_password
        current_user.password_changed_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"Password changed for user: {current_user.email}")
        
        return {"message": "Password changed successfully"}
        
    except (AuthenticationException, ValidationException):
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/logout-all")
async def logout_all_sessions(
    current_user: User = Depends(get_current_user),
    session_manager: RedisSessionManager = Depends(get_session_manager_dependency)
):
    """Logout user from all sessions"""
    try:
        # Delete all user sessions
        deleted_count = await session_manager.delete_user_sessions(current_user.id)
        
        logger.info(f"All sessions logged out for user: {current_user.email} ({deleted_count} sessions)")
        
        return {
            "message": "Logged out from all sessions",
            "sessions_deleted": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Logout all sessions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout from all sessions failed"
        )


@router.get("/sessions")
async def get_user_sessions(
    current_user: User = Depends(get_current_user),
    session_manager: RedisSessionManager = Depends(get_session_manager_dependency)
):
    """Get all active sessions for current user"""
    try:
        sessions = await session_manager.get_user_sessions(current_user.id)
        
        # Remove sensitive data from session info
        safe_sessions = []
        for session in sessions:
            safe_session = {
                "session_id": session["session_id"],
                "created_at": session["created_at"],
                "last_accessed": session["last_accessed"],
                "ip_address": session.get("ip_address"),
                "user_agent": session.get("user_agent")
            }
            safe_sessions.append(safe_session)
        
        return {
            "sessions": safe_sessions,
            "total": len(safe_sessions)
        }
        
    except Exception as e:
        logger.error(f"Get user sessions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user sessions"
        )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    session_manager: RedisSessionManager = Depends(get_session_manager_dependency)
):
    """Delete a specific session"""
    try:
        # Validate that the session belongs to the current user
        if not await session_manager.validate_session(session_id, current_user.id):
            raise NotFoundException("Session", session_id)
        
        # Delete the session
        success = await session_manager.delete_session(session_id)
        
        if not success:
            raise NotFoundException("Session", session_id)
        
        logger.info(f"Session deleted: {session_id} for user: {current_user.email}")
        
        return {"message": "Session deleted successfully"}
        
    except NotFoundException:
        raise
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )


# Optional: Email verification endpoints (placeholder for future implementation)
@router.post("/verify-email")
async def verify_email(
    verification_data: dict,  # Replace with proper schema when implementing
    db: AsyncSession = Depends(get_db)
):
    """Verify user email address"""
    # TODO: Implement email verification
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Email verification not yet implemented"
    )


@router.post("/resend-verification")
async def resend_verification(
    resend_data: ResendVerification,
    db: AsyncSession = Depends(get_db)
):
    """Resend email verification"""
    # TODO: Implement resend verification
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Resend verification not yet implemented"
    )


@router.post("/forgot-password")
async def forgot_password(
    email_data: dict,  # Replace with proper schema when implementing
    db: AsyncSession = Depends(get_db)
):
    """Initiate password reset process"""
    # TODO: Implement forgot password
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Password reset not yet implemented"
    )


@router.post("/reset-password")
async def reset_password(
    reset_data: dict,  # Replace with proper schema when implementing
    db: AsyncSession = Depends(get_db)
):
    """Reset password using reset token"""
    # TODO: Implement password reset
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Password reset not yet implemented"
    )