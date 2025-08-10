"""
User schemas for request/response validation
"""
from pydantic import BaseModel, EmailStr, Field, validator, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
import re

from app.models.user import UserRole, UserStatus


class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique username",
        pattern="^[a-zA-Z0-9_-]+$"
    )
    full_name: Optional[str] = Field(None, max_length=255, description="Full name")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    avatar_url: Optional[str] = Field(None, description="Avatar URL")


class UserCreate(UserBase):
    """Schema for user registration"""
    password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="User password"
    )
    confirm_password: str = Field(..., description="Password confirmation")
    role: Optional[UserRole] = Field(UserRole.USER, description="User role")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        # Check for at least one digit
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        # Check for common passwords
        common_passwords = ['password', '12345678', 'qwerty', 'abc123']
        if v.lower() in common_passwords:
            raise ValueError('Password is too common')
        
        return v


class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    full_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserAdminUpdate(UserUpdate):
    """Schema for admin updating user"""
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_superuser: Optional[bool] = None


class PasswordChange(BaseModel):
    """Schema for changing password"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    confirm_password: str = Field(..., description="Confirm new password")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class PasswordReset(BaseModel):
    """Schema for password reset"""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    confirm_password: str = Field(..., description="Confirm new password")
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(False, description="Remember login session")


class UserResponse(BaseModel):
    """Schema for user response"""
    id: UUID
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: UserRole
    status: UserStatus
    is_active: bool
    is_verified: bool
    is_superuser: bool
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    two_factor_enabled: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserDetailResponse(UserResponse):
    """Detailed user response with additional fields"""
    email_verified_at: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    preferences: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None
    api_key_created_at: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserResponse = Field(..., description="User information")


class TokenPayload(BaseModel):
    """JWT token payload"""
    sub: str  # User ID
    email: str
    username: str
    role: str
    exp: datetime
    iat: datetime
    jti: Optional[str] = None  # JWT ID for token revocation
    
    model_config = ConfigDict(from_attributes=True)


class EmailVerification(BaseModel):
    """Email verification schema"""
    token: str = Field(..., description="Email verification token")


class ResendVerification(BaseModel):
    """Resend email verification schema"""
    email: EmailStr = Field(..., description="Email address")


class TwoFactorSetup(BaseModel):
    """Two-factor authentication setup response"""
    secret: str = Field(..., description="2FA secret key")
    qr_code: str = Field(..., description="QR code image data URL")
    backup_codes: List[str] = Field(..., description="Backup recovery codes")


class TwoFactorVerify(BaseModel):
    """Two-factor authentication verification"""
    code: str = Field(..., min_length=6, max_length=6, description="6-digit 2FA code")


class UserList(BaseModel):
    """User list response with pagination"""
    users: List[UserResponse]
    total: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    
    model_config = ConfigDict(from_attributes=True)


class UserActivity(BaseModel):
    """User activity log entry"""
    user_id: UUID
    action: str = Field(..., description="Action performed")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    timestamp: datetime = Field(..., description="Activity timestamp")
    
    model_config = ConfigDict(from_attributes=True)


class APIKeyResponse(BaseModel):
    """API key generation response"""
    api_key: str = Field(..., description="Generated API key")
    created_at: datetime = Field(..., description="Creation timestamp")
    message: str = Field(..., description="Instructions for API key usage")