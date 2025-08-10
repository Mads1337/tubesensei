"""
User model for authentication and authorization
"""
from sqlalchemy import Column, String, Boolean, DateTime, Enum as SQLEnum, Index, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum
from typing import Optional

from app.models.base import BaseModel


class UserRole(str, enum.Enum):
    """User role enumeration"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    
    @classmethod
    def has_permission(cls, role: 'UserRole', required_role: 'UserRole') -> bool:
        """Check if a role has permission based on hierarchy"""
        hierarchy = {
            cls.ADMIN: 3,
            cls.USER: 2,
            cls.VIEWER: 1
        }
        return hierarchy.get(role, 0) >= hierarchy.get(required_role, 0)


class UserStatus(str, enum.Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(BaseModel):
    """User model for authentication and user management"""
    __tablename__ = "users"
    
    # Basic Information
    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User email address"
    )
    username = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique username"
    )
    full_name = Column(
        String(255),
        nullable=True,
        comment="User's full name"
    )
    
    # Authentication
    hashed_password = Column(
        String(255),
        nullable=False,
        comment="Bcrypt hashed password"
    )
    
    # Role and Permissions
    role = Column(
        SQLEnum(UserRole),
        default=UserRole.USER,
        nullable=False,
        index=True,
        comment="User role for authorization"
    )
    
    # Account Status
    status = Column(
        SQLEnum(UserStatus),
        default=UserStatus.PENDING,
        nullable=False,
        index=True,
        comment="Account status"
    )
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether the account is active"
    )
    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether email is verified"
    )
    is_superuser = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Superuser flag for system administration"
    )
    
    # Timestamps
    email_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When email was verified"
    )
    last_login_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful login timestamp"
    )
    last_activity_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last activity timestamp"
    )
    password_changed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        default=datetime.utcnow,
        comment="Last password change timestamp"
    )
    
    # Profile Information
    avatar_url = Column(
        String(500),
        nullable=True,
        comment="User avatar URL"
    )
    bio = Column(
        Text,
        nullable=True,
        comment="User biography/description"
    )
    preferences = Column(
        Text,  # JSON stored as text
        nullable=True,
        default="{}",
        comment="User preferences in JSON format"
    )
    
    # Security
    two_factor_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether 2FA is enabled"
    )
    two_factor_secret = Column(
        String(255),
        nullable=True,
        comment="2FA secret key (encrypted)"
    )
    recovery_codes = Column(
        Text,  # JSON array stored as text
        nullable=True,
        comment="2FA recovery codes (encrypted)"
    )
    
    # Login tracking
    login_attempts = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Failed login attempts counter"
    )
    locked_until = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Account locked until this timestamp"
    )
    
    # API Access
    api_key = Column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
        comment="API key for programmatic access"
    )
    api_key_created_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When API key was created"
    )
    
    # Metadata
    registration_ip = Column(
        String(45),  # Support IPv6
        nullable=True,
        comment="IP address used for registration"
    )
    last_login_ip = Column(
        String(45),
        nullable=True,
        comment="IP address of last login"
    )
    user_agent = Column(
        String(500),
        nullable=True,
        comment="Last user agent string"
    )
    
    # Soft delete
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Soft delete timestamp"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_users_email_password', 'email', 'hashed_password'),
        Index('idx_users_status_active', 'status', 'is_active'),
        Index('idx_users_role_active', 'role', 'is_active'),
        Index('idx_users_created_at', 'created_at'),
        Index('idx_users_last_login', 'last_login_at'),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, username={self.username}, role={self.role})>"
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert user to dictionary"""
        data = {
            "id": str(self.id),
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role.value if self.role else None,
            "status": self.status.value if self.status else None,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "is_superuser": self.is_superuser,
            "avatar_url": self.avatar_url,
            "bio": self.bio,
            "two_factor_enabled": self.two_factor_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
        }
        
        if include_sensitive:
            data.update({
                "email_verified_at": self.email_verified_at.isoformat() if self.email_verified_at else None,
                "password_changed_at": self.password_changed_at.isoformat() if self.password_changed_at else None,
                "login_attempts": self.login_attempts,
                "locked_until": self.locked_until.isoformat() if self.locked_until else None,
                "api_key": self.api_key,
                "api_key_created_at": self.api_key_created_at.isoformat() if self.api_key_created_at else None,
                "registration_ip": self.registration_ip,
                "last_login_ip": self.last_login_ip,
            })
        
        return data
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN or self.is_superuser
    
    @property
    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until:
            return datetime.utcnow() < self.locked_until
        return False
    
    @property
    def can_login(self) -> bool:
        """Check if user can login"""
        return (
            self.is_active and 
            not self.is_locked and 
            self.status == UserStatus.ACTIVE
        )
    
    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has required permission level"""
        if self.is_superuser:
            return True
        return UserRole.has_permission(self.role, required_role)
    
    def increment_login_attempts(self):
        """Increment failed login attempts"""
        self.login_attempts = (self.login_attempts or 0) + 1
    
    def reset_login_attempts(self):
        """Reset login attempts on successful login"""
        self.login_attempts = 0
        self.locked_until = None
    
    def lock_account(self, minutes: int = 15):
        """Lock account for specified minutes"""
        from datetime import timedelta
        self.locked_until = datetime.utcnow() + timedelta(minutes=minutes)
    
    def update_last_login(self, ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Update last login information"""
        self.last_login_at = datetime.utcnow()
        self.last_activity_at = datetime.utcnow()
        if ip_address:
            self.last_login_ip = ip_address
        if user_agent:
            self.user_agent = user_agent
        self.reset_login_attempts()
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity_at = datetime.utcnow()
    
    def generate_api_key(self) -> str:
        """Generate a new API key"""
        import secrets
        self.api_key = secrets.token_urlsafe(32)
        self.api_key_created_at = datetime.utcnow()
        return self.api_key
    
    def revoke_api_key(self):
        """Revoke API key"""
        self.api_key = None
        self.api_key_created_at = None
    
    def soft_delete(self):
        """Soft delete the user"""
        self.deleted_at = datetime.utcnow()
        self.is_active = False
        self.status = UserStatus.INACTIVE
    
    def restore(self):
        """Restore soft deleted user"""
        self.deleted_at = None
        self.is_active = True
        self.status = UserStatus.ACTIVE