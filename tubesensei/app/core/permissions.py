"""
Role-Based Access Control (RBAC) permission system for TubeSensei
"""
import enum
from typing import Set, Dict, List, Callable, Any, Optional
from functools import wraps, lru_cache
from fastapi import Depends, HTTPException, status
import logging

from app.models.user import User, UserRole
from app.core.auth import get_current_user
from app.core.exceptions import PermissionException, AuthenticationException

logger = logging.getLogger(__name__)


class Permission(str, enum.Enum):
    """System permissions enumeration"""
    
    # User permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"
    
    # Profile permissions
    PROFILE_READ = "profile:read"
    PROFILE_WRITE = "profile:write"
    
    # Video permissions
    VIDEO_READ = "video:read"
    VIDEO_WRITE = "video:write"
    VIDEO_DELETE = "video:delete"
    VIDEO_ADMIN = "video:admin"
    
    # Channel permissions
    CHANNEL_READ = "channel:read"
    CHANNEL_WRITE = "channel:write"
    CHANNEL_DELETE = "channel:delete"
    CHANNEL_ADMIN = "channel:admin"
    
    # Transcript permissions
    TRANSCRIPT_READ = "transcript:read"
    TRANSCRIPT_WRITE = "transcript:write"
    TRANSCRIPT_DELETE = "transcript:delete"
    TRANSCRIPT_ADMIN = "transcript:admin"
    
    # Processing permissions
    PROCESSING_READ = "processing:read"
    PROCESSING_WRITE = "processing:write"
    PROCESSING_DELETE = "processing:delete"
    PROCESSING_ADMIN = "processing:admin"
    
    # Filter permissions
    FILTER_READ = "filter:read"
    FILTER_WRITE = "filter:write"
    FILTER_DELETE = "filter:delete"
    FILTER_ADMIN = "filter:admin"
    
    # Admin interface permissions
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    ADMIN_DELETE = "admin:delete"
    ADMIN_FULL = "admin:full"
    
    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_WRITE = "analytics:write"
    ANALYTICS_ADMIN = "analytics:admin"
    
    # Settings permissions
    SETTINGS_READ = "settings:read"
    SETTINGS_WRITE = "settings:write"
    SETTINGS_ADMIN = "settings:admin"


class PermissionCategory(str, enum.Enum):
    """Permission categories for grouping"""
    USER_MANAGEMENT = "user_management"
    CONTENT_MANAGEMENT = "content_management"
    PROCESSING_MANAGEMENT = "processing_management"
    ADMIN_INTERFACE = "admin_interface"
    SYSTEM_ADMINISTRATION = "system_administration"
    API_ACCESS = "api_access"
    ANALYTICS = "analytics"
    SETTINGS = "settings"


# Role-Permission mappings
@lru_cache(maxsize=None)
def get_role_permissions() -> Dict[UserRole, Set[Permission]]:
    """Get role-permission mappings"""
    return {
        UserRole.VIEWER: {
            # Basic read permissions
            Permission.VIDEO_READ,
            Permission.CHANNEL_READ,
            Permission.TRANSCRIPT_READ,
            Permission.FILTER_READ,
            Permission.PROFILE_READ,
            Permission.API_READ,
        },
        
        UserRole.USER: {
            # All viewer permissions
            *get_role_permissions()[UserRole.VIEWER],
            
            # Additional user permissions
            Permission.USER_READ,
            Permission.PROFILE_WRITE,
            Permission.VIDEO_WRITE,
            Permission.CHANNEL_WRITE,
            Permission.TRANSCRIPT_WRITE,
            Permission.FILTER_WRITE,
            Permission.PROCESSING_READ,
            Permission.PROCESSING_WRITE,
            Permission.API_WRITE,
            Permission.ANALYTICS_READ,
        },
        
        UserRole.ADMIN: {
            # All user permissions
            *get_role_permissions()[UserRole.USER],
            
            # Administrative permissions
            Permission.USER_WRITE,
            Permission.USER_DELETE,
            Permission.USER_ADMIN,
            Permission.VIDEO_DELETE,
            Permission.VIDEO_ADMIN,
            Permission.CHANNEL_DELETE,
            Permission.CHANNEL_ADMIN,
            Permission.TRANSCRIPT_DELETE,
            Permission.TRANSCRIPT_ADMIN,
            Permission.FILTER_DELETE,
            Permission.FILTER_ADMIN,
            Permission.PROCESSING_DELETE,
            Permission.PROCESSING_ADMIN,
            Permission.ADMIN_READ,
            Permission.ADMIN_WRITE,
            Permission.ADMIN_DELETE,
            Permission.ADMIN_FULL,
            Permission.SYSTEM_READ,
            Permission.SYSTEM_WRITE,
            Permission.SYSTEM_ADMIN,
            Permission.API_ADMIN,
            Permission.ANALYTICS_WRITE,
            Permission.ANALYTICS_ADMIN,
            Permission.SETTINGS_READ,
            Permission.SETTINGS_WRITE,
            Permission.SETTINGS_ADMIN,
        }
    }


@lru_cache(maxsize=None)
def get_permission_categories() -> Dict[Permission, PermissionCategory]:
    """Get permission category mappings"""
    return {
        # User management
        Permission.USER_READ: PermissionCategory.USER_MANAGEMENT,
        Permission.USER_WRITE: PermissionCategory.USER_MANAGEMENT,
        Permission.USER_DELETE: PermissionCategory.USER_MANAGEMENT,
        Permission.USER_ADMIN: PermissionCategory.USER_MANAGEMENT,
        Permission.PROFILE_READ: PermissionCategory.USER_MANAGEMENT,
        Permission.PROFILE_WRITE: PermissionCategory.USER_MANAGEMENT,
        
        # Content management
        Permission.VIDEO_READ: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.VIDEO_WRITE: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.VIDEO_DELETE: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.VIDEO_ADMIN: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.CHANNEL_READ: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.CHANNEL_WRITE: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.CHANNEL_DELETE: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.CHANNEL_ADMIN: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.TRANSCRIPT_READ: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.TRANSCRIPT_WRITE: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.TRANSCRIPT_DELETE: PermissionCategory.CONTENT_MANAGEMENT,
        Permission.TRANSCRIPT_ADMIN: PermissionCategory.CONTENT_MANAGEMENT,
        
        # Processing management
        Permission.PROCESSING_READ: PermissionCategory.PROCESSING_MANAGEMENT,
        Permission.PROCESSING_WRITE: PermissionCategory.PROCESSING_MANAGEMENT,
        Permission.PROCESSING_DELETE: PermissionCategory.PROCESSING_MANAGEMENT,
        Permission.PROCESSING_ADMIN: PermissionCategory.PROCESSING_MANAGEMENT,
        Permission.FILTER_READ: PermissionCategory.PROCESSING_MANAGEMENT,
        Permission.FILTER_WRITE: PermissionCategory.PROCESSING_MANAGEMENT,
        Permission.FILTER_DELETE: PermissionCategory.PROCESSING_MANAGEMENT,
        Permission.FILTER_ADMIN: PermissionCategory.PROCESSING_MANAGEMENT,
        
        # Admin interface
        Permission.ADMIN_READ: PermissionCategory.ADMIN_INTERFACE,
        Permission.ADMIN_WRITE: PermissionCategory.ADMIN_INTERFACE,
        Permission.ADMIN_DELETE: PermissionCategory.ADMIN_INTERFACE,
        Permission.ADMIN_FULL: PermissionCategory.ADMIN_INTERFACE,
        
        # System administration
        Permission.SYSTEM_READ: PermissionCategory.SYSTEM_ADMINISTRATION,
        Permission.SYSTEM_WRITE: PermissionCategory.SYSTEM_ADMINISTRATION,
        Permission.SYSTEM_ADMIN: PermissionCategory.SYSTEM_ADMINISTRATION,
        
        # API access
        Permission.API_READ: PermissionCategory.API_ACCESS,
        Permission.API_WRITE: PermissionCategory.API_ACCESS,
        Permission.API_ADMIN: PermissionCategory.API_ACCESS,
        
        # Analytics
        Permission.ANALYTICS_READ: PermissionCategory.ANALYTICS,
        Permission.ANALYTICS_WRITE: PermissionCategory.ANALYTICS,
        Permission.ANALYTICS_ADMIN: PermissionCategory.ANALYTICS,
        
        # Settings
        Permission.SETTINGS_READ: PermissionCategory.SETTINGS,
        Permission.SETTINGS_WRITE: PermissionCategory.SETTINGS,
        Permission.SETTINGS_ADMIN: PermissionCategory.SETTINGS,
    }


class PermissionChecker:
    """Permission checking utilities"""
    
    @staticmethod
    def has_permission(user: User, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        if user.is_superuser:
            return True
        
        role_permissions = get_role_permissions()
        user_permissions = role_permissions.get(user.role, set())
        
        return permission in user_permissions
    
    @staticmethod
    def has_any_permission(user: User, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        if user.is_superuser:
            return True
        
        return any(PermissionChecker.has_permission(user, perm) for perm in permissions)
    
    @staticmethod
    def has_all_permissions(user: User, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions"""
        if user.is_superuser:
            return True
        
        return all(PermissionChecker.has_permission(user, perm) for perm in permissions)
    
    @staticmethod
    def get_user_permissions(user: User) -> Set[Permission]:
        """Get all permissions for a user"""
        if user.is_superuser:
            return set(Permission)
        
        role_permissions = get_role_permissions()
        return role_permissions.get(user.role, set())
    
    @staticmethod
    def get_permissions_by_category(
        user: User, 
        category: PermissionCategory
    ) -> Set[Permission]:
        """Get user permissions for a specific category"""
        user_permissions = PermissionChecker.get_user_permissions(user)
        permission_categories = get_permission_categories()
        
        return {
            perm for perm in user_permissions 
            if permission_categories.get(perm) == category
        }
    
    @staticmethod
    def can_access_resource(
        user: User, 
        resource_type: str, 
        action: str = "read"
    ) -> bool:
        """Check if user can access a resource with a specific action"""
        permission_name = f"{resource_type}:{action}"
        
        try:
            permission = Permission(permission_name)
            return PermissionChecker.has_permission(user, permission)
        except ValueError:
            logger.warning(f"Unknown permission: {permission_name}")
            return False
    
    @staticmethod
    def validate_resource_ownership(
        user: User, 
        resource_user_id: Any,
        allow_admin_access: bool = True
    ) -> bool:
        """Check if user owns a resource or has admin access"""
        # Convert to string for comparison
        user_id_str = str(user.id)
        resource_user_id_str = str(resource_user_id)
        
        # User owns the resource
        if user_id_str == resource_user_id_str:
            return True
        
        # Admin can access if allowed
        if allow_admin_access and user.is_admin:
            return True
        
        return False


# FastAPI Dependencies
def require_permission(permission: Permission):
    """Dependency factory for requiring specific permission"""
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        # Allow admin bypass regardless of DEBUG mode
        from app.core.config import get_settings
        settings = get_settings()
        # Always allow admin users
        if current_user.role == UserRole.ADMIN:
            return current_user
            
        if not PermissionChecker.has_permission(current_user, permission):
            raise PermissionException(
                f"Access denied. Required permission: {permission.value}",
                required_permission=permission.value
            )
        return current_user
    
    return permission_checker


async def require_any_permission(permissions: List[Permission]):
    """Dependency factory for requiring any of the specified permissions"""
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not PermissionChecker.has_any_permission(current_user, permissions):
            permission_names = [p.value for p in permissions]
            raise PermissionException(
                f"Access denied. Required permissions (any): {', '.join(permission_names)}",
                required_permission=f"any_of:{','.join(permission_names)}"
            )
        return current_user
    
    return permission_checker


async def require_all_permissions(permissions: List[Permission]):
    """Dependency factory for requiring all of the specified permissions"""
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not PermissionChecker.has_all_permissions(current_user, permissions):
            permission_names = [p.value for p in permissions]
            raise PermissionException(
                f"Access denied. Required permissions (all): {', '.join(permission_names)}",
                required_permission=f"all_of:{','.join(permission_names)}"
            )
        return current_user
    
    return permission_checker


# Resource-specific dependencies
async def require_admin_access(current_user: User = Depends(get_current_user)) -> User:
    """Require admin interface access"""
    # Allow admin users regardless of DEBUG mode
    from app.core.config import get_settings
    settings = get_settings()
    if current_user.role == UserRole.ADMIN:
        return current_user
        
    if not PermissionChecker.has_permission(current_user, Permission.ADMIN_READ):
        raise PermissionException(
            "Admin access required",
            required_permission=Permission.ADMIN_READ.value
        )
    return current_user


async def require_user_management(current_user: User = Depends(get_current_user)) -> User:
    """Require user management permissions"""
    required_permissions = [Permission.USER_READ, Permission.USER_WRITE]
    if not PermissionChecker.has_any_permission(current_user, required_permissions):
        raise PermissionException(
            "User management access required",
            required_permission="user_management"
        )
    return current_user


async def require_content_management(current_user: User = Depends(get_current_user)) -> User:
    """Require content management permissions"""
    required_permissions = [
        Permission.VIDEO_WRITE, Permission.CHANNEL_WRITE, Permission.TRANSCRIPT_WRITE
    ]
    if not PermissionChecker.has_any_permission(current_user, required_permissions):
        raise PermissionException(
            "Content management access required",
            required_permission="content_management"
        )
    return current_user


async def require_system_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require system administration permissions"""
    if not PermissionChecker.has_permission(current_user, Permission.SYSTEM_ADMIN):
        raise PermissionException(
            "System administration access required",
            required_permission=Permission.SYSTEM_ADMIN.value
        )
    return current_user


# Decorators for protecting functions
def require_permissions(*permissions: Permission):
    """Decorator for requiring permissions on functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to find user in kwargs or args
            user = kwargs.get('current_user') or kwargs.get('user')
            
            if not user:
                # Look for user in args
                for arg in args:
                    if isinstance(arg, User):
                        user = arg
                        break
            
            if not user:
                raise AuthenticationException("User context required for permission check")
            
            # Check permissions
            if not PermissionChecker.has_all_permissions(user, list(permissions)):
                permission_names = [p.value for p in permissions]
                raise PermissionException(
                    f"Access denied. Required permissions: {', '.join(permission_names)}",
                    required_permission=f"all_of:{','.join(permission_names)}"
                )
            
            return await func(*args, **kwargs) if hasattr(func, '__awaitable__') else func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_resource_ownership(
    resource_id_param: str = "resource_id",
    user_id_field: str = "user_id",
    allow_admin: bool = True
):
    """Decorator for requiring resource ownership or admin access"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user') or kwargs.get('user')
            if not user:
                raise AuthenticationException("User context required")
            
            # Get resource (assuming it's passed as a parameter)
            resource = kwargs.get('resource')
            if resource and hasattr(resource, user_id_field):
                resource_user_id = getattr(resource, user_id_field)
                
                if not PermissionChecker.validate_resource_ownership(
                    user, resource_user_id, allow_admin
                ):
                    raise PermissionException("Access denied. Resource ownership required")
            
            return await func(*args, **kwargs) if hasattr(func, '__awaitable__') else func(*args, **kwargs)
        
        return wrapper
    return decorator


# Utility functions for templates and UI
def get_user_permission_summary(user: User) -> Dict[str, Any]:
    """Get a summary of user permissions for UI display"""
    permissions = PermissionChecker.get_user_permissions(user)
    permission_categories = get_permission_categories()
    
    # Group permissions by category
    categorized_permissions = {}
    for permission in permissions:
        category = permission_categories.get(permission, PermissionCategory.SYSTEM_ADMINISTRATION)
        if category.value not in categorized_permissions:
            categorized_permissions[category.value] = []
        categorized_permissions[category.value].append(permission.value)
    
    return {
        "user_id": str(user.id),
        "role": user.role.value,
        "is_superuser": user.is_superuser,
        "is_admin": user.is_admin,
        "total_permissions": len(permissions),
        "permissions_by_category": categorized_permissions,
        "all_permissions": [p.value for p in permissions]
    }


def check_route_access(user: User, route_path: str, method: str = "GET") -> bool:
    """Check if user can access a specific route"""
    # Define route permission mappings
    route_permissions = {
        "/admin": Permission.ADMIN_READ,
        "/admin/users": Permission.USER_ADMIN,
        "/admin/videos": Permission.VIDEO_ADMIN,
        "/admin/channels": Permission.CHANNEL_ADMIN,
        "/admin/transcripts": Permission.TRANSCRIPT_ADMIN,
        "/admin/processing": Permission.PROCESSING_ADMIN,
        "/admin/settings": Permission.SETTINGS_ADMIN,
        "/api/users": Permission.USER_READ if method == "GET" else Permission.USER_WRITE,
        "/api/videos": Permission.VIDEO_READ if method == "GET" else Permission.VIDEO_WRITE,
        "/api/channels": Permission.CHANNEL_READ if method == "GET" else Permission.CHANNEL_WRITE,
        "/api/transcripts": Permission.TRANSCRIPT_READ if method == "GET" else Permission.TRANSCRIPT_WRITE,
    }
    
    # Find matching permission
    for route_pattern, required_permission in route_permissions.items():
        if route_path.startswith(route_pattern):
            return PermissionChecker.has_permission(user, required_permission)
    
    # Default to requiring user permission for API routes
    if route_path.startswith("/api/"):
        base_permission = Permission.API_READ if method == "GET" else Permission.API_WRITE
        return PermissionChecker.has_permission(user, base_permission)
    
    # Allow access to non-protected routes
    return True