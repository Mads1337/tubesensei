"""
Authentication test fixtures
"""
import pytest
from typing import Dict, AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
import bcrypt

from app.main_enhanced import app
from app.models.user import User, UserRole, UserStatus
from app.core.auth import auth_handler
from app.database import get_db


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user"""
    user = User(
        email="testuser@example.com",
        username="testuser",
        full_name="Test User",
        hashed_password=bcrypt.hashpw(b"testpass123", bcrypt.gensalt()).decode('utf-8'),
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
        is_active=True,
        is_verified=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create an admin user"""
    user = User(
        email="admin@example.com",
        username="adminuser",
        full_name="Admin User",
        hashed_password=bcrypt.hashpw(b"adminpass123", bcrypt.gensalt()).decode('utf-8'),
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        is_active=True,
        is_verified=True,
        is_superuser=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def viewer_user(db_session: AsyncSession) -> User:
    """Create a viewer user"""
    user = User(
        email="viewer@example.com",
        username="vieweruser",
        full_name="Viewer User",
        hashed_password=bcrypt.hashpw(b"viewerpass123", bcrypt.gensalt()).decode('utf-8'),
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
        is_active=True,
        is_verified=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def inactive_user(db_session: AsyncSession) -> User:
    """Create an inactive user"""
    user = User(
        email="inactive@example.com",
        username="inactiveuser",
        full_name="Inactive User",
        hashed_password=bcrypt.hashpw(b"testpass123", bcrypt.gensalt()).decode('utf-8'),
        role=UserRole.USER,
        status=UserStatus.INACTIVE,
        is_active=False,
        is_verified=True
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def auth_tokens(test_user: User) -> Dict[str, str]:
    """Generate authentication tokens for test user"""
    access_token = auth_handler.encode_token(
        str(test_user.id),
        test_user.email,
        test_user.username,
        test_user.role.value
    )
    refresh_token = auth_handler.encode_refresh_token(
        str(test_user.id),
        test_user.email
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token
    }


@pytest.fixture
async def auth_headers(auth_tokens: Dict[str, str]) -> Dict[str, str]:
    """Generate authorization headers for test user"""
    return {"Authorization": f"Bearer {auth_tokens['access_token']}"}


@pytest.fixture
async def admin_headers(admin_user: User) -> Dict[str, str]:
    """Generate authorization headers for admin user"""
    token = auth_handler.encode_token(
        str(admin_user.id),
        admin_user.email,
        admin_user.username,
        admin_user.role.value
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def user_headers(test_user: User) -> Dict[str, str]:
    """Generate authorization headers for regular user"""
    token = auth_handler.encode_token(
        str(test_user.id),
        test_user.email,
        test_user.username,
        test_user.role.value
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def viewer_headers(viewer_user: User) -> Dict[str, str]:
    """Generate authorization headers for viewer user"""
    token = auth_handler.encode_token(
        str(viewer_user.id),
        viewer_user.email,
        viewer_user.username,
        viewer_user.role.value
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def authenticated_client(auth_headers: Dict[str, str]) -> AsyncGenerator[AsyncClient, None]:
    """Create authenticated async HTTP client"""
    async with AsyncClient(app=app, base_url="http://test", headers=auth_headers) as ac:
        yield ac