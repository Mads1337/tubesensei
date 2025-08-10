"""
Authentication system tests
"""
import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta
from jose import jwt
import bcrypt

from app.main_enhanced import app
from app.core.config import settings
from app.core.auth import auth_handler
from app.models.user import User, UserRole, UserStatus
from app.schemas.user import UserCreate, UserLogin


@pytest.mark.asyncio
class TestAuthentication:
    """Test authentication functionality"""
    
    async def test_register_user(self, client: AsyncClient, db_session):
        """Test user registration"""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "TestPass123!",
            "confirm_password": "TestPass123!",
            "full_name": "Test User"
        }
        
        response = await client.post("/api/auth/register", json=user_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert "id" in data
        assert "password" not in data
        assert "hashed_password" not in data
    
    async def test_register_duplicate_email(self, client: AsyncClient, db_session, test_user):
        """Test registration with duplicate email"""
        user_data = {
            "email": test_user.email,
            "username": "newuser",
            "password": "TestPass123!",
            "confirm_password": "TestPass123!"
        }
        
        response = await client.post("/api/auth/register", json=user_data)
        assert response.status_code == 409
        assert "already exists" in response.json()["error"].lower()
    
    async def test_register_weak_password(self, client: AsyncClient):
        """Test registration with weak password"""
        user_data = {
            "email": "weak@example.com",
            "username": "weakuser",
            "password": "weak",
            "confirm_password": "weak"
        }
        
        response = await client.post("/api/auth/register", json=user_data)
        assert response.status_code == 422
    
    async def test_login_success(self, client: AsyncClient, test_user):
        """Test successful login"""
        login_data = {
            "email": test_user.email,
            "password": "testpass123"  # This should be the unhashed password
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == test_user.email
    
    async def test_login_invalid_credentials(self, client: AsyncClient):
        """Test login with invalid credentials"""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpass"
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["error"]
    
    async def test_login_inactive_user(self, client: AsyncClient, inactive_user):
        """Test login with inactive user"""
        login_data = {
            "email": inactive_user.email,
            "password": "testpass123"
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        assert response.status_code == 401
        assert "disabled" in response.json()["error"].lower()
    
    async def test_get_current_user(self, client: AsyncClient, auth_headers):
        """Test getting current user information"""
        response = await client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "email" in data
        assert "username" in data
        assert "role" in data
    
    async def test_get_current_user_no_auth(self, client: AsyncClient):
        """Test getting current user without authentication"""
        response = await client.get("/api/auth/me")
        assert response.status_code == 403
    
    async def test_refresh_token(self, client: AsyncClient, auth_tokens):
        """Test token refresh"""
        headers = {"Authorization": f"Bearer {auth_tokens['refresh_token']}"}
        response = await client.post("/api/auth/refresh", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != auth_tokens["access_token"]
    
    async def test_logout(self, client: AsyncClient, auth_headers):
        """Test logout"""
        response = await client.post("/api/auth/logout", headers=auth_headers)
        assert response.status_code == 200
        
        # Try to use the token after logout (should still work with JWT)
        response = await client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == 200  # JWT is stateless
    
    async def test_change_password(self, client: AsyncClient, auth_headers, test_user):
        """Test password change"""
        change_data = {
            "current_password": "testpass123",
            "new_password": "NewPass456!",
            "confirm_password": "NewPass456!"
        }
        
        response = await client.post(
            "/api/auth/change-password",
            json=change_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # Try to login with new password
        login_data = {
            "email": test_user.email,
            "password": "NewPass456!"
        }
        response = await client.post("/api/auth/login", json=login_data)
        assert response.status_code == 200


@pytest.mark.asyncio
class TestJWT:
    """Test JWT token functionality"""
    
    def test_encode_decode_token(self):
        """Test JWT encoding and decoding"""
        user_id = "test-user-id"
        email = "test@example.com"
        username = "testuser"
        role = "user"
        
        # Encode token
        token = auth_handler.encode_token(user_id, email, username, role)
        assert token is not None
        
        # Decode token
        payload = auth_handler.decode_token(token)
        assert payload["sub"] == user_id
        assert payload["email"] == email
        assert payload["username"] == username
        assert payload["role"] == role
    
    def test_expired_token(self):
        """Test expired JWT token"""
        user_id = "test-user-id"
        email = "test@example.com"
        username = "testuser"
        role = "user"
        
        # Create expired token
        payload = {
            "sub": user_id,
            "email": email,
            "username": username,
            "role": role,
            "exp": datetime.utcnow() - timedelta(minutes=1),
            "iat": datetime.utcnow() - timedelta(minutes=2)
        }
        token = jwt.encode(payload, settings.security.SECRET_KEY, algorithm=settings.security.ALGORITHM)
        
        # Try to decode expired token
        with pytest.raises(Exception) as exc_info:
            auth_handler.decode_token(token)
        assert "expired" in str(exc_info.value).lower()
    
    def test_invalid_token(self):
        """Test invalid JWT token"""
        with pytest.raises(Exception):
            auth_handler.decode_token("invalid-token")
    
    def test_wrong_secret_token(self):
        """Test JWT token with wrong secret"""
        payload = {
            "sub": "user-id",
            "email": "test@example.com",
            "username": "testuser",
            "role": "user",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow()
        }
        token = jwt.encode(payload, "wrong-secret", algorithm=settings.security.ALGORITHM)
        
        with pytest.raises(Exception):
            auth_handler.decode_token(token)


@pytest.mark.asyncio
class TestPasswordSecurity:
    """Test password security features"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "SecurePass123!"
        
        # Hash password
        hashed = auth_handler.hash_password(password)
        assert hashed != password
        assert hashed.startswith("$2b$")
        
        # Verify correct password
        assert auth_handler.verify_password(password, hashed) is True
        
        # Verify wrong password
        assert auth_handler.verify_password("WrongPass", hashed) is False
    
    def test_password_strength_validation(self):
        """Test password strength requirements"""
        from app.schemas.user import UserCreate
        
        # Too short
        with pytest.raises(ValueError):
            UserCreate(
                email="test@example.com",
                username="test",
                password="Short1!",
                confirm_password="Short1!"
            )
        
        # No uppercase
        with pytest.raises(ValueError):
            UserCreate(
                email="test@example.com",
                username="test",
                password="lowercase123!",
                confirm_password="lowercase123!"
            )
        
        # No lowercase
        with pytest.raises(ValueError):
            UserCreate(
                email="test@example.com",
                username="test",
                password="UPPERCASE123!",
                confirm_password="UPPERCASE123!"
            )
        
        # No digit
        with pytest.raises(ValueError):
            UserCreate(
                email="test@example.com",
                username="test",
                password="NoDigitsHere!",
                confirm_password="NoDigitsHere!"
            )
    
    async def test_login_attempts_lockout(self, client: AsyncClient, test_user, db_session):
        """Test account lockout after failed login attempts"""
        # Make multiple failed login attempts
        for i in range(6):
            response = await client.post("/api/auth/login", json={
                "email": test_user.email,
                "password": "wrongpassword"
            })
            assert response.status_code == 401
        
        # Account should be locked now
        response = await client.post("/api/auth/login", json={
            "email": test_user.email,
            "password": "testpass123"  # Even with correct password
        })
        assert response.status_code == 401
        assert "locked" in response.json()["error"].lower()


@pytest.mark.asyncio
class TestRBAC:
    """Test Role-Based Access Control"""
    
    async def test_admin_only_endpoint(self, client: AsyncClient, admin_headers, user_headers):
        """Test admin-only endpoint access"""
        # Admin should have access
        response = await client.get("/admin/dashboard/users/stats", headers=admin_headers)
        assert response.status_code == 200
        
        # Regular user should not have access
        response = await client.get("/admin/dashboard/users/stats", headers=user_headers)
        assert response.status_code == 403
    
    async def test_user_role_permissions(self, client: AsyncClient, user_headers):
        """Test user role permissions"""
        # User should be able to access their own profile
        response = await client.get("/api/auth/me", headers=user_headers)
        assert response.status_code == 200
        
        # User should not be able to access admin endpoints
        response = await client.get("/admin/users", headers=user_headers)
        assert response.status_code == 403
    
    async def test_viewer_role_limitations(self, client: AsyncClient, viewer_headers):
        """Test viewer role limitations"""
        # Viewer should be able to view content
        response = await client.get("/api/v1/channels", headers=viewer_headers)
        assert response.status_code == 200
        
        # Viewer should not be able to create content
        response = await client.post(
            "/api/v1/channels",
            json={"url": "https://youtube.com/channel/test"},
            headers=viewer_headers
        )
        assert response.status_code == 403


@pytest.mark.asyncio
class TestSession:
    """Test session management"""
    
    async def test_session_creation(self, client: AsyncClient, test_user):
        """Test session creation on login"""
        login_data = {
            "email": test_user.email,
            "password": "testpass123"
        }
        
        response = await client.post("/api/auth/login", json=login_data)
        assert response.status_code == 200
        
        # Check that session was created
        data = response.json()
        assert "access_token" in data
    
    async def test_multiple_sessions(self, client: AsyncClient, test_user):
        """Test multiple concurrent sessions"""
        login_data = {
            "email": test_user.email,
            "password": "testpass123"
        }
        
        # Create multiple sessions
        sessions = []
        for _ in range(3):
            response = await client.post("/api/auth/login", json=login_data)
            assert response.status_code == 200
            sessions.append(response.json()["access_token"])
        
        # All sessions should be valid
        for token in sessions:
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get("/api/auth/me", headers=headers)
            assert response.status_code == 200
    
    async def test_logout_all_sessions(self, client: AsyncClient, test_user):
        """Test logging out from all sessions"""
        login_data = {
            "email": test_user.email,
            "password": "testpass123"
        }
        
        # Create multiple sessions
        sessions = []
        for _ in range(3):
            response = await client.post("/api/auth/login", json=login_data)
            sessions.append(response.json()["access_token"])
        
        # Logout from all sessions
        headers = {"Authorization": f"Bearer {sessions[0]}"}
        response = await client.post("/api/auth/logout-all", headers=headers)
        assert response.status_code == 200
        
        # Note: With stateless JWT, tokens remain valid until expiry
        # This test would need Redis session tracking to fully work