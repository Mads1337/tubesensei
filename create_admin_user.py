#!/usr/bin/env python3
"""
Script to create an admin user for TubeSensei
"""
import asyncio
import sys
import os
sys.path.append('tubesensei')

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from uuid import uuid4
from datetime import datetime

# Import from the app
from app.models.user import User, UserRole, UserStatus
from app.core.auth import PasswordManager
from app.core.config import get_settings

async def create_admin():
    settings = get_settings()
    
    # Create database engine
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Check if admin user already exists
        result = await session.execute(
            select(User).where(User.email == "admin@tubesensei.com")
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            print("Admin user already exists!")
            print(f"Email: admin@tubesensei.com")
            print(f"Username: {existing_user.username}")
            return
        
        # Create new admin user
        admin_user = User(
            id=uuid4(),
            email="admin@tubesensei.com",
            username="admin",
            full_name="TubeSensei Admin",
            password_hash=PasswordManager.hash_password("admin123"),
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            is_email_verified=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(admin_user)
        await session.commit()
        
        print("✅ Admin user created successfully!")
        print(f"Email: admin@tubesensei.com")
        print(f"Password: admin123")
        print("\n⚠️  Please change the password after first login!")

if __name__ == "__main__":
    asyncio.run(create_admin())