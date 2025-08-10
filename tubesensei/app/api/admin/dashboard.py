"""
Admin dashboard API router for TubeSensei
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

from app.database import get_db
from app.models.user import User, UserRole, UserStatus
from app.models.video import Video
from app.models.channel import Channel
from app.models.transcript import Transcript
from app.models.processing_job import ProcessingJob, JobStatus
from app.models.processing_session import ProcessingSession
from app.schemas.user import UserResponse, UserList
from app.core.auth import get_current_user
from app.core.permissions import (
    require_admin_access, require_permission, require_system_admin,
    Permission, PermissionChecker, get_user_permission_summary
)
from app.core.session import get_session_manager_dependency, RedisSessionManager
from app.core.config import get_settings
from app.core.exceptions import PermissionException, NotFoundException

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(
    prefix="/admin/dashboard",
    tags=["Admin Dashboard"],
    dependencies=[Depends(require_admin_access)]
)


@router.get("/stats")
async def get_dashboard_stats(
    current_user: User = Depends(require_admin_access),
    db: AsyncSession = Depends(get_db),
    session_manager: RedisSessionManager = Depends(get_session_manager_dependency)
):
    """Get overall dashboard statistics"""
    try:
        # Users statistics
        total_users = await db.scalar(select(func.count(User.id)))
        active_users = await db.scalar(
            select(func.count(User.id)).where(
                and_(User.is_active == True, User.status == UserStatus.ACTIVE)
            )
        )
        new_users_today = await db.scalar(
            select(func.count(User.id)).where(
                User.created_at >= datetime.utcnow().date()
            )
        )
        
        # Content statistics
        total_videos = await db.scalar(select(func.count(Video.id)))
        total_channels = await db.scalar(select(func.count(Channel.id)))
        total_transcripts = await db.scalar(select(func.count(Transcript.id)))
        
        # Processing statistics
        total_jobs = await db.scalar(select(func.count(ProcessingJob.id)))
        running_jobs = await db.scalar(
            select(func.count(ProcessingJob.id)).where(
                ProcessingJob.status == JobStatus.RUNNING
            )
        )
        failed_jobs = await db.scalar(
            select(func.count(ProcessingJob.id)).where(
                ProcessingJob.status == JobStatus.FAILED
            )
        )
        
        # Session statistics
        session_stats = await session_manager.get_session_stats()
        
        return {
            "users": {
                "total": total_users or 0,
                "active": active_users or 0,
                "new_today": new_users_today or 0,
                "online": session_stats.get("active_users", 0)
            },
            "content": {
                "videos": total_videos or 0,
                "channels": total_channels or 0,
                "transcripts": total_transcripts or 0
            },
            "processing": {
                "total_jobs": total_jobs or 0,
                "running_jobs": running_jobs or 0,
                "failed_jobs": failed_jobs or 0
            },
            "sessions": session_stats,
            "system": {
                "timestamp": datetime.utcnow().isoformat(),
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get dashboard statistics"
        )


@router.get("/users/recent")
async def get_recent_users(
    limit: int = Query(10, ge=1, le=50, description="Number of recent users to fetch"),
    current_user: User = Depends(require_permission(Permission.USER_READ)),
    db: AsyncSession = Depends(get_db)
):
    """Get recently registered users"""
    try:
        result = await db.execute(
            select(User)
            .where(User.deleted_at.is_(None))
            .order_by(desc(User.created_at))
            .limit(limit)
        )
        users = result.scalars().all()
        
        return {
            "users": [UserResponse.model_validate(user) for user in users],
            "total": len(users)
        }
        
    except Exception as e:
        logger.error(f"Error getting recent users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recent users"
        )


@router.get("/users/stats")
async def get_user_statistics(
    current_user: User = Depends(require_permission(Permission.USER_READ)),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed user statistics"""
    try:
        # User counts by status
        status_counts = {}
        for user_status in UserStatus:
            count = await db.scalar(
                select(func.count(User.id)).where(User.status == user_status)
            )
            status_counts[user_status.value] = count or 0
        
        # User counts by role
        role_counts = {}
        for user_role in UserRole:
            count = await db.scalar(
                select(func.count(User.id)).where(User.role == user_role)
            )
            role_counts[user_role.value] = count or 0
        
        # Registration trends (last 7 days)
        registration_trends = []
        for i in range(7):
            date = datetime.utcnow().date() - timedelta(days=i)
            count = await db.scalar(
                select(func.count(User.id)).where(
                    func.date(User.created_at) == date
                )
            )
            registration_trends.append({
                "date": date.isoformat(),
                "count": count or 0
            })
        
        # Active users (logged in within last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        active_users_30d = await db.scalar(
            select(func.count(User.id)).where(
                and_(
                    User.last_login_at >= thirty_days_ago,
                    User.is_active == True
                )
            )
        )
        
        return {
            "by_status": status_counts,
            "by_role": role_counts,
            "registration_trends": list(reversed(registration_trends)),
            "active_users_30d": active_users_30d or 0
        }
        
    except Exception as e:
        logger.error(f"Error getting user statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user statistics"
        )


@router.get("/processing/stats")
async def get_processing_statistics(
    current_user: User = Depends(require_permission(Permission.PROCESSING_READ)),
    db: AsyncSession = Depends(get_db)
):
    """Get processing job statistics"""
    try:
        # Job counts by status
        status_counts = {}
        for job_status in JobStatus:
            count = await db.scalar(
                select(func.count(ProcessingJob.id)).where(
                    ProcessingJob.status == job_status
                )
            )
            status_counts[job_status.value] = count or 0
        
        # Processing trends (last 7 days)
        processing_trends = []
        for i in range(7):
            date = datetime.utcnow().date() - timedelta(days=i)
            completed_count = await db.scalar(
                select(func.count(ProcessingJob.id)).where(
                    and_(
                        func.date(ProcessingJob.completed_at) == date,
                        ProcessingJob.status == JobStatus.COMPLETED
                    )
                )
            )
            failed_count = await db.scalar(
                select(func.count(ProcessingJob.id)).where(
                    and_(
                        func.date(ProcessingJob.updated_at) == date,
                        ProcessingJob.status == JobStatus.FAILED
                    )
                )
            )
            processing_trends.append({
                "date": date.isoformat(),
                "completed": completed_count or 0,
                "failed": failed_count or 0
            })
        
        # Recent failed jobs
        result = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.status == JobStatus.FAILED)
            .order_by(desc(ProcessingJob.updated_at))
            .limit(5)
        )
        failed_jobs = result.scalars().all()
        
        return {
            "by_status": status_counts,
            "trends": list(reversed(processing_trends)),
            "recent_failures": [
                {
                    "id": str(job.id),
                    "type": job.job_type,
                    "error": job.error_message,
                    "updated_at": job.updated_at.isoformat() if job.updated_at else None
                }
                for job in failed_jobs
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting processing statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing statistics"
        )


@router.get("/content/stats")
async def get_content_statistics(
    current_user: User = Depends(require_permission(Permission.VIDEO_READ)),
    db: AsyncSession = Depends(get_db)
):
    """Get content statistics"""
    try:
        # Content creation trends (last 7 days)
        content_trends = []
        for i in range(7):
            date = datetime.utcnow().date() - timedelta(days=i)
            
            videos_count = await db.scalar(
                select(func.count(Video.id)).where(
                    func.date(Video.created_at) == date
                )
            )
            
            channels_count = await db.scalar(
                select(func.count(Channel.id)).where(
                    func.date(Channel.created_at) == date
                )
            )
            
            transcripts_count = await db.scalar(
                select(func.count(Transcript.id)).where(
                    func.date(Transcript.created_at) == date
                )
            )
            
            content_trends.append({
                "date": date.isoformat(),
                "videos": videos_count or 0,
                "channels": channels_count or 0,
                "transcripts": transcripts_count or 0
            })
        
        # Top channels by video count
        result = await db.execute(
            select(
                Channel.id,
                Channel.title,
                func.count(Video.id).label("video_count")
            )
            .outerjoin(Video, Video.channel_id == Channel.id)
            .group_by(Channel.id, Channel.title)
            .order_by(desc("video_count"))
            .limit(5)
        )
        top_channels = [
            {
                "id": str(row[0]),
                "title": row[1],
                "video_count": row[2] or 0
            }
            for row in result
        ]
        
        return {
            "trends": list(reversed(content_trends)),
            "top_channels": top_channels
        }
        
    except Exception as e:
        logger.error(f"Error getting content statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get content statistics"
        )


@router.get("/system/health")
async def get_system_health(
    current_user: User = Depends(require_permission(Permission.SYSTEM_READ)),
    db: AsyncSession = Depends(get_db),
    session_manager: RedisSessionManager = Depends(get_session_manager_dependency)
):
    """Get system health status"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Database health
        try:
            await db.execute(select(1))
            health_status["services"]["database"] = {
                "status": "healthy",
                "message": "Database connection successful"
            }
        except Exception as db_error:
            health_status["services"]["database"] = {
                "status": "unhealthy",
                "message": f"Database error: {str(db_error)}"
            }
            health_status["status"] = "degraded"
        
        # Redis health
        try:
            session_stats = await session_manager.get_session_stats()
            if session_stats.get("redis_connected", False):
                health_status["services"]["redis"] = {
                    "status": "healthy",
                    "message": "Redis connection successful",
                    "sessions": session_stats.get("total_sessions", 0)
                }
            else:
                health_status["services"]["redis"] = {
                    "status": "unhealthy",
                    "message": "Redis not connected"
                }
                health_status["status"] = "degraded"
        except Exception as redis_error:
            health_status["services"]["redis"] = {
                "status": "unhealthy",
                "message": f"Redis error: {str(redis_error)}"
            }
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Health check failed: {str(e)}"
        }


@router.get("/activities/recent")
async def get_recent_activities(
    limit: int = Query(20, ge=1, le=100, description="Number of activities to fetch"),
    current_user: User = Depends(require_permission(Permission.ADMIN_READ)),
    db: AsyncSession = Depends(get_db)
):
    """Get recent system activities"""
    try:
        # This would typically come from an audit log table
        # For now, we'll return recent user activities based on available data
        
        # Recent user registrations
        result = await db.execute(
            select(User.id, User.email, User.created_at)
            .where(User.deleted_at.is_(None))
            .order_by(desc(User.created_at))
            .limit(limit // 2)
        )
        registrations = result.all()
        
        # Recent processing jobs
        result = await db.execute(
            select(
                ProcessingJob.id,
                ProcessingJob.job_type,
                ProcessingJob.status,
                ProcessingJob.created_at,
                ProcessingJob.user_id
            )
            .order_by(desc(ProcessingJob.created_at))
            .limit(limit // 2)
        )
        jobs = result.all()
        
        activities = []
        
        # Add registration activities
        for user_id, email, created_at in registrations:
            activities.append({
                "id": f"user_reg_{user_id}",
                "type": "user_registration",
                "description": f"New user registered: {email}",
                "user_id": str(user_id),
                "timestamp": created_at.isoformat(),
                "metadata": {"email": email}
            })
        
        # Add job activities
        for job_id, job_type, status, created_at, user_id in jobs:
            activities.append({
                "id": f"job_{job_id}",
                "type": "processing_job",
                "description": f"Processing job {status.value}: {job_type}",
                "user_id": str(user_id) if user_id else None,
                "timestamp": created_at.isoformat(),
                "metadata": {
                    "job_type": job_type,
                    "status": status.value,
                    "job_id": str(job_id)
                }
            })
        
        # Sort activities by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "activities": activities[:limit],
            "total": len(activities)
        }
        
    except Exception as e:
        logger.error(f"Error getting recent activities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get recent activities"
        )


@router.get("/permissions/summary")
async def get_permissions_summary(
    current_user: User = Depends(require_system_admin)
):
    """Get permissions summary for the current admin user"""
    try:
        permission_summary = get_user_permission_summary(current_user)
        
        return {
            "user": permission_summary,
            "system_info": {
                "total_permissions": len(list(Permission)),
                "permission_categories": list(set(cat.value for cat in PermissionCategory)),
                "available_roles": [role.value for role in UserRole]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting permissions summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get permissions summary"
        )


@router.get("/config")
async def get_system_config(
    current_user: User = Depends(require_system_admin)
):
    """Get system configuration (non-sensitive values only)"""
    try:
        return {
            "app": {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
                "debug": settings.DEBUG
            },
            "features": {
                "registration_enabled": settings.FEATURES_ENABLE_REGISTRATION,
                "api_docs_enabled": settings.FEATURES_ENABLE_API_DOCS,
                "metrics_enabled": settings.FEATURES_ENABLE_METRICS,
                "health_checks_enabled": settings.FEATURES_ENABLE_HEALTH_CHECKS
            },
            "admin": {
                "path_prefix": settings.admin.ADMIN_PATH_PREFIX,
                "title": settings.admin.ADMIN_TITLE,
                "pagination_default": settings.admin.ADMIN_PAGINATION_DEFAULT,
                "pagination_max": settings.admin.ADMIN_PAGINATION_MAX
            },
            "security": {
                "access_token_expire_minutes": settings.security.ACCESS_TOKEN_EXPIRE_MINUTES,
                "session_expire_hours": settings.security.SESSION_EXPIRE_HOURS,
                "password_min_length": settings.security.PASSWORD_MIN_LENGTH,
                "login_attempts_max": settings.security.LOGIN_ATTEMPTS_MAX,
                "rate_limit_enabled": settings.security.RATE_LIMIT_ENABLED
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system configuration"
        )