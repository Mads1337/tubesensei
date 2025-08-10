from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
import asyncio
import json
from datetime import datetime

from app.core.auth import auth_handler
from app.services.monitoring_service import MonitoringService
from app.database import get_db


router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "dashboard": set(),
            "jobs": set(),
            "ideas": set()
        }
    
    async def connect(self, websocket: WebSocket, channel: str = "dashboard"):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[channel].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, channel: str = "dashboard"):
        """Remove connection"""
        self.active_connections[channel].discard(websocket)
    
    async def broadcast(self, message: dict, channel: str = "dashboard"):
        """Broadcast message to all connections in channel"""
        disconnected = set()
        
        for websocket in self.active_connections[channel]:
            try:
                await websocket.send_json(message)
            except:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.active_connections[channel].discard(websocket)
    
    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        await websocket.send_json(message)
    
    def get_connection_count(self, channel: str = None) -> int:
        """Get number of active connections"""
        if channel:
            return len(self.active_connections.get(channel, set()))
        return sum(len(conns) for conns in self.active_connections.values())


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for dashboard updates"""
    await manager.connect(websocket, "dashboard")
    
    try:
        async for db in get_db():
            monitoring = MonitoringService(db)
            
            # Send initial status
            await send_dashboard_update(websocket, monitoring)
            
            # Create update task
            update_task = asyncio.create_task(
                periodic_dashboard_updates(websocket, monitoring)
            )
            
            try:
                # Keep connection alive and handle messages
                while True:
                    # Wait for messages from client (heartbeat, etc.)
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0
                        )
                        
                        # Handle client messages
                        message = json.loads(data)
                        if message.get("type") == "ping":
                            await manager.send_personal(
                                {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                                websocket
                            )
                        elif message.get("type") == "refresh":
                            await send_dashboard_update(websocket, monitoring)
                            
                    except asyncio.TimeoutError:
                        # Send ping to check if connection is alive
                        await manager.send_personal(
                            {"type": "ping", "timestamp": datetime.utcnow().isoformat()},
                            websocket
                        )
                        
            except WebSocketDisconnect:
                update_task.cancel()
                await manager.disconnect(websocket, "dashboard")
            finally:
                await monitoring.cleanup()
                
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(websocket, "dashboard")


async def periodic_dashboard_updates(websocket: WebSocket, monitoring: MonitoringService):
    """Send periodic updates to dashboard"""
    try:
        while True:
            await asyncio.sleep(2)  # Update every 2 seconds
            await send_dashboard_update(websocket, monitoring)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Update task error: {e}")


async def send_dashboard_update(websocket: WebSocket, monitoring: MonitoringService):
    """Send dashboard update to websocket"""
    try:
        status_data = {
            "type": "status_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "system_status": await monitoring.get_system_status(),
                "processing_stats": await monitoring.get_processing_stats(),
                "queue_status": await monitoring.get_queue_status(),
                "recent_jobs": await monitoring.get_recent_jobs(5)
            }
        }
        
        await manager.send_personal(status_data, websocket)
    except Exception as e:
        error_message = {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal(error_message, websocket)


@router.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """WebSocket for specific job updates"""
    channel = f"job_{job_id}"
    await manager.connect(websocket, channel)
    
    try:
        async for db in get_db():
            from app.models.processing_job import ProcessingJob
            
            while True:
                # Get job status
                job = await db.get(ProcessingJob, job_id)
                
                if job:
                    job_data = {
                        "type": "job_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "id": str(job.id),
                            "status": job.status.value,
                            "progress": job.progress_percent,
                            "message": job.progress_message,
                            "error": job.error_message,
                            "completed": job.is_complete
                        }
                    }
                    
                    await manager.send_personal(job_data, websocket)
                    
                    # Stop updates if job is complete
                    if job.is_complete:
                        break
                
                await asyncio.sleep(1)
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket, channel)
    except Exception as e:
        print(f"Job WebSocket error: {e}")
        await manager.disconnect(websocket, channel)


@router.websocket("/ws/ideas")
async def ideas_websocket(websocket: WebSocket):
    """WebSocket for idea updates"""
    await manager.connect(websocket, "ideas")
    
    try:
        async for db in get_db():
            from app.services.idea_service import IdeaService
            
            idea_service = IdeaService(db)
            
            while True:
                # Send idea stats
                try:
                    categories = await idea_service.get_categories()
                    recent_ideas = await idea_service.list_ideas(limit=5)
                    
                    idea_data = {
                        "type": "idea_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "categories": categories,
                            "recent_count": recent_ideas["total"],
                            "recent_ideas": recent_ideas["items"][:5]
                        }
                    }
                    
                    await manager.send_personal(idea_data, websocket)
                    
                except Exception as e:
                    error_message = {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await manager.send_personal(error_message, websocket)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket, "ideas")
    except Exception as e:
        print(f"Ideas WebSocket error: {e}")
        await manager.disconnect(websocket, "ideas")


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status"""
    return {
        "total_connections": manager.get_connection_count(),
        "channels": {
            channel: manager.get_connection_count(channel)
            for channel in ["dashboard", "jobs", "ideas"]
        },
        "timestamp": datetime.utcnow().isoformat()
    }