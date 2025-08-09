"""
Task monitoring and metrics collection for TubeSensei workers
"""
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from threading import Lock
from collections import defaultdict, deque

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import redis

from app.config import settings

logger = logging.getLogger(__name__)

# Prometheus metrics
TASK_COUNTER = Counter(
    'celery_task_total',
    'Total number of tasks processed',
    ['task_name', 'status', 'queue']
)

TASK_DURATION = Histogram(
    'celery_task_duration_seconds',
    'Task execution time in seconds',
    ['task_name', 'queue'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 60.0, 120.0, 300.0, float('inf')]
)

QUEUE_SIZE = Gauge(
    'celery_queue_size',
    'Number of tasks in queue',
    ['queue_name']
)

ACTIVE_WORKERS = Gauge(
    'celery_active_workers',
    'Number of active workers',
    ['queue_name']
)

WORKER_MEMORY = Gauge(
    'celery_worker_memory_bytes',
    'Worker memory usage in bytes',
    ['worker_id', 'queue']
)

TRANSCRIPT_SUCCESS_RATE = Gauge(
    'transcript_extraction_success_rate',
    'Success rate of transcript extractions over last hour',
    ['time_window']
)

VIDEO_DISCOVERY_RATE = Gauge(
    'video_discovery_rate_per_minute',
    'Rate of video discovery per minute',
    ['channel_type']
)

API_QUOTA_USAGE = Gauge(
    'youtube_api_quota_usage_percent',
    'YouTube API quota usage percentage',
    ['api_type']
)

JOB_QUEUE_DEPTH = Gauge(
    'job_queue_depth',
    'Depth of job queues by priority',
    ['priority', 'job_type']
)

SYSTEM_INFO = Info(
    'tubesensei_system_info',
    'System information'
)

# Set system info
SYSTEM_INFO.info({
    'version': '1.0.0',
    'phase': '1D',
    'celery_version': '5.3.4',
    'redis_url': settings.REDIS_URL
})


class TaskMonitor:
    """Centralized task monitoring and metrics collection"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._task_stats = defaultdict(lambda: {
                'count': 0,
                'total_duration': 0.0,
                'last_execution': None,
                'error_count': 0,
                'success_count': 0
            })
            self._recent_tasks = deque(maxlen=1000)  # Keep last 1000 tasks
            self._redis_client = None
            self._start_time = datetime.utcnow()
            
            # Initialize Redis connection
            try:
                self._redis_client = redis.from_url(settings.REDIS_URL)
                self._redis_client.ping()
                logger.info("Connected to Redis for monitoring")
            except Exception as e:
                logger.error(f"Failed to connect to Redis for monitoring: {e}")
    
    @staticmethod
    def record_task_start(task_name: str, queue: str = "default"):
        """Record task start"""
        monitor = TaskMonitor()
        timestamp = datetime.utcnow()
        
        # Update internal stats
        monitor._task_stats[task_name]['last_start'] = timestamp
        
        # Update Prometheus metrics
        TASK_COUNTER.labels(task_name=task_name, status='started', queue=queue).inc()
        
        logger.debug(f"Task started: {task_name}")
    
    @staticmethod
    def record_task_complete(task_name: str, duration: float, queue: str = "default"):
        """Record successful task completion"""
        monitor = TaskMonitor()
        timestamp = datetime.utcnow()
        
        # Update internal stats
        stats = monitor._task_stats[task_name]
        stats['count'] += 1
        stats['success_count'] += 1
        stats['total_duration'] += duration
        stats['last_execution'] = timestamp
        
        # Add to recent tasks
        monitor._recent_tasks.append({
            'task_name': task_name,
            'status': 'completed',
            'duration': duration,
            'timestamp': timestamp,
            'queue': queue
        })
        
        # Update Prometheus metrics
        TASK_COUNTER.labels(task_name=task_name, status='completed', queue=queue).inc()
        TASK_DURATION.labels(task_name=task_name, queue=queue).observe(duration)
        
        logger.debug(f"Task completed: {task_name} in {duration:.2f}s")
    
    @staticmethod
    def record_task_fail(task_name: str, error: str, queue: str = "default"):
        """Record task failure"""
        monitor = TaskMonitor()
        timestamp = datetime.utcnow()
        
        # Update internal stats
        stats = monitor._task_stats[task_name]
        stats['count'] += 1
        stats['error_count'] += 1
        stats['last_execution'] = timestamp
        stats['last_error'] = error
        
        # Add to recent tasks
        monitor._recent_tasks.append({
            'task_name': task_name,
            'status': 'failed',
            'error': error,
            'timestamp': timestamp,
            'queue': queue
        })
        
        # Update Prometheus metrics
        TASK_COUNTER.labels(task_name=task_name, status='failed', queue=queue).inc()
        
        logger.debug(f"Task failed: {task_name} - {error}")
    
    @staticmethod
    def record_task_retry(task_name: str, attempt: int, queue: str = "default"):
        """Record task retry"""
        monitor = TaskMonitor()
        
        # Update Prometheus metrics
        TASK_COUNTER.labels(task_name=task_name, status='retried', queue=queue).inc()
        
        logger.debug(f"Task retried: {task_name} (attempt {attempt})")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics from Redis"""
        if not self._redis_client:
            return {}
        
        try:
            # Get queue lengths
            queues = ['discovery', 'transcripts', 'batch', 'metadata', 'celery']
            queue_stats = {}
            
            for queue in queues:
                length = self._redis_client.llen(queue)
                queue_stats[queue] = {
                    'length': length,
                    'name': queue
                }
                
                # Update Prometheus gauge
                QUEUE_SIZE.labels(queue_name=queue).set(length)
            
            return queue_stats
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get detailed task statistics"""
        stats = {}
        
        for task_name, task_stats in self._task_stats.items():
            avg_duration = (
                task_stats['total_duration'] / task_stats['count']
                if task_stats['count'] > 0 else 0
            )
            
            success_rate = (
                task_stats['success_count'] / task_stats['count'] * 100
                if task_stats['count'] > 0 else 0
            )
            
            stats[task_name] = {
                'total_executions': task_stats['count'],
                'successful_executions': task_stats['success_count'],
                'failed_executions': task_stats['error_count'],
                'success_rate_percent': round(success_rate, 2),
                'average_duration_seconds': round(avg_duration, 2),
                'total_duration_seconds': round(task_stats['total_duration'], 2),
                'last_execution': task_stats['last_execution'].isoformat() if task_stats.get('last_execution') else None,
                'last_error': task_stats.get('last_error')
            }
        
        return stats
    
    def get_recent_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent task activity"""
        recent = list(self._recent_tasks)[-limit:]
        
        # Convert datetime objects to ISO strings
        for task in recent:
            if 'timestamp' in task:
                task['timestamp'] = task['timestamp'].isoformat()
        
        return recent
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        uptime = datetime.utcnow() - self._start_time
        
        # Calculate rates over the last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_tasks = [
            task for task in self._recent_tasks
            if task['timestamp'] > one_hour_ago
        ]
        
        successful_recent = len([t for t in recent_tasks if t['status'] == 'completed'])
        failed_recent = len([t for t in recent_tasks if t['status'] == 'failed'])
        
        success_rate = (
            successful_recent / len(recent_tasks) * 100
            if recent_tasks else 0
        )
        
        # Update success rate metric
        TRANSCRIPT_SUCCESS_RATE.labels(time_window='1h').set(success_rate)
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_tasks_processed': sum(stats['count'] for stats in self._task_stats.values()),
            'tasks_last_hour': len(recent_tasks),
            'success_rate_last_hour': round(success_rate, 2),
            'average_tasks_per_minute': len(recent_tasks) / 60 if recent_tasks else 0,
            'queue_stats': self.get_queue_stats(),
            'task_stats': self.get_task_stats()
        }
    
    def update_worker_metrics(self, worker_info: Dict[str, Any]):
        """Update worker-specific metrics"""
        worker_id = worker_info.get('worker_id', 'unknown')
        queue = worker_info.get('queue', 'default')
        memory_usage = worker_info.get('memory_bytes', 0)
        
        # Update Prometheus metrics
        WORKER_MEMORY.labels(worker_id=worker_id, queue=queue).set(memory_usage)
        
        # Store worker info in Redis
        if self._redis_client:
            try:
                key = f"worker:{worker_id}"
                self._redis_client.hset(key, mapping={
                    'queue': queue,
                    'memory_bytes': memory_usage,
                    'last_seen': datetime.utcnow().isoformat()
                })
                self._redis_client.expire(key, 300)  # Expire after 5 minutes
            except Exception as e:
                logger.error(f"Error updating worker metrics: {e}")
    
    def update_api_quota_metrics(self, quota_info: Dict[str, Any]):
        """Update API quota usage metrics"""
        for api_type, usage_percent in quota_info.items():
            API_QUOTA_USAGE.labels(api_type=api_type).set(usage_percent)
    
    def update_job_queue_metrics(self, queue_depths: Dict[str, Dict[str, int]]):
        """Update job queue depth metrics"""
        for priority, job_types in queue_depths.items():
            for job_type, depth in job_types.items():
                JOB_QUEUE_DEPTH.labels(priority=priority, job_type=job_type).set(depth)
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        # Update queue stats before export
        self.get_queue_stats()
        
        # Update system stats
        system_stats = self.get_system_stats()
        
        # Export metrics
        return generate_latest()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        status = "healthy"
        issues = []
        
        # Check Redis connection
        redis_status = "ok"
        try:
            if self._redis_client:
                self._redis_client.ping()
            else:
                redis_status = "disconnected"
                issues.append("Redis not connected")
        except Exception as e:
            redis_status = "error"
            issues.append(f"Redis error: {str(e)}")
        
        # Check recent task failures
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_tasks = [
            task for task in self._recent_tasks
            if task['timestamp'] > one_hour_ago
        ]
        
        if recent_tasks:
            failed_tasks = [t for t in recent_tasks if t['status'] == 'failed']
            failure_rate = len(failed_tasks) / len(recent_tasks) * 100
            
            if failure_rate > 50:  # More than 50% failure rate
                status = "degraded"
                issues.append(f"High failure rate: {failure_rate:.1f}%")
            elif failure_rate > 80:  # More than 80% failure rate
                status = "unhealthy"
                issues.append(f"Critical failure rate: {failure_rate:.1f}%")
        
        # Check queue backlogs
        queue_stats = self.get_queue_stats()
        for queue_name, stats in queue_stats.items():
            if stats['length'] > 1000:  # More than 1000 tasks queued
                issues.append(f"Queue backlog: {queue_name} has {stats['length']} tasks")
                if status == "healthy":
                    status = "degraded"
        
        if issues and status == "healthy":
            status = "degraded"
        
        return {
            "status": status,
            "redis_connection": redis_status,
            "issues": issues,
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "last_check": datetime.utcnow().isoformat()
        }


# Global monitor instance
monitor = TaskMonitor()


def get_metrics_endpoint():
    """Get Prometheus metrics endpoint handler"""
    def metrics_handler():
        return monitor.export_prometheus_metrics()
    
    return metrics_handler


def get_health_endpoint():
    """Get health check endpoint handler"""
    def health_handler():
        return monitor.health_check()
    
    return health_handler


def get_stats_endpoint():
    """Get statistics endpoint handler"""
    def stats_handler():
        return monitor.get_system_stats()
    
    return stats_handler