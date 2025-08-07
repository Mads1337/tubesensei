"""
Worker Manager for TubeSensei Celery workers
Manages worker lifecycle, scaling, and monitoring
"""
import os
import signal
import subprocess
import psutil
import logging
from typing import Dict, Any, List, Optional, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

from app.config import settings
from app.workers.monitoring import TaskMonitor

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkerInfo:
    """Worker process information"""
    worker_id: str
    pid: Optional[int]
    status: WorkerStatus
    queue: str
    concurrency: int
    started_at: Optional[datetime]
    memory_usage_mb: float
    cpu_percent: float
    processed_tasks: int
    failed_tasks: int
    last_heartbeat: Optional[datetime]
    command: str
    log_file: Optional[str]


@dataclass
class QueueConfig:
    """Queue configuration"""
    name: str
    concurrency: int
    max_tasks_per_child: int
    prefetch_multiplier: int
    rate_limit: Optional[str] = None
    priority: int = 5


class WorkerManager:
    """
    Manages Celery worker processes, scaling, and monitoring.
    Provides programmatic control over worker lifecycle and performance.
    """
    
    def __init__(self):
        self.workers: Dict[str, WorkerInfo] = {}
        self.monitor = TaskMonitor()
        self.log_dir = "/tmp/tubesensei/logs"  # TODO: Make configurable
        self._ensure_log_directory()
        
        # Default queue configurations
        self.queue_configs = {
            "discovery": QueueConfig(
                name="discovery",
                concurrency=2,
                max_tasks_per_child=100,
                prefetch_multiplier=1,
                rate_limit="10/m"
            ),
            "transcripts": QueueConfig(
                name="transcripts",
                concurrency=4,
                max_tasks_per_child=50,
                prefetch_multiplier=1,
                rate_limit="30/m"
            ),
            "batch": QueueConfig(
                name="batch",
                concurrency=2,
                max_tasks_per_child=10,
                prefetch_multiplier=1
            ),
            "metadata": QueueConfig(
                name="metadata",
                concurrency=1,
                max_tasks_per_child=200,
                prefetch_multiplier=2,
                rate_limit="20/m"
            ),
            "celery": QueueConfig(  # Default queue
                name="celery",
                concurrency=2,
                max_tasks_per_child=100,
                prefetch_multiplier=1
            )
        }
    
    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        os.makedirs(self.log_dir, exist_ok=True)
    
    async def start_worker(
        self,
        queue: str,
        worker_id: Optional[str] = None,
        concurrency: Optional[int] = None,
        detached: bool = True
    ) -> str:
        """
        Start a Celery worker for a specific queue.
        
        Args:
            queue: Queue name to process
            worker_id: Optional custom worker ID
            concurrency: Override default concurrency
            detached: Run in background
            
        Returns:
            Worker ID
        """
        if queue not in self.queue_configs:
            raise ValueError(f"Unknown queue: {queue}")
        
        config = self.queue_configs[queue]
        worker_id = worker_id or f"{queue}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Check if worker already exists
        if worker_id in self.workers:
            existing = self.workers[worker_id]
            if existing.status in [WorkerStatus.RUNNING, WorkerStatus.STARTING]:
                raise ValueError(f"Worker {worker_id} already running")
        
        # Build command
        concurrency = concurrency or config.concurrency
        log_file = os.path.join(self.log_dir, f"{worker_id}.log")
        
        cmd = [
            "celery",
            "-A", "app.celery_app",
            "worker",
            "-n", f"{worker_id}@%h",
            "-Q", queue,
            "--concurrency", str(concurrency),
            "--max-tasks-per-child", str(config.max_tasks_per_child),
            "--prefetch-multiplier", str(config.prefetch_multiplier),
            "--loglevel", "info",
            "--logfile", log_file
        ]
        
        if config.rate_limit:
            # Note: Rate limits are set in celery_app.py, not here
            pass
        
        # Create worker info
        worker_info = WorkerInfo(
            worker_id=worker_id,
            pid=None,
            status=WorkerStatus.STARTING,
            queue=queue,
            concurrency=concurrency,
            started_at=datetime.utcnow(),
            memory_usage_mb=0.0,
            cpu_percent=0.0,
            processed_tasks=0,
            failed_tasks=0,
            last_heartbeat=None,
            command=" ".join(cmd),
            log_file=log_file
        )
        
        try:
            # Start process
            if detached:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid  # Create new process group
                )
            else:
                process = subprocess.Popen(cmd)
            
            worker_info.pid = process.pid
            worker_info.status = WorkerStatus.RUNNING
            
            # Store worker info
            self.workers[worker_id] = worker_info
            
            logger.info(f"Started worker {worker_id} for queue {queue} (PID: {process.pid})")
            return worker_id
            
        except Exception as e:
            worker_info.status = WorkerStatus.ERROR
            self.workers[worker_id] = worker_info
            logger.error(f"Failed to start worker {worker_id}: {e}")
            raise
    
    async def stop_worker(
        self,
        worker_id: str,
        graceful: bool = True,
        timeout: int = 30
    ) -> bool:
        """
        Stop a running worker.
        
        Args:
            worker_id: Worker ID to stop
            graceful: Use graceful shutdown (SIGTERM vs SIGKILL)
            timeout: Timeout for graceful shutdown
            
        Returns:
            True if successfully stopped
        """
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found")
            return False
        
        worker = self.workers[worker_id]
        if worker.status != WorkerStatus.RUNNING:
            logger.warning(f"Worker {worker_id} not running (status: {worker.status.value})")
            return False
        
        if not worker.pid:
            logger.error(f"Worker {worker_id} has no PID")
            return False
        
        try:
            # Update status
            worker.status = WorkerStatus.STOPPING
            
            # Get process
            process = psutil.Process(worker.pid)
            
            if graceful:
                # Send SIGTERM for graceful shutdown
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    logger.warning(f"Worker {worker_id} did not stop gracefully, forcing...")
                    process.kill()
            else:
                # Force kill
                process.kill()
            
            # Update status
            worker.status = WorkerStatus.STOPPED
            worker.pid = None
            
            logger.info(f"Stopped worker {worker_id}")
            return True
            
        except psutil.NoSuchProcess:
            # Process already dead
            worker.status = WorkerStatus.STOPPED
            worker.pid = None
            logger.info(f"Worker {worker_id} process already terminated")
            return True
            
        except Exception as e:
            worker.status = WorkerStatus.ERROR
            logger.error(f"Error stopping worker {worker_id}: {e}")
            return False
    
    async def restart_worker(
        self,
        worker_id: str,
        graceful: bool = True
    ) -> bool:
        """
        Restart a worker.
        
        Args:
            worker_id: Worker ID to restart
            graceful: Use graceful restart
            
        Returns:
            True if successfully restarted
        """
        if worker_id not in self.workers:
            logger.error(f"Worker {worker_id} not found")
            return False
        
        worker = self.workers[worker_id]
        queue = worker.queue
        concurrency = worker.concurrency
        
        # Stop worker
        stopped = await self.stop_worker(worker_id, graceful)
        if not stopped:
            return False
        
        # Start new worker with same configuration
        try:
            new_worker_id = await self.start_worker(
                queue=queue,
                concurrency=concurrency
            )
            
            # Remove old worker info
            del self.workers[worker_id]
            
            logger.info(f"Restarted worker {worker_id} as {new_worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart worker {worker_id}: {e}")
            return False
    
    async def scale_queue(
        self,
        queue: str,
        target_workers: int,
        target_concurrency: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Scale workers for a specific queue.
        
        Args:
            queue: Queue to scale
            target_workers: Target number of workers
            target_concurrency: Target concurrency per worker
            
        Returns:
            Scaling operation results
        """
        if queue not in self.queue_configs:
            raise ValueError(f"Unknown queue: {queue}")
        
        # Get current workers for this queue
        current_workers = [w for w in self.workers.values() if w.queue == queue and w.status == WorkerStatus.RUNNING]
        current_count = len(current_workers)
        
        results = {
            "queue": queue,
            "current_workers": current_count,
            "target_workers": target_workers,
            "started": [],
            "stopped": [],
            "errors": []
        }
        
        if target_workers > current_count:
            # Scale up - start new workers
            workers_to_add = target_workers - current_count
            
            for i in range(workers_to_add):
                try:
                    worker_id = await self.start_worker(
                        queue=queue,
                        concurrency=target_concurrency
                    )
                    results["started"].append(worker_id)
                except Exception as e:
                    results["errors"].append(f"Failed to start worker: {e}")
                    
        elif target_workers < current_count:
            # Scale down - stop excess workers
            workers_to_remove = current_count - target_workers
            workers_to_stop = current_workers[:workers_to_remove]
            
            for worker in workers_to_stop:
                try:
                    success = await self.stop_worker(worker.worker_id)
                    if success:
                        results["stopped"].append(worker.worker_id)
                    else:
                        results["errors"].append(f"Failed to stop worker: {worker.worker_id}")
                except Exception as e:
                    results["errors"].append(f"Error stopping worker {worker.worker_id}: {e}")
        
        # Update concurrency for existing workers if specified
        if target_concurrency:
            config = self.queue_configs[queue]
            config.concurrency = target_concurrency
            # Note: Changing concurrency requires restart in Celery
            logger.info(f"Updated default concurrency for queue {queue} to {target_concurrency}")
        
        logger.info(f"Scaled queue {queue} to {target_workers} workers")
        return results
    
    async def update_worker_stats(self):
        """Update worker statistics from system info"""
        for worker_id, worker in self.workers.items():
            if worker.status == WorkerStatus.RUNNING and worker.pid:
                try:
                    process = psutil.Process(worker.pid)
                    
                    # Update system metrics
                    worker.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                    worker.cpu_percent = process.cpu_percent()
                    worker.last_heartbeat = datetime.utcnow()
                    
                    # Update monitoring
                    self.monitor.update_worker_metrics({
                        "worker_id": worker_id,
                        "queue": worker.queue,
                        "memory_bytes": worker.memory_usage_mb * 1024 * 1024,
                        "cpu_percent": worker.cpu_percent
                    })
                    
                except psutil.NoSuchProcess:
                    # Process died
                    worker.status = WorkerStatus.STOPPED
                    worker.pid = None
                    logger.warning(f"Worker {worker_id} process not found, marking as stopped")
                    
                except Exception as e:
                    logger.error(f"Error updating stats for worker {worker_id}: {e}")
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive worker statistics.
        
        Returns:
            Dictionary with worker statistics
        """
        stats = {
            "total_workers": len(self.workers),
            "workers_by_status": {},
            "workers_by_queue": {},
            "total_concurrency": 0,
            "total_memory_mb": 0.0,
            "workers": []
        }
        
        # Count by status
        for status in WorkerStatus:
            count = len([w for w in self.workers.values() if w.status == status])
            stats["workers_by_status"][status.value] = count
        
        # Count by queue
        for queue in self.queue_configs.keys():
            count = len([w for w in self.workers.values() if w.queue == queue and w.status == WorkerStatus.RUNNING])
            stats["workers_by_queue"][queue] = count
        
        # Calculate totals and detailed stats
        for worker in self.workers.values():
            if worker.status == WorkerStatus.RUNNING:
                stats["total_concurrency"] += worker.concurrency
                stats["total_memory_mb"] += worker.memory_usage_mb
            
            stats["workers"].append(asdict(worker))
        
        # Add queue configurations
        stats["queue_configs"] = {name: asdict(config) for name, config in self.queue_configs.items()}
        
        return stats
    
    def get_queue_health(self) -> Dict[str, Any]:
        """
        Assess health of each queue.
        
        Returns:
            Queue health assessment
        """
        health = {}
        
        for queue_name, config in self.queue_configs.items():
            queue_workers = [w for w in self.workers.values() 
                           if w.queue == queue_name and w.status == WorkerStatus.RUNNING]
            
            total_concurrency = sum(w.concurrency for w in queue_workers)
            avg_memory = sum(w.memory_usage_mb for w in queue_workers) / len(queue_workers) if queue_workers else 0
            avg_cpu = sum(w.cpu_percent for w in queue_workers) / len(queue_workers) if queue_workers else 0
            
            # Assess health
            status = "healthy"
            issues = []
            
            if len(queue_workers) == 0:
                status = "critical"
                issues.append("No running workers")
            elif avg_memory > 1000:  # >1GB per worker
                status = "warning" if status == "healthy" else status
                issues.append(f"High memory usage: {avg_memory:.1f}MB")
            elif avg_cpu > 80:  # >80% CPU
                status = "warning" if status == "healthy" else status
                issues.append(f"High CPU usage: {avg_cpu:.1f}%")
            
            health[queue_name] = {
                "status": status,
                "worker_count": len(queue_workers),
                "total_concurrency": total_concurrency,
                "average_memory_mb": avg_memory,
                "average_cpu_percent": avg_cpu,
                "issues": issues
            }
        
        return health
    
    async def auto_scale(self) -> Dict[str, Any]:
        """
        Perform automatic scaling based on queue metrics.
        
        Returns:
            Auto-scaling results
        """
        # Get queue statistics from monitoring
        queue_stats = self.monitor.get_queue_stats()
        scaling_actions = []
        
        for queue_name, stats in queue_stats.items():
            if queue_name not in self.queue_configs:
                continue
            
            queue_length = stats.get('length', 0)
            current_workers = len([w for w in self.workers.values() 
                                 if w.queue == queue_name and w.status == WorkerStatus.RUNNING])
            
            # Simple scaling logic
            if queue_length > 100 and current_workers < 5:  # Scale up
                target_workers = min(current_workers + 1, 5)
                try:
                    result = await self.scale_queue(queue_name, target_workers)
                    scaling_actions.append({
                        "queue": queue_name,
                        "action": "scale_up",
                        "from": current_workers,
                        "to": target_workers,
                        "reason": f"Queue length: {queue_length}",
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Auto-scale up failed for {queue_name}: {e}")
                    
            elif queue_length < 10 and current_workers > 1:  # Scale down
                target_workers = max(current_workers - 1, 1)
                try:
                    result = await self.scale_queue(queue_name, target_workers)
                    scaling_actions.append({
                        "queue": queue_name,
                        "action": "scale_down",
                        "from": current_workers,
                        "to": target_workers,
                        "reason": f"Queue length: {queue_length}",
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Auto-scale down failed for {queue_name}: {e}")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "scaling_actions": scaling_actions,
            "queue_stats": queue_stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health check results
        """
        await self.update_worker_stats()
        
        worker_stats = self.get_worker_stats()
        queue_health = self.get_queue_health()
        
        # Overall health assessment
        overall_status = "healthy"
        issues = []
        
        # Check if any queues are critical
        critical_queues = [q for q, h in queue_health.items() if h["status"] == "critical"]
        if critical_queues:
            overall_status = "critical"
            issues.extend([f"Critical queue: {q}" for q in critical_queues])
        
        # Check for warning queues
        warning_queues = [q for q, h in queue_health.items() if h["status"] == "warning"]
        if warning_queues and overall_status == "healthy":
            overall_status = "warning"
            issues.extend([f"Warning queue: {q}" for q in warning_queues])
        
        # Check worker count
        if worker_stats["workers_by_status"].get("running", 0) == 0:
            overall_status = "critical"
            issues.append("No running workers")
        
        return {
            "status": overall_status,
            "issues": issues,
            "worker_stats": worker_stats,
            "queue_health": queue_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def shutdown_all_workers(self, graceful: bool = True, timeout: int = 30):
        """
        Shutdown all workers.
        
        Args:
            graceful: Use graceful shutdown
            timeout: Timeout for graceful shutdown
        """
        logger.info("Shutting down all workers...")
        
        running_workers = [w for w in self.workers.values() if w.status == WorkerStatus.RUNNING]
        
        for worker in running_workers:
            try:
                await self.stop_worker(worker.worker_id, graceful, timeout)
            except Exception as e:
                logger.error(f"Error stopping worker {worker.worker_id}: {e}")
        
        logger.info("All workers shutdown complete")


# Global worker manager instance
worker_manager = WorkerManager()


def get_worker_manager() -> WorkerManager:
    """Get the global worker manager instance"""
    return worker_manager