"""
Utility functions for Pansinayan server.
"""

import time
import psutil
import torch
import numpy as np
from typing import List
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def log_execution_time(func):
    """Decorator to log function execution time"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = (time.time() - start) * 1000
        logger.info(f"{func.__name__} took {duration:.2f}ms")
        return result
    return wrapper


def get_system_stats():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    stats = {
        "cpu_percent": cpu_percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "memory_percent": memory.percent
    }
    
    if torch.cuda.is_available():
        stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
    
    return stats


def validate_keypoints(keypoints: List[List[float]], max_length: int = 300) -> bool:
    """Validate keypoints array structure"""
    if not keypoints:
        return False, "Empty keypoints array"
    
    if len(keypoints) > max_length:
        return False, f"Sequence too long: {len(keypoints)} > {max_length}"
    
    if len(keypoints[0]) != 178:
        return False, f"Invalid feature dimension: {len(keypoints[0])} != 178"
    
    return True, None


def preprocess_keypoints(keypoints: List[List[float]]) -> np.ndarray:
    """Convert and preprocess keypoints"""
    # Convert to numpy array
    arr = np.array(keypoints, dtype=np.float32)
    
    # Clamp to [0, 1] range
    arr = np.clip(arr, 0.0, 1.0)
    
    return arr


class InferenceTimer:
    """Context manager for timing inference"""
    def __init__(self, name: str = "inference"):
        self.name = name
        self.start_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.debug(f"{self.name}: {self.elapsed_ms:.2f}ms")