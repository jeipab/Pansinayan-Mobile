"""
Utility functions for Pansinayan server.
"""

import time
import psutil
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Constants
EXPECTED_FEATURE_DIM = 178
DEFAULT_MAX_SEQUENCE_LENGTH = 300


def log_execution_time(func: Any) -> Any:
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = await func(*args, **kwargs)
        duration = (time.time() - start) * 1000
        logger.info(f"{func.__name__} took {duration:.2f}ms")
        return result
    return wrapper


def get_system_stats() -> Dict[str, Any]:
    """
    Get current system resource usage.
    
    Returns:
        Dictionary containing CPU, memory, and GPU statistics
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    stats: Dict[str, Any] = {
        "cpu_percent": cpu_percent,
        "memory_used_gb": memory.used / (1024**3),
        "memory_total_gb": memory.total / (1024**3),
        "memory_percent": memory.percent
    }
    
    if torch.cuda.is_available():
        stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
    
    return stats


def validate_keypoints(
    keypoints: List[List[float]], 
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
) -> Tuple[bool, Optional[str]]:
    """
    Validate keypoints array structure.
    
    Validates:
    - Non-empty array
    - Sequence length within limits
    - All frames have correct feature dimension (178)
    - Consistent dimensions across all frames
    
    Args:
        keypoints: List of keypoint frames, each frame is a list of floats
        max_length: Maximum allowed sequence length
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes
        - error_message: None if valid, error description if invalid
    """
    if not keypoints:
        return False, "Empty keypoints array"
    
    sequence_length = len(keypoints)
    if sequence_length > max_length:
        return False, f"Sequence too long: {sequence_length} > {max_length}"
    
    if sequence_length == 0:
        return False, "Sequence cannot be empty"
    
    # Validate first frame dimension
    if len(keypoints[0]) != EXPECTED_FEATURE_DIM:
        return False, (
            f"Invalid feature dimension: {len(keypoints[0])} != {EXPECTED_FEATURE_DIM}. "
            f"Expected {EXPECTED_FEATURE_DIM} features per frame."
        )
    
    # Validate all frames have same dimension
    for i, frame in enumerate(keypoints):
        if len(frame) != EXPECTED_FEATURE_DIM:
            return False, (
                f"Frame {i} has {len(frame)} features, expected {EXPECTED_FEATURE_DIM}. "
                f"All frames must have consistent dimensions."
            )
    
    return True, None


def preprocess_keypoints(keypoints: List[List[float]]) -> np.ndarray:
    """
    Convert and preprocess keypoints to numpy array.
    
    Performs:
    - Conversion to numpy array (float32)
    - Clamping values to [0, 1] range
    
    Args:
        keypoints: List of keypoint frames
        
    Returns:
        Numpy array of shape [T, 178] where T is sequence length
    """
    # Convert to numpy array
    arr = np.array(keypoints, dtype=np.float32)
    
    # Clamp to [0, 1] range to ensure valid input
    arr = np.clip(arr, 0.0, 1.0)
    
    return arr


class InferenceTimer:
    """
    Context manager for timing inference operations.
    
    Usage:
        with InferenceTimer("inference") as timer:
            # ... inference code ...
        print(f"Took {timer.elapsed_ms}ms")
    """
    
    def __init__(self, name: str = "inference") -> None:
        """
        Initialize timer.
        
        Args:
            name: Name identifier for this timer (for logging)
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed_ms: float = 0.0
    
    def __enter__(self) -> "InferenceTimer":
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop timing and log duration."""
        if self.start_time is not None:
            self.elapsed_ms = (time.time() - self.start_time) * 1000
            logger.debug(f"{self.name}: {self.elapsed_ms:.2f}ms")