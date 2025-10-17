"""
Retry utilities for fault tolerance.
"""

import asyncio
import logging
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


def retry_async(
    max_attempts: int = 3,
    delay_s: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator for async functions.
    
    Args:
        max_attempts: Maximum retry attempts
        delay_s: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay_s
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_s: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.timeout_s = timeout_s
        
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
    
    def __call__(self, func: Callable):
        """Decorator for circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                # Check if timeout expired
                if (time.time() - self.last_failure_time) > self.timeout_s:
                    self.state = "half_open"
                    logger.info("Circuit breaker: half-open")
                else:
                    raise RuntimeError("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                
                if self.state == "half_open":
                    self.state = "closed"
                    self.failures = 0
                    logger.info("Circuit breaker: closed")
                
                return result
                
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                    logger.error(f"Circuit breaker opened after {self.failures} failures")
                
                raise
        
        return wrapper