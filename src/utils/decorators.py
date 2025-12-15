"""
Common decorators for Smart Recycling Detection System.

This module provides reusable decorators for error handling,
logging, and performance monitoring.
"""

import functools
import time
from typing import Callable, Any, Optional
from config.logging_config import get_logger

logger = get_logger("decorators")


def handle_errors(
    error_message: str = "An error occurred",
    log_error: bool = True,
    reraise: bool = False,
    default_return: Any = None
):
    """
    Decorator to handle exceptions in functions.

    Args:
        error_message: Message to log on error
        log_error: Whether to log the error
        reraise: Whether to re-raise the exception
        default_return: Value to return on error if not re-raising

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"{error_message}: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def log_execution_time(log_level: str = "debug"):
    """
    Decorator to log function execution time.

    Args:
        log_level: Logging level ('debug', 'info', 'warning', 'error')

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                log_func = getattr(logger, log_level, logger.debug)
                log_func(f"{func.__name__} executed in {execution_time:.4f}s")

                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        return wrapper
    return decorator


def validate_input(*validators):
    """
    Decorator to validate function inputs.

    Args:
        validators: Validation functions that take (arg_value, arg_name) and return bool

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate arguments
            for i, (arg_name, arg_value) in enumerate(bound_args.arguments.items()):
                if i < len(validators) and validators[i]:
                    validator_func = validators[i]
                    if not validator_func(arg_value, arg_name):
                        raise ValueError(f"Validation failed for argument '{arg_name}': {arg_value}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function execution on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            raise last_exception
        return wrapper
    return decorator