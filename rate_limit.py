"""
Rate limiting configuration for the TINE backend API.

Provides rate limiting for:
- Auth endpoints: 5 requests/minute
- API endpoints: 60 requests/minute
- Detection endpoints: 30 requests/minute
"""

from flask import Flask, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps


# Rate limiter instance (to be initialized with app)
limiter = None


def init_rate_limiter(app: Flask) -> Limiter:
    """
    Initialize the rate limiter with Flask app.
    
    Args:
        app: Flask application instance
        
    Returns:
        Configured Limiter instance
    """
    global limiter
    
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["10000 per day", "2500 per hour"],
        storage_uri="memory://",
        strategy="fixed-window"
    )
    
    return limiter


def get_limiter() -> Limiter:
    """
    Get the rate limiter instance.
    
    Returns:
        Limiter instance
    """
    return limiter


# ============== Rate Limit Decorators ==============

def auth_rate_limit(f):
    """
    Decorator for authentication endpoints.
    Rate limit: 5 requests per minute
    """
    if limiter:
        return limiter.limit("5 per minute")(f)
    return f


def api_rate_limit(f):
    """
    Decorator for general API endpoints.
    Rate limit: 60 requests per minute
    """
    if limiter:
        return limiter.limit("60 per minute")(f)
    return f


def detection_rate_limit(f):
    """
    Decorator for detection/anomaly endpoints.
    Rate limit: 30 requests per minute
    """
    if limiter:
        return limiter.limit("30 per minute")(f)
    return f


def camera_stream_rate_limit(f):
    """
    Decorator for camera streaming endpoints.
    Rate limit: 10 requests per minute
    """
    if limiter:
        return limiter.limit("10 per minute")(f)
    return f


def sse_rate_limit(f):
    """
    Decorator for SSE (Server-Sent Events) endpoints.
    Rate limit: 5 connections per minute
    """
    if limiter:
        return limiter.limit("5 per minute")(f)
    return f


# ============== Custom Rate Limit Functions ==============

def apply_rate_limit(limit: str):
    """
    Apply a custom rate limit to an endpoint.
    
    Args:
        limit: Rate limit string (e.g., "10 per minute", "100 per hour")
        
    Returns:
        Decorator function
    """
    if limiter:
        return limiter.limit(limit)
    return lambda f: f


def check_rate_limit() -> bool:
    """
    Check if current request exceeds rate limit.
    
    Returns:
        True if within limit, False if exceeded
    """
    if limiter:
        try:
            # Get the current limit for this request
            limiter.check()
            return True
        except Exception:
            return False
    return True


# ============== Rate Limit Status ==============

def get_rate_limit_status() -> dict:
    """
    Get current rate limit status.
    
    Returns:
        Dictionary with rate limit information
    """
    if limiter:
        try:
            from flask import request
            # This will return limit info if available
            return {
                "enabled": True,
                "storage": "memory"
            }
        except Exception:
            pass
    
    return {
        "enabled": False,
        "storage": "none"
    }


# ============== Exempt from Rate Limiting ==============

def exempt_from_rate_limit(f):
    """
    Decorator to exempt an endpoint from rate limiting.
    """
    if limiter:
        return limiter.exempt(f)
    return f


__all__ = [
    'init_rate_limiter',
    'get_limiter',
    'auth_rate_limit',
    'api_rate_limit',
    'detection_rate_limit',
    'camera_stream_rate_limit',
    'sse_rate_limit',
    'apply_rate_limit',
    'check_rate_limit',
    'get_rate_limit_status',
    'exempt_from_rate_limit',
    'limiter'
]
