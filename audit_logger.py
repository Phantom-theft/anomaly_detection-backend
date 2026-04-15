"""
Audit logging system for the TINE backend API.

Tracks:
- User actions (login, logout, data access, modifications)
- API requests with user info
- Log storage with rotation

Log fields:
- timestamp, user_id, action, ip_address, endpoint, method
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
from functools import wraps
from flask import request, g, has_request_context


# Audit logger instance
audit_logger = None


# Audit action types
class AuditAction:
    # Authentication actions
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    LOGIN_FAILED = "LOGIN_FAILED"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    
    # User actions
    USER_CREATE = "USER_CREATE"
    USER_UPDATE = "USER_UPDATE"
    USER_DELETE = "USER_DELETE"
    USER_ACCESS = "USER_ACCESS"
    
    # Camera actions
    CAMERA_ADD = "CAMERA_ADD"
    CAMERA_REMOVE = "CAMERA_REMOVE"
    CAMERA_UPDATE = "CAMERA_UPDATE"
    CAMERA_ACCESS = "CAMERA_ACCESS"
    
    # Alert actions
    ALERT_CREATE = "ALERT_CREATE"
    ALERT_UPDATE = "ALERT_UPDATE"
    ALERT_ACCESS = "ALERT_ACCESS"
    ALERT_DELETE = "ALERT_DELETE"
    
    # Organization actions
    ORG_CREATE = "ORG_CREATE"
    ORG_UPDATE = "ORG_UPDATE"
    ORG_DELETE = "ORG_DELETE"
    
    # API actions
    API_REQUEST = "API_REQUEST"
    API_RESPONSE = "API_RESPONSE"
    API_ERROR = "API_ERROR"
    
    # Detection actions
    DETECTION_START = "DETECTION_START"
    DETECTION_STOP = "DETECTION_STOP"
    DETECTION_CONFIG_UPDATE = "DETECTION_CONFIG_UPDATE"
    
    # System actions
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SYSTEM_ACCESS = "SYSTEM_ACCESS"


def init_audit_logger(app, log_dir: str = "logs") -> logging.Logger:
    """
    Initialize the audit logger with Flask app.
    
    Args:
        app: Flask application instance
        log_dir: Directory to store audit logs
        
    Returns:
        Configured audit logger
    """
    global audit_logger
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger('audit')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create rotating file handler
    log_file = os.path.join(log_dir, 'audit.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding='utf-8'
    )
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    audit_logger = logger
    
    # Store logger in app config
    app.config['AUDIT_LOGGER'] = logger
    
    return logger


def get_audit_logger() -> Optional[logging.Logger]:
    """
    Get the audit logger instance.
    
    Returns:
        Audit logger instance
    """
    return audit_logger


# ============== Helper Functions ==============

def get_client_ip() -> str:
    """Get client IP address from request"""
    if has_request_context():
        # Check for forwarded header (if behind proxy)
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        return request.remote_addr or 'unknown'
    return 'no-request'


def get_current_user_id() -> Optional[str]:
    """Get current user ID from request context"""
    if has_request_context():
        if hasattr(g, 'current_user') and g.current_user:
            return g.current_user.get('uid')
        # Try to get from Firebase auth header
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            # This would typically decode the token
            return 'authenticated-user'
    return 'anonymous'


def get_request_info() -> Dict[str, Any]:
    """Get current request information"""
    if has_request_context():
        return {
            'endpoint': request.endpoint or 'unknown',
            'method': request.method,
            'path': request.path,
            'ip_address': get_client_ip(),
            'user_agent': request.headers.get('User-Agent', 'unknown')
        }
    return {
        'endpoint': 'no-request',
        'method': 'N/A',
        'path': 'N/A',
        'ip_address': 'no-request',
        'user_agent': 'N/A'
    }


# ============== Logging Functions ==============

def log_audit(
    action: str,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    level: str = 'INFO'
) -> None:
    """
    Log an audit event.
    
    Args:
        action: The action being logged
        user_id: User ID (optional, will use current user if not provided)
        details: Additional details to log
        level: Log level (INFO, WARNING, ERROR)
    """
    if not audit_logger:
        return
    
    # Get user ID if not provided
    if user_id is None:
        user_id = get_current_user_id()
    
    # Get request info
    request_info = get_request_info()
    
    # Build log entry
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'action': action,
        'ip_address': request_info['ip_address'],
        'endpoint': request_info['endpoint'],
        'method': request_info['method'],
        'details': details or {}
    }
    
    # Log the entry
    log_message = json.dumps(log_entry)
    
    if level == 'WARNING':
        audit_logger.warning(log_message)
    elif level == 'ERROR':
        audit_logger.error(log_message)
    else:
        audit_logger.info(log_message)


def log_api_request(
    endpoint: str,
    method: str,
    user_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an API request.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        user_id: User ID
        params: Request parameters
    """
    log_audit(
        action=AuditAction.API_REQUEST,
        user_id=user_id,
        details={
            'endpoint': endpoint,
            'method': method,
            'params': params
        }
    )


def log_user_action(
    action: str,
    user_id: str,
    target_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a user action.
    
    Args:
        action: Action performed
        user_id: User who performed the action
        target_id: ID of the target resource
        details: Additional details
    """
    log_details = details or {}
    if target_id:
        log_details['target_id'] = target_id
    
    log_audit(
        action=action,
        user_id=user_id,
        details=log_details
    )


def log_error(
    action: str,
    error: Exception,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an error event.
    
    Args:
        action: Action that caused the error
        error: Exception object
        user_id: User ID (if applicable)
        details: Additional details
    """
    log_details = details or {}
    log_details['error_type'] = type(error).__name__
    log_details['error_message'] = str(error)
    
    log_audit(
        action=action,
        user_id=user_id,
        details=log_details,
        level='ERROR'
    )


# ============== Decorator ==============

def audit_log(action: str):
    """
    Decorator to automatically log function calls.
    
    Args:
        action: The action to log
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = get_current_user_id()
            
            # Log the action
            log_audit(
                action=action,
                user_id=user_id,
                details={
                    'function': f.__name__,
                    'args': str(args)[:100],  # Limit args length
                    'kwargs': str(kwargs)[:100]
                }
            )
            
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                log_error(
                    action=f"{action}_FAILED",
                    error=e,
                    user_id=user_id
                )
                raise
        
        return decorated_function
    return decorator


# ============== Query Functions ==============

def get_audit_logs(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100
) -> list:
    """
    Query audit logs.
    
    Args:
        start_date: Start date for filtering
        end_date: End date for filtering
        user_id: Filter by user ID
        action: Filter by action type
        limit: Maximum number of logs to return
        
    Returns:
        List of audit log entries
    """
    if not audit_logger:
        return []
    
    # This is a simplified implementation
    # In production, you'd query from a database
    logs = []
    
    # For now, return empty list - would need to parse log file
    return logs


def get_user_activity(user_id: str, limit: int = 50) -> list:
    """
    Get activity logs for a specific user.
    
    Args:
        user_id: User ID
        limit: Maximum number of logs to return
        
    Returns:
        List of user activity logs
    """
    return get_audit_logs(user_id=user_id, limit=limit)


__all__ = [
    'init_audit_logger',
    'get_audit_logger',
    'AuditAction',
    'log_audit',
    'log_api_request',
    'log_user_action',
    'log_error',
    'audit_log',
    'get_audit_logs',
    'get_user_activity'
]
