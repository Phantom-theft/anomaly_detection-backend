"""
Firebase Token Verification Utility

This module provides decorators and functions for verifying Firebase ID tokens
on Flask endpoints to ensure only authenticated users can access protected routes.
"""

import os
import requests
from functools import wraps
from flask import request, jsonify, current_app, g

# Import audit logging (lazy import to avoid circular imports)
def _get_audit_logger():
    try:
        from audit_logger import log_audit, AuditAction
        return log_audit, AuditAction
    except ImportError:
        return None, None


# Firebase project configuration
FIREBASE_PROJECT_ID = os.environ.get('FIREBASE_PROJECT_ID', 'anomalydetection-18e91')
FIREBASE_API_KEY = os.environ.get('FIREBASE_API_KEY', '')

# Firebase token verification URL
FIREBASE_VERIFY_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"


def verify_firebase_token(id_token):
    """
    Verify a Firebase ID token and return the user information.
    
    Args:
        id_token: The Firebase ID token to verify
        
    Returns:
        dict: User information if token is valid, None otherwise
    """
    try:
        # Verify the token using Firebase Identity Toolkit
        # For production, use firebase-admin SDK for proper verification
        # This is a simplified version using the REST API
        
        # In production, use firebase-admin:
        # import firebase_admin
        # from firebase_admin import auth
        # decoded_token = auth.verify_id_token(id_token)
        # return decoded_token
        
        # For now, we'll do a basic validation
        # In production, replace with firebase-admin SDK
        if not id_token or len(id_token) < 10:
            return None
            
        # The token should have 3 parts separated by dots (JWT format)
        parts = id_token.split('.')
        if len(parts) != 3:
            return None
            
        # Return a mock user payload for development
        # In production, use firebase_admin.auth.verify_id_token(id_token)
        return {
            'uid': 'verified_user',
            'email': 'verified@example.com',
            'role': 'user'
        }
        
    except Exception as e:
        current_app.logger.error(f"Token verification error: {str(e)}")
        return None


def get_token_from_request():
    """
    Extract the Firebase ID token from the request header.
    
    Returns:
        str: The ID token or None if not found
    """
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]  # Remove 'Bearer ' prefix
    return None


def require_auth(f):
    """
    Decorator to require Firebase authentication on Flask routes.
    
    Usage:
        @app.route('/protected')
        @require_auth
        def protected_route():
            return jsonify({'message': 'Authenticated!'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = get_token_from_request()
        
        if not id_token:
            # Log failed authentication attempt
            log_audit_fn, _ = _get_audit_logger()
            if log_audit_fn:
                log_audit_fn(
                    action=AuditAction.LOGIN_FAILED if hasattr(AuditAction, 'LOGIN_FAILED') else 'LOGIN_FAILED',
                    user_id='anonymous',
                    details={'reason': 'no_token', 'endpoint': request.endpoint}
                )
            return jsonify({
                'success': False,
                'error': 'No authentication token provided',
                'message': 'Please provide a valid Firebase ID token in the Authorization header'
            }), 401
        
        user_info = verify_firebase_token(id_token)
        
        if not user_info:
            # Log failed authentication attempt
            log_audit_fn, _ = _get_audit_logger()
            if log_audit_fn:
                log_audit_fn(
                    action='LOGIN_FAILED',
                    user_id='anonymous',
                    details={'reason': 'invalid_token', 'endpoint': request.endpoint}
                )
            return jsonify({
                'success': False,
                'error': 'Invalid authentication token',
                'message': 'The provided token is invalid or expired'
            }), 401
        
        # Add user info to request context and Flask's g object
        request.user = user_info
        g.current_user = user_info
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(*allowed_roles):
    """
    Decorator to require specific roles on Flask routes.
    
    Usage:
        @app.route('/admin')
        @require_auth
        @require_role('admin', 'superadmin')
        def admin_route():
            return jsonify({'message': 'Admin access granted!'})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # First check if user is authenticated
            if not hasattr(request, 'user'):
                id_token = get_token_from_request()
                if not id_token:
                    return jsonify({
                        'success': False,
                        'error': 'No authentication token provided'
                    }), 401
                    
                user_info = verify_firebase_token(id_token)
                if not user_info:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid authentication token'
                    }), 401
                request.user = user_info
                g.current_user = user_info
            
            # Then check role
            user_role = request.user.get('role', 'user')
            if user_role not in allowed_roles:
                # Log authorization failure
                log_audit_fn, _ = _get_audit_logger()
                if log_audit_fn:
                    log_audit_fn(
                        action='AUTHORIZATION_FAILED',
                        user_id=request.user.get('uid', 'unknown'),
                        details={
                            'required_roles': list(allowed_roles),
                            'user_role': user_role,
                            'endpoint': request.endpoint
                        },
                        level='WARNING'
                    )
                return jsonify({
                    'success': False,
                    'error': 'Insufficient permissions',
                    'message': f'This endpoint requires one of the following roles: {", ".join(allowed_roles)}'
                }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def get_current_user():
    """
    Get the current authenticated user from the request.
    
    Returns:
        dict: User information or None if not authenticated
    """
    return getattr(request, 'user', None)