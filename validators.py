"""
Input validation schemas for the TINE backend API.
Provides validation for emails, passwords, camera URLs, and date ranges.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
from flask import request, jsonify


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, field: str = None, code: str = "VALIDATION_ERROR"):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(self.message)


# ============== Email Validation ==============

def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"
    
    # RFC 5322 compliant email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    
    if len(email) > 254:
        return False, "Email is too long"
    
    return True, ""


def validate_email_required(email: str) -> None:
    """Validate email and raise ValidationError if invalid"""
    is_valid, error = validate_email(email)
    if not is_valid:
        raise ValidationError(error, "email")


# ============== Password Validation ==============

def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validate password strength requirements.
    
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password is required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if len(password) > 128:
        return False, "Password is too long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, ""


def validate_password_required(password: str) -> None:
    """Validate password and raise ValidationError if invalid"""
    is_valid, error = validate_password_strength(password)
    if not is_valid:
        raise ValidationError(error, "password")


# ============== Camera URL Validation ==============

def validate_camera_url(url: str) -> Tuple[bool, str]:
    """
    Validate camera RTSP/RTSPS URL format.
    
    Args:
        url: Camera URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "Camera URL is required"
    
    # Check for valid RTSP URL format
    rtsp_pattern = r'^rtsp[s]?://[^\s/$.?#].[^\s]*$'
    
    if not re.match(rtsp_pattern, url, re.IGNORECASE):
        return False, "Invalid RTSP URL format. Must start with rtsp:// or rtsps://"
    
    # Check for valid hostname
    url_without_scheme = url.split('://', 1)[1] if '://' in url else url
    if not url_without_scheme or '/' in url_without_scheme and url_without_scheme.split('/')[0] == '':
        return False, "Invalid URL: missing hostname"
    
    return True, ""


def validate_camera_url_required(url: str) -> None:
    """Validate camera URL and raise ValidationError if invalid"""
    is_valid, error = validate_camera_url(url)
    if not is_valid:
        raise ValidationError(error, "rtsp_url")


# ============== Date Range Validation ==============

def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> Tuple[bool, str]:
    """
    Validate date range parameters.
    
    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    from datetime import datetime
    
    if start_date:
        try:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            return False, "Invalid start date format. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    
    if end_date:
        try:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            return False, "Invalid end date format. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    
    if start_date and end_date:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start > end:
            return False, "Start date must be before end date"
    
    return True, ""


def validate_date_range_required(start_date: Optional[str], end_date: Optional[str]) -> None:
    """Validate date range and raise ValidationError if invalid"""
    is_valid, error = validate_date_range(start_date, end_date)
    if not is_valid:
        raise ValidationError(error, "date_range")


# ============== Name Validation ==============

def validate_name(name: str, field_name: str = "name") -> Tuple[bool, str]:
    """
    Validate name field.
    
    Args:
        name: Name to validate
        field_name: Field name for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, f"{field_name} is required"
    
    if len(name) < 2:
        return False, f"{field_name} must be at least 2 characters"
    
    if len(name) > 100:
        return False, f"{field_name} must be less than 100 characters"
    
    # Check for valid characters (letters, spaces, hyphens, apostrophes)
    if not re.match(r"^[a-zA-Z\s\-']+$", name):
        return False, f"{field_name} contains invalid characters"
    
    return True, ""


# ============== Organization Validation ==============

def validate_organization_name(name: str) -> Tuple[bool, str]:
    """
    Validate organization name.
    
    Args:
        name: Organization name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Organization name is required"
    
    if len(name) < 2:
        return False, "Organization name must be at least 2 characters"
    
    if len(name) > 200:
        return False, "Organization name must be less than 200 characters"
    
    return True, ""


# ============== Camera Settings Validation ==============

def validate_camera_settings(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate camera settings data.
    
    Args:
        data: Camera settings data dictionary
        
    Returns:
        Dictionary of field errors
    """
    errors = {}
    
    # Validate name
    if 'name' in data:
        is_valid, error = validate_name(data['name'], "camera name")
        if not is_valid:
            errors['name'] = [error]
    
    # Validate RTSP URL
    if 'rtsp_url' in data or 'rtspUrl' in data:
        url = data.get('rtsp_url') or data.get('rtspUrl')
        is_valid, error = validate_camera_url(url)
        if not is_valid:
            errors['rtsp_url'] = [error]
    
    # Validate location
    if 'location' in data:
        location = data['location']
        if not location or len(location) < 2:
            errors['location'] = ["Location is required and must be at least 2 characters"]
        elif len(location) > 200:
            errors['location'] = ["Location must be less than 200 characters"]
    
    return errors


# ============== Alert Filters Validation ==============

def validate_alert_filters(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate alert filter parameters.
    
    Args:
        data: Filter parameters dictionary
        
    Returns:
        Dictionary of field errors
    """
    errors = {}
    
    # Validate date range
    start_date = data.get('startDate')
    end_date = data.get('endDate')
    
    if start_date or end_date:
        is_valid, error = validate_date_range(start_date, end_date)
        if not is_valid:
            errors['date_range'] = [error]
    
    # Validate type
    valid_types = ['all', 'theft', 'fall', 'violence', 'trespassing', 'other']
    if 'type' in data and data['type'] not in valid_types:
        errors['type'] = [f"Invalid alert type. Must be one of: {', '.join(valid_types)}"]

    
    # Validate status
    valid_statuses = ['all', 'new', 'acknowledged', 'resolved', 'dismissed']
    if 'status' in data and data['status'] not in valid_statuses:
        errors['status'] = [f"Invalid status. Must be one of: {', '.join(valid_statuses)}"]
    
    # Validate limit
    if 'limit' in data:
        try:
            limit = int(data['limit'])
            if limit < 1:
                errors['limit'] = ["Limit must be at least 1"]
            elif limit > 100:
                errors['limit'] = ["Limit must be at most 100"]
        except (ValueError, TypeError):
            errors['limit'] = ["Limit must be a number"]
    
    # Validate offset
    if 'offset' in data:
        try:
            offset = int(data['offset'])
            if offset < 0:
                errors['offset'] = ["Offset must be at least 0"]
        except (ValueError, TypeError):
            errors['offset'] = ["Offset must be a number"]
    
    return errors


# ============== Validation Decorator ==============

def validate_request(schema_validator):
    """
    Decorator to validate request data against a schema validator.
    
    Args:
        schema_validator: Function that takes data and returns error dict
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json() or {}
            errors = schema_validator(data)
            
            if errors:
                return jsonify({
                    "success": False,
                    "error": "Validation failed",
                    "code": "VALIDATION_ERROR",
                    "details": errors
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============== Export all validators ==============

__all__ = [
    'ValidationError',
    'validate_email',
    'validate_email_required',
    'validate_password_strength',
    'validate_password_required',
    'validate_camera_url',
    'validate_camera_url_required',
    'validate_date_range',
    'validate_date_range_required',
    'validate_name',
    'validate_organization_name',
    'validate_camera_settings',
    'validate_alert_filters',
    'validate_request'
]
