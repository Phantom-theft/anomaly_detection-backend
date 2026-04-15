"""
Error handling system for the TINE backend API.

Provides:
- Custom exception classes
- Global error handlers for Flask
- Structured error responses
- Input validation error handling
"""

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from functools import wraps
import traceback
import sys


# ============== Custom Exception Classes ==============

class TINEError(Exception):
    """Base exception for all TINE API errors"""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(TINEError):
    """Exception for authentication failures"""
    def __init__(self, message: str = "Authentication failed", code: str = "AUTH_ERROR"):
        super().__init__(message, code, 401)


class AuthorizationError(TINEError):
    """Exception for authorization failures"""
    def __init__(self, message: str = "Access denied", code: str = "FORBIDDEN"):
        super().__init__(message, code, 403)


class ValidationError(TINEError):
    """Exception for validation errors"""
    def __init__(self, message: str = "Validation failed", code: str = "VALIDATION_ERROR", errors: dict = None):
        super().__init__(message, code, 400)
        self.errors = errors or {}


class RateLimitError(TINEError):
    """Exception for rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded", code: str = "RATE_LIMIT_ERROR"):
        super().__init__(message, code, 429)


class ResourceNotFoundError(TINEError):
    """Exception for resource not found"""
    def __init__(self, message: str = "Resource not found", code: str = "NOT_FOUND"):
        super().__init__(message, code, 404)


class BadRequestError(TINEError):
    """Exception for bad requests"""
    def __init__(self, message: str = "Bad request", code: str = "BAD_REQUEST"):
        super().__init__(message, code, 400)


class InternalServerError(TINEError):
    """Exception for internal server errors"""
    def __init__(self, message: str = "Internal server error", code: str = "INTERNAL_ERROR"):
        super().__init__(message, code, 500)


class ServiceUnavailableError(TINEError):
    """Exception for service unavailable"""
    def __init__(self, message: str = "Service unavailable", code: str = "SERVICE_UNAVAILABLE"):
        super().__init__(message, code, 503)


# ============== Error Handler Setup ==============

def register_error_handlers(app: Flask) -> None:
    """
    Register all error handlers with the Flask app.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(TINEError)
    def handle_tine_error(error):
        """Handle custom TINE errors"""
        response = {
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message
            }
        }
        
        # Add validation errors if present
        if hasattr(error, 'errors') and error.errors:
            response["error"]["details"] = error.errors
        
        return jsonify(response), error.status_code
    
    @app.errorhandler(AuthenticationError)
    def handle_auth_error(error):
        """Handle authentication errors"""
        return jsonify({
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message
            }
        }), 401
    
    @app.errorhandler(AuthorizationError)
    def handle_authz_error(error):
        """Handle authorization errors"""
        return jsonify({
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message
            }
        }), 403
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Handle validation errors"""
        response = {
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message
            }
        }
        
        if error.errors:
            response["error"]["details"] = error.errors
        
        return jsonify(response), 400
    
    @app.errorhandler(RateLimitError)
    def handle_rate_limit_error(error):
        """Handle rate limit errors"""
        response = {
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message
            }
        }
        
        # Add rate limit info
        from flask_limiter import Limiter
        limiter = Limiter.key_func
        
        response["error"]["retry_after"] = 60  # Could be dynamic
        
        return jsonify(response), 429
    
    @app.errorhandler(ResourceNotFoundError)
    def handle_not_found_error(error):
        """Handle not found errors"""
        return jsonify({
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message
            }
        }), 404
    
    @app.errorhandler(BadRequestError)
    def handle_bad_request_error(error):
        """Handle bad request errors"""
        return jsonify({
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message
            }
        }), 400
    
    @app.errorhandler(404)
    def handle_404(error):
        """Handle 404 errors"""
        return jsonify({
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested endpoint was not found"
            }
        }), 404
    
    @app.errorhandler(405)
    def handle_405(error):
        """Handle 405 errors"""
        return jsonify({
            "success": False,
            "error": {
                "code": "METHOD_NOT_ALLOWED",
                "message": "The HTTP method is not allowed for this endpoint"
            }
        }), 405
    
    @app.errorhandler(500)
    def handle_500(error):
        """Handle 500 errors"""
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred"
            }
        }), 500
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle HTTP exceptions"""
        return jsonify({
            "success": False,
            "error": {
                "code": error.name.upper().replace(" ", "_"),
                "message": error.description
            }
        }), error.code
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle all unhandled exceptions"""
        # Log the exception
        app.logger.error(f"Unhandled exception: {str(error)}")
        app.logger.error(traceback.format_exc())
        
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred"
            }
        }), 500


# ============== Error Handler Decorator ==============

def handle_errors(f):
    """
    Decorator to handle errors in route functions.
    
    Returns structured JSON error responses.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TINEError:
            raise
        except ValidationError as e:
            raise
        except HTTPException:
            raise
        except Exception as e:
            # Log the error
            print(f"Error in {f.__name__}: {str(e)}", file=sys.stderr)
            raise InternalServerError(f"An error occurred in {f.__name__}")
    
    return decorated_function


# ============== Validation Error Handler ==============

def handle_validation_errors(errors: dict) -> None:
    """
    Raise a ValidationError with the given errors.
    
    Args:
        errors: Dictionary of field errors
    """
    if errors:
        raise ValidationError(
            message="Validation failed",
            errors=errors
        )


# ============== JSON Error Response Builder ==============

def error_response(
    message: str,
    code: str,
    status_code: int = 400,
    details: dict = None
) -> tuple:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        code: Error code
        status_code: HTTP status code
        details: Additional error details
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    return jsonify(response), status_code


def success_response(
    data: any = None,
    message: str = "Success",
    status_code: int = 200
) -> tuple:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        "success": True,
        "message": message
    }
    
    if data is not None:
        response["data"] = data
    
    return jsonify(response), status_code


# ============== Export all ==============

__all__ = [
    'TINEError',
    'AuthenticationError',
    'AuthorizationError',
    'ValidationError',
    'RateLimitError',
    'ResourceNotFoundError',
    'BadRequestError',
    'InternalServerError',
    'ServiceUnavailableError',
    'register_error_handlers',
    'handle_errors',
    'handle_validation_errors',
    'error_response',
    'success_response'
]
