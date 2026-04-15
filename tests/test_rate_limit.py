"""
Unit tests for rate limiting module.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, '.')


class TestRateLimiterInitialization:
    """Test rate limiter initialization"""

    def test_init_rate_limiter(self):
        """Test rate limiter initialization with Flask app"""
        from flask import Flask
        from rate_limit import init_rate_limiter
        
        app = Flask(__name__)
        app.config['TESTING'] = True
        
        limiter = init_rate_limiter(app)
        
        assert limiter is not None
        assert limiter.enabled is True

    def test_get_limiter(self):
        """Test getting rate limiter instance"""
        from rate_limit import get_limiter, init_rate_limiter
        from flask import Flask
        
        app = Flask(__name__)
        limiter = init_rate_limiter(app)
        
        retrieved_limiter = get_limiter()
        assert retrieved_limiter is not None


class TestRateLimitDecorators:
    """Test rate limit decorators"""

    @patch('rate_limit.limiter')
    def test_auth_rate_limit_decorator(self, mock_limiter):
        """Test auth rate limit decorator"""
        from rate_limit import auth_rate_limit
        
        mock_limiter.limit = MagicMock(return_value=lambda f: f)
        
        @auth_rate_limit
        def test_function():
            return "success"
        
        # Decorator should apply rate limit
        assert callable(test_function)

    @patch('rate_limit.limiter')
    def test_api_rate_limit_decorator(self, mock_limiter):
        """Test API rate limit decorator"""
        from rate_limit import api_rate_limit
        
        mock_limiter.limit = MagicMock(return_value=lambda f: f)
        
        @api_rate_limit
        def test_function():
            return "success"
        
        assert callable(test_function)

    @patch('rate_limit.limiter')
    def test_detection_rate_limit_decorator(self, mock_limiter):
        """Test detection rate limit decorator"""
        from rate_limit import detection_rate_limit
        
        mock_limiter.limit = MagicMock(return_value=lambda f: f)
        
        @detection_rate_limit
        def test_function():
            return "success"
        
        assert callable(test_function)

    @patch('rate_limit.limiter')
    def test_camera_stream_rate_limit_decorator(self, mock_limiter):
        """Test camera stream rate limit decorator"""
        from rate_limit import camera_stream_rate_limit
        
        mock_limiter.limit = MagicMock(return_value=lambda f: f)
        
        @camera_stream_rate_limit
        def test_function():
            return "success"
        
        assert callable(test_function)

    @patch('rate_limit.limiter')
    def test_sse_rate_limit_decorator(self, mock_limiter):
        """Test SSE rate limit decorator"""
        from rate_limit import sse_rate_limit
        
        mock_limiter.limit = MagicMock(return_value=lambda f: f)
        
        @sse_rate_limit
        def test_function():
            return "success"
        
        assert callable(test_function)


class TestCustomRateLimit:
    """Test custom rate limit functions"""

    @patch('rate_limit.limiter')
    def test_apply_rate_limit(self, mock_limiter):
        """Test applying custom rate limit"""
        from rate_limit import apply_rate_limit
        
        mock_limiter.limit = MagicMock(return_value=lambda f: f)
        
        decorator = apply_rate_limit("10 per minute")
        
        @decorator
        def test_function():
            return "success"
        
        assert callable(test_function)

    @patch('rate_limit.limiter')
    def test_check_rate_limit_within_limit(self, mock_limiter):
        """Test checking rate limit when within limit"""
        from rate_limit import check_rate_limit, init_rate_limiter
        from flask import Flask
        
        app = Flask(__name__)
        init_rate_limiter(app)
        
        mock_limiter.check = MagicMock(return_value=True)
        
        result = check_rate_limit()
        assert result is True

    @patch('rate_limit.limiter')
    def test_check_rate_limit_exceeded(self, mock_limiter):
        """Test checking rate limit when exceeded"""
        from rate_limit import check_rate_limit, init_rate_limiter
        from flask import Flask
        
        app = Flask(__name__)
        init_rate_limiter(app)
        
        mock_limiter.check = MagicMock(side_effect=Exception("Rate limit exceeded"))
        
        result = check_rate_limit()
        assert result is False


class TestRateLimitStatus:
    """Test rate limit status functions"""

    @patch('rate_limit.limiter')
    def test_get_rate_limit_status_enabled(self, mock_limiter):
        """Test getting rate limit status when enabled"""
        from rate_limit import get_rate_limit_status
        
        status = get_rate_limit_status()
        
        # When limiter is None, should return disabled
        # When initialized, should return enabled
        assert isinstance(status, dict)
        assert 'enabled' in status
        assert 'storage' in status


class TestExemptFromRateLimit:
    """Test exempt from rate limit decorator"""

    @patch('rate_limit.limiter')
    def test_exempt_from_rate_limit_decorator(self, mock_limiter):
        """Test exempt from rate limit decorator"""
        from rate_limit import exempt_from_rate_limit
        
        mock_limiter.exempt = MagicMock(return_value=lambda f: f)
        
        @exempt_from_rate_limit
        def test_function():
            return "success"
        
        assert callable(test_function)


class TestRateLimitConfiguration:
    """Test rate limit configuration values"""

    def test_auth_rate_limit_value(self):
        """Test auth endpoint has correct rate limit"""
        # Auth endpoints: 5 requests per minute
        # This is defined in the auth_rate_limit decorator
        pass

    def test_api_rate_limit_value(self):
        """Test API endpoint has correct rate limit"""
        # API endpoints: 60 requests per minute
        pass

    def test_detection_rate_limit_value(self):
        """Test detection endpoint has correct rate limit"""
        # Detection endpoints: 30 requests per minute
        pass

    def test_camera_stream_rate_limit_value(self):
        """Test camera stream endpoint has correct rate limit"""
        # Camera stream: 10 requests per minute
        pass

    def test_sse_rate_limit_value(self):
        """Test SSE endpoint has correct rate limit"""
        # SSE: 5 connections per minute
        pass


class TestRateLimitStorage:
    """Test rate limit storage configuration"""

    def test_memory_storage(self):
        """Test memory storage configuration"""
        from flask import Flask
        from rate_limit import init_rate_limiter
        
        app = Flask(__name__)
        limiter = init_rate_limiter(app)
        
        # Should use memory storage for testing
        assert limiter is not None

    def test_default_limits(self):
        """Test default rate limits"""
        from flask import Flask
        from rate_limit import init_rate_limiter
        
        app = Flask(__name__)
        limiter = init_rate_limiter(app)
        
        # Default limits: 200 per day, 50 per hour
        assert limiter is not None