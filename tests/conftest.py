"""
Pytest configuration and fixtures for backend tests.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch


# Add parent directory to path for imports
sys.path.insert(0, '.')


@pytest.fixture
def mock_firebase_auth():
    """Mock Firebase authentication"""
    with patch('firebase_auth.get_auth') as mock_auth:
        mock_user = MagicMock()
        mock_user.uid = 'test-uid'
        mock_user.email = 'test@example.com'
        mock_user.display_name = 'Test User'
        
        mock_auth.return_value = MagicMock(
            current_user=mock_user,
            create_user=MagicMock(return_value=mock_user),
            sign_in_with_email_and_password=MagicMock(return_value={'user': mock_user}),
        )
        yield mock_auth


@pytest.fixture
def mock_firestore():
    """Mock Firestore database"""
    with patch('firebase_auth.get_firestore') as mock_db:
        mock_collection = MagicMock()
        mock_doc = MagicMock()
        mock_doc.get = MagicMock(return_value=MagicMock(exists=True, data=lambda: {
            'uid': 'test-uid',
            'email': 'test@example.com',
            'role': 'admin',
            'org_id': 'org-1'
        }))
        
        mock_collection.doc.return_value = mock_doc
        mock_db.return_value = collection=MagicMock(return_value=mock_collection)
        yield mock_db


@pytest.fixture
def app():
    """Create Flask app for testing"""
    from flask import Flask
    from flask_cors import CORS
    
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['DEBUG'] = False
    CORS(app)
    
    # Register routes
    from app import register_routes
    register_routes(app)
    
    yield app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def mock_valid_token():
    """Mock valid Firebase token"""
    return 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test.mock.token'


@pytest.fixture
def auth_headers(mock_valid_token):
    """Headers with valid auth token"""
    return {
        'Authorization': f'Bearer {mock_valid_token}',
        'Content-Type': 'application/json'
    }


@pytest.fixture
def sample_camera_data():
    """Sample camera data for testing"""
    return {
        'name': 'Test Camera',
        'rtsp_url': 'rtsp://example.com/stream',
        'location': 'Test Location',
        'is_active': True,
        'organization_id': 'org-1'
    }


@pytest.fixture
def sample_alert_filter_data():
    """Sample alert filter data for testing"""
    return {
        'startDate': '2024-01-01T00:00:00Z',
        'endDate': '2024-12-31T23:59:59Z',
        'type': 'theft',
        'status': 'new',
        'limit': 50,
        'offset': 0
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        'uid': 'test-uid',
        'email': 'test@example.com',
        'display_name': 'Test User',
        'role': 'admin',
        'org_id': 'org-1'
    }


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests"""
    from rate_limit import limiter
    if limiter:
        limiter.reset()
    yield
    if limiter:
        limiter.reset()


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter"""
    with patch('rate_limit.limiter') as mock:
        mock.limit = MagicMock(return_value=lambda f: f)
        mock.exempt = MagicMock(return_value=lambda f: f)
        mock.check = MagicMock(return_value=True)
        mock.reset = MagicMock()
        yield mock


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )