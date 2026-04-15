"""
Unit tests for validators module.
"""

import pytest
from validators import (
    validate_email,
    validate_email_required,
    validate_password_strength,
    validate_password_required,
    validate_camera_url,
    validate_camera_url_required,
    validate_date_range,
    validate_date_range_required,
    validate_name,
    validate_organization_name,
    validate_camera_settings,
    validate_alert_filters,
    ValidationError,
)


class TestEmailValidation:
    """Test email validation functions"""

    def test_validate_email_valid(self):
        """Test valid email passes validation"""
        is_valid, error = validate_email('test@example.com')
        assert is_valid is True
        assert error == ""

    def test_validate_email_invalid_format(self):
        """Test invalid email format fails validation"""
        is_valid, error = validate_email('not-an-email')
        assert is_valid is False
        assert "Invalid email format" in error

    def test_validate_email_empty(self):
        """Test empty email fails validation"""
        is_valid, error = validate_email('')
        assert is_valid is False
        assert "Email is required" in error

    def test_validate_email_none(self):
        """Test None email fails validation"""
        is_valid, error = validate_email(None)
        assert is_valid is False
        assert "Email is required" in error

    def test_validate_email_too_long(self):
        """Test email over 254 characters fails"""
        long_email = 'a' * 250 + '@example.com'
        is_valid, error = validate_email(long_email)
        assert is_valid is False
        assert "too long" in error

    def test_validate_email_required_valid(self):
        """Test validate_email_required with valid email"""
        try:
            validate_email_required('test@example.com')
        except ValidationError:
            pytest.fail("Should not raise ValidationError for valid email")

    def test_validate_email_required_invalid(self):
        """Test validate_email_required with invalid email"""
        with pytest.raises(ValidationError) as exc_info:
            validate_email_required('invalid-email')
        assert exc_info.value.field == "email"

    def test_validate_email_with_subdomain(self):
        """Test email with subdomain passes"""
        is_valid, _ = validate_email('user@sub.domain.example.com')
        assert is_valid is True

    def test_validate_email_with_plus(self):
        """Test email with plus sign passes"""
        is_valid, _ = validate_email('user+tag@example.com')
        assert is_valid is True


class TestPasswordValidation:
    """Test password validation functions"""

    def test_validate_password_valid(self):
        """Test valid password passes validation"""
        is_valid, error = validate_password_strength('Password1!')
        assert is_valid is True
        assert error == ""

    def test_validate_password_too_short(self):
        """Test password under 8 characters fails"""
        is_valid, error = validate_password_strength('Pass1!')
        assert is_valid is False
        assert "at least 8 characters" in error

    def test_validate_password_no_uppercase(self):
        """Test password without uppercase fails"""
        is_valid, error = validate_password_strength('password1!')
        assert is_valid is False
        assert "uppercase" in error

    def test_validate_password_no_lowercase(self):
        """Test password without lowercase fails"""
        is_valid, error = validate_password_strength('PASSWORD1!')
        assert is_valid is False
        assert "lowercase" in error

    def test_validate_password_no_number(self):
        """Test password without number fails"""
        is_valid, error = validate_password_strength('Password!')
        assert is_valid is False
        assert "number" in error

    def test_validate_password_no_special(self):
        """Test password without special character fails"""
        is_valid, error = validate_password_strength('Password1')
        assert is_valid is False
        assert "special character" in error

    def test_validate_password_too_long(self):
        """Test password over 128 characters fails"""
        long_password = 'P' * 129
        is_valid, error = validate_password_strength(long_password)
        assert is_valid is False
        assert "too long" in error

    def test_validate_password_empty(self):
        """Test empty password fails"""
        is_valid, error = validate_password_strength('')
        assert is_valid is False
        assert "Password is required" in error

    def test_validate_password_required_valid(self):
        """Test validate_password_required with valid password"""
        try:
            validate_password_required('Password1!')
        except ValidationError:
            pytest.fail("Should not raise ValidationError for valid password")

    def test_validate_password_required_invalid(self):
        """Test validate_password_required with invalid password"""
        with pytest.raises(ValidationError) as exc_info:
            validate_password_required('weak')
        assert exc_info.value.field == "password"


class TestCameraURLValidation:
    """Test camera URL validation functions"""

    def test_validate_camera_url_valid_rtsp(self):
        """Test valid RTSP URL passes"""
        is_valid, error = validate_camera_url('rtsp://example.com/stream')
        assert is_valid is True
        assert error == ""

    def test_validate_camera_url_valid_rtsps(self):
        """Test valid RTSPS URL passes"""
        is_valid, error = validate_camera_url('rtsps://example.com/stream')
        assert is_valid is True

    def test_validate_camera_url_invalid_http(self):
        """Test HTTP URL fails"""
        is_valid, error = validate_camera_url('http://example.com/stream')
        assert is_valid is False
        assert "rtsp" in error.lower()

    def test_validate_camera_url_empty(self):
        """Test empty URL fails"""
        is_valid, error = validate_camera_url('')
        assert is_valid is False
        assert "required" in error.lower()

    def test_validate_camera_url_no_hostname(self):
        """Test URL without hostname fails"""
        is_valid, error = validate_camera_url('rtsp://')
        assert is_valid is False
        assert "hostname" in error.lower()

    def test_validate_camera_url_required(self):
        """Test validate_camera_url_required"""
        try:
            validate_camera_url_required('rtsp://example.com/stream')
        except ValidationError:
            pytest.fail("Should not raise for valid URL")

        with pytest.raises(ValidationError) as exc_info:
            validate_camera_url_required('')
        assert exc_info.value.field == "rtsp_url"


class TestDateRangeValidation:
    """Test date range validation functions"""

    def test_validate_date_range_valid(self):
        """Test valid date range passes"""
        is_valid, error = validate_date_range(
            '2024-01-01T00:00:00Z',
            '2024-12-31T23:59:59Z'
        )
        assert is_valid is True

    def test_validate_date_range_start_after_end(self):
        """Test start date after end date fails"""
        is_valid, error = validate_date_range(
            '2024-12-31T00:00:00Z',
            '2024-01-01T00:00:00Z'
        )
        assert is_valid is False
        assert "before" in error.lower()

    def test_validate_date_range_invalid_start(self):
        """Test invalid start date format fails"""
        is_valid, error = validate_date_range('invalid-date', '2024-12-31')
        assert is_valid is False

    def test_validate_date_range_invalid_end(self):
        """Test invalid end date format fails"""
        is_valid, error = validate_date_range('2024-01-01', 'not-a-date')
        assert is_valid is False

    def test_validate_date_range_empty(self):
        """Test empty dates pass"""
        is_valid, error = validate_date_range(None, None)
        assert is_valid is True


class TestNameValidation:
    """Test name validation functions"""

    def test_validate_name_valid(self):
        """Test valid name passes"""
        is_valid, error = validate_name('John Doe')
        assert is_valid is True

    def test_validate_name_too_short(self):
        """Test name under 2 characters fails"""
        is_valid, error = validate_name('J')
        assert is_valid is False

    def test_validate_name_too_long(self):
        """Test name over 100 characters fails"""
        is_valid, error = validate_name('J' * 101)
        assert is_valid is False

    def test_validate_name_empty(self):
        """Test empty name fails"""
        is_valid, error = validate_name('')
        assert is_valid is False

    def test_validate_name_invalid_chars(self):
        """Test name with invalid characters fails"""
        is_valid, error = validate_name('John@Doe')
        assert is_valid is False

    def test_validate_name_with_hyphen(self):
        """Test name with hyphen passes"""
        is_valid, error = validate_name('Mary-Jane')
        assert is_valid is True


class TestOrganizationValidation:
    """Test organization name validation"""

    def test_validate_organization_name_valid(self):
        """Test valid organization name passes"""
        is_valid, error = validate_organization_name('Test Organization')
        assert is_valid is True

    def test_validate_organization_name_too_short(self):
        """Test organization name under 2 chars fails"""
        is_valid, error = validate_organization_name('A')
        assert is_valid is False

    def test_validate_organization_name_too_long(self):
        """Test organization name over 200 chars fails"""
        is_valid, error = validate_organization_name('A' * 201)
        assert is_valid is False

    def test_validate_organization_name_empty(self):
        """Test empty organization name fails"""
        is_valid, error = validate_organization_name('')
        assert is_valid is False


class TestCameraSettingsValidation:
    """Test camera settings validation"""

    def test_validate_camera_settings_valid(self):
        """Test valid camera settings passes"""
        data = {
            'name': 'Front Camera',
            'rtsp_url': 'rtsp://example.com/stream',
            'location': 'Building A'
        }
        errors = validate_camera_settings(data)
        assert len(errors) == 0

    def test_validate_camera_settings_invalid_name(self):
        """Test camera settings with invalid name"""
        data = {
            'name': 'A',  # Too short
            'rtsp_url': 'rtsp://example.com/stream',
            'location': 'Building A'
        }
        errors = validate_camera_settings(data)
        assert 'name' in errors

    def test_validate_camera_settings_invalid_url(self):
        """Test camera settings with invalid URL"""
        data = {
            'name': 'Front Camera',
            'rtsp_url': 'http://example.com',  # Not RTSP
            'location': 'Building A'
        }
        errors = validate_camera_settings(data)
        assert 'rtsp_url' in errors


class TestAlertFiltersValidation:
    """Test alert filters validation"""

    def test_validate_alert_filters_valid(self):
        """Test valid alert filters passes"""
        data = {
            'type': 'theft',
            'status': 'new',
            'limit': 50,
            'offset': 0
        }
        errors = validate_alert_filters(data)
        assert len(errors) == 0

    def test_validate_alert_filters_invalid_type(self):
        """Test alert filters with invalid type"""
        data = {'type': 'invalid_type'}
        errors = validate_alert_filters(data)
        assert 'type' in errors

    def test_validate_alert_filters_invalid_status(self):
        """Test alert filters with invalid status"""
        data = {'status': 'invalid_status'}
        errors = validate_alert_filters(data)
        assert 'status' in errors

    def test_validate_alert_filters_limit_too_high(self):
        """Test alert filters with limit over 100"""
        data = {'limit': 200}
        errors = validate_alert_filters(data)
        assert 'limit' in errors

    def test_validate_alert_filters_offset_negative(self):
        """Test alert filters with negative offset"""
        data = {'offset': -1}
        errors = validate_alert_filters(data)
        assert 'offset' in errors

    def test_validate_alert_filters_date_range_error(self):
        """Test alert filters with invalid date range"""
        data = {
            'startDate': '2024-12-31',
            'endDate': '2024-01-01'
        }
        errors = validate_alert_filters(data)
        assert 'date_range' in errors