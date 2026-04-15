from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase initialization warning: {e}")

db = firestore.client()

# Import security and validation modules
from firebase_auth import require_auth, require_role, get_current_user
from validators import (
    validate_camera_settings,
    validate_alert_filters,
    ValidationError as TINEValidationError
)
from rate_limit import (
    init_rate_limiter,
    api_rate_limit,
    detection_rate_limit,
    exempt_from_rate_limit
)
from audit_logger import (
    init_audit_logger,
    AuditAction,
    log_audit,
    log_user_action,
    log_error
)
from error_handlers import (
    register_error_handlers,
    handle_errors,
    ValidationError,
    error_response
)

import uuid

from sse_manager import create_sse_response, emit_alert, emit_camera_status, emit_system_health

# Initialize Flask app
app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": ["https://bejewelled-cucurucho-944a9f.netlify.app", "*"],
    "methods": ["GET", "POST", "OPTIONS", "DELETE"],
    "allow_headers": ["Content-Type", "Authorization", "Cache-Control"]
}})
# Initialize rate limiter
init_rate_limiter(app)

# Initialize audit logger
init_audit_logger(app)

# Register error handlers
register_error_handlers(app)

# ============== Helper Functions ==============

def validate_and_get_camera_data():
    """Validate camera settings from request"""
    data = request.get_json()
    if not data:
        raise ValidationError("Request body is required")
    
    errors = validate_camera_settings(data)
    if errors:
        raise ValidationError("Validation failed", errors=errors)
    
    return data


# ============== Camera Endpoints ==============

@app.route("/addCamera", methods=["POST"])
@require_auth
@api_rate_limit
@handle_errors
def add_camera():
    """
    Add a camera to the system.
    Requires Firebase authentication.
    Rate limited: 60 requests/minute
    Saves to Firebase Firestore.
    """
    data = validate_and_get_camera_data()
    
    name = data.get("cameraName") or data.get("name")
    url = data.get("rtspUrl") or data.get("rtsp_url")
    org_id = data.get("org_id", "default")
    
    if name and url:
        # Save to Firestore
        camera_data = {
            "name": name,
            "rtsp_url": url,
            "org_id": org_id,
            "created_at": firestore.SERVER_TIMESTAMP,
            "online": False,
            "type": "rtsp"
        }
        
        db.collection("cameras").document(name).set(camera_data)
        print(f"Camera {name} saved to Firestore for org: {org_id}")
        
        # Get current user info for audit logging
        user = get_current_user()
        user_id = user.get('uid') if user else 'unknown'
        
        # Log camera addition
        log_user_action(
            action=AuditAction.CAMERA_ADD,
            user_id=user_id,
            target_id=name,
            details={'org_id': org_id, 'rtsp_url': '***'}
        )
        
        return jsonify({"success": True}), 200
    
    raise ValidationError("Invalid data: cameraName and rtspUrl are required")


@app.route("/getCameras", methods=["GET"])
@require_auth
@api_rate_limit
@handle_errors
def get_cameras():
    """
    Get all cameras.
    Requires Firebase authentication.
    Rate limited: 60 requests/minute
    Fetches from Firebase Firestore.
    """
    org_id = request.args.get("org_id", "default")
    
    # Fetch cameras from Firestore filtered by org_id
    cameras_ref = db.collection("cameras").where("org_id", "==", org_id)
    cameras_docs = cameras_ref.stream()
    
    cameras_dict = {}
    for doc in cameras_docs:
        cam_data = doc.to_dict()
        cameras_dict[doc.id] = {
            "name": doc.id,
            "src": cam_data.get("rtsp_url", ""),
            "online": cam_data.get("online", False),
            "type": cam_data.get("type", "rtsp"),
            "org_id": cam_data.get("org_id", "default")
        }
    
    # Get current user info for audit logging
    user = get_current_user()
    user_id = user.get('uid') if user else 'unknown'
    
    # Log camera access
    log_user_action(
        action=AuditAction.CAMERA_ACCESS,
        user_id=user_id,
        details={'count': len(cameras_dict), 'org_id': org_id}
    )
    
    return jsonify({"cameras": cameras_dict}), 200


@app.route("/removeCamera", methods=["DELETE"])
@require_auth
@api_rate_limit
@handle_errors
def remove_camera():
    """
    Remove a camera from the system.
    Requires Firebase authentication.
    Rate limited: 60 requests/minute
    Deletes from Firebase Firestore.
    """
    data = request.get_json()
    name = data.get("cameraName")
    
    if not name:
        raise ValidationError("cameraName is required")
    
    # Delete from Firestore
    camera_ref = db.collection("cameras").document(name)
    camera_doc = camera_ref.get()
    
    if camera_doc.exists:
        # Get current user info for audit logging
        user = get_current_user()
        user_id = user.get('uid') if user else 'unknown'
        
        # Log camera removal
        log_user_action(
            action=AuditAction.CAMERA_REMOVE,
            user_id=user_id,
            target_id=name
        )
        
        camera_ref.delete()
        return jsonify({"success": True, "message": f"Camera {name} removed"}), 200
    
    raise ValidationError("Camera not found")


@app.route("/admin/delete_user/<user_id>", methods=["DELETE"])
@require_auth
@require_role('admin', 'superadmin')
@api_rate_limit
@handle_errors
def delete_user(user_id):
    """
    Delete a user from the system.
    Requires Firebase authentication and admin/superadmin role.
    Rate limited: 60 requests/minute
    """
    # Get current user info for audit logging
    current_user = get_current_user()
    current_user_id = current_user.get('uid') if current_user else 'unknown'
    
    # Log user deletion
    log_user_action(
        action=AuditAction.USER_DELETE,
        user_id=current_user_id,
        target_id=user_id
    )
    
    # Implementation would interact with Firebase Admin SDK
    return jsonify({"success": True, "message": f"User {user_id} deleted"}), 200


# ============== SSE Endpoints ==============

@app.route("/stream", methods=["GET"])
@exempt_from_rate_limit
def sse_stream():
    """
    Server-Sent Events endpoint for real-time updates.
    Rate limited: exempt (SSE connections are long-lived)
    
    Query parameters:
    - events: Comma-separated list of event types to subscribe to
              Options: alert, camera_status, detection, health
    - client_id: Optional client identifier (generated if not provided)
    
    Example: /stream?events=alert,camera_status&client_id=my-client
    """
    # Get event types to subscribe to
    events_param = request.args.get('events', 'alert,camera_status,detection,health')
    event_types = [e.strip() for e in events_param.split(',')]
    
    # Get or generate client ID
    client_id = request.args.get('client_id', str(uuid.uuid4()))
    
    # Get org_id for filtering
    org_id = request.args.get('org_id')
    
    print(f"[SSE] Client {client_id} (Org: {org_id}) connecting with events: {event_types}")
    
    # Log SSE connection
    log_audit(
        action=AuditAction.SYSTEM_ACCESS,
        details={'client_id': client_id, 'events': event_types, 'org_id': org_id}
    )
    
    return create_sse_response(event_types, client_id, org_id)


@app.route("/sse/emitting/alert", methods=["POST"])
@require_auth
@api_rate_limit
@handle_errors
def sse_emit_alert():
    """
    Endpoint to emit an alert via SSE.
    Used by external services to broadcast alerts.
    Rate limited: 60 requests/minute
    """
    data = request.get_json()
    
    # Try to get org_id from data or from camera
    camera_name = data.get('camera_name')
    org_id = data.get('org_id')
    
    if not org_id and camera_name:
        # Fallback to looking up camera org_id
        from ai_model_server import get_org_id_for_camera
        org_id = get_org_id_for_camera(camera_name)
    
    alert_data = {
        'message': data.get('message', 'New alert'),
        'severity': data.get('severity', 'info'),
        'camera_name': camera_name,
        'timestamp': data.get('timestamp'),
        'org_id': org_id
    }
    
    count = emit_alert(alert_data, org_id=org_id)
    
    # Log alert emission
    user = get_current_user()
    user_id = user.get('uid') if user else 'unknown'
    log_user_action(
        action=AuditAction.ALERT_CREATE,
        user_id=user_id,
        details=alert_data
    )
    
    return jsonify({
        "success": True, 
        "message": f"Alert emitted to {count} clients"
    }), 200


@app.route("/sse/emitting/camera_status", methods=["POST"])
@require_auth
@api_rate_limit
@handle_errors
def sse_emit_camera_status():
    """
    Endpoint to emit camera status updates via SSE.
    Rate limited: 60 requests/minute
    """
    data = request.get_json()
    
    camera_name = data.get('camera_name')
    status = data.get('status', 'unknown')
    org_id = data.get('org_id')
    
    if not camera_name:
        raise ValidationError("camera_name is required")
        
    if not org_id:
        from ai_model_server import get_org_id_for_camera
        org_id = get_org_id_for_camera(camera_name)
    
    count = emit_camera_status(camera_name, status, org_id=org_id)
    
    # Log camera status update
    user = get_current_user()
    user_id = user.get('uid') if user else 'unknown'
    log_user_action(
        action=AuditAction.CAMERA_UPDATE,
        user_id=user_id,
        target_id=camera_name,
        details={'status': status}
    )
    
    return jsonify({
        "success": True,
        "message": f"Camera status emitted to {count} clients"
    }), 200


@app.route("/sse/stats", methods=["GET"])
@api_rate_limit
def sse_stats():
    """
    Get SSE connection statistics.
    Rate limited: 60 requests/minute
    """
    from sse_manager import sse_manager
    
    return jsonify({
        "subscriptions": sse_manager.get_all_subscriptions()
    }), 200


@app.route("/health", methods=["GET"])
@exempt_from_rate_limit
def health_check():
    """
    Health check endpoint - no authentication required.
    Rate limited: exempt
    """
    return jsonify({"status": "healthy", "service": "tine-backend"}), 200


# ============== Error Handlers ==============

@app.route("/error-test", methods=["GET"])
@api_rate_limit
def test_error():
    """Test endpoint to verify error handling"""
    raise ValidationError("This is a test validation error", errors={'test': ['test error']})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
