import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import numpy as np
import time
import threading
import os
import cv2
import sys
import json
import cloudinary
import cloudinary.uploader
import cloudinary.api
import smtplib
import shutil
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yt_dlp
import firebase_admin
from firebase_admin import credentials, firestore

from collections import deque
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Import security and validation modules from app.py
try:
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
    SECURITY_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Some security modules not available: {e}")
    SECURITY_AVAILABLE = False
    # Define dummy decorators if modules not available
    def require_auth(f): return f
    def require_role(*args): return lambda x: x
    def api_rate_limit(f): return f
    def handle_errors(f): return f
    class ValidationError(Exception): pass

# Import SSE Manager
try:
    from sse_manager import emit_alert, emit_camera_status, emit_detection
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    print("[SSE] SSE manager not available - running without real-time updates")

#  1. CONFIGURATION & CONSTANTS

ENABLE_EMAILS = True 

EMAIL_SENDER = os.environ.get('EMAIL_SENDER', 'christinerealino6@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'hexy ofyf kctc ardx') 

ANOMALY_MODEL_PATH = 'anomaly_detector.keras'
STEALING_MODEL_PATH = 'stealing_classifier.keras'
YOLO_MODEL = 'yolov8n-pose.pt'

LOG_FILE = "ai_model/detections_log.json"
OUTPUT_DIR = "ai_model/detections"
RAW_RECORDING_DIR = os.environ.get('RAW_RECORDING_DIR', 'D:/CCTV_Record')
ENABLE_RAW_RECORDING = os.environ.get('ENABLE_RAW_RECORDING', 'True').lower() == 'true'

# THRESHOLDS 
POSE_THRESHOLD = 0.18       
# Aggressive Mode
STEAL_THRESH = 0.20        
# Email Threshold
EMAIL_MIN_ACCURACY = 50 
SCAN_MIN_FRAMES = 5

# TIMERS 
HISTORY_SECONDS = 20               
FPS_ESTIMATE = 15
HISTORY_LEN = HISTORY_SECONDS * FPS_ESTIMATE
STILLNESS_LIMIT = 20 * FPS_ESTIMATE 

SCAN_WINDOW_SEC = 4 
SCAN_LEN = SCAN_WINDOW_SEC * FPS_ESTIMATE

LOGIC_SKIP = 2 

ALERT_MAX_DURATION = 10.0           
ROUTINE_COOLDOWN = 20.
DEBUG_MODE = True 


#  VIDEO CONFIGURATION 
TARGET_FPS = 30.0               
FRAME_DELAY = 1.0 / TARGET_FPS 
BUFFER_SECONDS = 20              
BUFFER_SIZE = int(TARGET_FPS * BUFFER_SECONDS)

# Global flag for email limit
EMAIL_LIMIT_REACHED = False

# Detection tunable settings
DETECTION_DIST_SPEED  = 8.0
DETECTION_PACING_MULT = 1.0
DETECTION_LOITER_W    = 0.35
DETECTION_LOITER_H    = 0.35


#  PRECISION LOGIC HELPERS

class BehaviorValidator:
    def __init__(self):
        self.alert_counters = {} 
        self.MIN_FRAMES_STEAL = 3
        self.MIN_FRAMES_SUSP = 5

    def get_temporal_validation(self, track_id, current_label):
        if track_id not in self.alert_counters:
            self.alert_counters[track_id] = {"label": "Normal", "count": 0}
        
        counter = self.alert_counters[track_id]
        if current_label == counter["label"]:
            counter["count"] += 1
        else:
            counter["label"] = current_label
            counter["count"] = 1

        if "Stealing" in counter["label"] and counter["count"] >= self.MIN_FRAMES_STEAL:
            return True
        if ("Suspicious" in counter["label"] or "Loitering" in counter["label"] or "Pacing" in counter["label"]) and counter["count"] >= self.MIN_FRAMES_SUSP:
            return True
            
        return False

validator = BehaviorValidator()


#  INITIALIZATION
app = Flask(__name__)
# Explicitly allow CORS for the frontend origin
CORS(app, resources={r"/*": {
    "origins": ["https://bejewelled-cucurucho-944a9f.netlify.app", "*"],
    "methods": ["GET", "POST", "OPTIONS", "DELETE"],
    "allow_headers": ["Content-Type", "Authorization", "Cache-Control"]
}})
# Initialize security modules if available
if SECURITY_AVAILABLE:
    try:
        init_rate_limiter(app)
        init_audit_logger(app)
        register_error_handlers(app)
        print("[OK] Security modules initialized")
    except Exception as e:
        print(f"[WARNING] Error initializing security modules: {e}")


cameras_dict = {}  # key = camera_name, value = RTSPVideoStream object
@app.route("/add_youtube", methods=["POST"])
def add_youtube():
    data = request.get_json()
    user_id = data.get("userId")
    camera_name = data.get("cameraName")
    youtube_url = data.get("youtubeUrl")
    org_id = data.get("org_id") # Get org_id from frontend

    if not all([user_id, camera_name, youtube_url]):
        return {"error": "Missing data"}, 400

    # Check runtime duplicates in memory
    if camera_name in cameras_dict:
        return {"error": "Camera name already exists"}, 400

    # Permanent Firestore check
    query_name = db.collection("cameras").where("name", "==", camera_name).stream()
    if any(query_name):
        return {"error": "Camera name already exists in database"}, 400

    # 1. Extract the raw stream URL using yt-dlp
    try:
        print(f"[INFO] Extracting stream URL for: {youtube_url}")
        # We prefer a lower resolution like 480p or 720p for faster AI processing
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True
        } 
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            raw_stream_url = info['url']
    except Exception as e:
        print(f"!! Failed to extract YouTube stream: {e}")
        return {"error": "Failed to extract stream from this YouTube link. It might be private or age-restricted."}, 500

    # Fallback lookup if frontend didn't send org_id
    if not org_id or org_id == "default":
        try:
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists:
                org_id = user_doc.to_dict().get("org_id", None)
        except: pass

    # 2. Add camera to memory using the RAW stream URL
    cameras_dict[camera_name] = RTSPVideoStream(raw_stream_url, original_url=youtube_url, name=camera_name, is_youtube=True, org_id=org_id)

    # 3. Save to Firestore (We save the original YouTube URL, not the raw one)
    db.collection("cameras").add({
        "name":       camera_name,
        "rtsp_url":   youtube_url,
        "is_youtube": True,
        "owner":      user_id,
        "org_id":     org_id,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    print(f"[INFO] YouTube Camera added: {camera_name}")
    return {"message": f"YouTube stream '{camera_name}' added successfully!"}, 200

@app.route("/addCamera", methods=["POST"])
def add_camera():
    data = request.get_json()
    user_id = data.get("userId")
    camera_name = data.get("cameraName")
    rtsp_url = data.get("rtspUrl")
    org_id = data.get("org_id") # Get org_id from frontend

    if not all([user_id, camera_name, rtsp_url]):
        return {"error": "Missing data"}, 400

    # Check runtime duplicates in memory
    if camera_name in cameras_dict:
        return {"error": "Camera name already exists"}, 400

    for cam in cameras_dict.values():
        if cam.src == rtsp_url:
            return {"error": "This RTSP URL is already added"}, 400

    # Permanent Firestore check for both camera name and RTSP
    query_name = db.collection("cameras").where("name", "==", camera_name).stream()
    if any(query_name):
        return {"error": "Camera name already exists in database"}, 400

    query_rtsp = db.collection("cameras").where("rtsp_url", "==", rtsp_url).stream()
    if any(query_rtsp):
        return {"error": "This RTSP URL already exists in database"}, 400

    # Fallback lookup if frontend didn't send org_id
    if not org_id or org_id == "default":
        try:
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists:
                org_id = user_doc.to_dict().get("org_id", None)
        except: pass

    # Add camera to memory
    cameras_dict[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name, org_id=org_id)

    # Save to Firestore
    db.collection("cameras").add({
        "name":       camera_name,
        "rtsp_url":   rtsp_url,
        "owner":      user_id,
        "org_id":     org_id,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    print(f"[INFO] Camera added: {camera_name} ({rtsp_url}) by {user_id}")
    return {"message": f"Camera '{camera_name}' added successfully!"}, 200



if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()


# VIDEO STREAM CLASS


class RawRecorder:
    def __init__(self, camera_name, storage_path, org_id="default", fps=15.0):
        self.camera_name = camera_name
        self.storage_path = storage_path
        self.org_id = org_id or "default"
        self.fps = fps
        self.writer = None
        self.current_day = None

    def _get_filename(self):
        # Creates a filename like: D:/CCTV_Record/org123/Camera1/2026-03-27/09_30_05.webm
        now = datetime.now()
        day_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H_%M_%S")
        
        # Ensure directory exists for this org, camera and day
        path = os.path.join(self.storage_path, self.org_id, self.camera_name, day_str)
        os.makedirs(path, exist_ok=True)
        
        return os.path.join(path, f"{time_str}.webm")

    def write(self, frame):
        if frame is None: return
        
        try:
            # --- ADD TIMESTAMP OVERLAY ---
            h, w = frame.shape[:2]
            # Copy frame to avoid modifying the original used by AI
            stamped_frame = frame.copy()
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Black background rectangle for the clock
            cv2.rectangle(stamped_frame, (10, h - 35), (280, h - 5), (0, 0, 0), -1)
            # White text
            cv2.putText(stamped_frame, timestamp_text, (20, h - 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            now = datetime.now()
            today = now.strftime("%Y%m%d")
            
            # --- TEST MODE: Rotate every 5 minutes ---
            current_period = f"{today}_{now.hour}_{now.minute // 5}"

            # Rotate file if period changes or writer isn't initialized
            if self.writer is None or current_period != getattr(self, 'current_period', None):
                if self.writer: 
                    self.writer.release()
                    print(f"[RECORDER] Released previous file for {self.camera_name}")
                
                self.current_period = current_period
                self.current_day = today
                filename = self._get_filename()
                
                # Check if path is writable
                path = os.path.dirname(filename)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                
                # Use VP80 (WebM) for maximum browser compatibility without DLL issues
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
                self.writer = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
                
                if self.writer.isOpened():
                    print(f"✅ [RECORDER] Started web-compatible recording (.webm): {filename}")
                else:
                    print(f"⚠️ [RECORDER] VP80 codec failed, falling back to mp4v...")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    filename_mp4 = filename.replace('.webm', '.mp4')
                    self.writer = cv2.VideoWriter(filename_mp4, fourcc, self.fps, (w, h))
                    if self.writer.isOpened():
                        print(f"✅ [RECORDER] Started recording with mp4v: {filename_mp4}")
                    else:
                        print(f"❌ [RECORDER] Failed to open VideoWriter for {filename}")
                        self.writer = None

            if self.writer:
                self.writer.write(stamped_frame)
                
        except Exception as e:
            print(f"❌ [RECORDER] Critical Error writing for {self.camera_name}: {e}")
            if self.writer:
                self.writer.release()
                self.writer = None

    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None


class RTSPVideoStream:
    def __init__(self, src, original_url=None, name=None, is_youtube=False, org_id="default"):
        self.src = src
        self.original_url = original_url if original_url else src 
        self.name = name
        self.is_youtube = is_youtube
        self.org_id = org_id
        
        # Start with a dummy state to allow frontend to request the feed
        self.ret = False
        self.frame = None
        self.stopped = False
        self.online = True # Assume online if we got this far with a URL
        
        # Initialize capture
        print(f"[INFO] Initializing stream for {self.name}...")
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        
        # --- NEW: Get video FPS to pace the playback ---
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0 or self.fps > 120:
            self.fps = 30.0
        self.frame_delay = 1.0 / self.fps

        # --- RAW RECORDER INITIALIZATION ---
        self.recorder = None
        if ENABLE_RAW_RECORDING:
            self.recorder = RawRecorder(self.name, RAW_RECORDING_DIR, org_id=self.org_id, fps=self.fps)
        
        threading.Thread(target=self.update, daemon=True).start()

    def get_fresh_url(self):
        print(f"[INFO] Refreshing YouTube stream for: {self.name}...")
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.original_url, download=False)
                return info.get('url')
        except Exception as e:
            print(f"!! Failed to refresh URL for {self.name}: {e}")
            return self.src

    def update(self):
        frame_count = 0
        error_count = 0
        while not self.stopped:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = True
                    self.frame = frame
                    self.online = True
                    error_count = 0
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"[DEBUG] {self.name}: Received {frame_count} frames.")

                    if self.recorder:
                        self.recorder.write(frame)

                    if self.is_youtube:
                        time.sleep(self.frame_delay)
                else:
                    error_count += 1
                    self.ret = False
                    if error_count > 10:
                        self.online = False
                        print(f"⚠️ Stream connection lost for {self.name} - retrying...")
                        self.cap.release()
                        time.sleep(2)
                        reconnect_url = self.get_fresh_url() if self.is_youtube else self.src
                        self.cap = cv2.VideoCapture(reconnect_url, cv2.CAP_FFMPEG)
            else:
                if self.cap: self.cap.release()
                time.sleep(3)
                print(f"[INFO] Attempting to connect to {self.name}...")
                reconnect_url = self.get_fresh_url() if self.is_youtube else self.src
                self.cap = cv2.VideoCapture(reconnect_url, cv2.CAP_FFMPEG)
                self.online = self.cap.isOpened()
                if self.online:
                    print(f"✅ Camera {self.name} connected!")

    def read(self):
        if not self.cap.isOpened():
            self.online = False
            return False, None
        return self.ret, self.frame
# --- Load cameras from Firestore ---


def load_cameras_from_firestore():
    cams = db.collection("cameras").stream()
    cameras = {}

    for cam_doc in cams:
        data = cam_doc.to_dict()

        try:
            camera_name = data.get("name")        
            rtsp_url = data.get("rtsp_url")
            # Check if this camera was flagged as a YouTube stream when it was added
            is_youtube = data.get("is_youtube", False)

            # FIX: Kung null ang org_id sa Firestore, hanapin sa owner ng camera
            # Dati: data.get("org_id", "default") — agad "default" kahit may owner
            org_id = data.get("org_id") or None
            if not org_id or org_id == "default":
                owner_uid = data.get("owner")
                if owner_uid:
                    try:
                        user_doc = db.collection("users").document(owner_uid).get()
                        if user_doc.exists:
                            org_id = user_doc.to_dict().get("org_id") or "default"
                            print(f"[LOAD] Resolved org_id from owner for '{camera_name}': {org_id}")
                        # Also fix the camera doc in Firestore so next load is instant
                        if org_id and org_id != "default":
                            cam_docs = db.collection("cameras").where("name", "==", camera_name).limit(1).stream()
                            for cam_doc in cam_docs:
                                cam_doc.reference.update({"org_id": org_id})
                                print(f"[LOAD] Auto-fixed org_id in Firestore for camera '{camera_name}'")
                    except Exception as e:
                        print(f"[LOAD] Could not resolve org_id from owner for '{camera_name}': {e}")
            if not org_id:
                org_id = "default"
            print(f"[LOAD] Camera '{camera_name}' → org_id: {org_id}")

            if not camera_name or not rtsp_url:
                print("⚠️ Skipping invalid camera document:", data)
                continue

            if is_youtube:
                # It's a YouTube link, so we need to extract a FRESH raw stream URL
                print(f"[INFO] Fetching fresh YouTube stream for: {camera_name}...")
                try:
                    ydl_opts = {
                        'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
                        'quiet': True,
                        'no_warnings': True,
                        'nocheckcertificate': True
                    } 
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(rtsp_url, download=False)
                        fresh_stream_url = info['url']
                        
                        cameras[camera_name] = RTSPVideoStream(fresh_stream_url, original_url=rtsp_url, name=camera_name, is_youtube=True, org_id=org_id)
                        print(f"✅ Loaded YouTube camera: {camera_name}")
                except Exception as e:
                    print(f"❌ Failed to refresh YouTube camera '{camera_name}'. The video might be offline: {e}")
            else:
                # It's a normal RTSP camera, load it directly
                cameras[camera_name] = RTSPVideoStream(rtsp_url, name=camera_name, org_id=org_id)
                print(f"✅ Loaded RTSP camera: {camera_name}")

        except Exception as e:
            print(f"❌ Failed to load camera: {e}")

    return cameras



cameras_dict = load_cameras_from_firestore()


cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'dog6t9mx5'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', '275631287775487'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', '-41zZjN0GR1xVpx5lhBjvomKDAM'),
    secure=True
)

print(">> Loading AI Models on Nitro V 15...")
try:
    lstm_model = load_model(ANOMALY_MODEL_PATH)
    yolo_model = YOLO(YOLO_MODEL)
    stealing_model = load_model(STEALING_MODEL_PATH) if os.path.exists(STEALING_MODEL_PATH) else None
    print(">> Models loaded successfully.")
except Exception as e:
    print(f"!! Error: {e}")
    sys.exit(1)





# HELPER FUNCTIONS
def get_user_emails_by_org(org_id):
    """Kukunin lang ang emails ng STANDARD USERS (role='user') na nasa parehong org_id."""
    emails = []
    print(f"[EMAIL-DEBUG] Searching users for Org ID: {org_id}")
    
    if not org_id or org_id == "none" or org_id == "default":
        print(f"[EMAIL] Invalid or missing Org ID: {org_id}. Cannot fetch emails.")
        return emails
        
    try:
        # Use only one 'where' to match your existing Firestore index
        # and filter the role manually in Python for better reliability
        users_ref = db.collection("users").where("org_id", "==", org_id).stream()
        
        found_users = 0
        for doc in users_ref:
            user_data = doc.to_dict()
            # Strict check: Only include users with role 'user'
            if user_data.get("role") == "user":
                found_users += 1
                if "email" in user_data:
                    emails.append(user_data["email"])
        
        print(f"[EMAIL-DEBUG] Found {found_users} users with role='user' in Org {org_id}. Emails collected: {len(emails)}")
        
        # Log if no standard users found
        if found_users == 0:
            print(f"[EMAIL] Warning: No users found with role='user' and org_id='{org_id}'.")
            # Extra check: are there ANY users in this org?
            all_org_users = db.collection("users").where("org_id", "==", org_id).limit(5).stream()
            any_user = [d.to_dict().get('role') for d in all_org_users]
            if any_user:
                print(f"[EMAIL-DEBUG] Org '{org_id}' has users with roles: {any_user}. (Only role='user' is allowed for alerts)")
            else:
                print(f"[EMAIL-DEBUG] Org '{org_id}' appears to have NO users registered at all.")

    except Exception as e: 
        print(f"!! Error fetching emails for Org {org_id}: {e}")
    return emails

def send_email_alert(label, cloud_url, camera_name, org_id=None):
    global EMAIL_LIMIT_REACHED
    if not ENABLE_EMAILS or EMAIL_LIMIT_REACHED: 
        print(f"[EMAIL] Email sending skipped. ENABLE_EMAILS={ENABLE_EMAILS}, LIMIT_REACHED={EMAIL_LIMIT_REACHED}")
        return
    
    # Priority: passed org_id, then lookup from camera
    target_org = org_id or get_org_id_for_camera(camera_name)
    print(f"[EMAIL-DEBUG] Attempting to send alert for {camera_name} in Org: {target_org}")
    
    recipient_list = get_user_emails_by_org(target_org)
    
    if not recipient_list: 
        print(f"[EMAIL] No standard users (role='user') found for Org ID: {target_org}. Skipping email.")
        return
        
    try:
        print(f"[EMAIL] Connecting to SMTP server for {len(recipient_list)} recipient(s)...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        
        for email in recipient_list:
            msg = MIMEMultipart()
            msg['From'] = f"Security System <{EMAIL_SENDER}>"
            msg['To'] = email
            msg['Subject'] = f"🚨 ALERT: {label} Detected!"
            
            body = f"""
            <html>
            <body>
                <h2 style='color: #dc2626;'>Security Alert Detected</h2>
                <p><b>Incident Type:</b> {label}</p>
                <p><b>Camera Name:</b> {camera_name}</p>
                <p><b>Organization ID:</b> {target_org}</p>
                <p><b>Time Detected:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p>You can view the detected video clip here:</p>
                <a href='{cloud_url}' style='background-color: #7c3aed; color: white; padding: 10px 20px; text-decoration: none; border-radius: 8px; font-weight: bold;'>View Evidence Clip</a>
                <p style='font-size: 12px; color: #6b7280; margin-top: 20px;'>This is an automated message from your Anomaly Detection System.</p>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            server.send_message(msg)
            print(f"[EMAIL] Alert successfully sent to: {email}")
            
        server.quit()
        print(f"[EMAIL] All emails sent successfully for Org: {target_org}.")
    except Exception as e: 
        print(f"!! SMTP Failed: {e}")
        if "limit" in str(e).lower() or "5.4.5" in str(e):
            EMAIL_LIMIT_REACHED = True
            print("!! EMAIL LIMIT REACHED. System will stop sending emails for this session.")

def get_org_id_for_camera(camera_name):
    """Kukunin ang org_id ng camera mula sa memory o Firestore.
    FIX: Nag-aayos na rin ng null org_id sa Firestore nang automatic.
    """
    # 0. PINAKA-MAAASAHAN: I-check ang in-memory cameras_dict muna
    # Ito ang pinakamabilis at hindi na kailangan pang mag-query sa Firestore
    if camera_name in cameras_dict:
        cam = cameras_dict[camera_name]
        if hasattr(cam, 'org_id') and cam.org_id and cam.org_id not in ("default", "none", None):
            return cam.org_id

    resolved_org_id = None

    try:
        # 1. Try by document ID
        doc_ref = db.collection("cameras").document(camera_name).get()
        if doc_ref.exists:
            data = doc_ref.to_dict()
            org_id = data.get("org_id")
            if org_id and org_id not in ("default", "none"):
                resolved_org_id = org_id
            else:
                # Walang org_id — hanapin sa owner
                owner_uid = data.get("owner")
                if owner_uid:
                    user_doc = db.collection("users").document(owner_uid).get()
                    if user_doc.exists:
                        resolved_org_id = user_doc.to_dict().get("org_id")
                        # Auto-fix ang Firestore para hindi na paulit-ulit ang lookup
                        if resolved_org_id:
                            doc_ref.reference.update({"org_id": resolved_org_id})
                            print(f"[ORG-FIX] Auto-fixed org_id for camera '{camera_name}': {resolved_org_id}")

        # 2. Try by 'name' field (ang paraan ng pag-save ng ai_model_server)
        if not resolved_org_id:
            docs = db.collection("cameras").where("name", "==", camera_name).limit(1).stream()
            for doc in docs:
                data = doc.to_dict()
                org_id = data.get("org_id")
                if org_id and org_id not in ("default", "none"):
                    resolved_org_id = org_id
                else:
                    owner_uid = data.get("owner")
                    if owner_uid:
                        user_doc = db.collection("users").document(owner_uid).get()
                        if user_doc.exists:
                            resolved_org_id = user_doc.to_dict().get("org_id")
                            # Auto-fix ang Firestore
                            if resolved_org_id:
                                doc.reference.update({"org_id": resolved_org_id})
                                print(f"[ORG-FIX] Auto-fixed org_id for camera '{camera_name}' (by name): {resolved_org_id}")

    except Exception as e:
        print(f"!! Error getting org_id for {camera_name}: {e}")

    # I-update din ang in-memory camera object para hindi na mag-lookup ulit
    if resolved_org_id and camera_name in cameras_dict:
        cameras_dict[camera_name].org_id = resolved_org_id
        print(f"[ORG-FIX] Updated in-memory org_id for '{camera_name}': {resolved_org_id}")

    return resolved_org_id or "none"


def save_to_firebase(label, cloud_url, confidence_score, camera_name, track_id=None, org_id=None):
    try:
                MODEL_TRAINING_ACCURACY = 0.89 
                
                # 2. REAL-TIME CONFIDENCE (Yung pabago-bagong hinala ng AI)
                conf_pct = int(confidence_score)
                if conf_pct >= 80:
                    conf_str = f"{conf_pct}% (HIGH)"
                elif conf_pct >= 50:
                    conf_str = f"{conf_pct}% (MODERATE)"
                else:
                    conf_str = f"{conf_pct}% (LOW)"

                local_timestamp  = datetime.now()
                # Use provided org_id or lookup as fallback
                # CRITICAL FIX: Ensure final_org_id is NEVER "none" as it breaks fetching
                final_org_id = org_id or get_org_id_for_camera(camera_name)
                if not final_org_id or final_org_id == "none":
                    final_org_id = "default"

                db.collection("detections").add({
                    "camera_name": camera_name,
                    "type":        label,
                    "video_url":   cloud_url,
                    "accuracy":    MODEL_TRAINING_ACCURACY, 
                    "confidence":  conf_str,
                    "timestamp":   local_timestamp,
                    "created_at":  firestore.SERVER_TIMESTAMP,
                    "org_id":      final_org_id,
                    "duration":    f"{ALERT_MAX_DURATION}s",
                    "track_id":    track_id
                })
                print(f"[FIREBASE] Alert log sent successfully for {camera_name} ({label}) | ID: {track_id} | Org: {final_org_id}")
    except Exception as e:
        print(f"!! Firebase Error: {e}")


def save_alert_clip(sliced_frames, label, track_id, confidence_score, fps=30, suspect_count=1, camera_name="", org_id=None):
    try:
        if not sliced_frames: return
        frames = list(sliced_frames)

        ts_display = time.strftime("%Y-%m-%d %H:%M:%S")
        ts_file = time.strftime("%Y%m%d_%H%M%S")
        
        safe_label = label.replace(":", "").replace(" ", "_")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        fn = f"{OUTPUT_DIR}/{camera_name}_{safe_label}_ID{track_id}_{ts_file}.mp4"
        h, w, _ = frames[0].shape
        write_fps = max(fps, 5.0) 
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))
        
        if not out.isOpened():
             fn = fn.replace(".mp4", ".avi")
             fourcc = cv2.VideoWriter_fourcc(*'XVID')
             out = cv2.VideoWriter(fn, fourcc, write_fps, (w, h))
        if not out.isOpened(): return

        header_h = 40  
        overlay_color = (0, 0, 0) 
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 
        font_thick = 1
        
        MODEL_TRAINING_ACCURACY = 0.89 

        for f in frames:
            frame_out = f.copy()
            cv2.rectangle(frame_out, (0, h - header_h), (w, h), overlay_color, -1)
            # Added ACC (Accuracy) to the header text
            header_text = f"{ts_display} | ID:{track_id} {label} | ACC:{int(MODEL_TRAINING_ACCURACY*100)}% | CONF:{confidence_score}%"
            (text_w, text_h), baseline = cv2.getTextSize(header_text, font, font_scale, font_thick)
            cv2.putText(frame_out, header_text, (10, h - int((header_h - text_h) / 2) - 5), font, font_scale, text_color, font_thick, cv2.LINE_AA)
            out.write(frame_out)
            
        out.release()
        time.sleep(0.3) # Binawasan natin ang sleep para mas mabilis
        
        # 1. Cloudinary Upload
        playable_url = None
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            print(f"[UPLOAD] File found: {fn} ({os.path.getsize(fn)} bytes). Uploading to Cloudinary...")
            upload_success = False
            attempts = 0
            res = None
            while attempts < 3 and not upload_success:
                try:
                    attempts += 1
                    res = cloudinary.uploader.upload(fn, resource_type="video", folder=f"ai_detections/{camera_name}")
                    upload_success = True
                except Exception as e: 
                    print(f"!! [CLOUDINARY-ERROR] Attempt {attempts} failed: {str(e)}")
                    # Check for common issues
                    if "unauthorized" in str(e).lower():
                        print("!! [DEBUG] Check your API Key and Secret. They may be incorrect.")
                    elif "not found" in str(e).lower():
                        print("!! [DEBUG] Cloud name might be incorrect.")
                    time.sleep(2)
                    
            if upload_success and res:
                raw_url = res.get("secure_url")
                if raw_url:
                    playable_url = raw_url.replace("/upload/", "/upload/vc_h264,f_mp4/")
                    print(f"[UPLOAD] Success: {playable_url}")
                    
                    # Notify the frontend via SSE that the video is ready
                    if SSE_AVAILABLE:
                        emit_alert({
                            "camera_name": camera_name,
                            "type": label,
                            "video_url": playable_url,
                            "confidence": f"{confidence_score}%",
                            "track_id": track_id,
                            "suspects": suspect_count,
                            "timestamp": ts_display,
                            "org_id": org_id or get_org_id_for_camera(camera_name)
                        }, org_id=org_id)
        else:
            print(f"!! [UPLOAD-ERROR] File not found or empty: {fn}")
                    
        # 2. Save to Firebase (Passing track_id AND org_id now!)
        save_to_firebase(label, playable_url, confidence_score, camera_name, track_id=track_id, org_id=org_id) 
                    
        # 3. Send Email (Para sa Standard Users lang)
        if playable_url and ENABLE_EMAILS and 'EMAIL_LIMIT_REACHED' in globals() and not EMAIL_LIMIT_REACHED:
            if confidence_score >= 50: 
                send_email_alert(label, playable_url, camera_name, org_id=org_id)
                            
    except Exception as e: 
        print(f"!! Critical Error in save_alert_clip: {e}")


def get_centroid(kpts):
    """Returns the center (x, y) of the person based on keypoints."""
    xs = kpts[0::2]
    ys = kpts[1::2]
    return np.mean(xs), np.mean(ys)

def check_head_scanning(kpts, scan_history):
    global SCAN_MIN_FRAMES
    """Detects sustained scanning (looking left then right)."""
    nose_x = kpts[0]
    left_sh_x = kpts[10] 
    right_sh_x = kpts[12] 
    shoulder_width = abs(right_sh_x - left_sh_x)
   
    if shoulder_width < 0.01: return False
   
    look_ratio = (nose_x - left_sh_x) / shoulder_width

    if look_ratio < 0.15: scan_history.append("LEFT")
    elif look_ratio > 0.85: scan_history.append("RIGHT")
    else: scan_history.append("CENTER")

    return scan_history.count("LEFT") > 5 and scan_history.count("RIGHT") > 5

def is_hand_near_face(kpts, height_px):
    nose_x, nose_y = kpts[0], kpts[1]
    lw_x, lw_y = kpts[18], kpts[19]
    rw_x, rw_y = kpts[20], kpts[21]
   
    dist_l = np.hypot(lw_x - nose_x, lw_y - nose_y) * height_px
    dist_r = np.hypot(rw_x - nose_x, rw_y - nose_y) * height_px
   
    return dist_l < (height_px * 0.15) or dist_r < (height_px * 0.15)

def is_hand_in_stashing_zone(kpts, h_px):
    """Checks if hands are near hips or pockets relative to torso size."""
    lw_x, lw_y = kpts[18], kpts[19]
    rw_x, rw_y = kpts[20], kpts[21]
    l_hip_x, l_hip_y = kpts[22], kpts[23]
    r_hip_x, r_hip_y = kpts[24], kpts[25]
   
    shoulder_y = (kpts[11] + kpts[13]) / 2 
    torso_h = abs((l_hip_y + r_hip_y)/2 - shoulder_y)
    
    # 55% of torso height is a reliable stashing zone
    dynamic_thresh = max(torso_h * 0.55, 0.05) 
   
    dist_l = np.hypot(lw_x - l_hip_x, lw_y - l_hip_y)
    dist_r = np.hypot(rw_x - r_hip_x, rw_y - r_hip_y)
   
    return min(dist_l, dist_r) < dynamic_thresh

# MAIN LOGIC LOOP
people_states = {}

# --- HELPER FOR DYNAMIC RECORDING & SNAPSHOTS ---
def trigger_dynamic_save(person, track_id, camera_name, org_id, fps):
    frames = person.get('recording_frames', [])
    label = person.get('recording_label', 'Anomaly')
    acc = person.get('recording_acc', 0)
    suspects = person.get('recording_suspects', 1)
    
    person['is_recording'] = False
    person['recording_frames'] = []
    person['last_save_time'] = time.time()
    
    if len(frames) < 15: return
    
    print(f"🎬 [FINALIZING CLIP] ID:{track_id} | Type:{label} | Duration:~{len(frames)/fps:.1f}s")
    threading.Thread(target=save_alert_clip, 
                      args=(frames, label, track_id, acc, fps, suspects, camera_name, org_id)).start()

def save_instant_snapshot(frame, camera_name, org_id, label, track_id):
    """Saves and uploads a single frame immediately when detection starts."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"{OUTPUT_DIR}/snap_{camera_name}_{track_id}_{ts}.jpg"
        cv2.imwrite(fn, frame)
        
        # Background upload
        def upload_task():
            try:
                res = cloudinary.uploader.upload(fn, folder=f"ai_detections/{camera_name}/snapshots")
                print(f"📸 [SNAPSHOT READY] {res.get('secure_url')}")
            except: pass
        threading.Thread(target=upload_task).start()
    except: pass

def gen_frames(camera_name):
    camera = cameras_dict.get(camera_name)
    if not camera: return
    
    frame_buffer = deque(maxlen=BUFFER_SIZE)
    frame_count = 0
    prev_frame_time = 0
    real_time_fps = 15.0 
    last_processed_frame = None
    
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05); continue
        if frame is last_processed_frame:
            time.sleep(0.01); continue
        last_processed_frame = frame
        
        # Calculate dynamic FPS for smooth video writing
        new_frame_time = time.time()
        if prev_frame_time > 0:
            real_time_fps = 0.9 * real_time_fps + 0.1 * (1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # Resize for AI speed
        orig_h, orig_w = frame.shape[:2]
        target_w = 640
        target_h = int(orig_h * (target_w / orig_w))
        frame = cv2.resize(frame, (target_w, target_h))
        height, width = frame.shape[:2]
        
        results = yolo_model.track(frame, persist=True, verbose=False, classes=[0], conf=0.45)
        active_ids_this_frame = []
    
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xyn.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
                
            for box, track_id, kpts, y_conf in zip(boxes, track_ids, keypoints, confidences):
                active_ids_this_frame.append(track_id)
                kpts_flat = kpts.flatten()
                
                if track_id not in people_states:
                    people_states[track_id] = {
                        'pose_seq': [], 'loc_hist': deque(maxlen=HISTORY_LEN),
                        'scan_hist': deque(maxlen=SCAN_LEN), 'stationary_counter': 0,
                        'current_label': "Normal", 'current_color': (0, 255, 0),
                        'smoothed_box': box, 'current_acc': 0,
                        'last_lstm_err': 0.0, 'last_steal_prob': 0.0,
                        'last_save_time': 0, 'is_recording': False,
                        'recording_frames': [], 'post_roll_counter': 0,
                        'alert_frame_count': 0
                    }

                person = people_states[track_id]
                
                # Smooth box movement
                person['smoothed_box'] = (0.5 * box) + (0.5 * person['smoothed_box'])
                person['pose_seq'].append(kpts_flat)
                if len(person['pose_seq']) > 30: person['pose_seq'].pop(0)

                # Track movement
                bx1, by1, bx2, by2 = map(int, person['smoothed_box'])
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                if len(person['loc_hist']) > 0:
                    lx, ly = person['loc_hist'][-1]
                    if np.hypot(cx-lx, cy-ly) > 2: person['loc_hist'].append((cx, cy))
                    person['stationary_counter'] = person['stationary_counter'] + 1 if np.hypot(cx-lx, cy-ly) < DETECTION_DIST_SPEED else 0
                else: person['loc_hist'].append((cx, cy))

                # AI Inference (LSTM/Stealing)
                if len(person['pose_seq']) == 30 and (frame_count % LOGIC_SKIP == 0):
                    inp = np.array([person['pose_seq']])
                    person['last_lstm_err'] = np.mean(np.abs(lstm_model.predict(inp, verbose=0) - inp))
                    person['last_steal_prob'] = stealing_model.predict(inp, verbose=0)[0][0] if stealing_model else 0
                
                raw_label, color, acc = "Normal", (0, 255, 0), int(y_conf * 100)
                
                # Check Behaviors
                if person['stationary_counter'] > STILLNESS_LIMIT:
                    raw_label, color, acc = "Loitering (Still)", (0, 255, 255), 90
                elif person['last_steal_prob'] > STEAL_THRESH:
                    raw_label, color = "Anomaly: Stealing", (128, 0, 128)
                    acc = min(int(50 + ((person['last_steal_prob'] - STEAL_THRESH) / (1.0 - STEAL_THRESH)) * 49) + (15 if is_hand_in_stashing_zone(kpts_flat, height) else 0), 99)
                elif len(person['loc_hist']) >= (HISTORY_LEN // 2):
                    xs, ys = [p[0] for p in person['loc_hist']], [p[1] for p in person['loc_hist']]
                    total_path = sum(np.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1]) for i in range(1, len(xs)))
                    if total_path > (height * DETECTION_PACING_MULT) and np.hypot(xs[-1]-xs[0], ys[-1]-ys[0]) < (total_path * 0.5):
                        raw_label, color, acc = "Anomaly:Pacing", (255, 140, 0), 88
                    elif (max(xs)-min(xs)) < (width * DETECTION_LOITER_W) and (max(ys)-min(ys)) < (height * DETECTION_LOITER_H):
                        raw_label, color, acc = "Anomaly:Loitering (Area)", (0, 255, 255), 92
                elif check_head_scanning(kpts_flat, person['scan_hist']):
                    raw_label, color, acc = "Anomaly: Scanning", (255, 165, 0), 85

                is_validated = validator.get_temporal_validation(track_id, raw_label)
                final_label = raw_label if is_validated else "Normal"
                
                # Update visual state
                person['current_label'] = final_label
                person['current_color'] = color if is_validated else (0, 255, 0)
                person['current_acc'] = acc
                active_suspects = sum(1 for p in people_states.values() if "Anomaly" in p.get('current_label', ''))

                # --- ALWAYS DRAW BOXES (Inside Loop) ---
                thickness = 3 if is_validated else 1
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), person['current_color'], thickness)
                cv2.putText(frame, f"ID {track_id} {person['current_label']} ({person['current_acc']}%)", 
                            (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person['current_color'], 2)
                
                for i in range(0, 17):
                    xk, yk = int(kpts_flat[i*2] * width), int(kpts_flat[i*2+1] * height)
                    if xk > 0 and yk > 0: cv2.circle(frame, (xk, yk), 3, (0, 255, 0), -1)

                # --- DYNAMIC RECORDING LOGIC ---
                if final_label != "Normal":
                    cooldown = 60.0 if "Stealing" in final_label else 30.0
                    if not person.get('is_recording', False) and (time.time() - person.get('last_save_time', 0)) > cooldown:
                        # START RECORDING: Start exactly when alert is triggered
                        person['is_recording'] = True
                        person['recording_frames'] = [] 
                        person['recording_label'] = final_label
                        person['recording_acc'] = acc
                        person['recording_suspects'] = active_suspects
                        person['post_roll_counter'] = 0
                        person['alert_frame_count'] = 0
                        
                        # Immediate Alert + Snapshot
                        if SSE_AVAILABLE:
                            emit_detection(camera_name, final_label, acc, org_id=camera.org_id)
                        save_instant_snapshot(frame.copy(), camera_name, camera.org_id, final_label, track_id)
                    
                    if person.get('is_recording', False):
                        person['recording_frames'].append(frame.copy())
                        person['alert_frame_count'] += 1
                        
                        if len(person['recording_frames']) > 900: # Max 30 seconds safety
                            trigger_dynamic_save(person, track_id, camera_name, camera.org_id, real_time_fps)
                
                elif person.get('is_recording', False):
                    # Anomaly stopped: end exactly now
                    trigger_dynamic_save(person, track_id, camera_name, camera.org_id, real_time_fps)

        # --- Handle suspects who left the frame while recording ---
        for t_id, p in list(people_states.items()):
            if p.get('is_recording', False) and t_id not in active_ids_this_frame:
                p['recording_frames'].append(frame.copy()) # Add current empty frame
                p['post_roll_counter'] += 1
                if p['post_roll_counter'] > 120: 
                    trigger_dynamic_save(p, t_id, camera_name, camera.org_id, real_time_fps)

        frame_count += 1
        frame_buffer.append(frame.copy())
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


#  FLASK ROUTES

@app.route('/cameras', methods=['GET', 'OPTIONS']) # Dagdagan ng OPTIONS
def get_cameras():
    if request.method == 'OPTIONS': return '', 200
    # ... rest of your code
    org_id = request.args.get("org_id", None)
    cam_list = []

    # Kunin lahat ng cameras mula sa Firestore
    try:
        if org_id:
            # I-filter ang cameras sa Firestore base sa org_id
            docs = db.collection("cameras").where("org_id", "==", org_id).stream()
        else:
            # Walang org_id — ibalik lahat (para sa superadmin)
            docs = db.collection("cameras").stream()
        
        firestore_cams = {doc.to_dict().get("name"): doc.to_dict() for doc in docs}
    except Exception as e:
        print(f"!! Firestore query error: {e}")
        firestore_cams = {}

    # I-check din ang memory para sa online status
    for name, firestore_data in firestore_cams.items():
        # Get online status from memory if available
        online = False
        cam_type = "rtsp"
        src = firestore_data.get("rtsp_url", "")
        
        if name in cameras_dict:
            cam = cameras_dict[name]
            online = getattr(cam, 'online', False)
            cam_type = "youtube" if getattr(cam, 'is_youtube', False) else "rtsp"
            src = getattr(cam, 'original_url', getattr(cam, 'src', src))
        
        cam_list.append({
            "name":   name,
            "online": online,
            "type":   cam_type,
            "src":    src,
            "org_id": firestore_data.get("org_id", "default"),
            "owner":  firestore_data.get("owner", "unknown")
        })

    return {"cameras": cam_list}, 200


@app.route('/delete_camera/<camera_name>', methods=['DELETE'])
def delete_camera(camera_name):
    # 1. Try to stop the stream if it's currently running in memory
    if camera_name in cameras_dict:
        try:
            cam = cameras_dict[camera_name]
            if hasattr(cam, 'stop'):
                cam.stop()
            elif hasattr(cam, 'stopped'):
                cam.stopped = True
            del cameras_dict[camera_name]
            print(f"[INFO] Stopped active stream for: {camera_name}")
        except Exception as e:
            print(f"!! Error stopping stream: {e}")

    # 2. Always attempt to delete from Firestore (Database)
    try:
        # Check by 'name' field
        docs = db.collection("cameras").where("name", "==", camera_name).stream()
        deleted_count = 0
        for doc in docs:
            doc.reference.delete()
            deleted_count += 1
            
        # Also try to check if camera_name was the Document ID
        doc_ref = db.collection("cameras").document(camera_name)
        if doc_ref.get().exists:
            doc_ref.delete()
            deleted_count += 1

        if deleted_count > 0:
            print(f"[INFO] Camera '{camera_name}' deleted from database.")
            return {"message": f"Camera '{camera_name}' deleted successfully!"}, 200
        else:
            return {"error": "Camera not found in database"}, 404

    except Exception as e:
        print(f"!! Firestore delete error: {e}")
        return {"error": f"Database error: {str(e)}"}, 500
@app.route('/video/<camera_name>')
def video_feed(camera_name):
    return Response(gen_frames(camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream', strict_slashes=False)
@exempt_from_rate_limit
def stream():
    if request.method == 'OPTIONS':
        return '', 200
    """
    SSE stream endpoint. If sse_manager is available, it uses the proper
    SSE response creator to allow for persistent real-time updates.
    """
    if SSE_AVAILABLE:
        try:
            from sse_manager import create_sse_response
            events_param = request.args.get('events', 'alert,camera_status,detection,health')
            event_types = [e.strip() for e in events_param.split(',')]
            client_id = request.args.get('client_id', f"ai-{time.time()}")
            org_id = request.args.get('org_id')
            
            response = create_sse_response(event_types, client_id, org_id)
            

            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Cache-Control, Content-Type, Authorization'
            # Explicitly force the MIME type to prevent browser abortion
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Accel-Buffering'] = 'no'
            return response
        except Exception as e:
            print(f"[SSE] Error creating SSE response: {e}")

    # Fallback/Heartbeat for when sse_manager is not fully available
    def generate():
        while True:
            yield "data: {\"type\": \"heartbeat\"}\n\n"
            time.sleep(30)
    res = Response(generate(), mimetype="text/event-stream")
    res.headers['Access-Control-Allow-Origin'] = '*' # Dagdagan din dito para sa fallback
    return res
   


@app.route('/logs', methods=['GET'])
@exempt_from_rate_limit
def get_logs():
    org_id = request.args.get("org_id", None)
    if not org_id:
        return jsonify([])
    
    try:
        # ALIGNED FIX: Ordering by 'created_at' to match your Firestore Composite Index
        query = db.collection("detections").where("org_id", "==", org_id)
        docs = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(20).stream()
        
        logs = []
        for doc in docs:
            d = doc.to_dict()
            # Convert timestamp for display if it exists
            if 'timestamp' in d and hasattr(d['timestamp'], 'strftime'):
                d['timestamp'] = d['timestamp'].strftime("%B %d, %Y at %I:%M:%S %p")
            d['id'] = doc.id 
            logs.append(d)
            
        return jsonify(logs)
    except Exception as e:
        print(f"!! Firebase Logs Fetch Error: {e}")
        # Fallback: Try fetching without ordering if index is still building
        try:
            docs = db.collection("detections").where("org_id", "==", org_id).limit(20).stream()
            logs = []
            for doc in docs:
                d = doc.to_dict()
                raw_ts = d.get('created_at') or d.get('timestamp')
                
                if 'timestamp' in d and hasattr(d['timestamp'], 'strftime'):
                    d['timestamp'] = d['timestamp'].strftime("%B %d, %Y at %I:%M:%S %p")
                d['id'] = doc.id
                d['_raw_ts'] = str(raw_ts) if raw_ts else ""
                logs.append(d)
            
            logs.sort(key=lambda x: x.get('_raw_ts', ''), reverse=True)
            return jsonify(logs)
        except Exception as e2:
            print(f"!! Fallback Fetch Error: {e2}")
            return jsonify([])


@app.route('/detection_settings', methods=['GET'])
def get_detection_settings():
    return {
        "stillness_limit_seconds": STILLNESS_LIMIT / FPS_ESTIMATE,
        "dist_speed_threshold":    DETECTION_DIST_SPEED,
        "history_seconds":         HISTORY_SECONDS,
        "pacing_path_mult":        DETECTION_PACING_MULT,
        "loiter_area_w":           DETECTION_LOITER_W,
        "loiter_area_h":           DETECTION_LOITER_H,
        "pose_threshold":          POSE_THRESHOLD, 
        "steal_threshold":         STEAL_THRESH,
        "scan_threshold":          SCAN_MIN_FRAMES,
        "video_fps":               TARGET_FPS # Bagong control para sa mabilis na videos
    }, 200


@app.route('/detection_settings', methods=['POST'])
def update_detection_settings():
    global STILLNESS_LIMIT, HISTORY_SECONDS, HISTORY_LEN
    global STEAL_THRESH, SCAN_MIN_FRAMES, POSE_THRESHOLD
    global DETECTION_DIST_SPEED, DETECTION_PACING_MULT
    global DETECTION_LOITER_W, DETECTION_LOITER_H, TARGET_FPS

    data = request.get_json()

    if "stillness_limit_seconds" in data:
        STILLNESS_LIMIT = float(data["stillness_limit_seconds"]) * FPS_ESTIMATE
    if "history_seconds" in data:
        HISTORY_SECONDS = float(data["history_seconds"])
        HISTORY_LEN     = int(HISTORY_SECONDS * FPS_ESTIMATE)
    if "steal_threshold" in data:
        STEAL_THRESH = float(data["steal_threshold"])
    if "pose_threshold" in data:
        POSE_THRESHOLD = float(data["pose_threshold"])
    if "scan_threshold" in data:
        SCAN_MIN_FRAMES = int(data["scan_threshold"])
    if "dist_speed_threshold" in data:
        DETECTION_DIST_SPEED = float(data["dist_speed_threshold"])
    if "pacing_path_mult" in data:
        DETECTION_PACING_MULT = float(data["pacing_path_mult"])
    if "loiter_area_w" in data:
        DETECTION_LOITER_W = float(data["loiter_area_w"])
    if "loiter_area_h" in data:
        DETECTION_LOITER_H = float(data["loiter_area_h"])
    if "video_fps" in data:
        TARGET_FPS = float(data["video_fps"])

    print(f"[INFO] Detection settings updated: {data}")
    return {"message": "Settings updated successfully!"}, 200

@app.route('/delete_user/<user_id>', methods=['DELETE'])    
def delete_user(user_id):
    try:
        from firebase_admin import auth as firebase_auth
        firebase_auth.delete_user(user_id)
        print(f"[INFO] User deleted from Auth: {user_id}")
        return {"message": f"User {user_id} deleted from Firebase Auth."}, 200
    except Exception as e:
        print(f"!! Error deleting user from Auth: {e}")
        return {"error": str(e)}, 500

from flask import send_from_directory
import os

# PALITAN ITO NG TOTOONG DRIVE LETTER NG SD CARD MO (e.g., 'E:/CCTV_Records' o 'D:/Records')
SD_CARD_PATH = "D:/CCTV_Record" 
BIN_PATH = "D:/CCTV_Record_Bin"
RETENTION_DAYS = 14

def enforce_retention_policy(org_id):
    """Checks the org folder and moves folders older than 14 days to the Bin."""
    org_path = os.path.join(SD_CARD_PATH, org_id)
    bin_org_path = os.path.join(BIN_PATH, org_id)

    if not os.path.exists(org_path):
        return

    now = datetime.now()

    for camera_name in os.listdir(org_path):
        cam_path = os.path.join(org_path, camera_name)
        if not os.path.isdir(cam_path): continue

        for date_str in os.listdir(cam_path):
            date_path = os.path.join(cam_path, date_str)
            if not os.path.isdir(date_path): continue

            try:
                folder_date = datetime.strptime(date_str, "%Y-%m-%d")
                if (now - folder_date).days > RETENTION_DAYS:
                    target_cam_bin = os.path.join(bin_org_path, camera_name)
                    os.makedirs(target_cam_bin, exist_ok=True)
                    target_date_bin = os.path.join(target_cam_bin, date_str)
                    print(f"🗑️ [ARCHIVE] Moving old recording {date_str} to Recycle Bin...")
                    shutil.move(date_path, target_date_bin)
            except ValueError:
                pass

@app.route('/get_recorded_cameras', methods=['GET', 'OPTIONS'])
def get_recorded_cameras():
    if request.method == 'OPTIONS': return '', 200
    org_id = request.args.get("org_id", "default").strip()
    is_bin = request.args.get("is_bin", "false").lower() == "true"
    
    if not is_bin:
        enforce_retention_policy(org_id)
        
    base_path = BIN_PATH if is_bin else SD_CARD_PATH
    org_path = os.path.join(base_path, org_id)
    
    if not os.path.exists(org_path):
        return {"cameras": []}, 200
        
    try:
        cameras = [d for d in os.listdir(org_path) if os.path.isdir(os.path.join(org_path, d))]
        cameras.sort()
        return {"cameras": cameras}, 200
    except Exception as e:
        print(f"❌ [SEARCH] Error listing recorded cameras: {e}")
        return {"error": str(e)}, 500

@app.route('/get_recordings', methods=['GET'])
def get_recordings():
    camera_name = request.args.get("camera", "").strip()
    date_str = request.args.get("date", "").strip() 
    org_id = request.args.get("org_id", "default").strip()
    is_bin = request.args.get("is_bin", "false").lower() == "true"
    
    if not camera_name or not date_str:
        return {"error": "Missing parameters"}, 400

    date_str = date_str.replace("/", "-")
    base_path = BIN_PATH if is_bin else SD_CARD_PATH
    org_path = os.path.join(base_path, org_id)
    actual_camera_folder = camera_name
    
    if not os.path.exists(os.path.join(org_path, camera_name)):
        try:
            if os.path.exists(org_path):
                existing_folders = os.listdir(org_path)
                for folder in existing_folders:
                    if folder.lower() == camera_name.lower():
                        actual_camera_folder = folder
                        break
        except:
            pass

    folder_path = os.path.join(org_path, actual_camera_folder, date_str)
    
    if not os.path.exists(folder_path):
        return {"files": []}, 200
        
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.webm'))]
        files.sort()
        return {"files": files}, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/play_record', methods=['GET'])
def play_record():
    camera_name = request.args.get("camera")
    date_str = request.args.get("date")
    file_name = request.args.get("file")
    org_id = request.args.get("org_id", "default")
    is_bin = request.args.get("is_bin", "false").lower() == "true"
    
    base_path = BIN_PATH if is_bin else SD_CARD_PATH
    folder_path = os.path.abspath(os.path.join(base_path, org_id, camera_name, date_str))
    
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        return "Video not found", 404
        
    mimetype = 'video/webm' if file_name.endswith('.webm') else 'video/mp4'
    response = send_from_directory(folder_path, file_name, mimetype=mimetype)
    response.headers['Accept-Ranges'] = 'bytes'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response
@app.route('/delete_raw_record', methods=['POST'])
def delete_raw_record():
    data = request.get_json()
    camera_name = data.get("camera")
    date_str = data.get("date")
    file_name = data.get("file")
    org_id = data.get("org_id", "default")

    if not all([camera_name, date_str, file_name, org_id]):
        return {"error": "Missing parameters"}, 400

    # Kunin ang pinanggalingan ng file (Source)
    src_folder = os.path.join(SD_CARD_PATH, org_id, camera_name, date_str)
    src_file = os.path.join(src_folder, file_name)

    # Kunin ang pupuntahan ng file (Destination / Recycle Bin)
    dest_folder = os.path.join(BIN_PATH, org_id, camera_name, date_str)
    dest_file = os.path.join(dest_folder, file_name)

    if not os.path.exists(src_file):
        return {"error": "File not found"}, 404

    try:
        # Siguraduhing may folder na sa Recycle Bin bago ilipat
        os.makedirs(dest_folder, exist_ok=True)
        shutil.move(src_file, dest_file)
        print(f"🗑️ [MANUAL ARCHIVE] Moved {file_name} to Recycle Bin.")
        return {"message": "Video moved to Recycle Bin"}, 200
    except Exception as e:
        print(f"❌ [ARCHIVE ERROR] {e}")
        return {"error": str(e)}, 500
@app.route('/restore_raw_record', methods=['POST'])
def restore_raw_record():
    import shutil
    import os
    SD_CARD_PATH = "D:/CCTV_Record" 
    BIN_PATH = "D:/CCTV_Record_Bin" 

    data = request.get_json()
    camera_name = data.get("camera")
    date_str = data.get("date")
    file_name = data.get("file")
    org_id = data.get("org_id", "default")

    if not all([camera_name, date_str, file_name, org_id]):
        return {"error": "Missing parameters"}, 400

    # Kunin galing sa Recycle Bin
    src_file = os.path.join(BIN_PATH, org_id, camera_name, date_str, file_name)
    # Ibabalik sa Main Storage
    dest_folder = os.path.join(SD_CARD_PATH, org_id, camera_name, date_str)
    dest_file = os.path.join(dest_folder, file_name)

    if not os.path.exists(src_file):
        return {"error": "File not found in Recycle Bin"}, 404

    try:
        os.makedirs(dest_folder, exist_ok=True)
        shutil.move(src_file, dest_file)
        print(f"♻️ [RESTORE] Successfully moved {file_name} back to Dashboard.")
        return {"message": "Video restored successfully"}, 200
    except Exception as e:
        print(f"❌ [RESTORE ERROR] {e}")
        return {"error": str(e)}, 500

@app.route('/permanent_delete_raw_record', methods=['POST'])
def permanent_delete_raw_record():
    import os
    BIN_PATH = "D:/CCTV_Record_Bin" 

    data = request.get_json()
    camera_name = data.get("camera")
    date_str = data.get("date")
    file_name = data.get("file")
    org_id = data.get("org_id", "default")

    if not all([camera_name, date_str, file_name, org_id]):
        return {"error": "Missing parameters"}, 400

    target_file = os.path.join(BIN_PATH, org_id, camera_name, date_str, file_name)

    if not os.path.exists(target_file):
        return {"error": "File not found"}, 404

    try:
        os.remove(target_file) # TULUYAN NANG BUBURAHIN SA HARD DRIVE
        print(f"🔥 [HARD DELETE] Permanently deleted {file_name}.")
        return {"message": "Video permanently deleted"}, 200
    except Exception as e:
        print(f"❌ [HARD DELETE ERROR] {e}")
        return {"error": str(e)}, 500
    
@app.route('/delete_alert_video', methods=['POST'])
def delete_alert_video():
    data = request.get_json()
    video_url = data.get("video_url")
    
    if not video_url or "cloudinary" not in video_url:
        return {"message": "No remote video to delete or invalid URL"}, 200

    try:
        # Extract public_id from Cloudinary URL
        # Example: https://res.cloudinary.com/cloud_name/video/upload/v12345/folder/public_id.mp4
        # We need everything after /upload/ (excluding version and extension)
        parts = video_url.split('/upload/')
        if len(parts) < 2:
            return {"error": "Invalid Cloudinary URL format"}, 400
            
        path_after_upload = parts[1]
        # Remove version (v1234567/) if present
        if path_after_upload.startswith('v') and '/' in path_after_upload:
            path_after_upload = path_after_upload.split('/', 1)[1]
            
        # Remove extension (.mp4, .avi, etc.)
        public_id = path_after_upload.rsplit('.', 1)[0]
        
        print(f"[CLOUDINARY] Attempting to delete video with Public ID: {public_id}")
        
        # Delete from Cloudinary
        result = cloudinary.uploader.destroy(public_id, resource_type="video")
        
        if result.get("result") == "ok":
            print(f"✅ [CLOUDINARY] Successfully deleted: {public_id}")
            return {"message": "Video deleted successfully"}, 200
        else:
            print(f"⚠️ [CLOUDINARY] Deletion response: {result}")
            return {"message": "Cloudinary reported: " + str(result.get("result"))}, 200

    except Exception as e:
        print(f"❌ [CLOUDINARY-ERROR] Failed to delete video: {e}")
        return {"error": str(e)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)