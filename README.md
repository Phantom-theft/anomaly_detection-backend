# TINE (Threat Identification & Notification Engine) - Backend

TINE is a robust, real-time anomaly detection backend designed for security and surveillance applications. It leverages state-of-the-art Computer Vision (YOLOv8) and custom Deep Learning models to identify suspicious activities (such as stealing or unauthorized movements) from live camera streams.

The system integrates real-time notifications via Server-Sent Events (SSE) and Email alerts, secure camera management using Firebase Firestore, and a modular architecture for scalability and security.

## 🚀 Key Features

- **AI-Powered Detection**:
  - Real-time pose estimation using **YOLOv8n-pose**.
  - Custom **Keras** models for anomaly detection and stealing classification.
  - Aggressive detection modes and configurable accuracy thresholds.
- **Camera Management**:
  - Support for multiple RTSP streams.
  - Organization-based filtering (`org_id`).
  - Secure storage of camera configurations in **Firebase Firestore**.
- **Real-time Notifications**:
  - **Server-Sent Events (SSE)** for instant client-side updates.
  - Automated **Email alerts** with accuracy statistics.
  - **Cloudinary** integration for alert video/snapshot storage and management.
- **Security & Integrity**:
  - **Firebase Authentication** for secure API access.
  - Role-Based Access Control (RBAC) for administrative tasks.
  - **Rate Limiting** to prevent API abuse.
  - **Audit Logging** for tracking system and user actions.
- **Comprehensive Testing**: Full suite of unit and integration tests using `pytest`.

## 🛠️  Tech Stack

- **Language**: Python 3.x
- **Framework**: Flask
- **AI/ML**: Keras (TensorFlow), Ultralytics (YOLOv8), OpenCV, NumPy
- **Database**: Firebase Firestore
- **Authentication**: Firebase Admin SDK
- **Storage**: Cloudinary (for snapshots and recordings)
- **Monitoring**: Custom Audit Logger & Rate Limiter

## 📂 Project Structure

```text
├── app.py                # Main Management API (Auth, Cameras, SSE, Admin)
├── ai_model_server.py    # AI Inference Engine & CV Processing Loop
├── firebase_auth.py      # Firebase Authentication & Role Management
├── sse_manager.py        # Server-Sent Events (SSE) Logic
├── rate_limit.py         # API Rate Limiting Middleware
├── audit_logger.py       # System & User Audit Logging
├── validators.py         # Request Validation Logic
├── error_handlers.py     # Centralized Error Handling
├── ai_model/             # Local storage for detection logs and snapshots
├── tests/                # Automated Test Suite (Pytest)
├── .env.example          # Environment variables template
└── *.keras / *.pt        # Pre-trained AI Models
```

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- [Firebase Project](https://console.firebase.google.com/) with Firestore and Auth enabled.
- [Cloudinary Account](https://cloudinary.com/) for media storage.
- SMTP server (e.g., Gmail) for email notifications.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd anomaly_detection-backend
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: Ensure you have `tensorflow`, `ultralytics`, `opencv-python`, `flask`, `flask-cors`, `firebase-admin`, `cloudinary`, and `python-dotenv` installed.*

### 3. Environment Configuration
Create a `.env` file based on `.env.example`:
```env
# Email Configuration
EMAIL_SENDER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_API_KEY=your-api-key
```

### 4. Firebase Service Account
Place your Firebase service account key in the root directory as `firebase_key.json`.

## 🏃 Running the Application

To start the AI processing engine and the main backend services:
```bash
python ai_model_server.py
```

## 🧪  Testing

The project uses `pytest` for testing. Run the following command to execute the test suite:
```bash
pytest
```
To check coverage:
```bash
pytest --cov=.
```

## 🔒 Security

- **Rate Limiting**: Most API endpoints are limited to 60 requests per minute.
- **Authentication**: Bearer Token (Firebase ID Token) is required for most endpoints.
- **Audit Logs**: All sensitive actions (adding cameras, deleting users) are logged in `logs/audit.log`.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
