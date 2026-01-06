# ICARO – Industrial Collision & Analysis Recognition Observer

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Name](#project-name)
3. [Man Down Detection](#man-down-detection)
4. [Classifier Specifications](#classifier-specifications)
5. [REST API Documentation](#rest-api-documentation)
6. [Backend Architecture](#backend-architecture)
7. [Frontend - Flutter Mobile App](#frontend---flutter-mobile-app)
8. [Wear OS Application](#wear-os-application)
9. [Data Flow](#data-flow)
10. [Setup and Installation](#setup-and-installation)

---

## Project Overview

**ICARO** is a human safety monitoring system designed for industrial and workplace environments. It provides real-time fall detection and abnormal posture recognition from video streams, with immediate alerting capabilities through mobile and wearable devices.

### Key Features
- **Multi-person fall detection** with pose quality checks and sliding window analysis
- **Horizontal posture detection** (man down) following fall events
- **Real-time video processing** using MediaPipe pose estimation
- **FastAPI backend** exposing RESTful control and telemetry endpoints
- **Cross-platform clients**: Flutter mobile app and Kotlin Wear OS companion
- **Firebase Cloud Messaging** for instant push notifications
- **MongoDB logging** for alert persistence and historical data
- **Wearable sensor integration** (heart rate, accelerometer) for health telemetry

### Technology Stack
- **Backend**: Python 3.10+, FastAPI, MediaPipe, scikit-learn
- **Machine Learning**: HistGradientBoostingClassifier, MLPClassifier
- **Mobile**: Flutter (Dart) with Firebase integration
- **Wearable**: Kotlin for Wear OS with Jetpack Compose
- **Database**: MongoDB for alert storage
- **Notifications**: Firebase Cloud Messaging (FCM)

---

## Project Name

**ICARO** stands for **Industrial Collision & Analysis Recognition Observer**.

The name represents:
- **Industrial**: Targeted at workplace and industrial safety environments
- **Collision**: Detection of falls and impacts
- **Analysis**: Real-time processing and classification of human poses
- **Recognition**: Identifying abnormal postures and dangerous situations
- **Observer**: Continuous monitoring and surveillance capability

The acronym also evokes the myth of Icarus, symbolizing the importance of fall prevention and safety awareness.

---

## Man Down Detection

### What is Man Down Detection?

"Man down" is a safety term referring to a person who has fallen and remains on the ground in a horizontal position, potentially incapacitated and unable to call for help. This is a critical safety concern in industrial environments, construction sites, healthcare facilities, and anywhere lone workers operate.

### ICARO's Two-Stage Detection

ICARO implements a sophisticated two-stage detection system:

#### Stage 1: Fall Detection
Monitors for rapid transitions from upright to fallen positions by analyzing:
- Body verticality and orientation changes
- Y-coordinate positions of key body landmarks (nose, shoulders, hips)
- Bounding box aspect ratio changes
- Torso length variations

#### Stage 2: Horizontal Posture Confirmation
After detecting a fall, the system monitors the person's posture to confirm they remain in a horizontal position:
- **Verticality ratio**: Measures height vs width of the body
- **Angle analysis**: 2D angle between shoulder and hip midpoints
- **Body span measurements**: Vertical and horizontal body dimensions
- **Bounding box aspect ratio**: Width-to-height ratio of the person's bounding box

### Why Two Stages?

1. **Reduces false positives**: Someone might briefly crouch or bend down
2. **Confirms actual emergency**: Ensures person remains down for extended period
3. **Contextual awareness**: Fall followed by prolonged horizontal posture indicates potential injury
4. **Post-fall monitoring**: Tracks person for 60 frames after fall detection to verify they don't get back up

---

## Classifier Specifications

ICARO uses two binary classifiers built with scikit-learn pipelines, trained on annotated fall video datasets.

### Fall Classifier

**Purpose**: Detect falling motion and distinguish from normal standing/walking.

**Architecture**:
- Pipeline: `SimpleImputer → StandardScaler → Classifier`
- Model types: HistGradientBoostingClassifier (HGB) or MLPClassifier
- Training: RandomizedSearchCV with GroupKFold cross-validation
- Best model selected based on F1-score on test set

**Input Features** (per frame, 6 features):
1. **y_nose**: Y-coordinate of nose landmark
2. **y_hips**: Y-coordinate of hip midpoint
3. **torso_len**: Distance between shoulders and hips
4. **bbox_ar**: Aspect ratio of bounding box (width/height)
5. **bbox_height**: Height of bounding box
6. **nose_to_hip_delta**: Vertical distance from nose to hips

**Window Aggregation** (sliding window of 9 frames):
- Mean, standard deviation, min, max, delta, and average speed for each feature
- Total: 6 features × 6 statistics = 36 aggregated features per window

**Quality Gates**:
- `fall_min_vis_point`: 0.80 (minimum visibility per landmark)
- `fall_min_pres_point`: 0.80 (minimum presence score per landmark)
- `fall_min_required_core_points`: 5 (must have at least 5 core landmarks visible)
- `min_window_quality`: 0.95 (high quality, mean visibility across window)

**Thresholds**:
- `fall_threshold`: 0.80 (probability threshold for positive classification)
- `consecutive_fall`: 5 (must detect fall in 5 consecutive windows)

**Output**: Binary classification (NORMAL=0, FALL=1) with probability score

---

### Horizontal Posture Classifier

**Purpose**: Confirm person is in horizontal/lying position after fall detection.

**Architecture**:
- Pipeline: `SimpleImputer → StandardScaler → Classifier`
- Model types: HistGradientBoostingClassifier or MLPClassifier
- Training: Same RandomizedSearchCV approach as fall classifier

**Input Features** (per frame, 9 features):
1. **verticality_ratio**: Body height span / body width span
2. **angle_2d**: 2D angle between shoulder-hip vector and vertical axis
3. **nose_to_hip**: Vertical distance from nose to hip center
4. **hip_to_foot**: Vertical distance from hip to foot center
5. **bbox_ar**: Bounding box aspect ratio
6. **body_y_span**: Total vertical span of body
7. **nose_y**: Y-coordinate of nose
8. **shoulder_mid_y**: Y-coordinate of shoulder midpoint
9. **quality_score**: Mean visibility × presence for key landmarks

**Window Aggregation** (sliding window of 9 frames):
- Same 6 statistics per feature: mean, std, min, max, delta, avg_speed
- Total: 9 features × 6 statistics = 54 aggregated features per window

**Quality Gates**:
- `horizontal_min_quality`: 0.65 (minimum quality for frame acceptance)
- `horizontal_min_good_keypoints`: 4 (minimum visible key landmarks)
- `min_window_quality`: 0.95 (high quality for window acceptance)

**Thresholds**:
- `horizontal_threshold`: 0.70 (probability threshold for horizontal classification)
- `consecutive_horizontal`: 4 (must detect horizontal in 4 consecutive windows)
- `post_fall_duration`: 60 (monitor for 60 frames after fall detection)

**Output**: Binary classification (NOT_HORIZONTAL=0, HORIZONTAL=1) with probability score

---

### Training Configuration

**Hyperparameter Search**:

HGB parameters:
- `max_depth`: [2, 3, 4, 5]
- `learning_rate`: [0.02, 0.05, 0.08, 0.12]
- `max_iter`: [200, 300, 450, 650]
- `min_samples_leaf`: [10, 20, 40, 80]
- `l2_regularization`: [0.0, 0.1, 0.5, 1.0]

MLP parameters:
- `hidden_layer_sizes`: [(64,), (128,), (256,), (128, 64), (256, 128), (128, 128)]
- `activation`: ["relu", "tanh"]
- `alpha`: [1e-5, 1e-4, 1e-3, 1e-2]
- `learning_rate_init`: [1e-4, 3e-4, 1e-3, 3e-3]
- `batch_size`: ["auto", 32]
- `max_iter`: [300, 500, 700]

**Training Strategy**:
1. Load annotated video dataset with fall start/end frame labels
2. Extract features with quality filtering
3. Create sliding windows with aggregated statistics
4. Split data with GroupShuffleSplit (ensures video-level separation)
5. Downsample majority class (max 2.5:1 ratio)
6. Apply quality-weighted sample weights
7. Train both HGB and MLP with RandomizedSearchCV
8. Select best model based on F1-score
9. Save to `data/icaro_models.joblib`

---

## REST API Documentation

Base URL: `http://<server_ip>:8000`

### 1. Get Service Status

**Endpoint**: `GET /api/v1/status`

**Description**: Returns current status of the pose detection pipeline.

**Response**:
```json
{
  "ok": true,
  "running": true,
  "active_video_path": "/path/to/video.avi",
  "running_mode": "VIDEO"
}
```

**Fields**:
- `ok`: Boolean indicating API health
- `running`: Whether pose pipeline is currently running
- `active_video_path`: Path to currently processed video
- `running_mode`: Either "VIDEO" or "LIVE_STREAM"

---

### 2. Start Pipeline

**Endpoint**: `POST /api/v1/start`

**Description**: Starts the pose detection pipeline with current configuration.

**Response**:
```json
{
  "ok": true,
  "started": true,
  "running": true,
  "active_video_path": "/path/to/video.avi",
  "running_mode": "VIDEO"
}
```

---

### 3. Stop Pipeline

**Endpoint**: `POST /api/v1/stop`

**Description**: Stops the currently running pose detection pipeline.

**Response**:
```json
{
  "ok": true,
  "running": false
}
```

---

### 4. Upload Video

**Endpoint**: `POST /api/v1/upload`

**Description**: Upload a video file for processing. Automatically restarts pipeline if running.

**Request**: Multipart form-data with file field

**Supported formats**: `.avi`, `.mp4`, `.mov`, `.mkv`

**Response**:
```json
{
  "ok": true,
  "saved_path": "/path/to/uploads/abc123.mp4",
  "active_video_path": "/path/to/uploads/abc123.mp4",
  "running_mode": "VIDEO",
  "running": true
}
```

**Error Responses**:
- `400`: Missing filename or unsupported file type
- `500`: Failed to save file

---

### 5. Set Running Mode - Live Stream

**Endpoint**: `PUT /api/v1/running-mode/live-stream`

**Description**: Switch to live camera stream mode. Restarts pipeline if running.

**Response**:
```json
{
  "ok": true,
  "running_mode": "LIVE_STREAM",
  "active_video_path": "/path/to/video.avi",
  "running": true
}
```

---

### 6. Set Running Mode - Video

**Endpoint**: `PUT /api/v1/running-mode/video`

**Description**: Switch to video file processing mode. Restarts pipeline if running.

**Response**:
```json
{
  "ok": true,
  "running_mode": "VIDEO",
  "active_video_path": "/path/to/video.avi",
  "running": true
}
```

---

### 7. Submit Heart Rate

**Endpoint**: `POST /api/v1/measure/{heartbeat}`

**Description**: Submit heart rate measurement from wearable device (in BPM).

**Path Parameters**:
- `heartbeat`: Integer representing beats per minute

**Example**: `POST /api/v1/measure/72`

**Response**:
```json
{
  "ok": true
}
```

**Side Effects**: Updates global `LatestHeartbeat.BPM` variable, logs to backend

---

### 8. Submit Movement Vector

**Endpoint**: `POST /api/v1/monitor/{x}-{y}-{z}`

**Description**: Submit accelerometer data from wearable device.

**Path Parameters**:
- `x`: Float, X-axis acceleration
- `y`: Float, Y-axis acceleration
- `z`: Float, Z-axis acceleration

**Example**: `POST /api/v1/monitor/0.123-0.456-9.810`

**Response**:
```json
{
  "ok": true
}
```

**Side Effects**: Updates global `LatestMovement.{X,Y,Z}` variables, logs to backend

---

### 9. Get Alerts

**Endpoint**: `GET /api/v1/alerts`

**Description**: Retrieve all stored alerts from MongoDB.

**Response**:
```json
{
  "alerts": [
    {
      "_id": {"$oid": "507f1f77bcf86cd799439011"},
      "title": "Fall Detected",
      "message": "Person has fallen and is not moving",
      "alert": true,
      "timestamp": {"$date": {"$numberLong": "1640000000000"}}
    }
  ]
}
```

**Fields**:
- `_id`: MongoDB ObjectId
- `title`: Alert title string
- `message`: Detailed alert message
- `alert`: Boolean, true for critical alerts
- `timestamp`: ISO 8601 timestamp or MongoDB date object

---

### CORS Configuration

The API allows cross-origin requests from:
- `localhost`
- `127.0.0.1`
- `192.168.1.*` (local network)

Both HTTP and HTTPS protocols supported with any port.

---

## Backend Architecture

### Core Components

#### 1. FastAPI Application (`src/app.py`)
- Async lifespan management with automatic pipeline cleanup
- CORS middleware for cross-origin requests
- Global state management for video path and running mode
- Asynchronous pipeline control with stop events

#### 2. Pose Landmark Extraction (`src/pose_landmark.py`)
- MediaPipe Pose Landmarker integration
- Supports both VIDEO and LIVE_STREAM modes
- Frame-by-frame pose estimation with 33 landmarks per person
- Outputs normalized 2D coordinates with visibility and presence scores

#### 3. Live Fall Detector (`src/live_fall_detector.py`)

**LiveManDownDetector Class**:
- Single-person fall and horizontal posture detection
- Sliding window feature buffers (deques)
- Configurable thresholds and quality gates
- State machine with consecutive hit counting
- Post-fall monitoring timer (60 frames)

**MultiPersonDetector Class**:
- Tracks multiple people simultaneously (up to 8)
- Person tracking by pose center (Euclidean distance matching)
- Per-person detector instances with individual state
- Distance threshold: 0.12 (normalized coordinates)

#### 4. Pipeline & Feature Extraction (`src/pipeline_horizontal_classification.py`)
- Feature extraction functions for fall and horizontal detection
- Window aggregation with NaN handling
- Dataset loading from JSON annotations
- Model training with RandomizedSearchCV
- Quality-weighted sampling
- Model persistence with joblib

#### 5. Push Notifications (`src/push_notification.py`)
- Firebase Admin SDK integration
- Topic-based messaging ("fall" topic)
- Global state for latest heartbeat and movement data
- Alert functions:
  - `send_push_notification()`: Generic fall alerts
  - `send_push_notification_heartbeat()`: Heart rate warnings
  - `send_monitoring_notification()`: Monitoring start signal

#### 6. MongoDB Integration (`src/mongodb.py`)
- Alert document storage
- Retrieval function for Flutter app consumption
- Timestamp and metadata preservation

### Execution Flow

1. **Startup**: FastAPI lifespan initializes event loop reference
2. **Start Request**: Creates async task running `run_pose_async()`
3. **Video Processing**:
   - MediaPipe extracts pose landmarks per frame
   - Multi-person detector creates/updates person tracks
   - Each person's detector receives pose landmarks
   - Feature extraction with quality checks
   - Sliding window aggregation
   - Model inference (fall and horizontal classifiers)
   - Consecutive hit counting
   - Alert triggering on threshold exceeds
4. **Alert Dispatch**:
   - Push notification via Firebase
   - MongoDB document insertion
   - Log output
5. **Stop Request**: Sets stop event, awaits task completion
6. **Shutdown**: Automatic cleanup via lifespan context

---

## Frontend - Flutter Mobile App

### Overview
The Flutter mobile app (`frontend/icaro/`) provides a monitoring interface for safety personnel to view fall alerts and receive real-time push notifications.

### Architecture

**Main Components**:
- `lib/main.dart`: Single-file app with all UI and logic
- `firebase_options.dart`: Auto-generated Firebase configuration
- Platform-specific Firebase configurations (Android/iOS)

### Key Features

#### 1. Firebase Cloud Messaging Integration
- Background message handler for notifications when app is closed
- Foreground message listener with in-app snackbar
- Topic subscription: "fall"
- Permission request for notifications

#### 2. Alert Display
- HTTP polling of backend `/api/v1/alerts` endpoint
- Support for multiple response formats:
  - List format: `[{...}, {...}]`
  - Keyed map format: `{"0": {...}, "1": {...}}`
  - Wrapped format: `{"alerts": [...]}`
- MongoDB ObjectId parsing
- Timestamp parsing with multiple format support
- Pull-to-refresh functionality

#### 3. UI Components
- **AppBar**: Title and refresh button
- **Alert List**: Scrollable list with icons (error for alerts, warning for info)
- **Empty State**: "No alerts" message
- **Error State**: Error display with retry button
- **Loading State**: Circular progress indicator

### Data Model

**AlertItem Class**:
```dart
class AlertItem {
  final String id;
  final String title;
  final String message;
  final bool alert;
  final DateTime timestamp;
}
```

### Configuration

**Endpoint**: `http://192.168.1.15:8000/api/v1/alerts`
- Hardcoded for local network deployment
- Should be configurable for production

**Dependencies**:
- `firebase_core: ^4.3.0`: Firebase initialization
- `firebase_messaging: ^16.1.0`: Push notifications
- `http: ^1.6.0`: REST API calls
- `permission_handler: ^12.0.1`: Runtime permissions

### Notification Flow

1. App subscribes to "fall" topic on Firebase
2. Backend sends notification via Firebase Admin SDK
3. If app is foreground: In-app snackbar notification
4. If app is background/terminated: System notification tray
5. User taps notification → App opens to alert list
6. User pulls to refresh → Fetches latest alerts from backend

---

## Wear OS Application

### Overview
The Wear OS app (`icaro-wearos/`) is a Kotlin-based smartwatch companion that provides wrist-worn alerts, heart rate monitoring, and accelerometer data collection.

### Architecture

**Technology Stack**:
- Kotlin with Jetpack Compose for Wear OS
- Firebase Cloud Messaging for notifications
- Hardware sensors: Heart rate, accelerometer
- OkHttp for HTTP requests
- Coroutines for asynchronous operations

**Main Components**:
1. `MainActivity.kt`: UI with Compose
2. `FirebaseNotificationService.kt`: Background service for FCM and sensors

### Features

#### 1. Real-Time Alert Display
- Shows "Monitoring..." in normal state
- Displays "FALL DETECTED!" in red when alert received
- Plays notification sound via RingtoneManager
- Vibration feedback on notifications

#### 2. Heart Rate Monitoring
- Subscribes to `Sensor.TYPE_HEART_RATE`
- Continuous monitoring when fall detected
- Displays live BPM on watch face
- Broadcasts heart rate updates to MainActivity
- 60-second timeout for sensor wake-up
- Averages multiple readings for accuracy

#### 3. Accelerometer Integration
- Subscribes to `Sensor.TYPE_ACCELEROMETER`
- Collects X, Y, Z acceleration values
- Uses `SENSOR_DELAY_GAME` for frequent updates
- Averages readings over collection period

#### 4. Backend Communication
- HTTP POST to `/api/v1/measure/{heartbeat}` with BPM value
- HTTP POST to `/api/v1/monitor/{x}-{y}-{z}` with acceleration data
- Automatic sensor data push on fall notification
- 30-second timeout for network requests

#### 5. User Interaction
- **"I'm OK" button**: Resets fall alert state
- Touch interaction dismisses alert
- TimeText shows current time at top

### UI States

**Normal State**:
- Text: "Monitoring..."
- Color: Primary theme color
- No buttons shown

**Alert State**:
- Text: "FALL DETECTED!"
- Color: Red
- Shows heart rate if available
- "I'm OK" button visible

### Firebase Notification Service

**Service Lifecycle**:
- Background service with SupervisorJob
- Service scope with Dispatchers.Default
- Automatic cleanup on service destroy

**On Message Received**:
1. Vibrate once (250ms pulse)
2. Check if "Start monitoring" message (skips fall broadcast)
3. Send "FALL_EVENT" broadcast to MainActivity
4. Launch coroutine for sensor measurement (60s timeout)
5. Register heart rate and accelerometer listeners
6. Wait for first valid heart rate reading
7. Collect sensor data asynchronously
8. Unregister listeners after timeout/completion
9. Calculate averages (mean HR, mean X/Y/Z)
10. POST data to backend API

**Sensor Configuration**:
- Heart rate: `SENSOR_DELAY_NORMAL` (updates every ~1-2 seconds)
- Accelerometer: `SENSOR_DELAY_GAME` (updates ~20ms, ~50Hz)
- First valid HR reading triggers completion signal
- Continues collecting accel data until timeout

### Broadcast Communication

**Intents**:
- `FALL_EVENT`: Signals fall detection to UI
- `HEART_RATE_UPDATE`: Live HR updates with `heart_rate` float extra
- Uses `RECEIVER_NOT_EXPORTED` flag for security

### Configuration

**Backend Endpoint**: `http://192.168.1.15:8000`
- Should match backend server IP on local network
- Requires backend and watch on same WiFi/network

**Permissions Required**:
- `BODY_SENSORS`: Heart rate sensor access
- `VIBRATE`: Notification vibration
- `INTERNET`: Backend communication
- Firebase Cloud Messaging permissions (automatic)

**Dependencies**:
- `firebase-messaging`: Cloud notifications
- `kotlinx-coroutines-android`: Async operations
- `okhttp`: HTTP client library
- `play-services-wearable`: Wear OS platform APIs
- Compose for Wear OS UI toolkit

### Sensor Reliability

**Heart Rate Challenges**:
- Sensor requires wrist contact and tight fit
- May take several seconds to get first reading
- 60-second timeout allows sensor warm-up
- Returns 0.0 if timeout expires without valid reading

**Accelerometer**:
- More reliable, immediate response
- Readings affected by wrist orientation
- Gravity component (~9.8 m/s²) visible in Z-axis when stationary

---

## Data Flow

### Complete Alert Flow

```
1. Video Frame Ingestion
   └─> MediaPipe Pose Estimation (33 landmarks per person)
       └─> Multi-Person Detector
           └─> Per-Person LiveManDownDetector
               ├─> Fall Feature Extraction
               │   └─> Sliding Window (9 frames)
               │       └─> Fall Classifier Inference
               │           └─> Consecutive Hits (5 required)
               │               └─> FALL EVENT DETECTED
               │                   ├─> Start Post-Fall Timer (60 frames)
               │                   └─> Enable Horizontal Monitoring
               │
               └─> Horizontal Feature Extraction (if post-fall timer active)
                   └─> Sliding Window (9 frames)
                       └─> Horizontal Classifier Inference
                           └─> Consecutive Hits (4 required)
                               └─> HORIZONTAL EVENT DETECTED
                                   └─> ALERT TRIGGERED

2. Alert Dispatch
   └─> push_notification.send_push_notification()
       ├─> Firebase Cloud Messaging
       │   ├─> Topic: "fall"
       │   ├─> Notification payload (title, body)
       │   └─> Data payload (action, urgency)
       │
       └─> MongoDB Alert Storage
           └─> Document: {title, message, alert: true, timestamp}

3. Mobile App Notification
   └─> Flutter App (foreground or background)
       ├─> Firebase onMessage listener
       │   └─> Show SnackBar (if foreground)
       │
       └─> Background message handler
           └─> System notification (if background/terminated)

4. Wear OS Notification
   └─> FirebaseNotificationService.onMessageReceived()
       ├─> Vibrate watch (250ms)
       ├─> Broadcast "FALL_EVENT" to MainActivity
       │   └─> Update UI to "FALL DETECTED!" (red text)
       │       └─> Play notification sound
       │
       └─> Launch sensor measurement coroutine
           ├─> Register heart rate sensor listener
           ├─> Register accelerometer sensor listener
           ├─> Wait up to 60 seconds for first HR reading
           ├─> Collect readings (HR, accel X/Y/Z)
           ├─> Calculate averages
           ├─> POST to /api/v1/measure/{bpm}
           └─> POST to /api/v1/monitor/{x}-{y}-{z}

5. Telemetry Integration
   └─> Backend receives wearable data
       └─> Updates LatestHeartbeat.BPM and LatestMovement.{X,Y,Z}
           └─> Available for future alert enrichment
               └─> send_push_notification_heartbeat() (if HR abnormal)
```

### Network Topology

```
┌─────────────────────────────────────────────────┐
│  Video Source (Camera / Video File)             │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Backend (192.168.1.15:8000)            │
│  ├─ MediaPipe Pose Estimation                   │
│  ├─ Multi-Person Fall Detection                 │
│  ├─ Alert Generation & MongoDB Storage          │
│  └─ Firebase Admin SDK                          │
└────────┬────────────────────────┬────────────────┘
         │                        │
         │ FCM Push               │ HTTP GET /alerts
         ▼                        ▼
┌─────────────────────┐   ┌──────────────────────┐
│  Firebase Cloud     │   │  Flutter Mobile App  │
│  Messaging          │   │  (Android/iOS)       │
└─────────┬───────────┘   └──────────────────────┘
          │
          ├─────────────────────┐
          │                     │
          ▼                     ▼
┌──────────────────┐   ┌─────────────────────────┐
│  Flutter Mobile  │   │  Wear OS App            │
│  App             │   │  (Smartwatch)           │
│  ├─ Alert UI     │   │  ├─ Alert UI            │
│  └─ Refresh List │   │  ├─ Heart Rate Sensor   │
└──────────────────┘   │  ├─ Accelerometer       │
                       │  └─ HTTP POST telemetry │
                       └────────────┬─────────────┘
                                    │
                                    │ POST /api/v1/measure/{bpm}
                                    │ POST /api/v1/monitor/{x}-{y}-{z}
                                    ▼
                       ┌─────────────────────────────┐
                       │  Backend Telemetry Storage  │
                       └─────────────────────────────┘
```

---

## Setup and Installation

### Backend Setup

1. **Create Python virtual environment**:
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # Windows
   source .venv/bin/activate  # Unix/MacOS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Firebase**:
   - Create Firebase project at https://console.firebase.google.com
   - Download `serviceAccountKey.json`
   - Place in `data/account/serviceAccountKey.json`

4. **Prepare models**:
   - Download MediaPipe pose model: `data/pose_landmarker_heavy.task`
   - Train or download classifiers: `data/icaro_models.joblib`
   - Optional: Run `python src/pipeline_horizontal_classification.py` to train

5. **Configure MongoDB** (optional):
   - Install MongoDB locally or use cloud service
   - Update connection string in `src/mongodb.py`

6. **Run tests**:
   ```bash
   python -m unittest discover -s test
   ```

7. **Start server**:
   ```bash
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
   ```

### Flutter Mobile App Setup

1. **Navigate to Flutter project**:
   ```bash
   cd frontend/icaro
   ```

2. **Install dependencies**:
   ```bash
   flutter pub get
   ```

3. **Configure Firebase**:
   - Add your Firebase project via FlutterFire CLI
   - Or manually place `google-services.json` (Android) and `GoogleService-Info.plist` (iOS)
   - Ensure FCM is enabled in Firebase Console

4. **Update backend endpoint**:
   - Edit `lib/main.dart`
   - Change `endpoint` constant to your backend IP

5. **Run app**:
   ```bash
   flutter run
   ```

### Wear OS App Setup

1. **Navigate to Wear OS project**:
   ```bash
   cd icaro-wearos
   ```

2. **Configure Firebase**:
   - Place `google-services.json` in `app/`
   - Ensure Wear OS device or emulator is registered in Firebase

3. **Update backend endpoint**:
   - Edit `app/src/main/java/it/unisa/icaro/presentation/FirebaseNotificationService.kt`
   - Update URLs with your backend IP

4. **Build and install**:
   ```bash
   ./gradlew installDebug
   ```
   Or use Android Studio with connected Wear OS device/emulator

5. **Grant permissions**:
   - Body sensors permission for heart rate
   - Notification permission
   - Internet access (automatic)

### Network Configuration

**Important**: All devices must be on the same local network for default configuration.

- Backend server: 192.168.1.15:8000 (update as needed)
- CORS allows: localhost, 127.0.0.1, 192.168.1.*
- For production: Configure proper domain, HTTPS, and authentication

---

## License

GPL-3.0-only. See [LICENCE.md](../LICENCE.md) for full license text.

---

## Contributing

Issues and pull requests are welcome. Please:
- Run test suite before submitting
- Follow existing code style
- Update documentation for new features
- Keep changes consistent with GPL-3.0 license

---

## Project Structure Summary

```
icaro/
├── src/                          # Backend Python source
│   ├── app.py                   # FastAPI application
│   ├── pose_landmark.py         # MediaPipe integration
│   ├── live_fall_detector.py   # Detection logic
│   ├── pipeline_horizontal_classification.py  # ML pipeline
│   ├── push_notification.py    # Firebase notifications
│   └── mongodb.py               # Database integration
├── data/                        # Models and datasets
│   ├── icaro_models.joblib     # Trained classifiers
│   ├── pose_landmarker_heavy.task  # MediaPipe model
│   └── archive/                 # Training videos
├── frontend/icaro/              # Flutter mobile app
│   ├── lib/main.dart           # App entry point
│   └── pubspec.yaml            # Dependencies
├── icaro-wearos/                # Wear OS application
│   ├── app/src/main/java/      # Kotlin source
│   └── build.gradle.kts        # Build configuration
├── test/                        # Unit tests
└── documentation/               # Project documentation
```

---

**Last Updated**: 2026-01-06
**Version**: 1.0
**Project**: ICARO – Industrial Collision & Analysis Recognition Observer
