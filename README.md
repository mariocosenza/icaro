# ICARO – Industrial Collision & Analysis Recognition Observer

<p align="center">
	<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
	<a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-ready-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
	<a href="https://flutter.dev/"><img src="https://img.shields.io/badge/Flutter-mobile-0468d7?logo=flutter&logoColor=white" alt="Flutter"></a>
	<a href="https://developer.android.com/wear"><img src="https://img.shields.io/badge/Kotlin-Wear%20OS-7f52ff?logo=kotlin&logoColor=white" alt="Kotlin WearOS"></a>
	<a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue" alt="GPLv3"></a>
</p>

ICARO (Industrial Collision & Analysis Recognition Observer) is a human safety stack that detects falls and abnormal postures in real time from video streams, sends alerts, and surfaces telemetry to client apps built with Flutter (mobile) and Kotlin (Wear OS). The backend uses FastAPI, MediaPipe pose estimation, and classical ML models for multi-person man-down detection.

## Features
- Multi-person fall/horizontal posture detection with pose quality checks and sliding windows.
- MediaPipe pose landmark ingestion (video file or live stream) with per-person tracking.
- FastAPI service exposing control/status endpoints and video upload.
- Notification hooks and MongoDB logging for alerts and vitals (heartbeat, movement).
- Cross-platform clients: Flutter mobile app (frontend/icaro) and Kotlin Wear OS companion (icaro-wearos).

## Repository Layout
- `src/` – FastAPI app, pose pipeline, ML models, notifications, and utilities.
- `data/` – Pose landmark models and sample videos (archive/), calibration outputs, joblib classifiers.
- `frontend/icaro/` – Flutter app for monitoring and alerting.
- `icaro-wearos/` – Kotlin Wear OS project for wrist-worn alerts/telemetry.
- `test/` – Unit tests for detectors and helpers.

## Backend (FastAPI + MediaPipe)
### Quick start
```bash
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt                 # ensure mediapipe/torch deps are present
python -m unittest discover -s test             # run tests
uvicorn src.app:app --reload                    # start API
```

### Key components
- `src/app.py` – FastAPI entrypoint; starts/stops pose pipeline, handles uploads, exposes alerts/vitals APIs.
- `src/pose_landmark.py` – MediaPipe pose runner (video/live), feeds landmarks into detection pipeline.
- `src/live_fall_detector.py` – Sliding-window fall/horizontal classifier, multi-person tracker, thresholds.
- `src/pipeline_horizontal_classification.py` – Feature extraction, quality scoring, model loading utilities.
- `src/push_notification.py`, `src/mongodb.py` – Alert delivery and persistence hooks.

### API overview
- `GET /api/v1/status` – Service status and active video path.
- `POST /api/v1/start` / `POST /api/v1/stop` – Control the pose pipeline.
- `PUT /api/v1/running-mode/live-stream|video` – Switch inference mode.
- `POST /api/v1/upload` – Upload a video to run detection.
- `POST /api/v1/measure/{heartbeat}` – Push latest BPM.
- `POST /api/v1/monitor/{x}-{y}-{z}` – Push latest motion vector.
- `GET /api/v1/alerts` – Retrieve stored alerts.

## Flutter Mobile App (frontend/icaro)
- Built with Flutter; Firebase config present under `frontend/icaro/android/app/google-services.json` and `firebase.json`.
- Provides a monitoring UI for alerts, vitals, and notifications from the backend.
- To run:
	```bash
	cd frontend/icaro
	flutter pub get
	flutter run
	```
	Configure your Firebase project as needed (see `frontend/icaro/firebase.json`).

## Kotlin Wear OS App (icaro-wearos)
- Wear OS companion delivering wrist alerts/telemetry.
- Standard Gradle project (`icaro-wearos/` with `app/build.gradle.kts`).
- To run:
	```bash
	cd icaro-wearos
	./gradlew installDebug  # connect emulator or device
	```

## Data & Models
- MediaPipe task files: `data/pose_landmarker_heavy.task`, `gesture_recognizer.task`.
- Trained classifiers: `data/icaro_models.joblib` (fall/horizontal models).
- Sample videos for testing: `data/archive/`.

## Testing
```bash
python -m unittest discover -s test
```
Tests cover fall detector logic, feature extraction, and helper utilities with mocks for the ML models.

## Contributing
Issues and PRs are welcome. Please keep changes consistent with the existing licensing and run the test suite before submitting.

## License
GPL-3.0-only. See [LICENCE.md](LICENCE.md).