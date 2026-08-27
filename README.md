# Pansinayan

Real-time Filipino Sign Language (FSL) recognition for Android.

Pansinayan turns live signing into gloss and category text using MediaPipe keypoints and CTC models (Transformer or GRU). Inference can run on-device with PyTorch Lite, or on a remote FastAPI server.

---

## Features

- Live camera recognition at 30 FPS
- Gloss + category output (for example `GOOD MORNING` / `GREETING`)
- Transformer (accuracy) or GRU (speed) models
- Offline (on-device) and online (server) inference, with fallback to offline
- Pose, hand, and face skeleton overlay
- Sign history with CSV export
- Optional screen recording

**Vocabulary:** 105 glosses across 10 categories — greeting, survival, number, calendar, days, family, relationships, color, food, and drink.

---

## How it works

```
Camera (30 FPS)
  → MediaPipe (89 keypoints: pose, hands, face)
  → Activity + sign-boundary detection
  → Sign-aligned window [1, T, 178]
  → CTC model (on-device or server)
  → Gloss + category on screen
```

Inference runs when a complete sign is detected, not on every frame. CTC decoding collapses per-frame predictions into glosses and categories.

---

## Models

| Model | Role | Size | Latency |
| --- | --- | --- | --- |
| Transformer CTC | Higher accuracy (default) | ~73 MB | ~150–250 ms |
| MediaPipe-GRU CTC | Faster on-device | ~10 MB | ~50–100 ms |

- **Input:** `[1, T, 178]` float32 — T is the sign-aligned window length (up to 300 frames)
- **Output:** gloss `[1, T, 106]` (105 signs + blank) and category `[1, T, 10]`

---

## Getting started

### Prerequisites

- Android Studio
- Android device or emulator, API 24+ (ARM recommended for PyTorch Mobile)
- Camera permission

### 1. Clone and open

Open the project in Android Studio and wait for Gradle sync.

### 2. Add required assets

Place these files in `app/src/main/assets/`:

**Recognition models (PyTorch Lite)**

- `SignTransformerCtc_best.ptl`
- `MediaPipeGRUCtc_best.ptl`
- `SignTransformerCtc_best.model.json` (already in the repo)
- `MediaPipeGRUCtc_best.model.json` (already in the repo)
- `label_mapping.json` (already in the repo)

**MediaPipe Tasks**

- `hand_landmarker.task`
- `pose_landmarker_full.task`
- `face_landmarker.task`

Download the Tasks models from [MediaPipe Models](https://ai.google.dev/edge/mediapipe/solutions/guide#models).

### 3. Build and run

1. Select a device
2. Run the app
3. Grant camera permission
4. Start recognition from the home screen

### Optional: online inference

The app can send keypoints to the FastAPI server in `server/`. See [server/SERVER_GUIDE.md](server/SERVER_GUIDE.md) for local setup and deployment. Set the server URL in the app settings before switching to online mode.

---

## Project structure

```
.
├── app/src/main/java/com/fslr/pansinayan/
│   ├── activities/      # Home, live recognition, history
│   ├── camera/          # CameraX capture
│   ├── mediapipe/       # 89-keypoint extraction
│   ├── recognition/     # Activity, boundaries, buffer, pipeline
│   ├── inference/       # PyTorch Lite, remote runner, CTC decode
│   ├── network/         # Server client
│   ├── database/        # Room history
│   └── views/           # Skeleton overlay
├── app/src/main/assets/ # Models, labels, MediaPipe tasks
├── models/              # PyTorch architecture definitions
├── server/              # FastAPI inference server
└── data/                # Shared label helpers
```

---

## Credits

Part of the **Pansinayan** project, built to support communication and accessibility for the Filipino Deaf community.
