# Pansinayan-Mobile

**Real-time Filipino Sign Language (FSL) Recognition on Android**

This repository contains the official Android application for **Pansinayan**, a system that translates Filipino Sign Language into text in real-time.
The app uses the device's camera, **MediaPipe** for keypoint extraction, and optimized **TFLite models (Transformer/GRU)** for live inference.

This project provides the complete, production-ready Android scaffolding code and the Python scripts needed to export your own trained models for deployment.

---

## 🚀 Features

- **Live Recognition:** Uses CameraX for a high-performance, real-time camera feed at 30 FPS.
- **Dual Recognition Modes:**
  - **Classification:** Isolated sign recognition (one sign per 5-second window)
  - **CTC:** Continuous sign recognition (multiple signs per window, fluent signing)
- **Dual-Model Support:** Real-time switching between Transformer (high accuracy) and GRU (lightweight) models.
- **Advanced Keypoint Extraction:** MediaPipe extracts 89 keypoints (25 pose + 21 left hand + 21 right hand + 22 face) = 178 data points.
- **Skeleton Visualization:** Real-time overlay showing pose (green) and hands (yellow). Toggle on/off, long-press for debug mode.
- **Occlusion Detection:** UI indicator when hand blocks face (affects accuracy).
- **Transcript Building:** Displays last 5 recognized signs with timestamps and confidence scores.
- **Persistent History:** All recognitions saved to Room database for review.
- **CSV Export:** Export recognition history for analysis.

---

## ⚙️ System Architecture

The app's recognition pipeline is designed for high performance and low latency:

```
Camera (CameraX @ 30 FPS)
   → MediaPipeProcessor (Extracts 89 keypoints × 2 = 178D)
   → SequenceBufferManager (Builds 150-frame sliding window = 5 seconds)
   → TFLiteModelRunner (Runs Transformer/GRU model)
   → Classification: TemporalRecognizer (Filters results) → Single sign
   → CTC: CTCDecoder (Segments sequence) → Multiple signs
   → UI Update (Transcript building)
```

**Dual Recognition Modes:**

- **Classification:** Predicts one sign per 5-second window (isolated signs with pauses)
- **CTC:** Predicts multiple signs per window (continuous fluent signing)

---

## 🧩 Core Components

### **1. Camera & Input Processing**

- **CameraManager** (`camera/CameraManager.kt`) - CameraX setup, 30 FPS capture, front/back switching
- **MediaPipeProcessor** (`mediapipe/MediaPipeProcessor.kt`) - Extracts 89 keypoints:
  - 25 pose points (body, shoulders, arms)
  - 21 left hand points
  - 21 right hand points
  - 22 face points (lips, eyes, eyebrows)
- **OverlayView** (`views/OverlayView.kt`) - Real-time skeleton visualization

### **2. Sequence Management**

- **SequenceBufferManager** (`inference/SequenceBufferManager.kt`) - Sliding window buffer:
  - Window: 150 frames (5 seconds at 30 FPS)
  - Gap interpolation for missing frames
  - Minimum 50 frames before inference

### **3. Model Inference**

- **TFLiteModelRunner** (`inference/TFLiteModelRunner.kt`) - TFLite inference engine:
  - Auto-detects model type (Classification vs CTC)
  - Input: `[1, 150, 178]` (batch, time, features)
  - Classification output: `[1, 105]` gloss + `[1, 10]` category
  - CTC output: `[1, 150, 106]` gloss + `[1, 150, 10]` category (per-frame)
  - GPU acceleration with fallback to CPU
- **CTCDecoder** (`inference/CTCDecoder.kt`) - CTC greedy decoder:
  - Collapses repeated predictions
  - Removes blank tokens (ID 105)
  - Assigns categories via frame distribution + majority voting

### **4. Recognition Logic**

- **RecognitionPipeline** (`recognition/RecognitionPipeline.kt`) - Orchestrates entire flow:
  - Runs inference every 10 frames (~0.33s intervals)
  - Routes to Classification or CTC handler
  - Health monitoring with auto-recovery
  - Thread-safe coroutine-based processing
- **TemporalRecognizer** (`recognition/TemporalRecognizer.kt`) - Classification smoothing:
  - Requires 5 consecutive same predictions
  - Confidence threshold: 60%
  - Cooldown: 1000ms between emissions
  - Filters noise and jitter

### **5. UI & Data Persistence**

- **MainActivity** (`activities/MainActivity.kt`) - Main recognition screen:
  - Live camera preview
  - Real-time prediction display
  - Transcript (last 5 signs)
  - Model switching (Transformer/GRU)
  - Screen recording integration
- **HistoryActivity** (`activities/HistoryActivity.kt`) - Recognition history with CSV export
- **AppDatabase** (`database/AppDatabase.kt`) - Room database for persistence
- **LabelMapper** (`utils/LabelMapper.kt`) - Maps IDs to human-readable labels

---

## 📊 Model Specifications

### Input

- **Shape:** `[1, 150, 178]`
- **Type:** Float32
- **Content:** 150 frames of 89 keypoints (x, y coordinates)

### Output (Classification)

- **Gloss:** `[1, 105]` - One of 105 FSL signs
- **Category:** `[1, 10]` - One of 10 semantic categories

### Output (CTC)

- **Gloss:** `[1, 150, 106]` - Per-frame predictions (105 signs + 1 blank)
- **Category:** `[1, 150, 10]` - Per-frame category predictions

### Supported Models

- **Transformer:** High accuracy, ~60-70ms inference
- **MediaPipe-GRU:** Lightweight, ~40-50ms inference
- Both support Classification and CTC modes (auto-detected)

---

## 🧩 Getting Started

Follow these steps to build and run the project on your local machine.

### Prerequisites

- Android Studio (latest version)
- Android device or Emulator (API 24+)
- Python 3.8+
- Required Python libraries: `torch`, `onnx`, `onnx-tf`, `tensorflow`, `numpy`, `pandas`

---

### Step 1: Export Your Models

Before building the app, export your trained PyTorch models to the TFLite format.

Install Python dependencies:

**Note:**
On Ubuntu systems, you may encounter an "externally-managed-environment" error. To fix this, create a virtual environment first:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install torch onnx onnx-tf tensorflow numpy pandas
```

**For other systems or if you prefer not to use a virtual environment:**

```bash
pip install torch onnx onnx-tf tensorflow numpy pandas
```

Run the export script (this converts your models and generates `label_mapping.json`):

```bash
python android/model_export_and_setup.py --model both
```

This will create a folder (e.g. `android/android_models/`) containing:

```
sign_transformer_quant.tflite
sign_mediapipe_gru_quant.tflite
label_mapping.json
```

---

### Step 2: Set Up the Android Project

1. **Install Android Studio**

   - Download and install the latest version of [Android Studio](https://developer.android.com/studio)
   - During installation, ensure the Android SDK, Android SDK Platform, and Android Virtual Device components are installed

2. **Open the Project**

   - Open **Android Studio**
   - Select **File > New > Project from Version Control**, then enter your repository URL
   - Alternatively, download the ZIP, extract it, and select **File > Open**, then navigate to the extracted folder

3. **Wait for Gradle Sync**

   - Android Studio will automatically sync the project and download dependencies
   - This may take a few minutes on the first run
   - If sync fails, click **Sync Project with Gradle Files** (elephant icon in the toolbar)

4. **Enable USB Debugging on Your Android Device** (for testing on a real phone)
   - On your Android phone, go to **Settings > About Phone**
   - Tap **Build Number** 7 times to enable Developer Options
   - Go back to **Settings > System > Developer Options** (or **Settings > Developer Options** on some devices)
   - Enable **USB Debugging**
   - Connect your phone to your computer via USB cable
   - On your phone, allow USB debugging when prompted (tap "OK" or "Allow")
   - In Android Studio, your device should appear in the device dropdown at the top

---

### Step 3: Copy Model Files

Copy the three generated files from Step 1 into:

```
app/src/main/assets/
```

Files:

- `sign_transformer_quant.tflite`
- `sign_mediapipe_gru_quant.tflite`
- `label_mapping.json`

---

### Step 4: Download MediaPipe Models

The app also requires MediaPipe’s base models for pose and hand tracking.

Download the following files and place them in `app/src/main/assets/`:

- `hand_landmarker.task`
- `pose_landmarker_full.task`

Your `assets/` folder should now contain **five files total**.

---

### Step 5: Build and Run

1. **Verify Gradle Sync**

   - Ensure Android Studio has finished syncing Gradle (check the bottom status bar)
   - If needed, click **Sync Project with Gradle Files** (elephant icon in the toolbar)

2. **Select Your Device**

   - **For Physical Device:** If you enabled USB debugging in Step 2, your phone should appear in the device dropdown at the top
   - **For Emulator:** Click the device dropdown → **Device Manager** → **Create Device** to set up an Android Virtual Device (AVD)
   - Select your target device from the dropdown menu

3. **Build and Run**

   - Click the **Run 'app'** button (green play icon) in the toolbar, or press **Shift + F10**
   - Android Studio will build the APK and install it on your device/emulator
   - First build may take several minutes

4. **Grant Permissions**

   - When the app launches, grant camera permission when prompted
   - The app requires camera access to capture sign language gestures

5. **Troubleshooting**
   - If your device doesn't appear, check USB debugging is enabled and cable is properly connected
   - Try revoking USB debugging authorizations on your phone and reconnecting
   - For emulator issues, ensure you have sufficient RAM and storage allocated in AVD settings

---

## 🎨 Using the Skeleton Overlay

The skeleton overlay helps you visualize keypoints in real-time:

1. **Toggle Visibility**: Use the "Show Skeleton" switch (bottom-left) to show/hide keypoints
2. **Default State**: The overlay is enabled by default for easier debugging
3. **Debug Mode**: Long-press the skeleton toggle to enable debug information overlay
4. **Color Coding**:
   - **Green**: Body pose keypoints (face, shoulders, arms, torso)
   - **Yellow**: Hand keypoints (21 points per hand)

---

## 🧱 Project Structure

```
.
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/fslr/pansinayan/
│   │   │   │   ├── activities/        # MainActivity, HistoryActivity, HomeActivity
│   │   │   │   ├── adapter/           # RecyclerView adapter for history
│   │   │   │   ├── camera/            # CameraManager (CameraX setup)
│   │   │   │   ├── database/          # Room database (AppDatabase, HistoryDao)
│   │   │   │   ├── inference/         # TFLiteModelRunner, CTCDecoder, SequenceBufferManager
│   │   │   │   ├── mediapipe/         # MediaPipeProcessor (keypoint extraction)
│   │   │   │   ├── recognition/       # RecognitionPipeline, TemporalRecognizer
│   │   │   │   ├── services/          # ScreenRecordService (video recording)
│   │   │   │   ├── utils/             # LabelMapper, ModelSelector
│   │   │   │   ├── views/             # OverlayView (skeleton visualization)
│   │   │   │   └── res/               # Layouts, drawables, values
│   │   │   └── assets/                # TFLite models, MediaPipe tasks, label mappings
│   │   └── ...
│   ├── build.gradle
│   └── ...
│
├── model_export_and_setup.py           # Python script to export models
└── README.md
```

---

## 📄 License

This project is released under an open license (add your license here, e.g. MIT, Apache 2.0).

---

## 💡 Credits

Developed as part of the **Pansinayan Project**, dedicated to improving accessibility and communication for the Filipino Deaf community.
