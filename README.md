# Pansinayan-Mobile

**Real-time Filipino Sign Language (FSL) Recognition on Android**

This repository contains the official Android application for **Pansinayan**, a system that translates Filipino Sign Language into text in real-time using CTC-based continuous recognition.
The app uses the device's camera, **MediaPipe** for keypoint extraction, and optimized **CTC TFLite models (Transformer/GRU)** for continuous sign recognition.

This project provides the complete, production-ready Android scaffolding code for CTC-based continuous sign language recognition.

---

## 🚀 Features

- **Continuous Recognition:** Uses CTC models for fluent, continuous sign language recognition without pauses
- **Dual-Model Support:** Real-time switching between Transformer CTC (high accuracy) and GRU CTC (lightweight) models
- **Advanced Keypoint Extraction:** MediaPipe extracts 89 keypoints (25 pose + 21 left hand + 21 right hand + 22 face) = 178 data points
- **Skeleton Visualization:** Real-time overlay showing pose (green) and hands (yellow). Toggle on/off, long-press for debug mode
- **Occlusion Detection:** UI indicator when hand blocks face (affects accuracy)
- **Continuous Transcript:** Displays continuous recognition results with phrase segmentation
- **Persistent History:** All recognitions saved to Room database for review
- **CSV Export:** Export recognition history for analysis

---

## ⚙️ System Architecture

The app's CTC recognition pipeline is designed for continuous sign language recognition:

```
Camera (CameraX @ 30 FPS)
   → MediaPipeProcessor (Extracts 89 keypoints × 2 = 178D)
   → CTCSequenceBufferManager (Builds 300-frame rolling window = 10 seconds)
   → CTCModelInterpreter (Runs Transformer/GRU CTC model)
   → CTCDecoder (Greedy CTC decoding with blank token removal)
   → ContinuousRecognitionManager (Orchestrates continuous recognition)
   → UI Update (Continuous transcript building)
```

**CTC Recognition Features:**

- **Continuous Recognition:** Predicts multiple signs per window without pauses
- **Rolling Buffer:** 300-frame window (10 seconds) for continuous processing
- **CTC Decoding:** Greedy algorithm with blank token removal
- **Phrase Segmentation:** Automatic phrase boundary detection

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

- **CTCSequenceBufferManager** (`inference/CTCSequenceBufferManager.kt`) - Rolling window buffer:
  - Window: 300 frames (10 seconds at 30 FPS)
  - Gap interpolation for missing frames
  - Minimum 30 frames before inference
  - Continuous processing without resets

### **3. CTC Model Inference**

- **CTCModelInterpreter** (`inference/CTCModelInterpreter.kt`) - CTC TFLite inference engine:
  - Input: `[1, 300, 178]` (batch, time, features)
  - Output: `[1, 300, 106]` (batch, time, classes=105 glosses + 1 blank)
  - GPU acceleration with fallback to CPU
  - Fixed sequence length with zero-padding
- **CTCDecoder** (`inference/CTCDecoder.kt`) - CTC greedy decoder:
  - Collapses repeated predictions
  - Removes blank tokens (ID 105)
  - Maps remaining indices to gloss strings
  - Confidence scoring support

### **4. Continuous Recognition Logic**

- **ContinuousRecognitionManager** (`recognition/ContinuousRecognitionManager.kt`) - Orchestrates CTC flow:
  - Runs inference every 15 frames (~0.5s intervals)
  - Manages continuous transcript building
  - Health monitoring with auto-recovery
  - Thread-safe coroutine-based processing
  - Recording state management

### **5. UI & Data Persistence**

- **MainActivity** (`activities/MainActivity.kt`) - Main CTC recognition screen:
  - Live camera preview
  - Continuous transcript display
  - CTC model switching (Transformer/GRU)
  - Screen recording integration
- **HistoryActivity** (`activities/HistoryActivity.kt`) - Recognition history with CSV export
- **AppDatabase** (`database/AppDatabase.kt`) - Room database for persistence
- **LabelMapper** (`utils/LabelMapper.kt`) - Maps IDs to human-readable labels

---

## 📊 Model Specifications

### Input

- **Shape:** `[1, 300, 178]`
- **Type:** Float32
- **Content:** 300 frames of 89 keypoints (x, y coordinates)

### Output (CTC)

- **Gloss:** `[1, 300, 106]` - Per-frame predictions (105 signs + 1 blank token)
- **Blank Token:** Index 105 represents silence/no sign

### Supported Models

- **Transformer CTC:** High accuracy, ~200-400ms inference
- **MediaPipe-GRU CTC:** Lightweight, ~100-200ms inference
- Both models support continuous recognition with CTC decoding

---

## 🧩 Getting Started

Follow these steps to build and run the project on your local machine.

### Prerequisites

- Android Studio (latest version)
- Android device or Emulator (API 24+)
- CTC-trained PyTorch models (Transformer and GRU)

---

### Step 1: Prepare CTC Models

This repository is designed for CTC-based continuous sign language recognition. You'll need to provide your own CTC-trained models.

#### CTC Model Requirements

- **Input Shape:** `[1, 300, 178]` (batch_size=1, sequence_length=300, features=178)
- **Output Shape:** `[1, 300, 106]` (batch_size=1, sequence_length=300, classes=105 glosses + 1 blank)
- **Quantization:** FP16 for mobile optimization
- **Keypoint Structure:** 89 keypoints × 2 coordinates = 178 features

#### Model Files Needed

Place your CTC models in the following locations:

```
app/src/main/assets/ctc/
├── sign_transformer_ctc_fp16.tflite
└── mediapipe_gru_ctc_fp16.tflite
```

**Note:** The previous classification model export pipeline has been removed. This repository now focuses exclusively on CTC-based continuous recognition.

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

### Step 3: Download MediaPipe Models

The app also requires MediaPipe’s base models for pose and hand tracking.

Download the following files and place them in `app/src/main/assets/`:

- `hand_landmarker.task`
- `pose_landmarker_full.task`

Your `assets/` folder should now contain **five files total**.

---

### Step 4: Build and Run

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
│   │   │   │   ├── activities/        # MainActivity, HistoryActivity
│   │   │   │   ├── adapter/           # RecyclerView adapter for history
│   │   │   │   ├── camera/            # CameraManager (CameraX setup)
│   │   │   │   ├── database/          # Room database (AppDatabase, HistoryDao)
│   │   │   │   ├── inference/         # CTCModelInterpreter, CTCDecoder, CTCSequenceBufferManager
│   │   │   │   ├── mediapipe/         # MediaPipeProcessor (keypoint extraction)
│   │   │   │   ├── recognition/       # ContinuousRecognitionManager
│   │   │   │   ├── services/          # ScreenRecordService (video recording)
│   │   │   │   ├── utils/             # LabelMapper, ModelSelector
│   │   │   │   ├── views/             # OverlayView (skeleton visualization)
│   │   │   │   └── res/               # Layouts, drawables, values
│   │   │   └── assets/                # CTC TFLite models, MediaPipe tasks, label mappings
│   │   └── ...
│   ├── build.gradle
│   └── ...
│
├── archived_models/                   # Archived classification models
├── data/                              # Label mapping utilities
├── models/                            # Model storage directory
├── scripts/                           # Utility scripts
└── README.md
```

---

## 📄 License

This project is released under an open license (add your license here, e.g. MIT, Apache 2.0).

---

## 💡 Credits

Developed as part of the **Pansinayan Project**, dedicated to improving accessibility and communication for the Filipino Deaf community.
