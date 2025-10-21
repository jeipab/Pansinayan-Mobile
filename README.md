# Pansinayan-Mobile

**Real-time Filipino Sign Language (FSL) Recognition on Android**

This repository contains the official Android application for **Pansinayan**, a system that translates Filipino Sign Language into text in real-time.
The app uses the device's camera, **MediaPipe** for keypoint extraction, and optimized **TFLite models (Transformer/GRU)** for live inference.

This project provides the complete, production-ready Android scaffolding code and the Python scripts needed to export your own trained models for deployment.

---

## 🚀 Features

- **Live Recognition:** Uses CameraX for a high-performance, real-time camera feed.
- **Dual-Model Support:** Allows real-time switching between a high-accuracy Transformer model and a lightweight GRU model.
- **Advanced Keypoint Extraction:** Employs MediaPipe for robust extraction of 78 keypoints (pose, left hand, right hand, and face), totaling 156 data points.
- **Skeleton Visualization:** Real-time skeleton overlay showing pose (green) and hand keypoints (yellow) for debugging and verification. Enabled by default with debug mode available via long-press.
- **Occlusion Detection:** UI indicator shows when the user's hand blocks their face (which can affect accuracy).
- **Persistent History:** All recognitions are saved to a local Room database for review.
- **CSV Export:** Users can export their recognition history as a CSV file for analysis.

---

## ⚙️ System Architecture

The app’s recognition pipeline is designed for high performance and low latency:

```
Camera (CameraX)
   → MediaPipeProcessor (Extracts 156 keypoints)
   → SequenceBufferManager (Builds 90-frame window)
   → TFLiteModelRunner (Runs Transformer/GRU model)
   → TemporalRecognizer (Filters results)
   → UI Update
```

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

1. Open **Android Studio**.
2. Select **File > New > Project from Version Control**, then enter your repository URL.

   - Alternatively, download the ZIP, extract it, and open this folder directly in Android Studio.

3. In the project panel, right-click `app/src/main/` → **New > Folder > Assets Folder** → **Finish**.

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

1. Wait for Android Studio to sync Gradle. If not, click **Sync Project with Gradle Files** (small elephant icon).
2. Connect your Android device or start an emulator.
3. Click the **Run 'app'** button (green play icon).
4. Grant camera permission when prompted.

---

## 🎨 Using the Skeleton Overlay

The skeleton overlay helps you visualize keypoints in real-time:

1. **Toggle Visibility**: Use the "Show Skeleton" switch (bottom-left) to show/hide keypoints
2. **Default State**: The overlay is enabled by default for easier debugging
3. **Debug Mode**: Long-press the skeleton toggle to enable debug information overlay
4. **Color Coding**:
   - **Green**: Body pose keypoints (face, shoulders, arms, torso)
   - **Yellow**: Hand keypoints (21 points per hand)

For detailed information, see [SKELETON_OVERLAY_GUIDE.md](SKELETON_OVERLAY_GUIDE.md).

---

## 🧱 Project Structure

```
.
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/fslr/pansinayan/
│   │   │   │   ├── activities/        # Home, Main, and History screens
│   │   │   │   ├── adapter/           # RecyclerView adapter for history
│   │   │   │   ├── camera/            # CameraX setup and management
│   │   │   │   ├── database/          # Room database for recognition history
│   │   │   │   ├── inference/         # TFLiteModelRunner and SequenceBufferManager
│   │   │   │   ├── mediapipe/         # MediaPipeProcessor for keypoint extraction
│   │   │   │   ├── recognition/       # Core pipeline and temporal logic
│   │   │   │   ├── utils/             # LabelMapper and ModelSelector
│   │   │   │   ├── views/             # OverlayView for skeleton drawing
│   │   │   │   └── res/               # Layouts, drawables, and values
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
