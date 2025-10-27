# Pansinayan-Mobile

**Real-time Filipino Sign Language (FSL) Recognition on Android**

**Pansinayan** translates Filipino Sign Language into text in real-time using MediaPipe keypoint extraction and optimized Transformer/GRU models. This project includes the complete Android app and model export scripts.

---

## 🚀 Features

- **Real-time Recognition:** 30 FPS camera feed with MediaPipe keypoint extraction
- **Continuous CTC:** Sliding window (150 frames) processes multiple signs per sequence
- **Dual Output:** Displays both **gloss** (word) and **category** for each sign
- **Dual-Model Support:** Switch between Transformer (accuracy) and GRU (speed) models
- **Skeleton Visualization:** Real-time pose and hand overlay with toggle
- **Transcript & History:** Shows last 5 signs and saves to database with CSV export

---

## ⚙️ System Architecture

```
Camera (30 FPS) → MediaPipe (89 keypoints) → Sliding Window (150 frames)
→ CTC Model → Dual Decoder → UI (Gloss + Category)
```

**How It Works:**

- **Sliding Window:** 150-frame window (5 seconds) slides forward by 10 frames every ~0.33s
- **Continuous Recognition:** Model processes overlapping windows to detect multiple signs
- **CTC Decoding:** Collapses frame-level predictions into sign sequences
- **Output:** Both gloss ("GOOD MORNING") and category ("GREETINGS") per sign

Example: Sign "GOOD MORNING EGG" → Outputs: "GOOD MORNING" [GREETINGS], "EGG" [FOOD]

---

## 🧩 Core Components

### **1. Input Processing**

- **MediaPipeProcessor** - Extracts 89 keypoints (25 pose + 21×2 hands + 22 face)
- **CameraManager** - CameraX 30 FPS capture with overlay visualization

### **2. Sequence Management**

- **SequenceBufferManager** - Maintains sliding 150-frame buffer (5 seconds)
- Slides forward by 10 frames every ~0.33s for continuous recognition

### **3. Model Inference**

- **TFLiteModelRunner** - Runs CTC models (Input: `[1, 150, 178]` → Output: per-frame gloss + category)
- **CTCDecoder** - Decodes CTC outputs: collapses repeats, removes blanks, segments signs

### **4. Recognition Pipeline**

- **RecognitionPipeline** - Orchestrates sliding window inference with health monitoring
- Runs every 10 frames, processes overlapping windows, outputs gloss + category pairs

### **5. UI & Persistence**

- **MainActivity** - Live preview, transcript, model switching
- **HistoryActivity** - Recognition history with CSV export
- **AppDatabase** - Room database for storing all recognitions

---

## 📊 Model Specifications

### Input

- **Shape:** `[1, 150, 178]` (150 frames × 89 keypoints × 2 coordinates)
- **Type:** Float32
- **Content:** MediaPipe keypoints over 5-second sliding window

### Output

- **Gloss:** `[1, 150, 106]` - Per-frame predictions (105 signs + 1 blank)
- **Category:** `[1, 150, 10]` - Per-frame predictions (10 categories)
- **Decoded:** Both decoded independently via CTC, aligned by frame timing

### Models

- **Transformer** - High accuracy (~60-70ms inference)
- **MediaPipe-GRU** - Lightweight (~40-50ms inference)

---

## 🧩 Getting Started

### Prerequisites

- Android Studio + Android device/emulator (API 24+)
- Python 3.8+ with required packages (see `scripts/requirements.txt`)

---

### Step 1: Export Your Models

1. **Install dependencies:**

   ```bash
   python3 -m venv venv  # Optional, for Ubuntu
   source venv/bin/activate
   pip install -r scripts/requirements.txt
   ```

2. **Place your PyTorch checkpoints** in the project root:

   - `SignTransformerCtc_best.pt`
   - `MediaPipeGRUCtc_best.pt`

3. **Run export script:**

   ```bash
   cd scripts
   python export_ctc_models.py --model both
   ```

   Options: `--model [transformer|mediapipe_gru|both]`, `--checkpoint <path>`, `--skip-quantization`

4. **Generated files** in `models/converted/`:
   - `sign_transformer_quant.tflite` ⭐ (use this)
   - `sign_mediapipe_gru_quant.tflite` ⭐
   - `label_mapping.json`

---

### Step 2: Android Setup

1. **Open in Android Studio** - Import project from version control or extract ZIP
2. **Wait for Gradle sync** - Download dependencies (~5 minutes first time)
3. **Enable USB Debugging** (physical device):
   - Settings → About Phone → Tap Build Number 7x
   - Developer Options → Enable USB Debugging

---

### Step 3: Copy Model Files

```bash
cp models/converted/*.tflite app/src/main/assets/
cp models/converted/label_mapping.json app/src/main/assets/
```

### Step 4: Download MediaPipe Models

Download and place in `app/src/main/assets/`:

- `hand_landmarker.task`
- `pose_landmarker_full.task`

Your `assets/` folder should contain 5 files total.

### Step 5: Build and Run

1. Verify Gradle sync complete
2. Select device from dropdown (physical device or emulator)
3. Click **Run** ▶️
4. Grant camera permission when prompted

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
│   │   │   │   ├── recognition/       # RecognitionPipeline
│   │   │   │   ├── services/          # ScreenRecordService (video recording)
│   │   │   │   ├── utils/             # LabelMapper, ModelSelector
│   │   │   │   ├── views/             # OverlayView (skeleton visualization)
│   │   │   │   └── res/               # Layouts, drawables, values
│   │   │   └── assets/                # TFLite models, MediaPipe tasks, label mappings
│   │   └── ...
│   ├── build.gradle
│   └── ...
│
├── scripts/
│   ├── export_ctc_models.py           # Python script to export CTC models
│   └── requirements.txt
└── README.md
```

---

## 💡 Credits

Developed as part of the **Pansinayan Project**, dedicated to improving accessibility and communication for the Filipino Deaf community.
