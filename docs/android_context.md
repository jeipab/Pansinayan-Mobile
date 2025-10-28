# Android Project Context — Pansinayan-Mobile

## 1. Architecture Overview

- The app uses a modular, plain Activities + Services structure with feature-focused packages (`camera`, `mediapipe`, `inference`, `recognition`, `database`, `utils`, `views`, `activities`). Not MVVM/MVP; no ViewModels or Repositories detected.
- UI is primarily XML with view binding. Jetpack Compose is enabled and theme files exist, but no composable screens are present or used.
- Core real-time pipeline is orchestrated by `recognition/RecognitionPipeline.kt`, which coordinates CameraX, MediaPipe Tasks (pose/hand/face), buffering, CTC model inference, decoding, and UI callbacks.
- Data persistence uses Room (`database` package) for recognition history.
- A foreground `ScreenRecordService` provides screen recording, integrated with the pipeline via broadcasts.

Packages:

- `activities/`: `HomeActivity`, `MainActivity`, `HistoryActivity`
- `adapter/`: RecyclerView adapter for history
- `camera/`: CameraX wrapper
- `database/`: Room entity/dao/db
- `inference/`: CTC runner, decoder, buffer, legacy TFLite runner, metadata
- `io/`: NPZ/NPY reader
- `mediapipe/`: MediaPipe keypoint extraction and occlusion detection
- `recognition/`: Pipeline, aggregator, temporal recognizer
- `services/`: Screen recording service
- `ui/theme/`: Compose theme files (unused for screens)
- `utils/`: Labels, model selector, screen recorder
- `views/`: Skeleton overlay view

## 2. Implemented Components

| Component                            | Class/File                                                                                         | Status         | Notes                                                                                                                                                                                                                |
| ------------------------------------ | -------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MediaPipe Holistic (pose/hands/face) | `app/src/main/java/com/fslr/pansinayan/mediapipe/MediaPipeProcessor.kt`                            | Complete       | Uses Tasks Vision for Pose, Hand, Face; returns 178 features; includes hand-face occlusion detection with temporal filtering; restart logic for freeze recovery.                                                     |
| Camera capture                       | `app/src/main/java/com/fslr/pansinayan/camera/CameraManager.kt`                                    | Complete       | CameraX preview + ImageAnalysis with FPS throttling; timeout monitor and auto-recovery; front/back switching.                                                                                                        |
| Sliding window buffer                | `app/src/main/java/com/fslr/pansinayan/inference/SequenceBufferManager.kt`                         | Complete       | Circular buffer; min-length gating; linear interpolation for gaps; window pop with stride and missing-ratio gate.                                                                                                    |
| CTC model runner                     | `app/src/main/java/com/fslr/pansinayan/inference/CtcModelRunner.kt`                                | Complete       | Loads TFLite CTC model + metadata; dynamic shape handling; outputs log-probs and optional categorical logits; GPU delegate optional.                                                                                 |
| Model metadata loader                | `app/src/main/java/com/fslr/pansinayan/inference/ModelMetadata.kt`                                 | Complete       | Loads `*.model.json` from assets; provides input_dim, num_ctc, blank_id, window/stride hints.                                                                                                                        |
| Greedy decoding                      | `app/src/main/java/com/fslr/pansinayan/inference/CtcGreedyDecoder.kt`                              | Complete       | Greedy CTC decode with average prob confidence per token; uses provided `blank_id`.                                                                                                                                  |
| CTC token aggregation                | `app/src/main/java/com/fslr/pansinayan/recognition/CtcAggregator.kt`                               | Complete       | Aggregates overlapping window tokens using IoU threshold; emits newly-added tokens.                                                                                                                                  |
| Recognition pipeline                 | `app/src/main/java/com/fslr/pansinayan/recognition/RecognitionPipeline.kt`                         | Complete       | Orchestrates camera→mediapipe→buffer→CTC→decode→aggregate; health monitor; manual restart hooks; emits UI callbacks with labels via `LabelMapper`. TemporalRecognizer is instantiated but not used in emission path. |
| Label mapping                        | `app/src/main/java/com/fslr/pansinayan/utils/LabelMapper.kt`                                       | Complete       | Loads `assets/label_mapping.json`; gloss and category lookups.                                                                                                                                                       |
| NPZ/NPY reader                       | `app/src/main/java/com/fslr/pansinayan/io/NpyNpzReader.kt`                                         | Complete       | Reads golden `.npz` test sequences (`X.npy`) into [T,178].                                                                                                                                                           |
| Room database                        | `app/src/main/java/com/fslr/pansinayan/database/*`                                                 | Complete       | `RecognitionHistory` entity, `HistoryDao`, `AppDatabase`. Used in `MainActivity` and `HistoryActivity`.                                                                                                              |
| History UI                           | `app/src/main/java/com/fslr/pansinayan/activities/HistoryActivity.kt`, `adapter/HistoryAdapter.kt` | Complete       | Lists history, filters by model, clear all, export CSV.                                                                                                                                                              |
| Main recognition UI                  | `app/src/main/java/com/fslr/pansinayan/activities/MainActivity.kt`                                 | Complete       | Binds pipeline to camera preview; displays result, confidence, transcript, stats; skeleton overlay; occlusion indicator; model radio buttons are UI-only (no pipeline switching yet).                                |
| Home/launcher UI                     | `app/src/main/java/com/fslr/pansinayan/activities/HomeActivity.kt`                                 | Complete       | Navigation to recognition and history screens.                                                                                                                                                                       |
| Skeleton overlay                     | `app/src/main/java/com/fslr/pansinayan/views/OverlayView.kt`                                       | Complete       | Draws pose, hands, and minimal face landmarks with connections.                                                                                                                                                      |
| Screen recording service             | `app/src/main/java/com/fslr/pansinayan/services/ScreenRecordService.kt`                            | Complete       | Foreground service with notifications; broadcasts start/stop; integrates with pipeline; uses `utils/ScreenRecorder`.                                                                                                 |
| Temporal recognizer                  | `app/src/main/java/com/fslr/pansinayan/recognition/TemporalRecognizer.kt`                          | Partial        | Implemented but not wired into current CTC emission path (aggregation is used instead).                                                                                                                              |
| Model selection                      | `app/src/main/java/com/fslr/pansinayan/utils/ModelSelector.kt`                                     | Partial        | Supports selection and metrics, but references classification paths; not integrated with pipeline. Radio buttons in `MainActivity` show TODO for real switching.                                                     |
| Legacy TFLite classification runner  | `app/src/main/java/com/fslr/pansinayan/inference/TFLiteModelRunner.kt`                             | Partial/Legacy | Single-label classifier (non-CTC). Not used by pipeline; default path points to `classification/...` which is not present in assets.                                                                                 |
| Android test parity                  | `app/src/androidTest/java/com/fslr/pansinayan/CtcGoldenParityTest.kt`                              | Partial        | Present to compare parity with golden assets; depends on `assets/golden/*`.                                                                                                                                          |

Notes on missing connections/TODOs:

- `MainActivity.switchModel()` has a TODO to implement actual model switching inside `RecognitionPipeline`.
- `TemporalRecognizer` is initialized/reset in pipeline but not in the emission loop; tokens are emitted from `CtcAggregator` directly.
- `TFLiteModelRunner` appears unused and points to non-existent `assets/classification` models.

## 3. App Configuration

- **Package name:** `com.fslr.pansinayan`
- **UI Framework:** XML views with viewBinding; Compose enabled but not used for screens
- **minSdkVersion:** 24
- **targetSdkVersion:** 34
- **compileSdkVersion:** 34
- **Dependencies:**
  - CameraX: `androidx.camera:camera-*-:1.3.1`
  - MediaPipe Tasks Vision: `com.google.mediapipe:tasks-vision:0.10.11`
  - TensorFlow Lite: `org.tensorflow:tensorflow-lite:2.14.0` (+ GPU, support)
  - ONNX Runtime (Android): `com.microsoft.onnxruntime:onnxruntime-android:1.18.0` (not used in code)
  - Room: `androidx.room:room-runtime/ktx:2.6.1` with KAPT compiler
  - Coroutines: `org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3`
  - Lifecycle Runtime KTX: `2.7.0`
  - Gson, Material, ConstraintLayout, CardView
  - Compose BOM + UI/M3 tooling (present but not used in screens)
- **Build features:** `viewBinding = true`, `compose = true`
- **Packaging:** `noCompress` for `tflite`, `onnx`, `task`; `packagingOptions.pickFirst` for common native libs (for ONNX)
- **ABIs:** `armeabi-v7a`, `arm64-v8a`
- **Assets Detected:**
  - CTC models and metadata: `assets/ctc/sign_transformer_ctc_fp16.tflite`, `assets/ctc/sign_transformer_ctc_fp16.model.json`, `assets/ctc/mediapipe_gru_ctc_fp16.tflite`, `assets/ctc/mediapipe_gru_ctc_fp16.model.json`
  - MediaPipe Tasks: `assets/pose_landmarker_full.task`, `assets/hand_landmarker.task`, `assets/face_landmarker.task`
  - Labels: `assets/label_mapping.json`, `assets/label_mapping_greeting.json`, `assets/labels/labels.json`
  - Golden test data: `assets/golden/*.json`, `assets/golden/*.npz`
- **Entry points:**
  - Launcher: `activities/HomeActivity`
  - Recognition: `activities/MainActivity`
  - History: `activities/HistoryActivity`
  - Service: `services/ScreenRecordService`
- **Permissions:** Camera ✅, Record audio ✅, Write external storage (≤ API 28) ✅, Post notifications ✅, Foreground service + media projection ✅

## 4. Partial or Conflicting Implementations

- Duplicate inference pathways:
  - `CtcModelRunner` (current) vs `TFLiteModelRunner` (legacy single-label classifier). Pipeline uses CTC; classification runner isn’t wired and references missing assets.
- Model selection mismatch:
  - `ModelSelector` references `assets/classification/...` but current assets live under `assets/ctc/...`. Radio UI shows model options but switching is not implemented in `RecognitionPipeline`.
- Temporal recognition vs aggregation:
  - `TemporalRecognizer` logic exists but token emission uses `CtcAggregator`; decide whether to keep both or deprecate TemporalRecognizer.
- ONNX Runtime dependency present but no ONNX models used or loaded in code.
- Compose libraries included but UI uses XML; Compose theme files exist without Composables.

## 5. Integration Suggestions

- UI
  - Keep existing Activities and XML layouts. Optionally migrate debug/status surfaces to Compose later, but not required.
  - Continue using `OverlayView` for skeleton; the occlusion indicator already reflects `MediaPipeProcessor.detectHandFaceOcclusion`.
- Model/Inference
  - Standardize on CTC models (`assets/ctc/*`). Remove or archive `TFLiteModelRunner` unless a non-CTC classifier is intentionally supported.
  - Update `ModelSelector` to enumerate available CTC models and metadata (`*.model.json`) under `assets/ctc/`, exposing a sealed list to the UI.
  - Implement model switching by adding a `switchModel(tflitePath, metadataPath)` method in `RecognitionPipeline` that reloads `CtcModelRunner` safely on a background thread.
- Pipeline
  - Maintain `SequenceBufferManager` window/stride from `CtcModelMetadata` hints. Consider moving iou/confidence thresholds to metadata-backed config for parity across models.
  - Decide on emission strategy: keep `CtcAggregator` or integrate `TemporalRecognizer` as a secondary stability gate; avoid duplicate gating.
  - Ensure health monitor covers model reload events (briefly pause frames when swapping models).
- Data
  - Keep Room history as-is. Consider adding a setting to toggle persistence or limit rows.
  - Record selected model name based on actual pipeline runner (Transformer/GRU from metadata).
- Build/Assets
  - Remove ONNX Runtime dependency if not planned to be used; or add ONNX models and a runner.
  - Remove unused Compose dependencies if not using Composables, or plan a small Compose screen to justify inclusion.
  - Ensure `aaptOptions.noCompress` includes all model/task extensions (already done).
- Tests
  - Keep `CtcGoldenParityTest` and golden assets. Add a small utility to run parity locally in a debug screen or instrumentation test.

---

This document summarizes the current Android implementation and highlights where to extend vs. replace when aligning with the final plan. The recommended path is to consolidate on the CTC pipeline, wire model switching to the existing radio UI, and retire legacy single-label classification code unless needed.
