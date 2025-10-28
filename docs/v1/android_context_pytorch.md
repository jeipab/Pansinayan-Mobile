# Android Context (Before PyTorch Integration)

## Architecture Overview

- **Pattern**: Controller/Service style centered in `RecognitionPipeline` coordinating components; not strict MVVM. No `ViewModel`/`Repository` classes detected.
- **UI Tech**: Primarily XML layouts with `ConstraintLayout` and custom views (`OverlayView`). Jetpack Compose is enabled in Gradle but used only for theme scaffolding (`ui/theme/Theme.kt`); core screens use XML.
- **Navigation**: Activities (`HomeActivity`, `MainActivity`, `HistoryActivity`); no Navigation Component observed.
- **Camera**: CameraX via `PreviewView` managed by `CameraManager`.

### Packages and Responsibilities

- **`com.fslr.pansinayan.activities`**: UI Activities, lifecycle, and binding to pipeline (`MainActivity` launches `RecognitionPipeline`).
- **`com.fslr.pansinayan.camera`**: CameraX setup, switching, and preview surface management (`CameraManager`).
- **`com.fslr.pansinayan.mediapipe`**: MediaPipe Tasks runners for pose/hand/face and keypoint extraction (`MediaPipeProcessor`). Provides `KeypointListener` callbacks and occlusion detection.
- **`com.fslr.pansinayan.inference`**:
  - `CtcModelRunner`: TFLite-based CTC sequence model runner with dynamic shape handling and optional GPU delegate.
  - `TFLiteModelRunner`: Legacy single-label classifier runner (not used by the main CTC pipeline).
  - `SequenceBufferManager`: Sliding window buffer, stride control, gap interpolation.
  - `CtcGreedyDecoder`: Greedy decoding of per-frame log-probs.
  - `ModelMetadata`/`ModelMetadataLoader`: Load model metadata from JSON.
- **`com.fslr.pansinayan.recognition`**: Orchestrator (`RecognitionPipeline`), temporal logic (`TemporalRecognizer`), CTC token aggregation (`CtcAggregator`), data classes.
- **`com.fslr.pansinayan.database`**: Room database (`AppDatabase`, `HistoryDao`, `RecognitionHistory`).
- **`com.fslr.pansinayan.utils`**: `LabelMapper`, `ModelSelector`, misc.

## Implemented Components

| Component                           | Class/File                                                                    | Status         | Notes                                                                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| MediaPipe Holistic (pose/hand/face) | `app/src/main/java/com/fslr/pansinayan/mediapipe/MediaPipeProcessor.kt`       | Complete       | Uses MediaPipe Tasks; extracts 178-D keypoints (50 pose + 42 LH + 42 RH + 44 face). Includes hand-face occlusion detection with temporal filtering. |
| Camera pipeline                     | `app/src/main/java/com/fslr/pansinayan/camera/CameraManager.kt`               | Complete       | CameraX setup; feeds frames to MediaPipe via `RecognitionPipeline`.                                                                                 |
| Sliding window buffer               | `app/src/main/java/com/fslr/pansinayan/inference/SequenceBufferManager.kt`    | Complete       | Windowing, stride trigger, gap interpolation, missing ratio.                                                                                        |
| CTC model runner                    | `app/src/main/java/com/fslr/pansinayan/inference/CtcModelRunner.kt`           | Complete       | TFLite interpreter; dynamic input handling; outputs per-frame CTC logits and optional category logits.                                              |
| CTC decoder                         | `app/src/main/java/com/fslr/pansinayan/inference/CtcGreedyDecoder.kt`         | Complete       | Greedy decode; confidence from per-frame max probs.                                                                                                 |
| Token aggregation                   | `app/src/main/java/com/fslr/pansinayan/recognition/CtcAggregator.kt`          | Complete       | IoU-based cross-window aggregation and de-duplication.                                                                                              |
| Temporal stabilizer                 | `app/src/main/java/com/fslr/pansinayan/recognition/TemporalRecognizer.kt`     | Complete       | Stability threshold, confidence gating, cooldown.                                                                                                   |
| Model selector                      | `app/src/main/java/com/fslr/pansinayan/utils/ModelSelector.kt`                | Partial        | Switches between Transformer/GRU via `RecognitionPipeline.switchModel`.                                                                             |
| UI for live recognition             | `app/src/main/res/layout/activity_main.xml`, `MainActivity.kt`, `OverlayView` | Complete       | Live preview (`PreviewView`), skeleton overlay, result/confidence cards, transcript, stats, model radio buttons, FABs.                              |
| Room history                        | `app/src/main/java/com/fslr/pansinayan/database/*`                            | Complete       | Persists recognition events; DAO for queries and clearing.                                                                                          |
| Legacy classifier                   | `app/src/main/java/com/fslr/pansinayan/inference/TFLiteModelRunner.kt`        | Partial/Legacy | Single-label classifier; not in main CTC flow.                                                                                                      |

## Dependencies

- From `app/build.gradle`:
  - **Android SDK**: `minSdk=24`, `targetSdk=34`, `compileSdk=34`.
  - **CameraX**: `androidx.camera:camera-* 1.3.1`.
  - **MediaPipe Tasks (Vision)**: `com.google.mediapipe:tasks-vision:0.10.11`.
  - **TensorFlow Lite**: `tensorflow-lite 2.14.0`, `-gpu 2.14.0`, `-support 0.4.4` (forced versions). Used by CTC and legacy classifier.
  - **ONNX Runtime**: `onnxruntime-android:1.18.0` present but not used in the active pipeline.
  - **Jetpack**: AppCompat, Material, ConstraintLayout, Lifecycle runtime KTX, Room (runtime, ktx, kapt).
  - **Coroutines**: `kotlinx-coroutines-android:1.7.3`.
  - **Compose**: BOM 2024.09.00 with material3/tooling; `compose=true` but UI uses XML.
  - **AAPT noCompress**: `tflite`, `onnx`, `task` preserved; packaging picks for native libs.
- Manifest permissions: Camera, Record Audio, Foreground Service (media projection), Notifications, legacy external storage for API ≤28.

### Assets and Models

- `app/src/main/assets/ctc/`:
  - `sign_transformer_ctc_fp16.tflite` with `sign_transformer_ctc_fp16.model.json`.
  - `mediapipe_gru_ctc_fp16.tflite` with `mediapipe_gru_ctc_fp16.model.json`.
  - Label maps: `label_mapping.json`, `label_mapping_greeting.json`, `labels/labels.json`.
- MediaPipe Tasks: `pose_landmarker_full.task`, `hand_landmarker.task`, `face_landmarker.task`.
- Exported references (not packaged): `exports/ctc/*.tflite`, `.onnx` and training `.pt` exist at repo root for reference but not bundled.

## Gaps & Overlaps

- **PyTorch Mobile overlap**: No `org.pytorch` dependency; current runtime is TFLite. Introducing PyTorch Mobile would overlap with `CtcModelRunner` responsibilities.
- **ONNX Runtime present but unused**: Could be removed or kept for experiments; not wired in the pipeline.
- **Compose enabled but unused**: UI remains XML; no `ViewModel` layer; state handled in `MainActivity` and pipeline.
- **Legacy TFLite classifier**: `TFLiteModelRunner` is not part of CTC flow; may be removed or adapted if PyTorch replaces all TFLite usage.

## Integration Readiness

- **Hook point for inference**: Replace `com.fslr.pansinayan.inference.CtcModelRunner` with a PyTorch-backed runner exposing the same API:
  - Input: `Array<FloatArray>` of shape `[T, 178]`.
  - Output: `CtcOutputs` with `logProbs` `[1, T, num_ctc]` and optional `catLogits` `[1, T, num_cat]`.
  - Methods: `run(sequence)`, `release()`, and `meta` loaded from JSON.
  - Maintain `window_size_hint`, `stride_hint`, `blank_id`, `num_ctc`, `num_cat` from metadata.
- **Where to integrate**: `RecognitionPipeline.initialize()` constructs `CtcModelRunner`. Swap to `CtcModelRunner` interface-compatible `PyTorchCtcRunner` and keep the rest intact (buffering, decoding, aggregation, temporal logic, UI callbacks).
- **Model selection**: `RecognitionPipeline.switchModel(...)` can continue to work by pointing to `.pt` or `.ptl` and the same `.model.json`. Consider a unified `ModelRunner` interface implemented by TFLite and PyTorch runners.
- **Preprocessing**: MediaPipe keypoint extraction and `SequenceBufferManager` are reusable as-is; ensure PyTorch input ordering matches current `[x,y]` layout of 178 features.
- **Postprocessing**: Reuse `CtcGreedyDecoder`, `CtcAggregator`, `TemporalRecognizer`, and `LabelMapper` unchanged if PyTorch output tensor shapes match.
- **Assets**: Place PyTorch mobile model in `app/src/main/assets/ctc/` (e.g., `sign_transformer_ctc_fp16.ptl`) and update `noCompress` if needed; keep metadata JSON for consistency.

### Suggested Refactors for Smooth Swap

- Introduce `ModelRunner` interface:
  - `fun run(sequence: Array<FloatArray>): CtcOutputs`
  - `val meta: CtcModelMetadata`
  - `fun release()`
- Implement `TFLiteCtcRunner` (existing) and `PyTorchCtcRunner` (new) to this interface; update `RecognitionPipeline` to depend on the interface.
- Remove unused ONNX dependency if not planned.

## Summary of Readiness

- MediaPipe pipeline, buffering, decoding, UI, and persistence are mature and reusable.
- TFLite-based CTC runner is the only component that needs replacement to integrate PyTorch Mobile.
- Integration risk is low if the PyTorch runner preserves I/O contract and metadata semantics.
