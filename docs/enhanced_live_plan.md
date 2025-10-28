# Enhanced Live Recognition Implementation Plan (Aligned with Android Context)

## 1. Integration Overview

This plan merges the verified live CTC blueprint with the current Android app. We will:

- Reuse the existing CTC pipeline: `CtcModelRunner` → `CtcGreedyDecoder` → `CtcAggregator` orchestrated by `RecognitionPipeline`.
- Keep MediaPipe Holistic keypoint extraction as-is and confirm the 178-D ordering/normalization.
- Extend `RecognitionPipeline` for runtime model switching and optional category head support.
- Integrate sliding-window scheduling into the existing coroutine loop (no new threads).
- Keep Activities/XML UI; wire radio buttons to actual model switching; persist to Room history.
- Clean up legacy single-label `TFLiteModelRunner` and mismatched `classification/*` references.

Assets stay under `app/src/main/assets/ctc/*`. Min/target/compile SDK and core dependencies remain as in `app/build.gradle` (compileSdk 34). Optional removal of ONNX/Compose if not used.

## 2. Reuse and Refactor Map

| Module                   | Existing File/Class                  | Action            | Notes                                                                                                         |
| ------------------------ | ------------------------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------- |
| Keypoint Extraction      | `mediapipe/MediaPipeProcessor.kt`    | Reuse             | Already emits 178 values; keep face-minimal indices and occlusion detection; ensure front-camera flip parity. |
| Sliding Window Buffer    | `inference/SequenceBufferManager.kt` | Reuse             | Keep circular buffer; set `windowSize/stride` from metadata; use existing coroutine cadence.                  |
| Model Metadata           | `inference/ModelMetadata.kt`         | Reuse             | Validate fields on load: `input_dim`, `num_ctc`, `blank_id`, `num_cat`, `window_size_hint/stride_hint`.       |
| Model Runner (CTC)       | `inference/CtcModelRunner.kt`        | Reuse             | Keep dynamic T handling; reuse ByteBuffers; warm-up after load/switch.                                        |
| Decoder (Greedy)         | `inference/CtcGreedyDecoder.kt`      | Reuse             | Keep `python_parity` confidence (mean of per-frame max prob).                                                 |
| Cross-window Aggregation | `recognition/CtcAggregator.kt`       | Reuse             | IoU-based merge; consider moving IoU threshold to metadata later.                                             |
| Temporal Recognizer      | `recognition/TemporalRecognizer.kt`  | Retain (optional) | Keep available as optional stability gate; not in default emission path.                                      |
| Orchestration            | `recognition/RecognitionPipeline.kt` | Extend            | Add `switchModel(tflite, json)`; compute category from `catLogits` if present; reset buffer on switch.        |
| UI (Main)                | `activities/MainActivity.kt`         | Extend            | Wire radio buttons to call pipeline `switchModel`; update `currentModel` and Room writes.                     |
| Overlay View             | `views/OverlayView.kt`               | Reuse             | Continue drawing landmarks; respect skeleton toggle and debug overlays.                                       |
| Data Layer               | `database/*`                         | Reuse             | Persist history; include model name and occlusion state.                                                      |
| Labels                   | `utils/LabelMapper.kt`               | Reuse             | Map gloss/category IDs from assets labels; verify lengths vs metadata.                                        |
| Model Selection          | `utils/ModelSelector.kt`             | Refactor          | Point to `assets/ctc/*`; expose `(displayName, tflitePath, metadataPath)`; remove `classification/*`.         |
| Legacy Classifier        | `inference/TFLiteModelRunner.kt`     | Deprecate         | Archive/remove after CTC path is fully wired.                                                                 |
| Camera                   | `camera/CameraManager.kt`            | Reuse             | Keep FPS throttling and restart logic.                                                                        |
| Tests                    | `androidTest/CtcGoldenParityTest.kt` | Complete          | Validate parity on golden assets for both models.                                                             |
| Build/Gradle             | `app/build.gradle`                   | Review            | Optional: remove ONNX and unused Compose deps; keep `noCompress` and ABIs.                                    |

## 3. Adjusted Implementation Steps

1. Assets and Metadata

- Keep models under `assets/ctc/`. Example pairs:
  - Transformer: `ctc/sign_transformer_ctc_fp16.tflite` + `ctc/sign_transformer_ctc_fp16.model.json`
  - GRU: `ctc/mediapipe_gru_ctc_fp16.tflite` + `ctc/mediapipe_gru_ctc_fp16.model.json`
- On load, validate `input_dim=178`, `blank_id`, `num_ctc`, `num_cat`, and labels length.

2. Keypoint Pipeline

- Reuse `MediaPipeProcessor`; confirm unmirrored model space and handedness mapping; keep minimal face indices.
- Output remains `[T,178]` with normalization in `[0,1]`.

3. Sliding Window

- Use `SequenceBufferManager` with `window_size_hint/stride_hint` from metadata; skip windows with >50% missing frames.
- Trigger `runCtcIfReady` from the existing coroutine using `ctcStride` (no extra thread).

4. Interpreter and Delegates

- Default CPU/XNNPACK with `threads=4`; optional GPU delegate in `CtcModelRunner` with safe fallback.
- Prefer FP16 models; use FP32 for parity tests.

5. Invocation

- Prepare `[1, T, 178]`; call `resizeInput` when supported; else pad/truncate to base T.
- Reuse input/output buffers across calls; warm-up after model load/switch.

6. Decoding and Aggregation

- Greedy decode with `CtcGreedyDecoder` and `blank_id` from metadata.
- Confidence mode: `python_parity` (already implemented).
- Aggregate window tokens with `CtcAggregator` (IoU ≥ 0.5), emitting only newly added tokens.

7. Category Head (if available)

- If `catLogits` exists, average over `[startT..endT]` for each token and `argmax` to get `categoryId`; map via `LabelMapper`.
- Populate `RecognizedSign.categoryId/categoryLabel`; fallback to `0` when head absent.

8. Timestamps

- Keep token spans from argmax-run union per decoder; UI may display midpoints or spans as needed.

9. Switching and Lifecycle

- Add `RecognitionPipeline.switchModel(tflitePath, metadataPath)`:
  - Pause frame processing; release old runner; instantiate new `CtcModelRunner`.
  - Update `ctcWindowSize/ctcStride`; rebuild `SequenceBufferManager`; `ctcAggregator.clear()`; warm-up.
  - Resume frames; reset health timers.
- In `MainActivity`, implement radio actions to call `pipeline.switchModel()` with correct pair paths and update `currentModel`.

10. Instrumentation/Parity

- Finalize `CtcGoldenParityTest` using `io/NpyNpzReader` and golden assets; test both models.
- Target parity tolerance: tokens and confidences ±0.02 vs Python.

## 4. Compatibility & Migration Notes

- Deprecate `inference/TFLiteModelRunner.kt`; remove from build if not used.
- Update `utils/ModelSelector.kt` to enumerate CTC pairs from `assets/ctc/` and stop referencing `classification/*`.
- In `activities/MainActivity.kt`, replace radio paths to use `ctc/...` and forward to pipeline `switchModel()` with both paths.
- Keep `recognition/TemporalRecognizer.kt` optional; do not duplicate gating with `CtcAggregator` by default.
- Consider removing `com.microsoft.onnxruntime:onnxruntime-android` if no ONNX models are planned.
- Consider removing unused Compose UI dependencies and plugin if Composables won’t be used; keep `viewBinding=true`.
- Ensure `aaptOptions.noCompress` includes `tflite`, `onnx`, `task` (already present) and keep ABIs `armeabi-v7a`, `arm64-v8a`.

## 5. Final Readiness Notes

- Dependencies verified against `app/build.gradle` (CameraX 1.3.1, MediaPipe 0.10.11, TFLite 2.14.0, Room 2.6.1, Coroutines 1.7.3, Lifecycle 2.7.0).
- Naming and packages match `com.fslr.pansinayan` and existing file layout.
- Plan extends existing classes instead of replacing them; legacy classifier path flagged for removal.
- Safe to implement directly: add `switchModel`, category head handling, and UI wiring; optional cleanup of ONNX/Compose.
