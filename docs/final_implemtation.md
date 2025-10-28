# 🧭 **Final Implementation Prompt for Cursor (Pansinayan-Mobile)**

> **Title:** Implement Real-Time Continuous Sign Language Recognition in Pansinayan-Mobile (Android, TFLite)

---

## 📘 **Context Recap**

Pansinayan-Mobile is the **Android** counterpart of the **Pansinayan-Web** (Python) project for **Filipino Sign Language Recognition (FSLR)**.
Both share the same **CTC-based models** (Transformer-CTC and MediaPipe-GRU-CTC).

The key difference:

- **Web/Python**: performs inference on preprocessed or uploaded videos.
- **Mobile**: performs **live recognition** via real-time camera keypoint capture (MediaPipe Holistic).

Models are converted from **PyTorch (`.pt`) → TFLite (`.tflite`)**.
Your task: implement a **clean, modular, and optimized live recognition pipeline** in the Android project following the verified architecture and ensuring full parity with the Python inference behavior.

---

## 🧩 **Files to Reference**

You have three verified documents forming the foundation of this prompt:

- **`py_context.md`** — Full description of the Python/Web model usage and inference process.
- **`live_plan.md`** — First detailed Android plan mapping Python logic to a live TFLite implementation.
- **`final_live_plan.md`** — Cross-verified and enhanced plan ensuring exact Python ↔ Android parity, including numerical and architectural alignment, confidence policies, and implementation blueprint.

Cursor should **read and integrate** these three markdowns as the authoritative source of implementation requirements.

---

## 🎯 **Goal**

Implement and integrate a **real-time continuous sign language recognition system** in the Android project `Pansinayan-Mobile`, meeting the following objectives:

1. **Live keypoint pipeline** using MediaPipe Holistic (pose, hands, face → 178D features).
2. **Sliding window buffering** to collect and manage keypoints in real time.
3. **Model inference pipeline** using **TensorFlow Lite** (Transformer-CTC / Mediapipe-GRU-CTC).
4. **Greedy CTC decoding** on-device (with optional beam search).
5. **Confidence and category calculation** consistent with the Python reference.
6. **Live transcript UI updates** and **session history tracking** via Room database.
7. **Runtime model switching** between available models.
8. **Performance tuning and delegate optimization** (CPU/XNNPACK by default, optional GPU/NNAPI).
9. **Validation and instrumentation** for numerical parity and runtime metrics.

---

## 🧱 **Implementation Instructions**

Follow the structure from `final_live_plan.md`, ensuring clean modular code and conflict-free integration.

### 1. **Core Architecture**

Implement the following components and ensure clear separation of concerns:

| Component                    | Description                                                                                                                                                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KeypointSource`             | Wrapper around MediaPipe Holistic that extracts pose (25), left hand (21), right hand (21), and face (22) landmarks per frame (total 89 points → 178D vector). Normalize to `[0,1]` and unmirror coordinates if needed. |
| `SlidingWindowBufferManager` | Maintains a circular buffer of keypoints. Emits windows of size `windowSize` every `stride` frames for inference. Handles interpolation, missing frames, and reset logic.                                               |
| `CTCModelRunner`             | Loads and manages TFLite models and delegates. Handles metadata (`model.json`), input resizing, tensor allocation, and invocation. Returns `log_probs` and `cat_logits`.                                                |
| `CTCDecoder`                 | Implements greedy decoding (`argmax → collapse repeats → remove blanks`). Supports confidence calculation (`python_parity` and `mobile_stable` modes).                                                                  |
| `SlidingWindowAggregator`    | Merges predictions across overlapping windows using IoU-based span merging (≥ 0.5 threshold). Ensures stable, time-consistent recognition results.                                                                      |
| `LabelRepository`            | Loads and validates label mappings from `labels.json`. Provides gloss and category lookup methods.                                                                                                                      |
| `RecognitionPipeline`        | Orchestrates the flow: capture → buffer → inference → decoding → aggregation → UI + history.                                                                                                                            |
| `RecognitionEvent`           | Room entity representing recognized tokens for persistence and session tracking.                                                                                                                                        |

---

### 2. **Model & Metadata Handling**

- Store models under `assets/models/ctc/`

  - `transformer_ctc_fp16.tflite`
  - `mediapipe_gru_ctc_fp16.tflite`

- Each model must have an adjacent `model.json` with:

  ```json
  {
    "input_dim": 178,
    "num_gloss": 10,
    "blank_id": 10,
    "num_ctc": 11,
    "num_cat": 1,
    "window_size_hint": 120,
    "stride_hint": 30,
    "decode_default": "greedy",
    "model_type": "transformer_ctc",
    "labels_file": "labels/labels.json",
    "version": "greetings_v2"
  }
  ```

- Validate counts and checksum against `labels.json` on load.
- Use FP16 models by default; FP32 for parity tests; INT8 optional with calibration dataset.

---

### 3. **Inference Logic**

- Prepare input tensor `[1, T, 178]` (float32).
- Run inference with `runForMultipleInputsOutputs`.
- Extract outputs:

  - `log_probs` `[1, T, num_ctc]`
  - `cat_logits` `[1, T, num_cat]`

- Postprocess:

  - Convert log-probs → probabilities via `exp()`.
  - Apply CTC greedy decoding and category averaging.

- Confidence computation:

  - Default = **`python_parity`** (mean of per-frame max probs in span).
  - Optional = **`mobile_stable`** (min per span).

---

### 4. **Sliding Window and Aggregation**

- Use `windowSize` and `stride` from model hints.
- Aggregate decoded tokens across windows via IoU ≥ 0.5.
- Maintain temporal consistency and stability (require K consecutive confirmations before emitting).
- Evict old tokens > 10 seconds past window horizon.

---

### 5. **UI and History Integration**

- Display recognized tokens in real time:

  - Gloss label, category, and confidence bar.

- Maintain transcript (scrollable) with recent tokens.
- Persist confirmed tokens into `RecognitionEvent` table with timestamps, frame indices, and session IDs.
- Debounce UI updates to 5–10 Hz for smoothness.
- Add indicators: capturing / buffering / recognizing / idle.

---

### 6. **Performance and Delegates**

- Default delegate: **XNNPACK (CPU, 2–4 threads)**.
- Optional toggles: **GPU**, **NNAPI**.
- Track performance metrics:

  - Capture FPS, inference ms (avg, P95), end-to-end latency.
  - Log delegate type and thread count.

- Run one **warmup inference** on model load for tensor allocation.

---

### 7. **Validation and Testing**

- Use golden validation set from Python NPZ → JSON conversion.
- Verify:

  - Sequence equivalence (same gloss order).
  - Confidence difference ≤ 0.02 absolute.
  - Timestamps uniformly distributed (unless refined mode enabled).

- Run instrumentation tests to validate decoder and aggregation logic.

---

### 8. **Model Switching**

- Switch between Transformer-CTC and Mediapipe-GRU models:

  - Pause capture.
  - Release delegates and interpreters.
  - Load new model + metadata.
  - Revalidate I/O shapes.
  - Resume recognition.

- Clear buffers and aggregators after each switch.

---

### 9. **Safety, Debugging, and Documentation**

- Handle shape mismatch or metadata errors gracefully with fallback and logging.
- Provide developer overlay showing buffer fill, inference ms, and FPS.
- Maintain structured logs for bug reporting and performance tuning.
- Document all configurations in `/docs/live_inference/` for maintainability.

---

## 🧮 **Numerical Parity Rules**

Ensure the following:

- `blank_id = num_gloss`, `num_ctc = num_gloss + 1`
- Log-softmax used in model export (no on-device log-softmax unless necessary)
- Per-frame `exp(log_probs)` sum ≈ 1.0 (tolerance ±1e-3 FP16, ±1e-4 FP32)
- Positional encodings validated for T range during export
- Confidence and category parity verified on golden set

---

## ✅ **Deliverables**

Cursor should produce:

1. Updated Kotlin and XML files implementing all modules cleanly and following Android best practices (MVVM or modular).
2. Integration with MediaPipe and TFLite Interpreter using efficient threading.
3. Documentation comments for every class and public function.
4. Summary markdown report (`implementation_summary.md`) listing:

   - Implemented classes and packages
   - Key design decisions
   - Performance metrics (after test run)
   - Future extension notes

---

## ⚙️ **Tone and Code Quality Requirements**

- Prioritize **clarity**, **maintainability**, and **real-time efficiency**.
- Use **coroutines / structured concurrency** for async inference.
- Avoid blocking the main thread.
- Follow **Kotlin + Jetpack conventions**.
- Keep UI elegant, responsive, and minimal.
- Code must be **self-explanatory**, **commented**, and **aligned with final_live_plan.md**.

---
