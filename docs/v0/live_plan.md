# Live Recognition Implementation Plan (Android, TFLite)

Link to Python/Web context: [py_context.md](c:/Users/Asus/Documents/GitHub/fslr-transformer-vs-iv3gru/py_context.md)

---

## Introduction and Context Link (from py_context.md)

- Python reference implements Continuous Sign Language Recognition using CTC models (Transformer-CTC and MediaPipe-GRU-CTC) over MediaPipe keypoints [T, 178].
- Outputs per frame:
  - CTC log-probabilities: [B, T, num_ctc] with blank at index `blank_id = num_gloss` and `num_ctc = num_gloss + 1`.
  - Per-frame category logits: [B, T, num_cat].
- Decoding: greedy (argmax → collapse repeats → remove blanks) or beam search; sliding-window inference with aggregation across windows.
- Android goal: replicate this flow in live mode using TFLite models and camera-driven keypoint streams.

Assumed default configs (from context):

- Input dim: 178 (89 keypoints × x,y normalized [0,1])
- For full set: `num_gloss = 105`, `blank_id = 105`, `num_ctc = 106`, example beam width 10
- Subsets (e.g., greetings) may reduce class counts; we must load metadata per model.

---

## High-Level System Design for Live Recognition

Dataflow (live):

- Camera (30 FPS) → MediaPipe Holistic → normalize to [0,1] → keypoints frame [178]
- Sliding window buffer (length `windowSize`, stride `stride`) → batch to [1, T, 178]
- TFLite Interpreter (Transformer-CTC or Mediapipe-GRU-CTC)
- Outputs per frame: `log_probs[T, num_ctc]`, `cat_logits[T, num_cat]`
- CTC decoder (greedy default, optional beam) → predicted token sequence with confidences
- Sliding-window aggregation (merge tokens across overlapping windows by temporal position/confidence)
- UI: live transcript + per-token category; History: persisted session with timestamps

Key Android components:

- KeypointSource (MediaPipe wrapper)
- SlidingWindowBufferManager (produce windows and trigger inference)
- CTCModelRunner (TFLite Interpreter lifecycle, pre/post-processing)
- CTCDecoder (greedy/beam)
- SlidingWindowAggregator (temporal aggregation)
- LabelMapper/LabelRepository (IDs → labels, loaded from JSON assets)
- ModelMetadataProvider (reads model.json for dims/classes/blank)
- RecognitionPipeline (orchestrator)
- UI ViewModel + Room DB for history

---

## Input Pipeline Design

Continuous keypoint collection:

- Use MediaPipe Holistic on-device to emit per-frame 178-D features already normalized to [0,1].
- Represent each frame as `FloatArray(178)` with capture timestamp (ms) and frame index.

MediaPipe configuration:

- Tasks: `pose_landmarker_full.task`, `hand_landmarker.task` (left/right), and face landmarks if used for the 22 facial points.
- FPS target: 30; back off gracefully on lower-end devices (auto-detect achieved FPS and adapt `stride`).
- Landmark selection (total 89 points):
  - Pose: 25 indices (subset of full 33; match Python selection)
  - Hands: 21 each (left + right)
  - Face: 22 (subset for eyebrows/eyes/mouth per training)
- Ordering and feature vector:
  - Concatenate in fixed order: `[pose(25) x,y] + [left_hand(21) x,y] + [right_hand(21) x,y] + [face(22) x,y]` → 178-D
  - Normalize coordinates to [0,1] by image width/height; drop visibility if models were trained without it
- Optional smoothing: apply light EMA (e.g., α=0.2) on coordinates to reduce jitter, only if training used similar smoothing.

Sliding window buffering:

- Parameters (configurable per model/profile):
  - `windowSize`: default 120–150 frames (4–5 seconds at 30 FPS)
  - `stride`: default 10–20 frames (0.33–0.66 s)
  - `maxBuffer`: `windowSize + stride` to support overlap
  - `maxGapFrames`: tolerate short dropouts; interpolate/zero-fill where needed
- Behavior:
  - On each new frame, append to a ring buffer; when `buffer.size >= windowSize` and `framesSinceLastInference >= stride` → emit the most recent `windowSize` frames as a window and update `lastInferenceFrame`.
  - Reset buffer on model switch, camera switch, or explicit user reset.
  - Drop frames only if the pipeline lags; prefer backpressure (skip emitting) over dropping MediaPipe frames.

Handling missing frames and resets:

- Interpolate short gaps (≤ `maxGapFrames`) with previous frame values; otherwise zero-fill.
- Skip inference if more than 50% of frames in a window are missing/zeroed.
- Reset conditions: app backgrounded/foregrounded, camera re-init, model reload; clear buffer and aggregator.
- Session segmentation: if no hands detected for N seconds (e.g., 3s), finalize current transcript chunk and start a new session segment for history.

Threading:

- Keypoint capture on Camera thread/MediaPipe thread → dispatch to a single-producer buffer.
- Inference on a dedicated background executor/coroutine dispatcher; UI updates on Main.

---

## Model Invocation (TFLite Integration)

Model assets and metadata:

- Store models under `assets/models/ctc/`:
  - `transformer_ctc_fp16.tflite`
  - `mediapipe_gru_ctc_fp16.tflite`
- Alongside each model, include JSON metadata: `model.json` containing `{ input_dim, num_gloss, blank_id, num_ctc, num_cat, window_size_hint, stride_hint, decode_default }`.
- Labels in `assets/labels/labels.json` (JSON in assets). Suggested schema:

```json
{
  "gloss_labels": ["hello", "good", "morning", "..."],
  "category_labels": ["greeting", "question", "..."],
  "version": "greetings_v2"
}
```

- Validation on load: `gloss_labels.size == num_gloss`, `category_labels.size == num_cat`.

Model metadata schema (model.json):

```json
{
  "input_dim": 178,
  "num_gloss": 105,
  "blank_id": 105,
  "num_ctc": 106,
  "num_cat": 10,
  "window_size_hint": 120,
  "stride_hint": 30,
  "decode_default": "greedy",
  "model_type": "transformer_ctc",
  "labels_file": "labels/labels.json",
  "version": "greetings_v2"
}
```

Label loading (high-level):

- Read `labels/labels.json` from assets via `AssetManager` and parse once at app start.
- Provide `LabelRepository` that exposes `getGlossLabel(id)` and `getCategoryLabel(id)`; validate lengths against `model.json`.

Interpreter setup:

- Prefer XNNPACK (CPU) with multiple threads for small sequence models; enable GPU delegate optionally if tested beneficial on device class; expose NNAPI toggle for Android 10+.
- Pre-allocate input `ByteBuffer` sized for `[1, T, 178]` float32; reuse across inferences to avoid GC.
- Validate tensor shapes at load:
  - Input: `[1, T, 178]` (T can be fixed or dynamic; if dynamic, use `resizeInput` per window)
  - Output0: `[1, T, num_ctc]`
  - Output1: `[1, T, num_cat]`

Dynamic tensor resizing:

- If the model supports dynamic `T`, call `interpreter.resizeInput(0, intArrayOf(1, windowSize, 178)); interpreter.allocateTensors()` when `windowSize` changes.
- If fixed `T`, ensure `windowSize` in pipeline matches the model’s expected length from `model.json`.

Delegate selection:

- Default to XNNPACK with `numThreads = min(4, availableCores)`.
- Offer toggles in settings: NNAPI (Android 10+) and GPU delegate; use feature detection with safe fallback to CPU.
- Persist user choice; log delegate type in session stats.

Preprocessing:

- Ensure keypoints are float32 in [0,1]; if MediaPipe yields doubles, cast to float.
- If the model expects standardized inputs (mean/std), apply the same normalization used in training; otherwise pass-through.

Invocation:

- Copy frames into input buffer in time-major order `[t][feature]`.
- Run `runForMultipleInputsOutputs` to fetch both outputs in one call.
- Postprocessing: if outputs are log-probs (log_softmax), convert to probabilities via `exp`; if raw logits, either apply log-softmax on device (costly) or ensure training exported log-probs.

Error handling:

- On shape mismatch or missing tensors, surface a recoverable error (toast + logs) and revert to idle; block further inference until resolved.

---

## Sliding Window Algorithm

Per-window decoding:

- For each window, decode gloss sequence using greedy by default:
  1. `argmax` over `num_ctc` per frame
  2. collapse consecutive duplicates
  3. remove `blank_id`
- Compute per-token confidence: average of max per-frame probabilities over frames contributing to the token or the min across its span for conservative confidence.
- Category per token: average category logits over the token's frame span, then argmax.

Token-to-frame span estimation:

- Without framewise alignments, approximate by uniformly splitting the T frames across N decoded tokens; assign contiguous frame ranges per token.
- Optionally refine by detecting token boundary changes in the argmax path before collapsing (where class changes), mapping collapsed tokens to contiguous argmax segments.

Cross-window aggregation:

- For each decoded token, estimate its representative frame position within the window (e.g., the midpoint of its assigned span).
- Global map: `position → list<TokenPrediction>` for positions along the stream.
- When multiple windows contribute tokens at similar positions, keep the highest-confidence token; maintain ordering by frame position.
- Evict predictions older than a horizon (e.g., 10 seconds) to bound memory.

Refined merging across windows:

- Represent each token as a span `[startFrame, endFrame]` with a predicted `glossId` and confidence.
- When adding a new token, compare with existing tokens by temporal IoU = `intersection/union` of spans; if IoU > threshold (e.g., 0.5) and same `glossId`, merge by taking max confidence and unioned span.
- If conflicting `glossId` at same position, prefer higher confidence; optionally keep both with slight position offset to avoid flicker, then prune on stabilization.

Latency vs accuracy knobs:

- Increase `stride` for lower CPU usage (higher latency); decrease for tighter responsiveness.
- Increase `windowSize` for more context (accuracy) at the cost of latency/memory.
- Enable beam search for post-action review modes; keep greedy for live.

Stability heuristics:

- Only emit a token to UI when it persists for K consecutive window updates at the same approximate position or confidence exceeds a high threshold (e.g., 0.9).
- Provide a “low-latency” mode that emits immediately, and a “stable” mode that waits for confirmation.

---

## Performance Optimization

Computation and memory:

- Reuse `ByteBuffer` and output arrays; avoid per-call allocations. Pool arrays for `T × classes`.
- Keep `windowSize` as low as acceptable (e.g., 120) to reduce compute; tune per model.
- Prefer FP16 / int8 quantized models for mobile; ensure dequantization is handled in the model or interpreter.

Threading and backpressure:

- Dedicated inference executor; never call Interpreter on the main thread.
- If inference lags behind frame rate, drop stale windows (keep last) rather than queuing.
- Adapt `stride` upward under sustained load to maintain real-time characteristics.

Delegates and device classes:

- Mid/high-end devices: try GPU; low-end: XNNPACK CPU with 2–4 threads.
- Use NNAPI where GPU is unavailable but NNAPI drivers exist.

Instrumentation:

- Track moving averages for: capture FPS, window emission rate, inference time (avg/P95), end-to-end latency.
- Expose a developer overlay to visualize these metrics during testing.

---

## Real-Time Output Mapping

CTC decoding in Kotlin:

- Greedy: iterate frames, collect argmax IDs and their max probabilities; collapse repeats; filter blanks; return token IDs and confidences.
- Beam (optional): small beam width (e.g., 5–10) over per-frame probabilities; prune by cumulative log-probability; higher latency.

Per-token category mapping:

- For each token’s frame span, average category logits across frames; take argmax; emit category ID and confidence (softmax of averaged logits or average of per-frame softmax).

Timestamps:

- Use capture timestamps; for a token covering frames `[start..end]`, set `start_ms = timestamps[start]`, `end_ms = timestamps[end]`.
- If timestamps are not tracked per frame, approximate using FPS and frame indices.

Label mapping:

- Map token IDs via `labels.json.gloss_labels[tokenId]`; map categories via `labels.json.category_labels[categoryId]`.
- Ensure label arrays align with the model’s indices; validate sizes on model load. Note: `blank_id` refers to the extra CTC class and does not have a gloss label entry.

Output payload to UI:

- `List<RecognizedSign>` where each item contains `{glossId, glossLabel, categoryId, categoryLabel, confidence, startMs, endMs, frameIndex}`.

UI update strategies:

- Diff-based transcript updates: append only new tokens; avoid redrawing entire list.
- Use chips with color-coded categories and a small confidence bar; provide accessibility labels for TalkBack.
- Debounce updates to ~5–10 Hz to reduce jank while remaining responsive.

---

## UI Reflection and History Tracking

Real-time UI:

- Show latest token prominently with its category and confidence.
- Maintain a transcript (last N tokens) horizontally scrollable; color-code by category/confidence.
- Optional per-token chip with timestamp; tap for details.

History/session logging:

- Room entity `RecognitionEvent`: `{ id, sessionId, glossId, glossLabel, categoryId, categoryLabel, confidence, startMs, endMs, frameIndex, createdAt }`.
- Persist tokens as they are confirmed by the aggregator; batch inserts per window to reduce I/O.
- Allow export (CSV/JSON) with full transcript and timing.

Session management:

- Group events by `sessionId`; create a new session on app start or explicit “New Session”.
- Persist window ID and sequence index per token for reproducibility.

UX states:

- Indicators for: capturing (camera), buffering (window filling), recognizing (inference active), idle/paused.
- Model badge showing active model and decoding mode.

---

## Model Switching and Resource Management

Runtime switching:

- Expose a toggle between Transformer-CTC and Mediapipe-GRU-CTC.
- On switch: pause recognition, release current interpreter/delegates, load new model and labels, clear buffers/aggregator, resume capture.
- Validate I/O shapes and metadata before resuming; update `windowSize/stride` based on model hints if provided.

Safety and atomicity:

- State machine: {Idle, Capturing, Inference}. Switch only from Idle.
- On switch request: pause capture → cancel pending inferences → close interpreter/delegates → load new model + labels + metadata → validate → clear buffers → resume.
- Verify `labels.json` version matches `model.json.version` (or compatible table) before enabling.

Compatibility and metadata:

- Require model-adjacent `model.json` with `{num_gloss, blank_id, num_ctc, num_cat, input_dim}`; verify consistency with labels.
- If `num_ctc != num_gloss + 1`, block load and show actionable error.

Resource management:

- Preallocate and reuse `ByteBuffer` and output arrays.
- Limit aggregator memory by evicting entries beyond a frame horizon.
- Close delegates on lifecycle stop and on model switch.
- Use a single-threaded inference executor to avoid concurrent interpreter calls.

---

## Validation Strategy

Golden-sequence parity tests:

- Export NPZ sequences from Python reference; convert to a neutral JSON/NPY the Android app can read (to avoid NPZ parsing on-device).
- Run both Python and Android with the same window/stride and decoding mode; compare predicted sequences and token counts.
- Accept minor differences in confidences due to float math; assert sequence equality and near-equal confidences within tolerance (e.g., ±0.02 absolute).

Dimensional checks:

- On model load, assert input/output tensor shapes match expectations; log detailed shapes and metadata.
- During inference, verify `[T]` of outputs equals input `T`.

Decoder unit tests:

- Synthetic logits covering repeats, blanks, and category averaging; assert collapsed sequences and blank removal.
- Include cases where blank dominates mid-span to ensure proper removal; verify category averaging over non-uniform spans.

End-to-end tests:

- Instrumentation test that runs a short prerecorded keypoint stream (file) through the full pipeline and validates the final transcript.

Performance validation:

- Measure average and P95 inference time, end-to-end latency (frame arrival → token emission), and CPU/GPU utilization; tune threads and stride accordingly.
- Collect metrics per delegate; keep device-specific profiles for recommended defaults.

---

## Potential Issues and Mitigation Plan

Known risks and mitigations:

- Shape mismatches (dynamic vs fixed T): support `resizeInput(T)` and reallocate buffers accordingly.
- Output domain mismatch (log-probs vs logits): prefer exporting log-probs; otherwise apply log-softmax or softmax on-device consistently.
- Beam search latency: keep greedy default in live; allow beam in review/offline modes.
- Label/index drift: bind models to specific label files via checksum or version in metadata and validate on load.
- MediaPipe variability (missing hands/face): robust interpolation/zero-fill; skip windows with excessive missing data.
- Device variability: provide runtime toggles for NNAPI/GPU/XNNPACK; choose defaults per device class.
- Memory pressure: reuse buffers, cap aggregator memory, avoid large allocations per frame.

Operational monitoring:

- Debug overlay (dev builds): buffer fill level, window emissions, inference ms, FPS, token counts.
- Structured logs (timings, shapes) and a hidden screen to dump last N minutes of transcript for bug reports.

---

Appendix: Key Parameters (suggested defaults)

- Transformer-CTC: windowSize=120, stride=30, greedy decode
- Mediapipe-GRU-CTC: windowSize=120, stride=30, greedy decode
- FPS assumed 30; adaptively estimate if variable.
