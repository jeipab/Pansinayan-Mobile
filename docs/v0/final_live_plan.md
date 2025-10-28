# Final Live Recognition Plan (Android, TFLite)

Link to Python/Web context: `c:/Users/Asus/Documents/GitHub/fslr-transformer-vs-iv3gru/py_context.md`
Link to prior Android plan: `c:/Users/Asus/AndroidStudioProjects/Pansinayan/live_plan.md`

---

## Verified Equivalences (Python ↔ Android)

- **Input features**: `[T, 178]` float32, normalized to `[0,1]`, derived from 89 keypoints × `(x,y)`; same landmark subsets and ordering: `pose(25)` + `left_hand(21)` + `right_hand(21)` + `face(22)` → concatenated as `[pose x,y][left x,y][right x,y][face x,y]`.
- **Batching**: Inference uses `[1, T, 178]` on both; `T` can be dynamic or fixed per model.
- **CTC conventions**: `blank_id = num_gloss`, `num_ctc = num_gloss + 1`; decoders perform `argmax → collapse repeats → remove blanks`.
- **Outputs**: Two heads per frame: `log_probs[T, num_ctc]` (CTC log-softmax) and `cat_logits[T, num_cat]` (category).
- **Category assignment**: For a token’s span, average per-frame category logits, then `argmax` to select the category; consistent across both.
- **Sliding-window inference**: Overlapping windows with aggregation by temporal position to form the final sequence; same strategy and semantics.
- **Label mapping**: IDs → human labels resolved from model-bound label tables; lengths must match `num_gloss` and `num_cat`.
- **Tensor layout note**: Inputs are copied in time-major order `[t][feature]` into a batch-first tensor `[1, T, 178]`. Outputs are read as `[1, T, C]` and then squeezed to `[T, C]` for decoding, matching the Python reference.
- **Confidence policy**: Default is `python_parity` — average of per-frame max probabilities over each token span.

---

## Current Status Summary (as implemented)

- **Delegates**: Default CPU (XNNPACK), `threads=4`. GPU/NNAPI optional, guarded with safe fallback; GPU off by default to avoid device/classpath issues.
- **CTC Runner**: Handles fixed/dynamic `T`; caps to model base length (e.g., 300) if positional encodings are fixed; otherwise resizes input when supported.
- **Output handling**: Selects CTC/category output tensors by shape, reads via direct ByteBuffers, reshapes/permutates to `[T, C]` robustly.
- **Windowing/Aggregation**: Uses `window_size_hint/stride_hint` (e.g., 120/40). Skips windows with >50% missing frames. Aggregates across windows by temporal IoU ≥ 0.5, keeps highest-confidence token.
- **Parity tests**: Instrumentation test loads golden NPZ (X.npy, float32, C-order) from assets, runs windowed decode, aggregates, and (optionally) compares to expected with ±0.02 tolerance.
- **Labels/metadata**: `assets/labels/labels.json` (arrays) and per-model `*.model.json` with checksum and correct counts are in place.
- **Known TODO**: Verify GRU TFLite load (no Flex) and run a small window; optional uniform timestamp emission for strict parity.

---

## Identified Gaps or Inconsistencies

- **Window/stride defaults differ in references**: Python shows both `window_size=60, stride=15` (in config) and `window_size=120, stride=40` (example); Android plan defaults to `windowSize=120–150`, `stride=10–20`.
- **Confidence calculation (greedy)**: Python notes “max probability per predicted token”; Android draft proposes average or min over the token span.
- **Logits vs log-probs**: Python typically outputs log-softmax; Android includes fallback for raw logits. Risk of mismatch if exported model changes.
- **Front-camera mirroring**: Training assumes canonical `(x,y)` orientation; mobile previews often mirror front camera—can flip handedness and x-axis unless corrected.
- **Dynamic `T` handling**: Some exports may fix `T`; others support dynamic `T` requiring `resizeInput` flow.
- **GRU op support in TFLite**: PyTorch GRU → TFLite can require conversion via TensorFlow; ONNX-to-TFLite may not map GRU cleanly without unrolling or Flex ops.
- **Quantization drift**: FP16/INT8 introduce numeric drift vs PyTorch FP32; can affect CTC argmax near ties.
- **Timestamp policy**: Python uniformly distributes frames across tokens; Android suggests midpoint/IoU merging plus optional refined spans. Need explicit alignment for parity tests.

---

## Fixes/Adjustments Proposed

- **Authoritative metadata per model**: Require `model.json` fields `{ input_dim, num_gloss, blank_id, num_ctc, num_cat, window_size_hint, stride_hint, decode_default, model_type, labels_file, version }`. Always adopt these hints over hardcoded defaults.
- **Confidence parity mode**: Implement two selectable modes with `default = python_parity`:
  - `python_parity`: exponentiate log-probs, then compute the average of per-frame max probabilities within each token span.
  - `mobile_stable`: minimum of per-frame max probabilities within span (more conservative UI). Keep as optional.
- **Enforce log-softmax outputs**: Export models with final `log_softmax` for the CTC head. Reject/flag models lacking it; do not compute log-softmax on device unless necessary.
- **Front-camera normalization**:
  - Disable mirror in preview or post-correct coordinates so that `x` increases left→right in model space and left/right hand identity is preserved.
  - Validate handedness from MediaPipe; if preview is mirrored, swap left/right hands in the feature vector to match training.
- **Dynamic `T` resilience**: If interpreter supports dynamic shapes, call `resizeInput(0, [1, windowSize, 178])` on changes; otherwise, enforce fixed `windowSize == model.json.window_size_hint`.
- **GRU conversion path**:
  - Prefer TensorFlow/Keras reimplementation for GRU and export to TFLite with built-in GRU op.
  - If staying PyTorch, use Torch → ONNX (opset ≥ 14) → TF → TFLite with tests verifying identical output slices; avoid Flex if possible.
- **Quantization controls**:
  - Provide FP32/FP16 builds; for INT8, use representative dataset of keypoint sequences (not images) and verify CTC parity on golden samples.
  - For live, default to FP16 unless target device shows regressions.
- **Timestamp policy alignment**: Default to Python policy (uniform allocation of frames across tokens). Keep refined span/IoU merging as an optional “refined timestamps” mode.
- **Argmax-to-span mapping for parity**: Before collapsing repeats, segment the argmax path into maximal constant-class runs; map each collapsed token to the union of its underlying runs (excluding `blank_id`) to define spans consistently with Python.
- **On-device distribution checks**: Validate per-frame CTC distribution sums to ~1.0 after `exp(log_probs)` (tolerance ±1e-4 in FP32, ±3e-3 in FP16). Log anomalies to aid debugging.

---

## Enhancement Recommendations

- **Adaptive stride controller**: Monitor end-to-end latency; increase `stride` under sustained load to hold latency targets while preserving accuracy.
- **Landmark smoothing with occlusion gating**: EMA with `α=0.15–0.25`, gated off when large pose/hand jumps indicate motion to avoid lag.
- **Stability throttle**: Emit tokens to UI only after K consecutive window confirmations or if confidence ≥ threshold (e.g., 0.9). Expose Low-Latency vs Stable modes.
- **Device-class profiles**: Persist per-device recommended delegate (XNNPACK/NNAPI/GPU), threads, and stride based on profiled P95.
- **Model/labels checksum binding**: Hash both to prevent label drift; block mismatches.
- **Warmup and buffer priming**: Run one dry inference on app start and after model switch to allocate tensors and stabilize latency.

---

## Updated Android Implementation Blueprint

1. **Assets and Metadata**

   - Place models under `assets/models/ctc/` with adjacent `model.json` and a shared `labels/labels.json`.
   - On load, validate: `input_dim==178`, `num_ctc==num_gloss+1`, `blank_id==num_gloss`, label lengths match counts.

2. **Keypoint Pipeline**

   - MediaPipe Holistic at ~30 FPS; ensure coordinates normalized to `[0,1]` in unmirrored model space.
   - Build `FloatArray(178)` per frame using canonical ordering; apply optional EMA smoothing.

3. **Sliding Window**

   - `windowSize = model.json.window_size_hint` (e.g., 120), `stride = model.json.stride_hint` (e.g., 30–40).
   - Interpolate short gaps; zero-fill otherwise; skip inference if >50% missing.
   - Maintain `lastInferenceFrame` and only emit a new window when `framesSinceLastInference >= stride` to bound latency.

4. **Interpreter and Delegates**

   - XNNPACK CPU by default (`threads = min(4, cores)`); toggles for NNAPI and GPU with safe fallback.
   - Prefer FP16 models; keep FP32 for parity tests; INT8 only after calibration validation.

5. **Invocation**

   - Prepare `[1, T, 178]` input; if dynamic `T`, `resizeInput` on changes and re-`allocateTensors()`.
   - Single call returns `log_probs[1, T, num_ctc]` and `cat_logits[1, T, num_cat]`.
   - Reuse input/output buffers across calls to avoid GC; prewarm with a dry run after model load.

6. **Decoding and Aggregation**

   - Greedy: per-frame `argmax` over `num_ctc` → collapse duplicates → drop `blank_id`.
   - Confidence (default `python_parity`): for each token span, compute `mean_t max_c exp(log_probs[t, c])` over frames t in span.
   - Category: average per-frame `cat_logits` over span → `argmax`.
   - Cross-window: group tokens by temporal position, prefer higher confidence, optional IoU-based span merging.
   - Stability: emit a token only after K consecutive window confirmations of the same token at similar position, or if confidence ≥ high threshold.

7. **Timestamps**

   - Default uniform allocation across tokens; optional refined mode using argmax-transition spans.

8. **Outputs and UI**

   - Map IDs via labels; emit `RecognizedSign { glossId, glossLabel, categoryId, categoryLabel, confidence, startMs, endMs, frameIndex }`.
   - Debounce UI updates to ~5–10 Hz; stability throttle as configured.

9. **Switching and Lifecycle**

   - Switch models atomically from Idle; release delegates; clear buffers; reload metadata; revalidate shapes.

10. **Instrumentation**

- Track capture FPS, window rate, inference ms (avg/P95), end-to-end latency; expose a developer overlay.
  - Log delegate type, thread count, and per-call tensor shapes; sample memory usage to detect regressions.

---

## Numerical Parity & Conversion Notes

- **Tolerances**: Expect small differences between PyTorch FP32 and TFLite FP16. For per-frame class probabilities, tolerate ±0.01 absolute; for token confidences, ±0.02 absolute.
- **Softmax temperature**: Ensure no inadvertent scaling before `log_softmax`. Temperature must be 1.0; verify training/export graphs.
- **Padding/masking**: If models were trained with attention masks, ensure export preserves masking behavior. For fixed-T models, pad frames with zeros at the tail; do not pad with repeats.
- **Positional encoding** (Transformer): Validate that exported positional embeddings cover `T` (resize if needed during training/export, not on-device).
- **Representative dataset (INT8)**: Build from diverse keypoint windows (still, slow, fast motions; left/right-hand signs; occlusions). At least 500–1k windows recommended for stable calibration.
- **GRU numerical checks**: Compare PyTorch vs TFLite hidden-state traces on short sequences to confirm gate behavior after conversion.

---

## Validation Checklist

- [ ] Model metadata validated: `input_dim`, `num_gloss`, `blank_id`, `num_ctc`, `num_cat`, labels length, checksums.
- [ ] Tensor shapes checked at runtime: input `[1, T, 178]`, outputs `[1, T, num_ctc]` and `[1, T, num_cat]` with matching `T`.
- [ ] Camera orientation verified: front camera mirroring off or coordinates corrected; left/right hands preserved.
- [ ] Confidence parity: greedy decoding confidence matches Python within ±0.02 absolute on golden sequences.
- [ ] Category parity: per-token category matches Python on golden sequences; tolerance for probabilities.
- [ ] Sliding-window parity: same window/stride as `model.json` hints; identical sequences across systems.
- [ ] Delegate stability: compare FP32 (baseline) vs FP16 (default) vs INT8 (optional) on golden set; no sequence regressions.
- [ ] Performance targets: average and P95 inference time, end-to-end latency within device-class budgets; stride adapts under load.
- [ ] Robustness: behavior verified for missing frames, model switch, app background/foreground, and error surfaces.
- [ ] Unit tests: decoder collapse/blank removal, category averaging, timestamp assignment; instrumentation test with prerecorded stream.
- [ ] Distribution sanity: per-frame `exp(log_probs)` sums to ~1 (within tolerance); no NaNs/INF during runs.
- [ ] Argmax-span mapping parity: collapsed-token spans match argmax-run union policy on both Python and Android.
- [ ] Quantization: INT8 model validated with representative dataset; CTC sequences identical to FP32/FP16 on golden set.

---

Notes:

- Use `model.json.decode_default` to set greedy/beam; keep beam width modest (5–10) for offline/review modes.
- Prefer exporting CTC `log_softmax` in the model to avoid on-device numerical variance and cost.
