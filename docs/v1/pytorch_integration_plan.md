# Unified PyTorch Integration Plan

## Overview

- Replace the current TensorFlow Lite CTC runner with a PyTorch Mobile runner while keeping the existing Android pipeline (MediaPipe → buffering → decoding → UI) intact.
- Preserve I/O contract from the Python/Web models: inputs as keypoint sequences [1, T, 178]; outputs as CTC log-probabilities [1, T, num_ctc] (and optional per-frame category logits).
- Introduce a `ModelRunner` abstraction implemented by both the existing TFLite runner and the new PyTorch runner to enable side-by-side testing and easy migration.

## Mapping: Python → Android

- **Inputs**
  - Python: float32 tensor [B=1, T, 178] (normalized in [0,1]).
  - Android: `Array<FloatArray>` shaped [T, 178] from MediaPipe; convert to PyTorch `Tensor` [1, T, 178] float32.
- **Preprocessing**
  - Python: MediaPipe-style keypoints; clamped to [0,1]; optional EMA smoothing, outlier removal, and short-gap interpolation (≤5 frames).
  - Android: Already implemented in `MediaPipeProcessor` + `SequenceBufferManager` (clamp, smoothing, gap interpolation). Ensure ordering: pose(25) + left(21) + right(21) + face(22) → 89×2 = 178.
- **Inference**
  - Python: Torch model forward returns log-softmaxed CTC logits [1, T, C].
  - Android: PyTorch Mobile `Module.forward` returns `Tensor` [1, T, C].
- **Decoding**
  - Python: Greedy CTC (argmax per frame → collapse repeats → remove blanks), with `blank_id` from config.
  - Android: Implement the same greedy decoder in Kotlin; reuse `blank_id` from model metadata JSON.
- **Labels / Metadata**
  - Python: Labels, class counts, and blank_id from configs.
  - Android: Continue using `ModelMetadataLoader` to load `num_ctc`, `blank_id`, `window_size_hint`, `stride_hint`, and label maps from the existing JSON.

## Architecture Adjustments

- **Introduce `ModelRunner` interface**
  - `fun run(sequence: Array<FloatArray>): CtcOutputs`
  - `val meta: CtcModelMetadata`
  - `fun release()`
- **New implementation: `PyTorchCtcRunner`**
  - Loads a TorchScript `.pt`/`.ptl` module from `app/src/main/assets/ctc/`.
  - Converts `[T,178]` to a contiguous `Tensor` `[1,T,178]` (float32) and performs forward.
  - Returns `CtcOutputs(logProbs=[1,T,C], catLogits=[1,T,K]? )`.
- **Recognition pipeline**
  - `RecognitionPipeline` depends on `ModelRunner` instead of the concrete TFLite runner.
  - `ModelSelector` selects between Transformer/GRU and TFLite/PyTorch via file extension (`.tflite` vs `.pt/.ptl`).
- **Dependencies**
  - Add PyTorch Mobile dependency (choose one):
    - Full: `org.pytorch:pytorch_android:<version>`
    - Lite: `org.pytorch:pytorch_android_lite:<version>` (smaller binary; preferred if compatible)
  - Keep TFLite during migration; remove once parity is confirmed.
- **Assets and packaging**
  - Place TorchScript model at `app/src/main/assets/ctc/<model>.pt(l)`.
  - Ensure `aaptOptions.noCompress += ["pt", "ptl"]` so the file is not compressed.
  - Reuse existing `*.model.json` for metadata.

## Implementation Steps

### 1) Model Loading

- Add PyTorch dependency in `app/build.gradle` and keep existing MediaPipe/Room/coroutines.
- Implement `PyTorchCtcRunner`:

```kotlin
interface ModelRunner {
    val meta: CtcModelMetadata
    fun run(sequence: Array<FloatArray>): CtcOutputs
    fun release()
}

class PyTorchCtcRunner(
    private val context: Context,
    private val assetPath: String, // e.g., "ctc/sign_transformer_ctc_mobile.ptl"
    override val meta: CtcModelMetadata
) : ModelRunner {
    private val module: org.pytorch.Module by lazy {
        org.pytorch.Module.load(assetFilePath(context, assetPath))
    }

    override fun run(sequence: Array<FloatArray>): CtcOutputs {
        val t = sequence.size
        val d = if (t > 0) sequence[0].size else 178
        val inputTensor = toInputTensor(sequence) // [1,T,178] float32
        val output = module.forward(inputTensor) as org.pytorch.Tensor
        val shape = output.shape() // [1,T,C]
        val logProbs = output.dataAsFloatArray()
        return CtcOutputs(
            logProbs = NdArray.wrap(logProbs, shape),
            catLogits = null
        )
    }

    override fun release() { /* Module is unloaded by GC; keep method for symmetry */ }
}
```

- Utility to load assets and build tensors:

```kotlin
fun assetFilePath(context: Context, assetPath: String): String {
    // Copy from assets to a readable file path if needed; or use AssetLoader helpers.
    // Existing TFLite asset utils can be adapted for ".pt".
    return AssetFileLoader.ensureOnDisk(context, assetPath)
}

fun toInputTensor(sequence: Array<FloatArray>): org.pytorch.Tensor {
    val t = sequence.size
    val d = if (t > 0) sequence[0].size else 178
    val buffer = java.nio.FloatBuffer.allocate(t * d)
    for (i in 0 until t) {
        buffer.put(sequence[i])
    }
    buffer.rewind()
    return org.pytorch.Tensor.fromBlob(buffer, longArrayOf(1L, t.toLong(), d.toLong()))
}
```

### 2) Preprocessing

- Reuse `MediaPipeProcessor` for keypoints and normalization; ensure values are clamped to [0,1].
- Reuse `SequenceBufferManager` for windowing, stride triggers, and short-gap interpolation.
- Confirm final input layout `[T,178]` = pose(25), left(21), right(21), face(22) with order matching Python.

### 3) Sliding Window

- Keep existing window size/stride from metadata (`window_size_hint`, `stride_hint`).
- On stride trigger, materialize `[T,178]` and pass into `ModelRunner.run(...)`.

### 4) Inference

- Convert `[T,178]` to `Tensor` `[1,T,178]` and call `module.forward(tensor)`.
- Expect `log_probs` `[1,T,C]` already log-softmaxed; no extra activation needed.
- Keep inference on a background thread via coroutines:

```kotlin
private val inferenceDispatcher = kotlinx.coroutines.newSingleThreadContext("pytorch-infer")

suspend fun infer(sequence: Array<FloatArray>): CtcOutputs =
    withContext(inferenceDispatcher) { modelRunner.run(sequence) }
```

### 5) Decoding (CTC Greedy)

- Implement Kotlin greedy CTC decoder equivalent to Python:

```kotlin
fun greedyCtcDecode(logProbs: FloatArray, shape: LongArray, blankId: Int): IntArray {
    // shape = [1, T, C]
    val t = shape[1].toInt()
    val c = shape[2].toInt()
    val tokens = IntArray(t)
    var last = -1
    var outSize = 0
    // Argmax per frame
    for (i in 0 until t) {
        var maxIdx = 0
        var maxVal = Float.NEGATIVE_INFINITY
        val base = i * c
        for (j in 0 until c) {
            val v = logProbs[base + j]
            if (v > maxVal) { maxVal = v; maxIdx = j }
        }
        // Collapse repeats and remove blanks
        if (maxIdx != blankId && maxIdx != last) {
            tokens[outSize++] = maxIdx
        }
        last = maxIdx
    }
    return tokens.copyOf(outSize)
}
```

- Reuse existing `CtcAggregator` and `TemporalRecognizer` for cross-window aggregation and stability.

### 6) UI Updates

- Keep existing XML UI and `OverlayView`. No changes needed beyond surfacing model name/source (TFLite vs PyTorch) and any additional per-frame categories if available.
- Add an option in `MainActivity` or Settings to switch between runners for A/B testing.

## Validation Plan

- **Golden sample parity**
  - Export 3–5 representative sequences `[T,178]` to JSON from Python pipeline.
  - Add an Android debug action to load these sequences and run both TFLite and PyTorch runners.
  - Compare outputs:
    - Per-frame argmax track equality ratio ≥ 99%.
    - Decoded token sequence exact match.
    - Average per-frame log-prob MSE ≤ 1e-4 (allow small numeric drift).
- **Live parity**
  - Run side-by-side (toggle runner) on the same live capture; verify same decoded transcripts and similar confidences.
- **Normalization check**
  - Assert all inputs to model ∈ [0,1]; log min/max per window in debug.
- **Performance metrics**
  - Measure end-to-end latency per window; target within ±10% of TFLite baseline or <30 ms for typical `T≤150` windows.

## Migration Notes

- **Keep both runners initially** to validate parity. Default to PyTorch once confidence is high.
- **Dependencies**: Add PyTorch Mobile (full or lite). Remove ONNX Runtime if unused. Remove TFLite after migration to reduce APK size.
- **Packaging**: Ensure `noCompress` includes `"pt"`, `"ptl"`. Place models under `assets/ctc/` with matching `.model.json`.
- **ProGuard/R8**: Add keep rules for `org.pytorch` if needed (usually minimal for TorchScript).
- **Threading**: Use a single-threaded coroutine dispatcher for inference to avoid contention; warm up the model on app start to JIT-initialize kernels.
- **Interface stability**: Keep `ModelRunner` I/O contract stable so future models (e.g., classification) can be added by implementing the same interface.

---

### Appendix: Kotlin Integration Notes

- Asset loading helpers already used for TFLite can be adapted for `.pt(l)` files.
- If category per-frame logits are present in the model, extend `CtcOutputs` to carry `[1,T,K]` and surface them in the UI if desired.
- If the PyTorch model expects lengths, wrap the model in TorchScript to accept only `[1,T,178]` and internally derive full lengths, matching our on-device usage.
