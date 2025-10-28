## PyTorch Mobile Integration Summary (PyTorch-only)

- Added `org.pytorch:pytorch_android_lite:2.3.0` and configured `aaptOptions.noCompress` for `pt`/`ptl` in `app/build.gradle`.
- Introduced `ModelRunner` interface used by both TFLite and PyTorch.
- Implemented `PyTorchModelRunner` to load `.pt/.ptl` from `assets/ctc/`, create input tensor `[1,T,178]`, and return CTC log-probabilities `[1,T,C]`.
- Added `PreprocessingUtils` (clamp/normalize/shape guard) and `CTCDecoder` wrapper (delegates to `CtcGreedyDecoder`).
- Adapted `CtcModelRunner` to implement `ModelRunner`.
- Refactored `RecognitionPipeline` to depend on `ModelRunner` and strictly use PyTorch (`ctc/SignTransformerCtc_best.pt`). Fallback removed.
- `MainActivity.switchModel(...)` accepts only `.pt/.ptl` paths now.

### File paths

- `app/src/main/java/com/fslr/pansinayan/inference/ModelRunner.kt`
- `app/src/main/java/com/fslr/pansinayan/inference/PyTorchModelRunner.kt`
- `app/src/main/java/com/fslr/pansinayan/inference/PreprocessingUtils.kt`
- `app/src/main/java/com/fslr/pansinayan/inference/CTCDecoder.kt`
- `app/src/main/java/com/fslr/pansinayan/inference/CtcModelRunner.kt` (updated)
- `app/src/main/java/com/fslr/pansinayan/recognition/RecognitionPipeline.kt` (updated)
- `app/src/main/java/com/fslr/pansinayan/activities/MainActivity.kt` (updated)

### Asset placement

- Place TorchScript models in `app/src/main/assets/ctc/`:
  - `SignTransformerCtc_best.pt` + `SignTransformerCtc_best.model.json`
  - `MediaPipeGRUCtc_best.pt` + `MediaPipeGRUCtc_best.model.json`
- Keep existing metadata JSONs:
  - `ctc/sign_transformer_ctc_fp16.model.json`
  - `ctc/mediapipe_gru_ctc_fp16.model.json`

### Developer overlay

- Existing stats overlay shows average inference time and pipeline stats from `RecognitionPipeline.getStats()`.
- Model name can be surfaced via current selection in `MainActivity` and `RecognitionPipeline` (unchanged UI wiring).

### Performance and parity checks

- Input shape `[1,T,178]` verified in runner construction; preprocessing provides clamp and optional normalization hooks.
- Greedy CTC decoding parity ensured by reusing `CtcGreedyDecoder` (log-softmax expected from model).
- Inference executed off main thread inside `RecognitionPipeline` coroutine scope.

### Quick test steps

1. Copy `.pt` files to `app/src/main/assets/ctc/SignTransformerCtc_best.pt` and `app/src/main/assets/ctc/MediaPipeGRUCtc_best.pt`.
2. Build and run; pipeline requires `SignTransformerCtc_best.pt` and will fail-fast if missing.
3. Use radio buttons to switch models; PyTorch paths are already wired in `MainActivity`.
