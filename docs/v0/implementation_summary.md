# Pansinayan-Mobile — Enhanced Live Recognition Implementation Summary

## Implemented/Updated Files

- Updated: `app/src/main/java/com/fslr/pansinayan/recognition/RecognitionPipeline.kt`

  - Added runtime model switching via `switchModel(tflitePath, metadataPath, preferGpu)`
  - Integrated optional category head averaging and argmax per token
  - Tracked inference performance (avg time) and exposed in stats
  - Kept sliding-window gating and missing-frame ratio handling

- Updated: `app/src/main/java/com/fslr/pansinayan/activities/MainActivity.kt`

  - Wired model radio buttons to actual CTC pairs under `assets/ctc/*`
  - Calls `RecognitionPipeline.switchModel(...)`
  - Continues to display per-token confidence and category; persists history

- Updated: `app/src/main/java/com/fslr/pansinayan/utils/ModelSelector.kt`
  - Switched to CTC paths: `ctc/sign_transformer_ctc_fp16.tflite` and `ctc/mediapipe_gru_ctc_fp16.tflite`
  - Added metadata paths for each model type

No changes required (already aligned per Android context and plan):

- `inference/CtcModelRunner.kt` (dynamic T, shared buffers, optional GPU)
- `inference/CtcGreedyDecoder.kt` (python-parity confidence)
- `recognition/CtcAggregator.kt` (IoU-based aggregation)
- `inference/SequenceBufferManager.kt` (sliding window + interpolation)
- `io/NpyNpzReader.kt` (golden test reader)
- Room database and history UI

## Key Classes and Functions

- `RecognitionPipeline.switchModel(tflitePath, metadataPath, preferGpu=false)`
  - Pauses frames, releases old runner, loads new `CtcModelRunner`, rebuilds `SequenceBufferManager`, clears `CtcAggregator`, warms up, and resumes.
- `RecognitionPipeline.runCtcIfReady(currentFrame)`
  - Runs CTC, decodes tokens, averages optional category head over token spans, and emits `RecognizedSign` with `categoryId/categoryLabel` and confidence.
- `MainActivity.switchModel(tflitePath, metadataPath)`
  - Invokes pipeline switching using CTC model/metadata pairs.
- `ModelSelector.ModelType`
  - Now includes `metadataPath` for each CTC model.

## Integration Notes and Testing Outcomes

- Model switching updates window/stride from new metadata and resets buffer/aggregator to avoid cross-model contamination.
- Category head is optional; when absent, `categoryId` defaults to 0; labels are fetched via `LabelMapper`.
- Performance instrumentation: average inference time is tracked in `RecognitionPipeline` and shown in stats UI.
- Room persistence remains: each emitted token is saved with model name (`Transformer` or `GRU`) and occlusion state.
- Live UI continues to show transcript with per-token confidence and category.

Manual sanity checks performed:

- Verified radio switching loads `assets/ctc/*` pairs and resumes recognition.
- Verified tokens emit with expected labels and confidences; category populated on models with category head.

## Deviations/Improvements vs Plan

- `ModelSelector` was updated to CTC paths and metadata but is not yet used to drive UI selection; `MainActivity` directly wires known CTC pairs for simplicity.
- Legacy `inference/TFLiteModelRunner.kt` remains in tree but is unused; can be removed in a future cleanup pass.
- ONNX and Compose dependencies were not removed in this step; optional cleanup can follow after full validation.

## Next Steps (Optional)

- Finalize and run `androidTest` parity (`CtcGoldenParityTest`) against golden assets, assert ±0.02 confidence tolerance.
- Consider moving IoU/category thresholds to metadata for per-model tuning.
- Remove legacy ONNX/Compose dependencies if not needed.
