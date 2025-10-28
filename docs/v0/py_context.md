## Overview

Pansinayan-Web implements Continuous Sign Language Recognition using Connectionist Temporal Classification (CTC) models in PyTorch. Inputs are MediaPipe keypoints packaged as NPZ files. This is the reference for the Android implementation.

## Model Architecture Summary

- Transformer-CTC: Encoder over keypoints [B, T, 178] with CTC head producing log-probabilities [B, T, num_ctc] and a per-frame category head [B, T, num_cat].
- MediaPipe-GRU-CTC: Two-layer GRU over [B, T, 178] with CTC head [B, T, num_ctc] and a per-frame category head [B, T, num_cat].
- IV3-GRU variants exist but are not targeted for mobile and are omitted here.

## Inference Flow

- Predictor: `evaluation/prediction/predict_ctc.py` → `CTCPredictor`.
  - Initialize with `model_type` in {`transformer_ctc`, `mediapipe_gru_ctc`}, `checkpoint_path`, and device. `blank_id` auto-detected from `streamlit_app/core/config.py` (`CTC_CONFIG` or `CTC_CONFIG_SUBSET`).
  - Load NPZ `X` → [T, 178], convert to Tensor and add batch → [1, T, 178].
  - Forward pass returns:
    - `log_probs`: [B, T, num_ctc] (CTC log-probabilities).
    - `cat_logits`: [B, T, num_cat] (per-frame categories).
  - Decode methods:
    - Greedy: argmax per timestep → collapse repeats → remove blanks.
    - Beam search: keep top-k paths; collapse best path.
  - Sliding window (long sequences): process overlapping windows (window_size/stride), decode per-window, then aggregate predictions by frame position to form a final sequence.
  - Postprocessing:
    - Timestamps: uniform allocation of frames across predicted tokens.
    - Categories: per-sign category by averaging per-frame probabilities over the frames covering each predicted token.
  - Optional ground truth: compute WER, error breakdown, and temporal alignment accuracy.

Detailed parameters and configuration

- CTC configuration (from `streamlit_app/core/config.py`):
  - Full set: `CTC_CONFIG = { num_gloss_classes: 105, blank_token_id: 105, num_ctc_classes: 106, beam_width: 10, window_size: 60, window_stride: 15 }`
  - Subset (e.g., Greetings): `CTC_CONFIG_SUBSET = { num_gloss_classes: 10, blank_token_id: 10, num_ctc_classes: 11, ... }`
- Model registry: `MODEL_CONFIG['transformer_ctc']` and `MODEL_CONFIG['mediapipe_gru_ctc']` define checkpoints, input_dim (178), and that training_mode is `ctc`.
- Label mapping: `data/labels/label_mapping.py` reads `data/labels_reference.csv` for gloss/category names.

## Input/Output Specification

- Inputs
  - NPZ `X`: float32 keypoints with shape [T, 178] (89 points × x,y in [0,1]).
  - Optional `mask` [T, 89] bool for visibility (used in visualization/preprocessing).
  - Optional ground truth JSON for evaluation.
- Outputs (CTC)
  - `predicted_sequence`: List[int] (gloss IDs after CTC collapsing)
  - `predicted_labels`: List[str] (mapped from IDs via labels CSV)
  - `confidence_scores`: List[float] (per-token confidence)
  - `predicted_categories`: List[int]
  - `category_confidences`: List[float]
  - `predicted_timestamps`: List[{index, gloss, start_ms, end_ms, duration_ms}]
  - `num_predicted`: int
  - If GT provided: `wer`, `num_insertions`, `num_deletions`, `num_substitutions`, and `temporal_alignment_accuracy` when fully correct.
  - Decoding options: `decode_method` ∈ {`greedy`, `beam_search`}, `beam_width` (int).

Notes on shapes and datatypes

- Inputs: float32 tensors, batch-first; lengths passed internally as `[B]` when needed.
- Log-probs: typically `log_softmax` over classes per timestep; confidence extraction converts to probs via `exp`.

## Preprocessing/Feature Extraction

- MediaPipe keypoints: `preprocessing/extractors/keypoints_features.py` extracts pose (25), hands (21 each), face (22) → 89 points → 178-D vector per frame (float32, normalized [0,1]).
- NPZ packaging: `X` [T, 178], optional `mask` [T, 89], `timestamps_ms` [T], and `meta` (JSON). Continuous sequences can be generated/handled for validation and demo.

## Postprocessing and Output Interpretation

- Decoders: `evaluation/ctc_utils.py`
  - Greedy: `greedy_ctc_decoder(log_probs, blank_id, input_lengths)`
  - Beam: `beam_search_ctc_decoder(log_probs, blank_id, beam_width, input_lengths)`
  - Collapsing uses: remove consecutive duplicates then remove blanks (`blank_id = num_gloss`).
- Label mapping: `data/labels/label_mapping.py` loads `data/labels_reference.csv` to map gloss/category IDs to labels.
- Confidence: greedy uses max probability per predicted token; beam uses exponentiated average log-probability per token.
- Timestamps: uniform distribution of total frames over predicted tokens.
- Categories: per-sign category from per-frame logits averaged over the frames for each token.

Sliding-window aggregation details

- Windows: `[start, start+window_size)` with stride overlap; each window yields a decoded sequence and confidences.
- Aggregation: estimate a representative frame position per predicted token inside the window; group all tokens by frame position across windows, keep the most confident per position; sort positions to form the final sequence.

Error handling and robustness

- Missing keys: `predict_ctc.py` checks for required NPZ keys (`X` for 178-D). Raises clear errors if absent.
- Length mismatches: input lengths handled for decoders; positional encoding resizing is handled when loading checkpoints for Transformer-CTC.
- Ground truth mismatch: WER and alignment computed only when lengths and content allow; alignment set to 0.0 when WER > 0.

## Example Flow (Code)

```python
from pathlib import Path
from evaluation.prediction.predict_ctc import CTCPredictor

ctc = CTCPredictor(
    model_type='transformer_ctc',
    checkpoint_path='trained_models/transformer/greetings_ctc_v2/SignTransformerCtc_best.pt'
)
res = ctc.predict_sequence_sliding_window(
    npz_path=Path('seq.npz'),
    decode_method='greedy',
    window_size=120,
    stride=40,
    fps=30,
    temporal_tolerance=500,
)
# Example result (simplified):
# {
#   "predicted_sequence": [4, 1, 7],
#   "predicted_labels": ["hello", "good", "morning"],
#   "predicted_categories": [0, 0, 0],
#   "confidence_scores": [0.93, 0.89, 0.92]
# }
```

## Dependencies

- PyTorch, NumPy, pandas (labels), MediaPipe (preprocessing), Streamlit (UI), tqdm.
- Configuration: `streamlit_app/core/config.py`
  - `MODEL_CONFIG['transformer_ctc']`, `MODEL_CONFIG['mediapipe_gru_ctc']` (checkpoints, dims)
  - `CTC_CONFIG` (full set) and `CTC_CONFIG_SUBSET` (e.g., Greetings) define `blank_token_id`, `num_ctc_classes`, default `window_size`/`stride`.
- Labels: `data/labels_reference.csv`.

Performance considerations

- Greedy decoding is fast and suitable for real-time; beam search improves accuracy at the cost of latency (tunable `beam_width`).
- Sliding windows control memory and latency; typical defaults for greetings: `window_size≈120` frames, `stride≈40`.
- Use CPU/GPU automatically (device selection in predictors); ensure float32 throughout.

## Key Observations and Takeaways for Android adaptation

- Use MediaPipe on-device to generate 178-D keypoints per frame; normalize to [0,1].
- Implement greedy CTC decoding on-device (argmax → collapse → remove blanks) for speed; optionally add beam search.
- Maintain consistent CTC config: `blank_id = num_gloss`, `num_ctc_classes = num_gloss + 1` (subset vs full vocabulary).
- For long sequences, apply sliding window with overlap; aggregate decoded tokens by temporal position.
- Ship label mappings and model with matching class indices; ensure category head alignment.
- Inputs: [T, 178] float32; Outputs: sequences of gloss IDs/labels with confidences and optional categories/timestamps.

Android mapping checklist

- Input pipeline: Camera → MediaPipe Holistic → normalize to [0,1] → buffer frames (T) → [T,178] float32 → model.
- Model: TorchScript/ONNX export using the same architecture (Transformer-CTC or MediaPipe-GRU-CTC) with category head included.
- Decoding: implement greedy/beam in Kotlin/C++ (argmax, duplicate collapse, blank removal); maintain `blank_id`.
- Sliding window: maintain ring buffer of frames; run inference every stride; aggregate as described.
- Output: Map IDs to labels with an embedded labels table; emit JSON like `{ "sequence": [...], "labels": [...], "categories": [...], "confidences": [...] }`.
