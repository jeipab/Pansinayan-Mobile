# PyTorch Model Usage Context (Web/Python)

## Model Architectures

- **SignTransformer (classification)**
  - Input: [B, T, 178] keypoint sequences (89 keypoints  2 coords: x,y)
  - Embedding  PositionalEncoding  LayerNorm  N TransformerEncoder  Pooling (mean/max/cls)
  - Heads: gloss_head: [B, E] -> [B, num_gloss], category_head: [B, E] -> [B, num_cat]
  - Typical dims: emb_dim=256, 
_heads=8, 
_layers=4, max_len ~300
  - File: models/transformer.py (class SignTransformer)

- **SignTransformerCtc (continuous/CTC)**
  - Input: [B, T, 178]
  - Same encoder backbone but no temporal pooling; per-frame output
  - Head: ctc_head: [B, T, E] -> [B, T, num_ctc_classes], then LogSoftmax
  - Typical dims: emb_dim=512, 
_layers=6, 
um_ctc_classes=105+1=106 (blank=105)
  - File: models/transformer.py (class SignTransformerCtc)

- **MediaPipeGRU (classification)**
  - Input: [B, T, 178]
  - Optional linear projection  GRU1  dropout  GRU2  dropout  final hidden
  - Heads: gloss_head and category_head from final hidden
  - Defaults: hidden1=256, hidden2=128, optional idirectional=False
  - File: models/mediapipe_gru.py (class MediaPipeGRU)

- **MediaPipeGRUCtc (continuous/CTC)**
  - Input: [B, T, 178]
  - Optional projection  (uni-GRU2 with dropout + LayerNorm)  per-frame logits
  - Heads: ctc_head: [B, T, H] -> [B, T, num_ctc_classes] (+ optional per-frame category_head)
  - Defaults (subset GREETINGS often): hidden1512, hidden2512, 
um_ctc_classes=10+1=11, lank=10
  - File: models/mediapipe_gru.py (class MediaPipeGRUCtc)

- **InceptionV3GRU (classification)**
  - Input: either raw frames [B, T, 3, 299, 299] (ImageNet norm) or precomputed features [B, T, 2048]
  - InceptionV3 (fc=Identity)  GRU2  heads
  - Defaults: hidden116, hidden212 (mobile-light), dropout=0.3
  - File: models/iv3_gru.py (class InceptionV3GRU)

- **InceptionV3GRUCtc (continuous/CTC)**
  - Input: frames or features as above; bidirectional GRUs; per-frame CTC logits
  - Optional per-frame category_head
  - Defaults (subset often): hidden1512, hidden2256, 
um_ctc_classes=11, lank=10
  - File: models/iv3_gru.py (class InceptionV3GRUCtc)

## Inference Flow

- **Model construction and checkpoint loading**
  - Checkpoints are loaded with 	orch.load(checkpoint, map_location=...).
  - State dict key is resolved in order: model_state_dict, state_dict, model, or raw dict.
  - Models are re-instantiated with inferred input_dim and class counts when possible.
  - model.load_state_dict(...), then model.eval().
  - References:
    - evaluation/prediction/predict.py (ModelPredictor for classification models)
    - evaluation/prediction/predict_ctc.py (CTCPredictor for CTC models)

- **Classification forward (Transformer/MediaPipeGRU/IV3-GRU)**
  - Inputs: [1, T, 178] (keypoints) or [1, T, 2048] (features)
  - Outputs: gloss_logits [1, num_gloss], cat_logits [1, num_cat]
  - Predictions: rgmax per head; probabilities via softmax if needed

- **CTC forward (TransformerCtc/MediaPipeGRUCtc/IV3GRUCtc)**
  - Input: [1, T, D] where D=178 (keypoints) or 2048 (features)
  - Output: log_probs [1, T, num_ctc_classes] (log-softmaxed)
  - Decoding: greedy or beam search; uses lank_id from config or model subset
  - Optional per-frame categories if checkpoint contains category_head

Example (classification, keypoints):
`python
import numpy as np, torch
from models import SignTransformer
ckpt = torch.load('trained_models/transformer/.../SignTransformer_best.pt', map_location='cpu')
model = SignTransformer(input_dim=178, num_gloss=105, num_cat=10)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.eval()
X = torch.from_numpy(np.load('path/to/clip.npz')['X']).float().unsqueeze(0)  # [1,T,178]
with torch.no_grad():
    gloss_logits, cat_logits = model(X)
    gloss_id = gloss_logits.argmax(-1).item()
    cat_id = cat_logits.argmax(-1).item()
`

Example (CTC, keypoints, greedy decoding):
`python
import numpy as np, torch
from models import SignTransformerCtc
from evaluation.ctc_utils import greedy_ctc_decoder
ckpt = torch.load('trained_models/transformer/.../SignTransformerCtc_best.pt', map_location='cpu')
model = SignTransformerCtc(input_dim=178, num_ctc_classes=106)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.eval()
X = torch.from_numpy(np.load('path/to/seq.npz')['X']).float().unsqueeze(0)  # [1,T,178]
with torch.no_grad():
    log_probs = model(X)  # [1,T,106]
    input_lengths = torch.tensor([X.shape[1]], dtype=torch.long)
    seq_ids = greedy_ctc_decoder(log_probs, blank_id=105, input_lengths=input_lengths)[0]
`

## Preprocessing

- **Keypoint extraction (MediaPipe  178-dim):**
  - Pose upper-body 252=50, Left hand 212=42, Right hand 212=42, Face minimal 222=44  total 178 dims.
  - Coordinates are normalized to [0,1] by MediaPipe; code clamps to [0,1].
  - Optional steps prior to saving/usage:
    - Visibility mask per keypoint (89) for confidence.
    - EMA smoothing (smooth_keypoints_ema) on visible points.
    - Outlier removal (alidate_and_clean_keypoints) by max inter-frame jump (default 0.3 in normalized space).
    - Linear interpolation of short gaps up to max_gap=5 frames (interpolate_gaps) for Android parity.
  - File: preprocessing/extractors/keypoints_features.py

- **InceptionV3 visual features (2048-dim):**
  - Frames are resized (299299) and ImageNet-normalized; backbone returns 2048-d per-frame.
  - File: preprocessing/extractors/iv3_features.py

- **NPZ layout:**
  - Keypoints: X with shape [T, 178]
  - Features: X2048 with shape [T, 2048]
  - Some flows support combined [T, 2204] by concatenating X and X2048 when model input_dim=2204.
  - For validation/inference, sequences may be truncated (e.g., to 300 frames in some utilities).

- **Batch/length handling:**
  - Models expect [1, T, D] on input; RNN-based models optionally take lengths=[T].
  - CTCLoss expects [T, B, C]; code uses [B, T, C] during forward and permutes when training.

## Outputs and Decoding

- **Classification models:**
  - Outputs: gloss_logits [B, num_gloss], cat_logits [B, num_cat]
  - Preds: rgmax per head; confidences via softmax
  - Label mapping: data/labels/labels_reference.csv loaded by data/labels/label_mapping.py

- **CTC models:**
  - Output: log_probs [B, T, num_ctc_classes] (already log_softmax)
  - Config: Full set 
um_gloss=105, lank_id=105, 
um_ctc_classes=106.
    - Subset GREETINGS: 
um_gloss=10, lank_id=10, 
um_ctc_classes=11.
  - Decoding:
    - Greedy: rgmax per timestep  collapse duplicates  remove blanks
    - Beam search: optional beam width (default 10)
  - Optional per-frame category_head for dual-task sequence category predictions; post-aggregation maps to sign-level categories.

## Relevant Code References

- Architectures: models/transformer.py, models/mediapipe_gru.py, models/iv3_gru.py, models/__init__.py
- CTC decoding utils: evaluation/ctc_utils.py
- Predictors: evaluation/prediction/predict.py, evaluation/prediction/predict_ctc.py
- Configs (CTC/class counts/blank ids): streamlit_app/core/config.py
- Label mapping: data/labels/label_mapping.py and data/labels/labels_reference.csv
- Preprocessing: preprocessing/extractors/keypoints_features.py, preprocessing/extractors/iv3_features.py
- Trained model usage examples: 	rained_models/TRAINED_MODEL_GUIDE.md, evaluation/prediction/PREDICTION_GUIDE.md

## Android Adaptation Notes

- **On-device inputs:**
  - Preferred: replicate keypoint pipeline and feed [1, T, 178] float32 normalized in [0,1].
  - Alternate (heavier): precompute [1, T, 2048] with InceptionV3 (not recommended for mobile).

- **Export for PyTorch Mobile:**
  - Use TorchScript (	orch.jit.trace or 	orch.jit.script) on the exact Python model class.
  - Example for Transformer-CTC:
    `python
    model = SignTransformerCtc(input_dim=178, num_ctc_classes=106)
    model.load_state_dict(torch.load('...')['model_state_dict'])
    model.eval()
    example = torch.randn(1, 150, 178)
    scripted = torch.jit.trace(model, example)  # or script(model)
    scripted.save('SignTransformerCtc_mobile.pt')
    `
  - For RNNs with optional lengths, prefer a scripted forward without the lengths arg on-device; or provide a wrapper that internally computes/uses full length.

- **Inference on Android (PyTorch Mobile):**
  - Load model: Module module = Module.load(assetFilePath('SignTransformerCtc_mobile.pt'));
  - Prepare tensor: shape [1, T, 178], dtype float32, CPU, contiguous.
  - Forward: single input (or tuple if wrapper). Output tensor shape [1, T, num_ctc] for CTC; [1, num_gloss] and [1, num_cat] for classification.

- **Preprocessing parity:**
  - Ensure MediaPipe landmark normalization to [0,1], same keypoint ordering (pose 25, left 21, right 21, face 22), same concatenation order.
  - Optional smoothing and gap interpolation: EMA smoothing and linear interpolation up to 5 frames matched by Web version.
  - Sequence truncation/padding: keep [T<=max_len]; CTC models in web support max_len up to ~1000.

- **Decoding on-device (CTC):**
  - Implement greedy decoding: rgmax per frame  collapse repeats  remove lank_id.
  - Use lank_id=105 for full model, lank_id=10 for GREETINGS subset.
  - Confidence: per-token max probability across frames of that token (approx) or average log prob for beam result.

- **Label mapping:**
  - Bundle a JSON of gloss_id -> label from labels_reference.csv for on-device display.

- **I/O specs to replicate**
  - Classification (keypoints): input [1, T, 178]  logits [1, num_gloss], [1, num_cat]
  - CTC (keypoints): input [1, T, 178]  log_probs [1, T, num_ctc] (log-softmax)
  - Feature-based variants: same with D=2048

---

Notes align with Web/Python usage and are intended to be directly portable to Android via PyTorch Mobile using TorchScript .pt models (not TFLite).