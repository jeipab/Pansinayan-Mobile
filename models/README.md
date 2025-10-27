# Model Storage Directory

This directory contains model definition files for CTC-based continuous sign language recognition.

## Current Structure

PyTorch model checkpoints are stored in the **project root folder**, not in this directory:

```
Pansinayan-Mobile/
├── SignTransformerCtc_best.pt    # Transformer CTC checkpoint (218 MB)
├── MediaPipeGRUCtc_best.pt       # GRU CTC checkpoint (30 MB)
├── models/                        # Model definition files (this directory)
│   ├── transformer.py
│   └── mediapipe_gru.py
└── exports/                       # Exported TFLite models
    └── ctc/
        ├── sign_transformer_ctc_fp16.tflite
        ├── sign_transformer_ctc_fp32.tflite
        ├── mediapipe_gru_ctc_fp16.tflite
        ├── mediapipe_gru_ctc_fp32.tflite
        └── ctc_export_results.json
```

**Note**: This directory only contains the model architecture definitions (`transformer.py`, `mediapipe_gru.py`). The actual model checkpoints (`.pt` files) are stored in the project root for easier access by the export scripts.

## CTC Model Specifications

- **Input Shape**: `[1, 300, 178]` (batch_size=1, sequence_length=300, features=178)
- **Output Shape**: `[1, 300, 106]` (batch_size=1, sequence_length=300, classes=105 glosses + 1 blank)
- **Quantization**: FP16 for mobile optimization
- **Keypoint Structure**: 89 keypoints × 2 coordinates = 178 features

## Usage

1. Place your PyTorch CTC checkpoints in the **project root folder**:

   - `SignTransformerCtc_best.pt`
   - `MediaPipeGRUCtc_best.pt`

2. Run the CTC conversion script from `scripts/` directory:

   ```bash
   python scripts/export_ctc_models.py
   ```

3. Exported TFLite models will be placed in `exports/ctc/`

4. Copy exported models to `app/src/main/assets/ctc/` for Android runtime use:
   ```bash
   cp exports/ctc/*.tflite app/src/main/assets/ctc/
   ```

## Model Definitions

This directory contains the architecture definitions:

- **`transformer.py`**: Transformer-based CTC model architecture
- **`mediapipe_gru.py`**: GRU-based CTC model architecture

These files define the model structure and are loaded when converting PyTorch checkpoints to TFLite format.

## Version Control

- **Include in git**: `models/*.py` (small architecture files)
- **Exclude from git**:
  - `*.pt` files (large PyTorch checkpoints)
  - `exports/` directory (generated TFLite files)

Add to `.gitignore`:

```
*.pt
*.pth
exports/
```
