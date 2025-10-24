# Model Storage Directory

This directory contains PyTorch model checkpoints and converted Android models for CTC-based continuous sign language recognition.

## Structure

```
models/
├── checkpoints/                 # Raw PyTorch models (not in version control)
│   ├── transformer_ctc/
│   │   ├── best_model.pth      # Transformer CTC checkpoint
│   │   └── config.json         # Model configuration
│   └── gru_ctc/
│       ├── best_model.pth      # GRU CTC checkpoint
│       └── config.json         # Model configuration
└── converted/                   # Android-ready CTC models
    ├── sign_transformer_ctc_fp16.tflite
    ├── mediapipe_gru_ctc_fp16.tflite
    └── label_mapping.json
```

## CTC Model Specifications

- **Input Shape**: `[1, 300, 178]` (batch_size=1, sequence_length=300, features=178)
- **Output Shape**: `[1, 300, 106]` (batch_size=1, sequence_length=300, classes=105 glosses + 1 blank)
- **Quantization**: FP16 for mobile optimization
- **Keypoint Structure**: 89 keypoints × 2 coordinates = 178 features

## Usage

1. Place your PyTorch CTC checkpoints in `checkpoints/[model_type]_ctc/`
2. Run the CTC conversion script from `scripts/` directory
3. Converted CTC models will be placed in `converted/`
4. Copy converted models to `app/src/main/assets/ctc/` for runtime use

## Version Control

- **Include in git**: `converted/` directory (small converted models)
- **Exclude from git**: `checkpoints/` directory (large PyTorch files)
- Add to `.gitignore`:
  ```
  models/checkpoints/
  *.pth
  *.pt
  ```
