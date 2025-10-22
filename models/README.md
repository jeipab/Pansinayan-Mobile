# Model Storage Directory

This directory contains PyTorch model checkpoints and converted Android models.

## Structure

```
models/
├── checkpoints/                 # Raw PyTorch models (not in version control)
│   ├── transformer/
│   │   ├── best_model.pth      # Transformer checkpoint
│   │   └── config.json         # Model configuration
│   └── gru/
│       ├── best_model.pth      # GRU checkpoint
│       └── config.json         # Model configuration
└── converted/                   # Android-ready models
    ├── sign_transformer_quant.tflite
    ├── sign_mediapipe_gru_quant.tflite
    └── label_mapping.json
```

## Usage

1. Place your PyTorch checkpoints in `checkpoints/[model_type]/`
2. Run the conversion script from `scripts/` directory
3. Converted models will be placed in `converted/`
4. Copy converted models to `app/src/main/assets/` for runtime use

## Version Control

- **Include in git**: `converted/` directory (small converted models)
- **Exclude from git**: `checkpoints/` directory (large PyTorch files)
- Add to `.gitignore`:
  ```
  models/checkpoints/
  *.pth
  *.pt
  ```
