# Model Export Scripts

This directory contains Python scripts for converting PyTorch models to Android-compatible formats.

## Files

- `model_export_and_setup.py` - Main conversion script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place your PyTorch checkpoints in the models directory:

   ```
   models/checkpoints/transformer/best_model.pth
   models/checkpoints/gru/best_model.pth
   ```

3. Run conversion:

   ```bash
   python model_export_and_setup.py --model both
   ```

4. Copy converted models to assets:
   ```bash
   cp ../models/converted/*.tflite ../app/src/main/assets/
   cp ../models/converted/label_mapping.json ../app/src/main/assets/
   ```

## Directory Structure

```
scripts/
├── model_export_and_setup.py    # Main conversion script
├── requirements.txt             # Python dependencies
└── README.md                    # This file

models/
├── checkpoints/                 # Raw PyTorch models
│   ├── transformer/
│   │   ├── best_model.pth
│   │   └── config.json
│   └── gru/
│       ├── best_model.pth
│       └── config.json
└── converted/                   # Android-ready models
    ├── sign_transformer_quant.tflite
    ├── sign_mediapipe_gru_quant.tflite
    └── label_mapping.json
```

## Requirements

- Python 3.8+
- PyTorch models with compatible architecture
- Model classes available in project root (SignTransformer, MediaPipeGRU, etc.)

## Troubleshooting

- **Import errors**: Ensure model classes are available in project root
- **Checkpoint not found**: Place `.pth` files in `models/checkpoints/[model_type]/`
- **Conversion fails**: Check ONNX and TensorFlow dependencies are installed
