# CTC Model Export Pipeline

## Quick Start

1. **Check Python version**:

   ```bash
   python --version
   python3 --version
   python3.11 --version
   ```

2. **Install Python 3.11 (Linux/Mac)**:

   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev -y
   python3.11 --version
   ```

3. **Clone and setup**:

   ```bash
   git clone https://github.com/jeipab/Pansinayan-Mobile.git
   cd Pansinayan-Mobile
   python3.11 -m venv .venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r scripts/requirements.txt
   ```

4. **Place CTC checkpoints** in project root:

   ```
   Pansinayan-Mobile/
   ├── SignTransformerCtc_best.pt    # Transformer CTC checkpoint (218 MB)
   └── MediaPipeGRUCtc_best.pt       # GRU CTC checkpoint (30 MB)
   ```

   **Note**: The script looks for these files in the root folder by default.

5. **Export CTC models**:

   ```bash
   python scripts/export_ctc_models.py
   ```

   This will convert the PyTorch checkpoints and export TFLite models to `exports/ctc/`

## CTC Model Specifications

- **Input**: `[1, 300, 178]` - Keypoint sequences (89 keypoints × 2)
- **Output**: `[1, 300, 106]` - CTC logits (105 glosses + 1 blank token)
- **Blank Token**: Index 105

## Usage Options

```bash
# Basic export (uses default root folder checkpoints)
python scripts/export_ctc_models.py

# Custom model paths (if checkpoints are elsewhere)
python scripts/export_ctc_models.py --transformer path/to/model.pt --gru path/to/model.pt

# Custom sequence length
python scripts/export_ctc_models.py --sequence-length 500

# Skip verification (faster export)
python scripts/export_ctc_models.py --no-verify

# Custom output directory
python scripts/export_ctc_models.py --output-dir path/to/exports
```

## Output

Exported TFLite models are saved in `exports/ctc/`:

- `sign_transformer_ctc_fp16.tflite` (~2-4 MB)
- `sign_transformer_ctc_fp32.tflite` (~4-8 MB)
- `mediapipe_gru_ctc_fp16.tflite` (~1-2 MB)
- `mediapipe_gru_ctc_fp32.tflite` (~2-4 MB)
- `ctc_export_results.json` (export metadata and verification results)

## Android Integration

Copy CTC models to Android assets:

```bash
cp exports/ctc/*.tflite app/src/main/assets/ctc/
```

Use in Android:

```kotlin
val interpreter = CTCModelInterpreter(context)
val logits = interpreter.runInference(sequence)
val decoded = ctcDecoder.decode(logits)
```

## Pipeline Flow

```
PyTorch CTC → ONNX → TensorFlow → TFLite (FP16/FP32)
```

- **Dynamic sequences**: Variable-length input with padding
- **FP16 quantization**: Mobile optimization
- **Verification**: Shape validation and dummy inference
- **Error handling**: Multiple fallback strategies

## Troubleshooting

- **Checkpoint not found**: Ensure `SignTransformerCtc_best.pt` and `MediaPipeGRUCtc_best.pt` are in the project root folder
- **Model loading**: Ensure CTC models have correct output shape `[1, 300, 106]`
- **Dependencies**: Install all requirements from `scripts/requirements.txt`
- **Memory issues**: Use `--no-verify` for faster export or reduce batch size
- **Export fails**: Check ONNX/TensorFlow compatibility and ensure you have sufficient disk space (models are ~250 MB total)

## Note

This repository focuses on CTC-based continuous sign language recognition. The previous classification model export pipeline has been removed.
